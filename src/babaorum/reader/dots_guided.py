"""Guidage CTC d'un décodeur libre, sous forme de « LogitsProcessor » de transformers (P0 de
docs/READER_DOTS_CTC_PLAN.txt). Aucune dépendance à transformers à l'import : l'objet est appelé par
generate() comme tout LogitsProcessor, avec (input_ids [N, L], scores [N, V]).

À chaque pas, pour chaque hypothèse n (préfixe généré g_n, texte décodé par le tokenizer) :
  - candidats = les K jetons de plus haut score LLM parmi les jetons PERMIS, plus TOUJOURS les jetons de fin
    (eos) ; tout le reste à −∞ (« partial scoring » d'ESPnet). Jetons interdits : les jetons spéciaux qui ne
    sont pas des fins et les ids hors du tokenizer (fournis par l'appelant : ils se décodent en '' et
    passeraient sans coût CTC) ;
  - candidat v : score' = (1 − λ) · score[v] + λ · Δ_v + PEN · (hors codec ajoutés par v), avec
    Δ_v = préfixe_CTC(texte(g_n + v)) − préfixe_CTC(texte(g_n)) ; fin : Δ = complet_CTC − préfixe_CTC (et
    PEN si le texte finit par un caractère UTF-8 incomplet) ; la somme des Δ le long d'une hypothèse vaut
    λ · log P_CTC(texte final) ; la pénalité hors codec est HORS du mélange λ (fixe pendant la grille) ;
  - Δ est borné en bas à FLOOR (−50) : une hypothèse dont aucun candidat ne tient dans le treillis n'est
    pas tuée (−∞ partout ferait produire n'importe quoi au faisceau), elle est très pénalisée.
En beam search, transformers 4.51.3 passe des LOG-probabilités au processeur et cumule ce qu'il rend
(_beam_search : log_softmax → logits_processor → + running_beam_scores).
Texte vu par le CTC : texte décodé, NFC, caractère UTF-8 incomplet final (U+FFFD) mis de côté (compté au jeton
suivant), caractères hors codec RETIRÉS puis blancs compactés (dans cet ordre : « a x b » → « a b »).
Scripts non latins (décision pA du 18/10) : la fumée P0 (modèle aléatoire réduit) a montré qu'avec une
pénalité de −10, émettre un caractère hors codec (arabe, CJK…) coûtait MOINS que d'expliquer un caractère du
codec au CTC, si bien que le faisceau s'en remplissait — la voie d'invention que la laisse doit fermer. Ces
jetons sont donc INTERDITS par un masque fixé d'avance. Dans les carnets Kaggle : allowed_vocab (masque de
VOCABULAIRE, pA : caractères complets du codec, plus les seuls fragments d'octets qui en composent un) ; la
pénalité PEN (= FLOOR) ne frappe plus qu'un caractère hors codec recomposé de fragments permis. non_latin_ids
(lettre de nom Unicode non « LATIN ») reste la voie de la fumée locale --mini, moins stricte.
Plafond (audit pT B2) : avec max_new (le max_new_tokens passé à generate), au DERNIER pas permis chaque
candidat non-fin paie aussi complet − préfixe (et PEN s'il laisse un caractère UTF-8 incomplet) : une hypothèse
arrêtée au plafond est scorée comme une fin, à égalité avec celles qui finissent par un jeton de fin. Les
incréments peuvent être positifs (recomposition NFC, retrait d'un hors codec) : l'arrêt du faisceau de
transformers, qui suppose des scores non croissants, reste une approximation (signalée au rapport).
Limite connue : un caractère combinant arrivé seul (e puis U+0301) recompose le préfixe en NFC ; les Δ
intermédiaires sont alors approximatifs, la somme finale reste exacte.
"""

from __future__ import annotations

import math
import unicodedata

from babaorum.reader.ctc_prefix import CTCPrefixScorer

FLOOR = -50.0


def allowed_vocab(decode_one, vocab_size: int, ok_char, encode=None) -> set[int]:
    """Masque de VOCABULAIRE (pA, 18/10, remplace non_latin_ids dans les carnets) : ids permis = jetons qui se
    décodent en caractères UTF-8 COMPLETS, non vides, tous acceptés par ok_char (codec du treillis, repli,
    espace) ; plus les FRAGMENTS d'octets (décodage contenant U+FFFD) qui composent un caractère accepté, trouvés
    en tokenisant chaque caractère accepté seul et précédé d'une espace (encode : texte → ids). Sur le train
    presse, 192 jetons sur 592 091 sont des fragments, pour ⸗, Œ, ò, Ê, ì et le tilde combinant : un masque
    « caractères complets seulement » les rendrait impossibles à écrire. Tout le reste est interdit."""
    ok, pieces = set(), set()
    for i in range(vocab_size):
        try:
            t = decode_one(i)
        except Exception:
            continue
        if t and "\ufffd" not in t and all(ok_char(c) for c in t):
            ok.add(i)
    if encode is not None:
        cands = {chr(c) for c in range(0x80, 0x3000)} | {"\u0303", "\u2e17"}
        for ch in sorted(c for c in cands if ok_char(c)):
            for text in (ch, " " + ch):
                for i in encode(text):
                    if "\ufffd" in decode_one(i):
                        pieces.add(i)
    return ok | pieces


def non_latin_ids(decode_one, vocab_size: int) -> set[int]:
    """Ids dont le texte décodé contient une LETTRE (catégorie L*) de nom Unicode non « LATIN … » ; chiffres,
    ponctuation, symboles, marques combinantes et fragments UTF-8 incomplets (U+FFFD) ne sont pas jugés."""
    out = set()
    for i in range(vocab_size):
        try:
            t = decode_one(i)
        except Exception:  # id sans jeton : laissé au masque des ids hors tokenizer
            continue
        for ch in t:
            if unicodedata.category(ch).startswith("L") and not unicodedata.name(ch, "").startswith("LATIN"):
                out.add(i)
                break
    return out


class CTCGuide:
    def __init__(self, scorer: CTCPrefixScorer, decode, prompt_len: int, eos_ids: set[int], lam: float = 0.5,
                 top_k: int = 10, pen: float = FLOOR, forbidden: set[int] | None = None,
                 max_new: int | None = None):
        assert 0.0 <= lam <= 1.0 and top_k >= 1 and pen <= 0
        self.s, self.decode, self.p0 = scorer, decode, prompt_len
        self.eos = set(eos_ids)
        self.forbidden = sorted(set(forbidden or ()) - self.eos)
        self.lam, self.k, self.pen = lam, top_k, pen
        self.max_new = max_new  # plafond de generate : au dernier pas permis, toute hypothèse SE TERMINE
        self._memo: dict[str, tuple[float, int, bool]] = {}

    def ctc_text(self, text: str) -> tuple[str, int, bool]:
        """(texte gardé pour le CTC, caractères hors codec retirés, fin UTF-8 incomplète)."""
        t = unicodedata.normalize("NFC", text)
        cut = t.endswith("�")
        if cut:
            t = t[:-1]
        kept = [c for c in t if c.isspace() or self.s.labels(c) is not None]
        bad = len(t) - len(kept)
        return " ".join("".join(kept).split()), bad, cut

    def _score(self, text: str) -> tuple[float, int, bool]:
        if text not in self._memo:
            kept, bad, cut = self.ctc_text(text)
            self._memo[text] = (self.s.prefix(kept), bad, cut)
        return self._memo[text]

    def __call__(self, input_ids, scores):
        import torch

        out = torch.full_like(scores, -math.inf)
        allowed = scores.clone()
        if self.forbidden:
            allowed[:, [i for i in self.forbidden if i < scores.shape[-1]]] = -math.inf
        top = torch.topk(allowed, min(self.k, scores.shape[-1]), dim=-1).indices
        eos = [e for e in self.eos if e < scores.shape[-1]]
        for n in range(scores.shape[0]):
            gen = input_ids[n, self.p0:].tolist()
            text = self.decode(gen)
            base, bad0, cut0 = self._score(text)
            for v in set(top[n].tolist()) | set(eos):
                if not math.isfinite(allowed[n, v].item()):  # interdit ou déjà −∞
                    continue
                if v in self.eos:
                    kept, _, _ = self.ctc_text(text)
                    delta, extra = self.s.full(kept) - base, self.pen * cut0
                else:
                    text1 = self.decode(gen + [v])
                    new, bad1, cut1 = self._score(text1)
                    delta, extra = new - base, self.pen * (bad1 - bad0)
                    if self.max_new is not None and len(gen) + 1 >= self.max_new:
                        # audit pT B2 : fin au plafond = fin réelle : score complet, fragment UTF-8 final pénalisé
                        kept1, _, _ = self.ctc_text(text1)
                        delta += self.s.full(kept1) - new
                        extra += self.pen * cut1
                delta = delta if math.isfinite(delta) else FLOOR
                out[n, v] = (1 - self.lam) * scores[n, v] + self.lam * max(delta, FLOOR) + extra
        return out
