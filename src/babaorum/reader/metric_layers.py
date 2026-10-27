"""Métrique en couches pour la lecture de lignes (demande pA et Marcel, 17/10/2026).

Inspirée des profils de normalisation de cinoc (cinoc/formats/text/normalization.py : normalisation
SYMÉTRIQUE, hypothèse et vérité passent par la même fonction ; table appliquée en un seul passage), sans
dépendance à cinoc. Pour une paire (hypothèse, vérité) :

- CER à chaque couche CUMULÉE (L0 → L5), distance d'édition sur somme des longueurs de vérité normalisée ;
  une ligne dont la vérité normalisée est vide sort du support (convention de axel.reader.metric). Le
  support change donc de couche en couche (lignes « — » vides dès L4, œ → oe allonge, ponctuation retirée) :
  la baisse L3 → L4 mêle normalisation et support. D'où aussi un CER à SUPPORT FIXE : mêmes lignes à toutes
  les couches (vérité non vide à L5) ;
- WER (distance d'édition sur les mots, couche L0 et couche L4) ;
- erreurs en SAC DE MOTS, indépendantes de l'ordre (couche L4) : par ligne, manquants = Σ max(0,
  n_vérité − n_hyp), inventés = Σ max(0, n_hyp − n_vérité) par mot ; taux = Σ_lignes max(manquants,
  inventés) / mots de la vérité (une substitution compte pour un, comme dans le WER ; le max est pris PAR
  LIGNE, pour que des erreurs de sens opposé sur deux lignes ne s'annulent pas) ;
- classement de chaque différence entre mots (alignement des mots de L0) : rattrapée par une couche
  (casse, accents, typographie, ponctuation, orthographe, segmentation) ou vraie erreur (substitution,
  mot manquant, mot inventé). Unité = le mot (comme le WER), pas l'événement : « aujourd'hui » contre
  « aujourd hui » compte 2. Lignes à vérité vide : hors du classement et du sac de mots, comptées à part
  (lignes, mots écrits dessus), comme score() d'axel.
Rapport à D-150 (R6/R7) : D-150 = NFC, minuscules, ß/ꝛ/œ/æ repliés, ſ → s, non-mot → espace, ACCENTS
GARDÉS ; ce n'est aucune couche ici (L4 retire en plus les accents) : D-150 est plus dur que L4.

Couches (cumulatives ; l'ordre est fixé ici et ne varie pas) :
  L0 brut        NFC, invisibles retirés, blancs compactés ;
  L1 casse       + minuscules (str.lower : casefold plierait aussi ſ → s et ß → ss, qui relèvent de L3) ;
  L2 accents     + diacritiques retirés (NFD, catégorie Mn) ;
  L3 typographie + ſ → s, ligatures dépliées (œ, æ, ﬁ, ﬂ…), apostrophes et guillemets unifiés, tirets
                   unifiés en « - » ;
  L4 ponctuation + tout caractère non-mot devient une espace, blancs compactés (comme D-150/HIPE) ;
  L5 orthographe + modernisation par une table de mots courte et explicite (MODERNISATION) : BORNE BASSE
                   partielle (une trentaine de mots), pas une modernisation complète.
La typographie passe AVANT la ponctuation (l'ordre de la demande les inverse) : sinon L4 effacerait
apostrophes et tirets et L3 ne mesurerait plus que ſ et ligatures ; dans cet ordre, une apostrophe
courbe contre droite est comptée « typographie », une virgule manquante « ponctuation ».
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field

from rapidfuzz.distance import Levenshtein

LAYERS = ("L0_brut", "L1_casse", "L2_accents", "L3_typographie", "L4_ponctuation", "L5_orthographe")

#: Typographie (appliquée en un seul passage, clé la plus longue d'abord).
TYPO: dict[str, str] = {
    "ſ": "s", "œ": "oe", "æ": "ae", "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st",
    "’": "'", "‘": "'", "ʼ": "'", "´": "'", "`": "'", "“": '"', "”": '"', "„": '"', "«": '"', "»": '"',
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "―": "-", "¬": "-", "−": "-",
    "ß": "ss", "ĳ": "ij", "ꝛ": "r", "′": "'", "‹": '"', "›": '"', "‚": '"', "‟": '"',
}
#: Modernisation orthographique (français XIXe-XXe) : paires de mots EXPLICITES, sans règle productive
#: (une règle -oit → -ait toucherait « droit », « froid » …). Appliquée mot à mot, après L4.
MODERNISATION: dict[str, str] = {
    "etoit": "etait", "etoient": "etaient", "avoit": "avait", "avoient": "avaient", "faisoit": "faisait",
    "seroit": "serait", "auroit": "aurait", "pouvoit": "pouvait", "devoit": "devait", "disoit": "disait",
    "tems": "temps", "enfans": "enfants", "parens": "parents", "presens": "presents", "momens": "moments",
    "gouvernemens": "gouvernements", "evenemens": "evenements", "sentimens": "sentiments", "loix": "lois",
    "anglois": "anglais", "hollandois": "hollandais", "connoitre": "connaitre",
    "paroitre": "paraitre", "foible": "faible", "foibles": "faibles", "foiblesse": "faiblesse",
    "connoissance": "connaissance", "connoissances": "connaissances",
}
_TYPO_RE = re.compile("|".join(re.escape(k) for k in sorted(TYPO, key=len, reverse=True)))


def _hygiene(t: str) -> str:
    t = unicodedata.normalize("NFC", t or "")
    t = "".join(c for c in t if unicodedata.category(c) not in ("Cf", "Cc") or c in "\t\n")
    return " ".join(t.split())


def _strip_diacritics(t: str) -> str:
    return unicodedata.normalize("NFC", "".join(c for c in unicodedata.normalize("NFD", t)
                                                if unicodedata.category(c) != "Mn"))


def layers(text: str) -> dict[str, str]:
    """Le texte à chaque couche cumulée (voir l'en-tête)."""
    out = {}
    t = _hygiene(text)
    out["L0_brut"] = t
    t = unicodedata.normalize("NFC", t.lower())
    out["L1_casse"] = t
    t = _strip_diacritics(t)
    out["L2_accents"] = t
    t = _TYPO_RE.sub(lambda m: TYPO[m.group(0)], t)
    out["L3_typographie"] = t
    t = " ".join(re.sub(r"[^\w]", " ", t).split())
    out["L4_ponctuation"] = t
    t = " ".join(MODERNISATION.get(w, w) for w in t.split())
    out["L5_orthographe"] = t
    return out


@dataclass
class Tally:
    """Sommes sur un ensemble de lignes (erreurs, support) par couche, et comptes de différences."""

    char: dict[str, list[int]] = field(default_factory=lambda: {k: [0, 0] for k in LAYERS})
    char_fixe: dict[str, list[int]] = field(default_factory=lambda: {k: [0, 0] for k in LAYERS})
    word: dict[str, list[int]] = field(default_factory=lambda: {"L0_brut": [0, 0], "L4_ponctuation": [0, 0]})
    bow: list[int] = field(default_factory=lambda: [0, 0, 0, 0])  # manquants, inventés, Σ max par ligne, mots vérité
    classes: Counter = field(default_factory=Counter)
    vide: dict[str, int] = field(default_factory=lambda: {"lignes_verite_vide_L0": 0, "mots_ecrits_L0": 0,
                                                          "lignes_verite_vide_L4": 0, "mots_ecrits_L4": 0})
    lines: int = 0

    def add(self, hyp: str | None, truth: str) -> None:
        h, t = layers(hyp or ""), layers(truth)
        self.lines += 1
        for k in LAYERS:
            if t[k]:
                self.char[k][0] += Levenshtein.distance(h[k], t[k])
                self.char[k][1] += len(t[k])
            if t["L5_orthographe"]:  # support fixe : mêmes lignes à toutes les couches
                self.char_fixe[k][0] += Levenshtein.distance(h[k], t[k])
                self.char_fixe[k][1] += len(t[k])
        for k in self.word:
            tw = t[k].split()
            if tw:
                self.word[k][0] += Levenshtein.distance(h[k].split(), tw)
                self.word[k][1] += len(tw)
        if t["L0_brut"]:
            self.classes.update(classify(hyp or "", truth))
        else:
            self.vide["lignes_verite_vide_L0"] += 1
            self.vide["mots_ecrits_L0"] += len(h["L0_brut"].split())
        if t["L4_ponctuation"]:
            ch, ct = Counter(h["L4_ponctuation"].split()), Counter(t["L4_ponctuation"].split())
            miss = sum(max(0, ct[w] - ch[w]) for w in ct)
            extra = sum(max(0, ch[w] - ct[w]) for w in ch)
            self.bow[0] += miss
            self.bow[1] += extra
            self.bow[2] += max(miss, extra)
            self.bow[3] += sum(ct.values())
        else:
            self.vide["lignes_verite_vide_L4"] += 1
            self.vide["mots_ecrits_L4"] += len(h["L4_ponctuation"].split())

    def report(self) -> dict:
        rate = lambda e, n: 100 * e / n if n else None  # noqa: E731
        return {"lignes": self.lines,
                "cer_pc": {k: rate(*v) for k, v in self.char.items()},
                "cer_support": {k: v[1] for k, v in self.char.items()},
                "cer_support_fixe_pc": {k: rate(*v) for k, v in self.char_fixe.items()},
                "wer_pc": {k: rate(*v) for k, v in self.word.items()},
                "sac_de_mots": {"manquants": self.bow[0], "inventes": self.bow[1], "mots_verite": self.bow[3],
                                "taux_pc": rate(self.bow[2], self.bow[3])},
                "differences": dict(sorted(self.classes.items())),
                "verite_vide": dict(self.vide)}


def _first_equal_layer(a: str, b: str) -> str | None:
    """Première couche à laquelle deux morceaux de texte deviennent égaux (None : jamais)."""
    la, lb = layers(a), layers(b)
    for k in LAYERS[1:]:
        if la[k] == lb[k]:
            return k
    return None


_CLASS_OF = {"L1_casse": "casse", "L2_accents": "accents", "L3_typographie": "typographie",
             "L4_ponctuation": "ponctuation", "L5_orthographe": "orthographe"}


def _nospace(words: list[str]) -> str:
    return layers(" ".join(words))["L5_orthographe"].replace(" ", "")


def _blocks(a: list[str], b: list[str]) -> list[tuple[int, int, int, int, bool]]:
    """Blocs (a0, a1, b0, b1, égal) : opérations non « equal » consécutives fusionnées."""
    out: list[list] = []
    for op in Levenshtein.opcodes(a, b):
        eq = op.tag == "equal"
        if not eq and out and not out[-1][4] and out[-1][1] == op.src_start and out[-1][3] == op.dest_start:
            out[-1][1], out[-1][3] = op.src_end, op.dest_end
        else:
            out.append([op.src_start, op.src_end, op.dest_start, op.dest_end, eq])
    return [tuple(x) for x in out]


def _key(w: str) -> str:
    """Clé d'alignement d'un mot de L0 : sa forme L5 ; pour un mot de ponctuation seule (vide à L4), sa
    forme L3 précédée d'un marqueur (« — » et « - » ont la même clé, « , » et « ; » non)."""
    lw = layers(w)
    return lw["L5_orthographe"] if lw["L4_ponctuation"] else "\x00" + lw["L3_typographie"]


def classify(hyp: str, truth: str) -> Counter:
    """Différences entre mots ; unité = le mot (vérité, ou hypothèse pour les inventés).

    UN SEUL alignement de la ligne, sur les clés normalisées (_key : forme L5 des mots, forme L3 marquée
    des mots de ponctuation seule) ; opérations non « equal » consécutives fusionnées en blocs.
    - Paire de clés égales mais de mots bruts différents : classée par la PREMIÈRE couche qui l'égalise
      (casse, accents, typographie, ponctuation interne, orthographe ; casse et accents ensemble comptent
      une fois, en « accents » ; « — » contre « - » = typographie).
    - Bloc inégal de n mots de vérité contre m mots lus, compté max(n, m) : « segmentation » si les mots
      (hors ponctuation seule) deviennent égaux une fois les espaces retirées à L5 (espace déplacée, trait
      d'union omis ; la casse ou les accents qui l'accompagnent y sont absorbés) ; sinon par paires dans
      l'ordre, min(n, m) paires : deux ponctuations = « ponctuation », sinon « substitution » ; surplus :
      ponctuation seule = « ponctuation », mot de vérité = « mot_manquant », mot lu = « mot_invente ».
    Chaque mot est compté au plus une fois. Si le total dépasse le WER brut de la ligne (un autre
    alignement est meilleur au niveau brut), la ligne est classée avec les MÊMES règles sur l'alignement
    des mots bruts (L0) : le total ne dépasse alors jamais le WER L0.
    """
    hw, tw = layers(hyp)["L0_brut"].split(), layers(truth)["L0_brut"].split()
    out = _classify_on(tw, hw, _key)
    if sum(out.values()) > Levenshtein.distance(hw, tw):
        out = _classify_on(tw, hw, lambda w: w)
    return out


def _classify_on(tw: list[str], hw: list[str], key) -> Counter:
    out: Counter = Counter()
    punct = lambda w: not layers(w)["L4_ponctuation"]  # noqa: E731

    def pair(a: str, b: str) -> str:
        k = _first_equal_layer(a, b)
        if k:
            return _CLASS_OF[k]
        return "ponctuation" if punct(a) and punct(b) else "substitution"

    for t0, t1, h0, h1, eq in _blocks([key(w) for w in tw], [key(w) for w in hw]):
        t_seg, h_seg = tw[t0:t1], hw[h0:h1]
        if eq:
            for a, b in zip(h_seg, t_seg):
                if a != b:
                    out[pair(a, b)] += 1
            continue
        tw_ = [w for w in t_seg if not punct(w)]
        hw_ = [w for w in h_seg if not punct(w)]
        if (tw_ and hw_ and _nospace(tw_) == _nospace(hw_)  # égaux sans espaces, mais l'espacement diffère
                and layers(" ".join(tw_))["L5_orthographe"] != layers(" ".join(hw_))["L5_orthographe"]):
            out["segmentation"] += max(len(t_seg), len(h_seg))
            continue
        for a, b in zip(h_seg, t_seg):
            out[pair(a, b)] += 1
        for w in t_seg[len(h_seg):]:
            out["ponctuation" if punct(w) else "mot_manquant"] += 1
        for w in h_seg[len(t_seg):]:
            out["ponctuation" if punct(w) else "mot_invente"] += 1
    return Counter({k: v for k, v in out.items() if v})
