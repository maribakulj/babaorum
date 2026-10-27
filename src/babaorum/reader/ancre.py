"""Ancre — lecteur ancré : encodeur gelé + tête CTC + décodeur d'attention léger,
décodage joint par score de préfixe CTC (Watanabe et al. 2017 ; Wick et al.
2021 pour l'écriture manuscrite) avec un « couloir » : un préfixe dont le
score CTC tombe à plus de ``corridor`` nats sous le meilleur préfixe CTC de
même longueur est coupé. Le décodeur de langue ne choisit donc qu'entre
lectures que l'image soutient. La géométrie vient de l'alignement forcé du
texte final sur le treillis CTC (positions par caractère).

Entrée : features [B, T, D] d'un encodeur gelé (jetons pré-fusion, une
colonne par 14 px, E17) et longueurs valides. Aucune dépendance à axel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

BLANK = 0  # classe blanche du CTC ; le décodeur utilise les mêmes indices
NEG = -1e30


class CTCHead(nn.Module):
    """Tête d'E17 (même architecture que ``axel.reader.ctc.Head``)."""

    def __init__(self, n_out: int, din: int = 1536, hid: int = 256, up: int = 2) -> None:
        super().__init__()
        self.up = up
        self.inp = nn.Sequential(nn.LayerNorm(din), nn.Dropout(0.1), nn.Linear(din, hid))
        self.rnn = nn.LSTM(hid, hid, num_layers=2, bidirectional=True, batch_first=True, dropout=0.1)
        self.out = nn.Linear(2 * hid, n_out)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """-> (log-probas [B, T·up, C], mémoire [B, T·up, 2·hid] pour le décodeur)."""
        z = self.inp(x).repeat_interleave(self.up, dim=1)
        n = (lengths.cpu() * self.up).long()
        packed = nn.utils.rnn.pack_padded_sequence(z, n, batch_first=True, enforce_sorted=False)
        y = nn.utils.rnn.pad_packed_sequence(self.rnn(packed)[0], batch_first=True, total_length=z.shape[1])[0]
        return self.out(y).log_softmax(-1), y


class AttnDecoder(nn.Module):
    """Décodeur autorégressif léger, attention croisée sur la mémoire de la
    tête CTC. Vocabulaire = codec CTC ; l'indice 0 (blanc) sert de BOS/EOS."""

    def __init__(self, n_out: int, dmem: int = 512, d: int = 256, layers: int = 3, heads: int = 4,
                 max_len: int = 512) -> None:
        super().__init__()
        self.emb = nn.Embedding(n_out, d)
        self.pos = nn.Embedding(max_len, d)
        self.mem = nn.Linear(dmem, d)
        layer = nn.TransformerDecoderLayer(d, heads, 4 * d, dropout=0.1, batch_first=True, norm_first=True)
        self.dec = nn.TransformerDecoder(layer, layers)
        self.out = nn.Linear(d, n_out)

    def forward(self, prev: torch.Tensor, memory: torch.Tensor, mem_pad: torch.Tensor) -> torch.Tensor:
        """prev [B, U] (BOS en tête) -> log-probas [B, U, C] du jeton suivant."""
        u = prev.shape[1]
        h = self.emb(prev) + self.pos(torch.arange(u, device=prev.device))[None]
        causal = torch.triu(torch.ones(u, u, dtype=torch.bool, device=prev.device), 1)
        y = self.dec(h, self.mem(memory), tgt_mask=causal, memory_key_padding_mask=mem_pad)
        return self.out(y).log_softmax(-1)


class Ancre(nn.Module):
    def __init__(self, n_out: int, din: int = 1536, hid: int = 256, dec_layers: int = 3) -> None:
        super().__init__()
        self.ctc = CTCHead(n_out, din, hid)
        self.dec = AttnDecoder(n_out, 2 * hid, layers=dec_layers)

    def losses(self, x: torch.Tensor, lengths: torch.Tensor, targets: list[list[int]]) -> dict[str, torch.Tensor]:
        dev = x.device
        logp, mem = self.ctc(x, lengths)
        tl = lengths.to(dev) * self.ctc.up
        flat = torch.tensor([c for t in targets for c in t], dtype=torch.long, device=dev)
        tlen = torch.tensor([len(t) for t in targets], dtype=torch.long, device=dev)
        ctc = nn.functional.ctc_loss(logp.transpose(0, 1).float(), flat, tl, tlen, blank=BLANK,
                                     reduction="mean", zero_infinity=True)
        u = max(len(t) for t in targets) + 1
        prev = torch.zeros(len(targets), u, dtype=torch.long)
        gold = torch.full((len(targets), u), -100, dtype=torch.long)
        for i, t in enumerate(targets):
            prev[i, 1 : len(t) + 1] = torch.tensor(t, dtype=torch.long)
            gold[i, : len(t)] = torch.tensor(t, dtype=torch.long)
            gold[i, len(t)] = BLANK  # EOS
        prev, gold = prev.to(dev), gold.to(dev)
        pad = torch.arange(logp.shape[1], device=dev)[None] >= tl[:, None]
        att = nn.functional.nll_loss(self.dec(prev, mem, pad).transpose(1, 2), gold, ignore_index=-100)
        return {"ctc": ctc, "att": att}


# -- score de préfixe CTC ------------------------------------------------------


@dataclass
class CTCPrefixState:
    """Variables avant r^n_t(g), r^b_t(g) du préfixe g (log), sur tous les t."""

    rn: np.ndarray
    rb: np.ndarray
    last: int
    score: float  # log P(g... | x) : préfixe g suivi de n'importe quoi


def ctc_prefix_init(lp: np.ndarray) -> CTCPrefixState:
    t = lp.shape[0]
    rn = np.full(t, NEG)
    rb = np.cumsum(lp[:, BLANK])
    return CTCPrefixState(rn, rb, -1, 0.0)


def ctc_prefix_extend(lp: np.ndarray, st: CTCPrefixState, c: int) -> CTCPrefixState:
    """Score de préfixe de g+c (Watanabe 2017, algorithme 2), O(T)."""
    T = lp.shape[0]
    rn = np.full(T, NEG)
    rb = np.full(T, NEG)
    prev_tot = np.logaddexp(st.rn, st.rb)
    # phi_t = r_t(g) sauf si c répète le dernier caractère : alors seul r^b
    phi = st.rb.copy() if c == st.last else prev_tot
    start = st.last < 0
    rn[0] = lp[0, c] if start else NEG
    psi = rn[0]
    for t in range(1, T):
        rn[t] = np.logaddexp(rn[t - 1], phi[t - 1]) + lp[t, c]
        rb[t] = np.logaddexp(rn[t - 1], rb[t - 1]) + lp[t, BLANK]
        psi = np.logaddexp(psi, phi[t - 1] + lp[t, c])
    return CTCPrefixState(rn, rb, c, float(psi))


def ctc_prefix_extend_many(lp: np.ndarray, sts: list[CTCPrefixState], cs: list[int]) -> list[CTCPrefixState]:
    """:func:`ctc_prefix_extend` pour N couples (état, caractère) à la fois :
    même récurrence, vectorisée sur les candidats (boucle sur T seulement)."""
    if not sts:
        return []
    T = lp.shape[0]
    n = len(sts)
    c = np.asarray(cs)
    last = np.asarray([s.last for s in sts])
    prn = np.stack([s.rn for s in sts])  # [N, T]
    prb = np.stack([s.rb for s in sts])
    phi = np.where((c == last)[:, None], prb, np.logaddexp(prn, prb))
    xc = lp[:, c].T  # [N, T]
    xb = lp[:, BLANK]
    rn = np.full((n, T), NEG)
    rb = np.full((n, T), NEG)
    rn[:, 0] = np.where(last < 0, xc[:, 0], NEG)
    psi = rn[:, 0].copy()
    for t in range(1, T):
        rn[:, t] = np.logaddexp(rn[:, t - 1], phi[:, t - 1]) + xc[:, t]
        rb[:, t] = np.logaddexp(rn[:, t - 1], rb[:, t - 1]) + xb[t]
        psi = np.logaddexp(psi, phi[:, t - 1] + xc[:, t])
    return [CTCPrefixState(rn[i], rb[i], int(c[i]), float(psi[i])) for i in range(n)]


def ctc_full(st: CTCPrefixState) -> float:
    """log P(g | x) exact pour le préfixe complet (fin de ligne)."""
    return float(np.logaddexp(st.rn[-1], st.rb[-1]))


# -- décodage joint ancré ------------------------------------------------------


def greedy_ids(lp: np.ndarray) -> list[int]:
    """Glouton CTC replié : argmax par trame, répétitions et blancs retirés."""
    out: list[int] = []
    prev = BLANK
    for c in lp.argmax(1):
        c = int(c)
        if c != prev and c != BLANK:
            out.append(c)
        prev = c
    return out


def ctc_of(lp: np.ndarray, ids: list[int]) -> float:
    """log P(ids | x) exact (score CTC complet d'une suite de classes)."""
    st = ctc_prefix_init(lp)
    for c in ids:
        st = ctc_prefix_extend(lp, st, c)
    return ctc_full(st)


def end_ok(fc: float, witness: float, delta: float) -> bool:
    """Porte des fins : admise ssi le score CTC complet vaut au moins témoin − Δ."""
    return fc >= witness - delta


@dataclass
class Hyp:
    ids: tuple[int, ...]
    ctc: CTCPrefixState
    att: float
    score: float = 0.0
    done: bool = False
    final_ctc: float = field(default=NEG)


@torch.no_grad()
def joint_decode(model: Ancre, x: torch.Tensor, length: int, alpha: float = 0.5, beam: int = 8,
                 topk: int = 8, corridor: float = 15.0, max_len: int | None = None,
                 lm=None, lm_weight: float = 0.0) -> list[Hyp]:
    """Une ligne. score = α·CTC + (1−α)·attention (+ λ·LM externe).
    ``corridor`` (Δ) : coupe de recherche sur les préfixes vivants, et porte des fins.
    Une fin n'est admise que si son score CTC complet vaut au moins témoin − Δ ; le
    témoin est le glouton CTC replié, relevé par la meilleure fin admise. Aucune fin
    admise : liste vide (abstention)."""
    model.eval()
    lengths = torch.tensor([length])
    logp, mem = model.ctc(x[:, :length], lengths)
    lp = logp[0].double().cpu().numpy()
    T = lp.shape[0]
    max_len = max_len or T // 2 + 2
    dev = x.device
    pad = torch.zeros(1, T, dtype=torch.bool, device=dev)
    witness = ctc_of(lp, greedy_ids(lp))
    hyps = [Hyp((), ctc_prefix_init(lp), 0.0)]
    finished: list[Hyp] = []
    for _ in range(max_len):
        prev = torch.tensor([[BLANK] + list(h.ids) for h in hyps], device=dev)
        att = model.dec(prev, mem.expand(len(hyps), -1, -1), pad.expand(len(hyps), -1))[:, -1].double().cpu().numpy()
        ends: list[tuple[Hyp, float]] = []
        pend: list[tuple[Hyp, int, float]] = []
        for h, a in zip(hyps, att):
            for c in np.argsort(-a)[:topk]:
                c = int(c)
                if c == BLANK:  # fin : admise plus bas, par la porte
                    ends.append((h, float(a[c])))
                else:
                    pend.append((h, c, float(a[c])))
        sts = ctc_prefix_extend_many(lp, [p[0].ctc for p in pend], [p[1] for p in pend])
        cand: list[Hyp] = []
        for (h, c, ac), st in zip(pend, sts):
            s = alpha * st.score + (1 - alpha) * (h.att + ac)
            if lm is not None:
                s += lm_weight * lm(h.ids, c)
            cand.append(Hyp(h.ids + (c,), st, h.att + ac, s))
        if cand:
            best_ctc = max(h.ctc.score for h in cand)
            cand = [h for h in cand if h.ctc.score >= best_ctc - corridor]
        for h, ac in ends:
            fc = ctc_full(h.ctc)
            if end_ok(fc, witness, corridor):
                s = alpha * fc + (1 - alpha) * (h.att + ac)
                if lm is not None:
                    s += lm_weight * lm(h.ids, None)
                finished.append(Hyp(h.ids, h.ctc, h.att + ac, s, True, fc))
        if finished:  # témoin = max(témoin, meilleure fin admise) ; retrait sous le seuil
            witness = max(witness, max(f.final_ctc for f in finished))
            finished = [f for f in finished if end_ok(f.final_ctc, witness, corridor)]
        cand.sort(key=lambda h: -h.score)
        hyps = cand[:beam]
        if not hyps or (finished and max(f.score for f in finished) > hyps[0].score):
            break
    return sorted(finished, key=lambda h: -h.score)


# -- géométrie : alignement forcé ------------------------------------------------


def forced_align(lp: np.ndarray, ids: list[int]) -> list[tuple[int, int, float]]:
    """Viterbi CTC : pour chaque caractère, (premier pas, dernier pas, log-proba
    moyenne de ses pas). ``[]`` si le texte est infaisable pour T."""
    T = lp.shape[0]
    ext = [BLANK]
    for c in ids:
        ext += [c, BLANK]
    S = len(ext)
    if not ids:
        return []
    dp = np.full((T, S), NEG)
    bp = np.zeros((T, S), dtype=np.int64)
    dp[0, 0] = lp[0, BLANK]
    dp[0, 1] = lp[0, ext[1]]
    for t in range(1, T):
        for s in range(S):
            best, arg = dp[t - 1, s], s
            if s >= 1 and dp[t - 1, s - 1] > best:
                best, arg = dp[t - 1, s - 1], s - 1
            if s >= 2 and ext[s] != BLANK and ext[s] != ext[s - 2] and dp[t - 1, s - 2] > best:
                best, arg = dp[t - 1, s - 2], s - 2
            dp[t, s] = best + lp[t, ext[s]]
            bp[t, s] = arg
    s = S - 1 if dp[T - 1, S - 1] >= dp[T - 1, S - 2] else S - 2
    if dp[T - 1, s] <= NEG / 2:
        return []
    path = [s]
    for t in range(T - 1, 0, -1):
        s = bp[t, s]
        path.append(s)
    path.reverse()
    out: list[list] = [[None, None, []] for _ in ids]
    for t, s in enumerate(path):
        if s % 2 == 1:
            k = s // 2
            if out[k][0] is None:
                out[k][0] = t
            out[k][1] = t
            out[k][2].append(lp[t, ext[s]])
    return [(a, b, float(np.mean(v))) for a, b, v in out]


def readability_margin(lp: np.ndarray) -> float:
    """Marge moyenne (top1 − top2, en nats) sur les pas non blancs : une
    image illisible donne un treillis plat (E16, E18). −inf si aucune trame non
    blanche en tête (point 8 du lot 2a) : la porte « marge ≥ θ » l'écarte."""
    srt = np.sort(lp, axis=1)
    keep = lp.argmax(1) != BLANK
    if not keep.any():  # rien à lire : abstention avant décodage
        return -math.inf
    return float((srt[keep, -1] - srt[keep, -2]).mean())


# -- faisceau joint sur plusieurs treillis CTC (encodeurs différents) -------------


@dataclass
class MultiHyp:
    text: str
    states: list[CTCPrefixState]
    lm: float
    score: float = 0.0
    done: bool = False
    ctc: float = field(default=NEG)


def _extend_ids(lp: np.ndarray, sts: list[CTCPrefixState], seqs: list[list[int]]) -> list[CTCPrefixState]:
    """Étend chaque état par une suite d'indices (un caractère du texte peut
    valoir plusieurs classes dans un codec : é = e + accent en NFD)."""
    out = list(sts)
    depth = max((len(s) for s in seqs), default=0)
    for d in range(depth):
        idx = [i for i, s in enumerate(seqs) if len(s) > d]
        new = ctc_prefix_extend_many(lp, [out[i] for i in idx], [seqs[i][d] for i in idx])
        for i, st in zip(idx, new):
            out[i] = st
    return out


def _multi_full(lattices: list, text: str) -> float | None:
    """Σ w_i·log P_i(text) (score CTC complet pondéré) ; None si un caractère
    n'est pas écrivable par un des codecs."""
    total = 0.0
    for lp, w, enc in lattices:
        st = ctc_prefix_init(lp)
        for ch in text:
            seq = enc(ch)
            if not seq:
                return None
            st = _extend_ids(lp, [st], [seq])[0]
        total += w * ctc_full(st)
    return total


def _multi_witness(lattices: list, candidates: list[str]) -> float:
    """Témoin multi : meilleure Σ w·ctc_full parmi les gloutons candidats (une
    chaîne commune). Un glouton dont un id n'a pas de caractère simple est écarté ;
    le texte vide reste toujours candidat."""
    texts = {""}
    for lp, _, enc in lattices:
        inv = {}
        for c in candidates:
            seq = enc(c)
            if seq and len(seq) == 1:
                inv[seq[0]] = c
        ids = greedy_ids(lp)
        if all(k in inv for k in ids):
            texts.add("".join(inv[k] for k in ids))
    return max(v for v in (_multi_full(lattices, t) for t in texts) if v is not None)


def multi_ctc_beam(lattices: list[tuple[np.ndarray, float, "callable"]], candidates: list[str],
                   lm=None, lm_weight: float = 0.0, beta: float = 0.0, beam: int = 8,
                   corridor: float = 20.0, max_len: int = 400) -> list[MultiHyp]:
    """Faisceau synchrone sur les caractères du texte. ``lattices`` : (log-probas
    [T_i, C_i], poids w_i, encode_i(car) -> liste d'indices ou None). Score d'un
    préfixe g : Σ w_i·log P_i(g…) + λ·log P_LM(g) + β·|g| ; fin : Σ w_i·log P_i(g).
    Un caractère qu'un codec ne sait pas écrire rend ce préfixe impossible pour
    lui (−∞) : rien n'est écrit qu'aucun treillis n'appuie. ``corridor`` (Δ) : coupe
    de recherche sur les préfixes vivants, et porte des fins : une fin n'est admise
    que si Σ w·ctc_full vaut au moins témoin − Δ ; témoin = meilleur glouton candidat,
    relevé par la meilleure fin admise. Aucune fin admise : liste vide."""
    root = MultiHyp("", [ctc_prefix_init(lp) for lp, _, _ in lattices], 0.0)
    witness = _multi_witness(lattices, candidates)
    hyps, finished = [root], []
    for _ in range(max_len):
        # fins de l'étape : porte à score complet, témoin courant
        for h in hyps:
            full = sum(w * ctc_full(st) for st, (_, w, _) in zip(h.states, lattices))
            if end_ok(full, witness, corridor):
                lmv = h.lm + (lm(h.text, None) if lm is not None else 0.0)
                finished.append(MultiHyp(h.text, h.states, lmv, full + lm_weight * lmv + beta * len(h.text),
                                         True, full))
        if finished:  # témoin = max(témoin, meilleure fin admise) ; retrait sous le seuil
            witness = max(witness, max(f.ctc for f in finished))
            finished = [f for f in finished if end_ok(f.ctc, witness, corridor)]
        pairs = [(h, c) for h in hyps for c in candidates]
        new_states: list[list[CTCPrefixState | None]] = [[None] * len(lattices) for _ in pairs]
        for li, (lp, _, enc) in enumerate(lattices):
            ok = [(k, enc(c)) for k, (_, c) in enumerate(pairs)]
            ok = [(k, s) for k, s in ok if s]
            ext = _extend_ids(lp, [pairs[k][0].states[li] for k, _ in ok], [s for _, s in ok])
            for (k, _), st in zip(ok, ext):
                new_states[k][li] = st
        live = []
        for (h, c), sts in zip(pairs, new_states):
            if any(s is None for s in sts):
                continue
            vis = sum(w * st.score for st, (_, w, _) in zip(sts, lattices))
            lmv = h.lm + (lm(h.text, c) if lm is not None else 0.0)
            live.append((vis, MultiHyp(h.text + c, sts, lmv, vis + lm_weight * lmv + beta * (len(h.text) + 1))))
        if not live:
            break
        best_vis = max(v for v, _ in live)
        live = [h for v, h in live if v >= best_vis - corridor]
        live.sort(key=lambda h: -h.score)
        hyps = live[:beam]
        if finished and max(f.score for f in finished) >= hyps[0].score + 5.0 * max(1.0, beta):
            break
    return sorted(finished, key=lambda h: -h.score)
