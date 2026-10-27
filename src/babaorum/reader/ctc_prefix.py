"""Score de PRÉFIXE CTC pour guider un décodeur libre (docs/READER_DOTS_CTC_PLAN.txt, P0).

Pour un treillis CTC (log-probabilités [T, C], blanc = 0) et un texte g :
  préfixe(g) = log P(la sortie CTC commence par g) = log Σ_{s} P(g · s) sur toutes les suites s ;
  complet(g) = log P(la sortie CTC est exactement g) (= axel.image.lattice.ctc_logprob).
Récurrence avant de Watanabe et al. (2017, « hybrid CTC/attention », algorithme 2), en log, sur deux
vecteurs par préfixe : r_n[t] (chemins finissant par le dernier label de g au temps t) et r_b[t] (finissant
par un blanc). Étendre g par un label c coûte O(T) ; les préfixes sont gardés en cache, si bien que le
décodeur qui étend ses hypothèses jeton par jeton ne refait jamais un préfixe déjà calculé.
Texte → labels par un encodeur (encoder_for de reader_r1d_joint : caractère → liste d'indices, ou None
hors codec). Un caractère hors codec rend le préfixe NON SCORABLE (None) : la pénalité est décidée par
l'appelant, pas ici. Numpy seul. Récurrences VECTORISÉES (logaddexp.accumulate sur les cumuls ; vérifiées
contre la force brute par les tests). Les log-probabilités sont bornées à −1e4 (un −∞ du treillis, float16,
rendrait −∞ − −∞ indéfini dans les cumuls).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

NEG = -np.inf
BLANK = 0


class CTCPrefixScorer:
    """Scores de préfixe et complets d'un treillis, avec cache par suite de labels."""

    def __init__(self, logprobs: np.ndarray, enc: Callable[[str], list[int] | None]):
        self.lp = np.maximum(np.asarray(logprobs, dtype=np.float64), -1e4)
        self.T = self.lp.shape[0]
        self.enc = enc
        rb = np.cumsum(self.lp[:, BLANK])  # préfixe vide : que des blancs jusqu'à t
        self._cache: dict[tuple[int, ...], tuple[np.ndarray, np.ndarray, float]] = {
            (): (np.full(self.T, NEG), rb, 0.0)}

    def _extend(self, g: tuple[int, ...], c: int) -> tuple[np.ndarray, np.ndarray, float]:
        rn_g, rb_g, _ = self._state(g)
        lp_c, lp_b = self.lp[:, c], self.lp[:, BLANK]
        # phi[t] : masse au temps t d'où l'on peut émettre c comme NOUVEAU label après g
        phi = rb_g.copy() if (g and g[-1] == c) else np.logaddexp(rn_g, rb_g)
        rn0 = lp_c[0] if not g else NEG
        # rn[t] = logaddexp(rn[t-1], phi[t-1]) + lp_c[t] ; avec C = cumsum(lp_c), u = rn − C vérifie
        # u[t] = logaddexp(u[t-1], phi[t-1] − C[t-1]) : un logaddexp.accumulate
        C = np.cumsum(lp_c)
        x = np.empty(self.T)
        x[0] = rn0 - C[0]
        x[1:] = phi[:-1] - C[:-1]
        rn = np.logaddexp.accumulate(x) + C
        # rb[t] = logaddexp(rb[t-1], rn[t-1]) + lp_b[t], rb[0] = −∞ ; même chose avec B = cumsum(lp_b)
        B = np.cumsum(lp_b)
        y = np.empty(self.T)
        y[0] = NEG
        y[1:] = rn[:-1] - B[:-1]
        rb = np.logaddexp.accumulate(y) + B
        psi = float(np.logaddexp.reduce(np.concatenate(([rn0], phi[:-1] + lp_c[1:]))))
        return rn, rb, psi

    def _state(self, g: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray, float]:
        if g not in self._cache:
            self._cache[g] = self._extend(g[:-1], g[-1])
        return self._cache[g]

    def labels(self, text: str) -> tuple[int, ...] | None:
        ids: list[int] = []
        for ch in text:
            e = self.enc(ch)
            if e is None:
                return None
            ids += e
        return tuple(ids)

    def prefix(self, text: str) -> float | None:
        """log P(la sortie commence par text) ; None si text sort du codec ; 0 pour le texte vide."""
        g = self.labels(text)
        if g is None:
            return None
        for k in range(1, len(g) + 1):  # remplit le cache dans l'ordre (pas de récursion profonde)
            self._state(g[:k])
        return self._state(g)[2]

    def full(self, text: str) -> float | None:
        """log P(la sortie est exactement text) ; None si text sort du codec."""
        g = self.labels(text)
        if g is None:
            return None
        self.prefix(text)
        rn, rb, _ = self._state(g)
        return float(np.logaddexp(rn[-1], rb[-1]))
