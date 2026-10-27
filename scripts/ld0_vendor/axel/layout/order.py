"""Ordre de lecture des blocs par coupe XY récursive.

À chaque niveau : si une bande horizontale vide traverse tout le groupe,
on coupe en bandes de haut en bas (un titre sur toute la largeur passe
avant les colonnes) ; sinon, si une bande verticale vide le traverse, on
coupe en colonnes de gauche à droite ; sinon, tri par (haut, gauche). Les
boîtes sont rétrécies de ``shrink`` (part de leur taille) avant la
projection, pour qu'un léger chevauchement entre blocs voisins ne bouche
pas la gouttière.
"""

from __future__ import annotations

from collections.abc import Sequence

from axel.layout.geom import Box


def _gaps(spans: Sequence[tuple[float, float]]) -> list[float]:
    """Les points de coupe : milieux des trous de la projection."""
    s = sorted(spans)
    cuts: list[float] = []
    end = s[0][1]
    for a, b in s[1:]:
        if a > end:
            cuts.append((end + a) / 2)
        end = max(end, b)
    return cuts


def _shrunk(b: Box, shrink: float) -> Box:
    dx = (b[2] - b[0]) * shrink
    dy = (b[3] - b[1]) * shrink
    return (b[0] + dx, b[1] + dy, b[2] - dx, b[3] - dy)


def xy_cut(boxes: Sequence[Box], shrink: float = 0.02) -> list[int]:
    """Les indices de ``boxes`` dans l'ordre de lecture."""
    sb = [_shrunk(b, shrink) for b in boxes]

    def rec(idx: list[int]) -> list[int]:
        if len(idx) <= 1:
            return idx
        ycuts = _gaps([(sb[i][1], sb[i][3]) for i in idx])
        if ycuts:
            return _split(idx, ycuts, axis=1)
        xcuts = _gaps([(sb[i][0], sb[i][2]) for i in idx])
        if xcuts:
            return _split(idx, xcuts, axis=0)
        return sorted(idx, key=lambda i: (sb[i][1], sb[i][0]))

    def _split(idx: list[int], cuts: list[float], axis: int) -> list[int]:
        groups: list[list[int]] = [[] for _ in range(len(cuts) + 1)]
        for i in idx:
            c = (sb[i][axis] + sb[i][axis + 2]) / 2
            k = sum(1 for t in cuts if c > t)
            groups[k].append(i)
        out: list[int] = []
        for g in groups:
            out.extend(rec(g))
        return out

    return rec(list(range(len(boxes))))
