"""Tests du diagnostic D-vides (scripts/reader_dvides.py) : copie forcée du faisceau.

Cas jouet à deux trames (a puis b) : avec faisceau 1 et max_len 2, « ab » n'est atteint
que par le chemin forcé, à la dernière itération ; sa fin doit être admise dans cette
même itération, avant l'arrêt. Sans forçage, la copie doit rendre la liste de ancre.
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))
sys.path.insert(0, str(HERE / "scripts"))

from babaorum.reader.ancre import multi_ctc_beam  # noqa: E402
import reader_dvides as dv  # noqa: E402

ENC = {"a": [1], "b": [2]}


def toy_lattice():
    eps = 0.01
    lp = np.log(np.array([[eps, 1 - 2 * eps, eps], [eps, eps, 1 - 2 * eps]], dtype=np.float64))
    return [(lp, 1.0, ENC.get)]


def test_forced_complete_prefix_admitted_in_same_iteration():
    lattices = toy_lattice()
    out = dv.forced_beam(lattices, ["a", "b"], None, 0.0, 0.0, beam=1, corridor=1.0,
                         max_len=2, forced="ab")
    texts = [h.text for h in out]
    assert texts == ["ab"]


def test_unforced_copy_matches_multi_ctc_beam():
    lattices = toy_lattice()
    for beam, max_len in ((1, 2), (8, 3)):
        ref = multi_ctc_beam(lattices, ["a", "b"], beam=beam, corridor=1.0, max_len=max_len)
        cpy = dv.forced_beam(lattices, ["a", "b"], None, 0.0, 0.0, beam=beam, corridor=1.0, max_len=max_len)
        assert [(h.text, round(h.score, 9)) for h in cpy] == [(h.text, round(h.score, 9)) for h in ref]
