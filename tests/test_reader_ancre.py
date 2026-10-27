import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from babaorum.reader.ancre import (Ancre, ctc_full, ctc_of, ctc_prefix_extend, ctc_prefix_init,
                           end_ok, forced_align, greedy_ids, joint_decode, multi_ctc_beam,
                           readability_margin)


def rand_lp(T=12, C=5, seed=0):
    g = np.random.default_rng(seed)
    z = g.normal(size=(T, C)) * 2
    return z - np.logaddexp.reduce(z, axis=1, keepdims=True)


def prefix(lp, ids):
    st = ctc_prefix_init(lp)
    for c in ids:
        st = ctc_prefix_extend(lp, st, c)
    return st


def test_full_matches_torch():
    lp = rand_lp()
    for ids in ([1], [1, 2], [2, 2], [1, 3, 3, 4]):
        ref = -torch.nn.functional.ctc_loss(torch.tensor(lp)[:, None], torch.tensor([ids]),
                                            torch.tensor([lp.shape[0]]), torch.tensor([len(ids)]),
                                            reduction="sum").item()
        assert abs(ctc_full(prefix(lp, ids)) - ref) < 1e-6


def test_prefix_mass_conservation():
    lp = rand_lp(T=8, C=4, seed=3)
    for g in ([], [1], [2, 2], [1, 3]):
        st = prefix(lp, g)
        total = np.logaddexp.reduce([ctc_prefix_extend(lp, st, c).score for c in range(1, 4)]
                                    + [ctc_full(st)] if g else
                                    [ctc_prefix_extend(lp, st, c).score for c in range(1, 4)]
                                    + [np.sum(lp[:, 0])])
        assert abs(total - (st.score if g else 0.0)) < 1e-6


def test_forced_align_monotone():
    lp = rand_lp(T=20, C=5, seed=1)
    sp = forced_align(lp, [1, 2, 2, 3])
    assert len(sp) == 4 and all(a <= b for a, b, _ in sp)
    assert all(sp[i][1] < sp[i + 1][0] for i in range(3))


def test_model_trains_and_decodes():
    torch.manual_seed(0)
    m = Ancre(n_out=6, din=16, hid=16, dec_layers=1)
    x = torch.randn(3, 10, 16)
    L = torch.tensor([10, 8, 6])
    tg = [[1, 2, 3], [4, 4], [5]]
    opt = torch.optim.Adam(m.parameters(), 3e-3)
    first = None
    for _ in range(150):
        loss = m.losses(x, L, tg)
        tot = loss["ctc"] + loss["att"]
        first = first or tot.item()
        opt.zero_grad(); tot.backward(); opt.step()
    assert tot.item() < 0.3 * first
    out = joint_decode(m, x[:1], 10, alpha=0.5, beam=4, topk=4)
    assert list(out[0].ids) == [1, 2, 3]
    out = joint_decode(m, x[1:2], 8, alpha=0.5, beam=4, topk=4)
    assert list(out[0].ids) == [4, 4]


def test_extend_many_matches_single():
    from babaorum.reader.ancre import ctc_prefix_extend_many
    lp = rand_lp(T=15, C=5, seed=7)
    base = [prefix(lp, g) for g in ([], [1], [2, 2], [3])]
    cs = [1, 2, 2, 4]
    many = ctc_prefix_extend_many(lp, base, cs)
    for b, c, m in zip(base, cs, many):
        one = ctc_prefix_extend(lp, b, c)
        assert abs(one.score - m.score) < 1e-9 and np.allclose(one.rn, m.rn) and np.allclose(one.rb, m.rb)


def toy_lp(rows):
    """Log-probas [T, 3] : classes 0 = blanc, 1 = a, 2 = b."""
    return np.log(np.array(rows, dtype=np.float64))


ENC = {"a": [1], "b": [2]}


def test_contre_exemple_deux_trames_multi():
    # Δ = 1 : « a » (log P = −3,93) ne doit pas sortir, « ab » (−0,04) doit sortir.
    eps = 0.01
    lp = toy_lp([[eps, 1 - 2 * eps, eps], [eps, eps, 1 - 2 * eps]])
    out = multi_ctc_beam([(lp, 1.0, ENC.get)], ["a", "b"], beam=8, corridor=1.0, max_len=3)
    texts = [h.text for h in out]
    assert texts and texts[0] == "ab" and "a" not in texts


class ToyModel(torch.nn.Module):
    """Modèle jouet : CTC fixe, langue dont l'EOS (classe 0) domine après un caractère."""

    def __init__(self, lp, eta=0.0005):
        super().__init__()
        self.lp = lp
        self.eta = eta

    def ctc(self, x, lengths):
        return torch.from_numpy(self.lp)[None], torch.zeros((1, self.lp.shape[0], 1), dtype=torch.float64)

    def dec(self, prev, memory, pad):
        rows = []
        for row in prev:
            hist = tuple(row[1:].tolist())
            rows.append(np.log([self.eta, 1 - 2 * self.eta, self.eta] if not hist
                               else [1 - 2 * self.eta, self.eta, self.eta]))
        return torch.from_numpy(np.stack(rows))[:, None].expand(-1, prev.shape[1], -1)


def test_contre_exemple_deux_trames_joint():
    eps = 0.01
    lp = toy_lp([[eps, 1 - 2 * eps, eps], [eps, eps, 1 - 2 * eps]])
    out = joint_decode(ToyModel(lp), torch.zeros(1, 2, 1), 2, alpha=0.5, beam=8, topk=3,
                       corridor=1.0, max_len=3)
    assert out and list(out[0].ids) == [1, 2]
    assert all(list(h.ids) != [1] for h in out)


def test_temoin_jouet_12_trames_refus_a_delta_10():
    # Filtre des fins appelé directement : la porte par règle 0 n'est pas en cause ici.
    lp = toy_lp([[0.98, 0.015, 0.005]] * 12)
    assert greedy_ids(lp) == []
    witness = ctc_of(lp, greedy_ids(lp))
    fin = ctc_of(lp, [2, 2, 2])
    assert fin < witness - 10
    assert not end_ok(fin, witness, 10.0)


def test_marge_treillis_plat_et_trames_blanches():
    flat = toy_lp([[0.4, 0.3, 0.3]] * 16)
    assert readability_margin(flat) == -np.inf
    assert readability_margin(toy_lp([[0.1, 0.8, 0.1]] * 4)) > 0
