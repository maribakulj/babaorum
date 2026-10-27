import numpy as np

from babaorum.reader.ctc_prefix import CTCPrefixScorer

LABELS = ("", "a", "b", "c")


def enc(ch):
    return [LABELS.index(ch)] if ch in LABELS[1:] else None


def lattice(T, seed):
    x = np.random.default_rng(seed).normal(size=(T, len(LABELS)))
    return x - np.logaddexp.reduce(x, axis=1, keepdims=True)


def brute_full(lp, ids):
    """log P(exactement ids) par énumération de tous les chemins (petits T)."""
    import itertools

    tot = -np.inf
    for path in itertools.product(range(lp.shape[1]), repeat=lp.shape[0]):
        out, prev = [], None
        for p in path:
            if p != prev and p != 0:
                out.append(p)
            prev = p
        if tuple(out) == tuple(ids):
            tot = np.logaddexp(tot, sum(lp[t, p] for t, p in enumerate(path)))
    return tot


def test_full_matches_bruteforce():
    lp = lattice(5, 0)
    s = CTCPrefixScorer(lp, enc)
    for text in ("a", "ab", "aa", "abc", "cab"):
        assert np.isclose(s.full(text), brute_full(lp, [LABELS.index(c) for c in text]))


def test_full_matches_axel():
    from axel.image.lattice import Lattice, ctc_logprob

    lp = lattice(30, 1)
    s = CTCPrefixScorer(lp, enc)
    lat = Lattice(lp.astype(np.float32), np.arange(30, dtype=np.float32), 1.0, LABELS)
    for text in ("abcab", "aab", "cccc", "b"):
        assert np.isclose(s.full(text), ctc_logprob(lat, [LABELS.index(c) for c in text]), atol=1e-4)


def test_prefix_total_probability():
    # P(préfixe g) = P(exactement g) + Σ_c P(préfixe g·c)
    lp = lattice(12, 2)
    s = CTCPrefixScorer(lp, enc)
    for g in ("", "a", "ab", "aa", "cba"):
        rhs = np.logaddexp.reduce([s.full(g)] + [s.prefix(g + c) for c in "abc"])
        assert np.isclose(s.prefix(g), rhs)
        assert s.prefix(g) >= s.full(g) - 1e-12


def test_out_of_codec():
    s = CTCPrefixScorer(lattice(6, 3), enc)
    assert s.prefix("ax") is None and s.full("x") is None and s.prefix("") == 0.0
