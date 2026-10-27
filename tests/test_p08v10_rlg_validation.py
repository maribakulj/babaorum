"""NOTE-B-P08-oracle-E27-correction v2.3: the validation sequence and the INVERSE decision of R-LG (p1)."""
import hashlib
import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("rlg", ROOT / "scripts/p08v10_rlg_validation.py")
rlg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rlg)


def test_the_instrument_scorer_is_the_pinned_one():
    assert hashlib.sha256((ROOT / "scripts/p08v9_instrument_score.py").read_bytes()).hexdigest() == \
        "aa92b23711ccd9480a21915c83e64046621b01d1e8be30f030524bd25568d0f4"


def test_sequence_skips_tuning_groups_keeps_order_and_draws_only_outside_them():
    pop = [{"chars": c} for c in (10, 20, 30, 40, 50, 60)]
    draws = {"population": pop, "stage1": [0, 1, 1], "stage2": [3, 0, 5, 1, 3]}
    v = rlg.validation_sequence(draws, seed=7, n=200)
    assert v["stage2_kept"] == [3, 5, 3] and v["stage2_skipped"] == [0, 1]
    assert v["sequence"][:3] == [3, 5, 3] and len(v["sequence"]) == 203
    assert not set(v["extra"]) & {0, 1} and v["restricted_size"] == 4
    assert v == rlg.validation_sequence(draws, seed=7, n=200)                     # deterministic


def test_extra_draws_are_pps_on_the_restricted_population():
    pop = [{"chars": c} for c in (1000, 1, 3)]
    v = rlg.validation_sequence({"population": pop, "stage1": [0], "stage2": []}, seed=1, n=20000)
    share = np.mean(np.array(v["extra"]) == 2)
    assert abs(share - 0.75) < 0.02


def test_inverse_decision_validates_when_e27_is_almost_always_wrong():
    assert rlg.decide_inverse([1.0] * 80, [1.0] * 80)["outcome"] == "R-LG validé"
    x = [0.0 if i in (13, 40, 67) else 1.0 for i in range(80)]                    # 3 real splits: lower 0.8655
    assert rlg.decide_inverse(x, x)["validated"]
    y = [0.0 if i in (10, 30, 50, 70) else 1.0 for i in range(80)]                # 4: lower 0.8495, not validated
    d = rlg.decide_inverse(y, y)
    assert not d["validated"] and not d["rejected"] and d["outcome"] == "non conclu : note B"


def test_inverse_decision_rejects_when_e27_is_often_right_and_never_calls_decide(monkeypatch):
    monkeypatch.setattr(rlg.sc, "decide", lambda *a, **k: (_ for _ in ()).throw(AssertionError("decide must not be called")))
    x = [0.0] * 40 + [1.0] * 40
    rng = np.random.default_rng(0)
    rng.shuffle(x)
    d = rlg.decide_inverse(list(x), list(x))
    assert d["rejected"] and d["outcome"] == "R-LG rejeté : note B"


def test_bounds_use_e_minus_for_validation_and_e_plus_for_rejection():
    e_minus = [0.8] * 80                                                          # indeterminate counted for E27: low
    e_plus = [1.0] * 80
    d = rlg.decide_inverse(e_minus, e_plus)
    assert not d["validated"] and not d["rejected"]
    assert d["cs_m_plus"] == [1 - d["cs_e_minus"][1], 1 - d["cs_e_minus"][0]]


def test_an_empty_sequence_goes_to_note_b():
    x = [0.0, 0.0, 0.0] + [1.0] * 77
    assert rlg.decide_inverse(x, x)["outcome"] == "séquence vide : note B"


def test_the_pre_registered_constants_and_defaults():
    import inspect
    assert (rlg.SEED_EXTRA, rlg.N_EXTRA, rlg.THRESHOLD_E, rlg.GRID_STEP) == (20261031, 40, 0.85, 0.0005)
    sig = inspect.signature(rlg.validation_sequence).parameters
    assert sig["seed"].default == 20261031 and sig["n"].default == 40


def test_default_seed_regression_on_a_synthetic_population():
    pop = [{"chars": c} for c in range(1, 21)]
    v = rlg.validation_sequence({"population": pop, "stage1": [0, 1], "stage2": []})
    w = np.arange(3, 21, dtype=float)
    expected = [int(j) + 2 for j in np.random.default_rng(20261031).choice(18, size=40, replace=True, p=w / w.sum())]
    assert v["extra"] == expected and len(v["extra"]) == 40


def test_m_minus_bounds_and_the_anomaly_branch(monkeypatch):
    d = rlg.decide_inverse([0.95] * 80, [1.0] * 80)
    assert d["cs_m_minus"] == [1 - d["cs_e_plus"][1], 1 - d["cs_e_plus"][0]]
    seqs = iter([(0.9, 0.95), (0.5, 0.6)])                                       # crafted: lower(e-) high AND upper(e+) low
    monkeypatch.setattr(rlg.sc, "cs", lambda x: next(seqs))
    d = rlg.decide_inverse([1.0], [1.0])
    assert not d["validated"] and not d["rejected"] and d["outcome"].startswith("anomalie")


# VT-002 (pA, 24/10, before the judges): V lines with an identical polygon count as one line
def test_canonical_v_groups_identical_polygons_only():
    sq = [(0, 0), (10, 0), (10, 5), (0, 5)]
    assert rlg.canonical_v({1: sq, 2: [(0, 0), (9, 0), (9, 5), (0, 5)], 48: list(sq), 49: sq}) == {1: 1, 2: 2, 48: 1, 49: 1}


def test_without_duplicates_status_equals_the_instrument():
    rng = np.random.default_rng(3)
    for _ in range(300):
        n_v, n_p = int(rng.integers(1, 7)), int(rng.integers(1, 6))
        ans = {p: (None if rng.random() < 0.2 else frozenset(int(x) for x in rng.choice(np.arange(1, n_v + 1), size=int(rng.integers(0, n_v + 1)),
                                                                                   replace=False))) for p in range(1, n_p + 1)}
        group_v = sorted({int(x) for x in rng.choice(np.arange(1, n_v + 1), size=int(rng.integers(1, n_v + 1)), replace=False)})
        canon = {v: v for v in range(1, n_v + 1)}
        assert rlg.status_dedup(ans, group_v, canon) == rlg.sc.judge_status(ans, group_v, n_v)


def test_a_p_holding_both_copies_of_a_duplicate_is_not_a_merge():
    canon = {1: 1, 2: 2, 3: 2}                                                    # V2 and V3 share one polygon
    ans = {1: frozenset({1}), 2: frozenset({2, 3})}
    assert rlg.sc.judge_status(ans, [2, 3], 3) == {2: True, 3: True}             # the instrument alone would call it a merge
    assert rlg.status_dedup(ans, [2, 3], canon) == {2: False}
    ans_u = {1: frozenset({1}), 2: None}                                          # unreadable P: FULL = canonical {1, 2}
    assert rlg.status_dedup(ans_u, [2], canon) == {2: None}
