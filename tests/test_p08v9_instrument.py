"""Unit tests of the instrument scripts (NOTE-B-P08-instrument-E27 v2.8) on synthetic data only: no real draw, planche or annotation."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S = _load("p08v9_instrument_score")
PROTO = Path.home() / "Documents/reviews/babaorum-pipeline/instrument_E27_betting_cs_prototype.py"


# answers (rule C2)
def test_parse_answer_closed_rules():
    ans, ign = S.parse_answer('{"P1": [1, 2], "P2": [3], "P3": "illisible", "X": 1}', 3, 4)
    assert ans == {1: frozenset({1, 2}), 2: frozenset({3}), 3: None} and ign == ["X"]
    assert S.parse_answer("pas du json", 2, 3)[0] == {1: None, 2: None}
    assert S.parse_answer('["P1"]', 1, 3)[0] == {1: None}
    ans, _ = S.parse_answer('{"P1": [1], "P1": [2], "P2": [2]}', 2, 3)      # repeated key -> that P unreadable
    assert ans == {1: None, 2: frozenset({2})}
    for bad in ('{"P1": null}', '{"P1": [9]}', '{"P1": [0]}', '{"P1": [1, 1]}', '{"P1": "1"}', '{"P1": [true]}', '{"P2": [1]}'):
        assert S.parse_answer(bad, 1, 3)[0] == {1: None}, bad


# statuses under the two extreme attributions
def test_split_merge_and_unreadable_extremes():
    assert S.judge_status({1: frozenset({1}), 2: frozenset({1})}, [1], 3) == {1: True}            # split: 2 P read V1
    assert S.judge_status({1: frozenset({1, 2})}, [1, 2], 3) == {1: True, 2: True}                # merge
    assert S.judge_status({1: frozenset({1}), 2: frozenset({2})}, [1, 2], 3) == {1: False, 2: False}   # clean 1:1
    assert S.judge_status({1: frozenset({1, 3})}, [1], 3) == {1: True}                            # merge with a V outside the group
    # an unreadable P only makes undetermined the V whose status depends on it (pJ 2, v2.2) ...
    st = S.judge_status({1: frozenset({1}), 2: None}, [1, 2], 3)
    assert st == {1: None, 2: None}
    st = S.judge_status({1: frozenset({1}), 2: frozenset({1}), 3: None}, [1], 3)
    assert st == {1: True}                                                                       # already split either way
    # ... and the full attribution covers V lines outside the group (pJ, v2.3)
    assert S.judge_status({1: frozenset({1}), 2: None}, [1], 3)[1] is None


def test_group_scores_bounds_and_indeterminate():
    sc = S.group_scores({1: True, 2: False, 3: None}, {1: True, 2: False, 3: True}, {1: 10, 2: 30, 3: 60})
    assert sc["e_plus"] == pytest.approx(1 - 10 / 100) and sc["e_minus"] == pytest.approx(1 - 70 / 100) and sc["u"] == pytest.approx(0.6)
    sc = S.group_scores({1: True}, {1: False}, {1: 5})                                            # disagreement -> indeterminate
    assert sc["verdict"] == {1: "indeterminate"} and sc["e_minus"] == 0 and sc["e_plus"] == 1


def test_agreement_and_kappa():
    assert S.agreement([(True, True), (False, False)]) == {"n": 2, "raw": 1.0, "kappa": 1.0}
    r = S.agreement([(True, False), (False, True)])
    assert r["raw"] == 0 and r["kappa"] < 0


# the decision sequence is the audited prototype
def test_cs_is_identical_to_the_audited_prototype():
    spec = importlib.util.spec_from_file_location("proto", PROTO)
    proto = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(proto)
    rng = np.random.default_rng(3)
    for x in ([0.14] * 42, [0.0] * 84, list(rng.random(42)), list(rng.beta(1, 6, 84))):
        assert S.cs(x) == tuple(float(v) for v in proto.cs(x))


def test_decision_branches():
    assert S.decide([0.14] * 42, [0.14] * 42, 1)["outcome"] == "étape 2"                         # pT counterexample: no false acceptance
    assert S.decide([0.0] * 42, [0.0] * 42, 1)["outcome"] == "étape 2"                           # acceptance impossible at 42
    assert S.decide([0.0] * 84, [0.0] * 84, 2)["accept"]
    assert S.decide([1.0] * 42, [1.0] * 42, 1)["reject"]
    r = S.decide([1.0] * 4 + [0.0] * 80, [1.0] * 4 + [0.0] * 80, 2)                           # empty set -> neither
    assert np.isnan(r["cs_e_plus"][0]) and r["outcome"].startswith("instrument non validé")
