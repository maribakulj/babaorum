"""LOT-P08 oracles on a toy page where each oracle has a known effect (a stub character error rate, no axel)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from babaorum.p08_oracles import FORWARD, REVERSE, Group, Links, decompose_page, parts_sum_error


def cer(pred, gt):
    """Edit distance of the joined texts over the true length: ordered, simple, deterministic."""
    a, b = "\n".join(pred), "\n".join(gt)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1] / max(len(b), 1)


GT = [("g1", "alpha"), ("g2", "beta"), ("g3", "gamma"), ("g4", "delta"), ("g5", "omega")]


def test_a_perfect_output_has_all_parts_zero():
    pred = [(f"p{i}", t) for i, (_, t) in enumerate(GT, 1)]
    links = Links({f"p{i}": f"g{i}" for i in range(1, 6)}, [], [], [])
    d = decompose_page(pred, GT, links, cer)
    assert d.cer["0"] == 0 and all(abs(v) < 1e-12 for v in d.parts.values())


def test_reading_oracle_removes_only_reading_errors():
    pred = [("p1", "alpxa"), ("p2", "beta"), ("p3", "gamma"), ("p4", "delta"), ("p5", "omega")]
    links = Links({f"p{i}": f"g{i}" for i in range(1, 6)}, [], [], [])
    d = decompose_page(pred, GT, links, cer)
    assert d.parts["lecture"] == pytest.approx(1 / len("alpha\nbeta\ngamma\ndelta\nomega"))
    assert d.parts["decoupe"] == d.parts["manquees"] == d.parts["insertions"] == 0 and d.cer["4"] == 0


def test_empty_prediction_is_all_missed():
    links = Links({}, [], [g for g, _ in GT], [])
    d = decompose_page([], GT, links, cer)
    assert d.parts["manquees"] == pytest.approx(d.cer["0"]) and d.cer["0"] == 1.0 and d.cer["4"] == 0
    assert parts_sum_error(d) == pytest.approx(0, abs=1e-12)


def test_partial_line_gets_its_true_text_once_and_is_not_removed_or_reinserted():
    pred = [("p1", "alpha"), ("p2", "bet"), ("p3", "gamma"), ("p4", "delta"), ("p5", "omega")]
    links = Links({"p1": "g1", "p3": "g3", "p4": "g4", "p5": "g5"}, [], [], [], {"p2": "g2"})
    d = decompose_page(pred, GT, links, cer)
    assert d.parts["partielles"] > 0 and d.parts["lecture"] == 0 and d.parts["insertions"] == 0 and d.cer["4"] == 0


def test_a_line_already_rewritten_by_reading_is_not_treated_again_by_the_partial_oracle():
    pred = [("p1", "alpxa"), ("p2", "beta"), ("p3", "gamma"), ("p4", "delta"), ("p5", "omega")]
    links = Links({f"p{i}": f"g{i}" for i in range(1, 6)}, [], [], [], {"p1": "g1"})
    d = decompose_page(pred, GT, links, cer)
    assert d.parts["partielles"] == 0 and d.parts["lecture"] > 0  # the first oracle in the fixed order keeps it
    r = decompose_page(pred, GT, links, cer, order=REVERSE)
    assert r.parts["lecture"] == 0 and r.parts["partielles"] > 0  # in the reverse order the partial oracle comes first


def test_true_order_control_stops_when_a_true_line_is_covered_by_no_oracle():
    pred = [("p1", "alpha"), ("p2", "gamma"), ("p3", "delta"), ("p4", "omega")]
    links = Links({"p1": "g1", "p2": "g3", "p3": "g4", "p4": "g5"}, [], [], [])  # g2 is neither matched, partial, grouped nor missed
    with pytest.raises(ValueError, match="true-order control"):
        decompose_page(pred, GT, links, cer)


def test_positions_part_is_zero_when_reinserted_lines_already_sit_in_front_of_their_successor():
    pred = [("p1", "alpha"), ("p2", "gamma"), ("p3", "delta"), ("p4", "omega")]
    links = Links({"p1": "g1", "p2": "g3", "p3": "g4", "p4": "g5"}, [], ["g2"], [])
    d = decompose_page(pred, GT, links, cer)
    assert d.parts["part_positions"] == pytest.approx(0, abs=1e-12) and d.true_order_cer == 0


def test_cutting_oracle_repairs_a_fragmented_line_and_a_merged_pair():
    pred = [("p1", "alp"), ("p2", "ha"), ("p3", "beta gamma"), ("p4", "delta"), ("p5", "omega")]
    links = Links({"p4": "g4", "p5": "g5"},
                  [Group("coupure", ("p1", "p2"), ("g1",)), Group("fusion_verticale", ("p3",), ("g2", "g3"))], [], [])
    d = decompose_page(pred, GT, links, cer)
    assert d.parts["decoupe"] > 0 and d.parts["lecture"] == 0 and d.parts["manquees"] == 0 and d.cer["4"] == 0


def test_missed_line_is_inserted_after_the_counterpart_of_the_previous_true_line():
    pred = [("p1", "alpha"), ("p2", "gamma"), ("p3", "delta"), ("p4", "omega")]
    links = Links({"p1": "g1", "p2": "g3", "p3": "g4", "p4": "g5"}, [], ["g2"], [])
    d = decompose_page(pred, GT, links, cer)
    assert d.cer["after_manquees"] == 0 and d.parts["manquees"] > 0  # inserted between alpha and gamma: the exact true order


def test_two_consecutive_missed_lines_keep_the_true_order_and_a_leading_one_goes_first():
    pred = [("p1", "delta"), ("p2", "omega")]
    links = Links({"p1": "g4", "p2": "g5"}, [], ["g1", "g2", "g3"], [])
    d = decompose_page(pred, GT, links, cer)
    assert d.cer["after_manquees"] == 0


def test_insertions_are_removed_last_and_the_rest_is_the_order_residual():
    pred = [("p0", "noise noise"), ("p1", "alpha"), ("p2", "beta"), ("p3", "gamma"), ("p4", "delta"), ("p5", "omega")]
    links = Links({f"p{i}": f"g{i}" for i in range(1, 6)}, [], [], ["p0"])
    d = decompose_page(pred, GT, links, cer)
    assert d.parts["insertions"] > 0 and d.cer["4"] == 0 and d.parts["ordre"] == 0


def test_parts_sum_to_the_total_in_both_orders_and_a_split_of_missed_adds_up():
    pred = [("p1", "alpxa"), ("p2", "be"), ("p3", "ta"), ("p4", "deltz"), ("p5", "junk")]
    links = Links({"p1": "g1", "p4": "g4"}, [Group("coupure", ("p2", "p3"), ("g2",))], ["g3", "g5"], ["p5"])
    inzone = lambda g: g in ("g3",)
    for order in (FORWARD, REVERSE):
        d = decompose_page(pred, GT, links, cer, order=order, zone_of_missed=inzone)
        assert parts_sum_error(d) == pytest.approx(0, abs=1e-12)
        assert d.parts["manquees_dans_zone"] + d.parts["manquees_hors_zone"] == pytest.approx(d.parts["manquees"], abs=1e-12)


def test_part_positions_does_not_depend_on_the_order_of_the_input():
    """Reinserted lines that share a true successor are put in true order, so permuting the input lists changes nothing."""
    pred = [("p3", "gamma"), ("p5", "omega")]
    kwargs = dict(matched={"p3": "g3", "p5": "g5"}, extra=[])
    a = decompose_page(pred, GT, Links(groups=[], missed=["g1", "g2", "g4"], **kwargs), cer)
    b = decompose_page(pred, GT, Links(groups=[], missed=["g4", "g2", "g1"], **kwargs), cer)
    assert a.parts["part_positions"] == pytest.approx(b.parts["part_positions"], abs=1e-15)
    grp = [Group("coupure", ("p1", "p2"), ("g1",)), Group("fusion_verticale", ("p3",), ("g2", "g3"))]
    c = decompose_page([("p1", "alp"), ("p2", "ha"), ("p3", "beta gamma"), ("p4", "delta"), ("p5", "omega")], GT,
                       Links({"p4": "g4", "p5": "g5"}, grp, [], []), cer)
    d = decompose_page([("p1", "alp"), ("p2", "ha"), ("p3", "beta gamma"), ("p4", "delta"), ("p5", "omega")], GT,
                       Links({"p5": "g5", "p4": "g4"}, grp[::-1], [], []), cer)
    assert c.parts["part_positions"] == pytest.approx(d.parts["part_positions"], abs=1e-15)
    assert c.cer["4"] == pytest.approx(d.cer["4"], abs=1e-15)


# --- LOT-P08 v9: oracle 3 in four disjoint passes, bootstrap by groups, disagreement pages reported apart ---

from babaorum.p08_oracles import MISSED_CLASSES, bootstrap_by_groups, split_disagreements  # noqa: E402


def test_the_four_missed_passes_are_disjoint_and_sum_to_the_missed_part():
    pred = [("p1", "alpha"), ("p2", "gamma"), ("p3", "delta")]
    links = Links({"p1": "g1", "p2": "g3", "p3": "g4"}, [], ["g2", "g5"], [])
    classes = {"g1": "ecartees", "g2": "dans_zone", "g5": "sans_centre"}
    # g5 gets sans_centre, g2 dans_zone; every missed line exactly once
    d = decompose_page(pred, GT, links, cer, missed_class=classes | {"g2": "dans_zone"})
    parts = [d.parts[f"manquees_{c}"] for c in MISSED_CLASSES]
    assert abs(sum(parts) - d.parts["manquees"]) < 1e-12
    assert parts_sum_error(d) == pytest.approx(0, abs=1e-12)


def test_a_missed_line_without_a_class_is_refused():
    pred = [("p1", "alpha")]
    links = Links({"p1": "g1"}, [], ["g2"], [])
    with pytest.raises(ValueError, match="needs a class"):
        decompose_page(pred, GT, links, cer, missed_class={})


def test_reverse_order_runs_the_missed_passes_in_reverse():
    gt3 = [("g1", "alpha"), ("g2", "beta"), ("g3", "gamma")]
    pred = [("p1", "alpha")]
    links = Links({"p1": "g1"}, [], ["g2", "g3"], [])
    classes = {"g2": "hors_zone", "g3": "ecartees"}
    d = decompose_page(pred, gt3, links, cer, order=REVERSE, missed_class=classes)
    assert parts_sum_error(d) == pytest.approx(0, abs=1e-12)
    assert d.order == REVERSE


def test_bootstrap_by_groups_draws_whole_groups_and_is_seeded():
    rows = [{"page": "a1", "group": ("1", "x"), "chars": 10, "lecture": 0.1},
            {"page": "a2", "group": ("1", "x"), "chars": 10, "lecture": 0.1},
            {"page": "b1", "group": ("2", "y"), "chars": 30, "lecture": 0.5}]
    one = bootstrap_by_groups(rows, ["lecture"], draws=200, seed=7)
    two = bootstrap_by_groups(rows, ["lecture"], draws=200, seed=7)
    assert one == two and one["groups"] == 2
    lo, hi = one["ci95"]["lecture"]
    assert 0.1 <= lo <= hi <= 0.5


def test_a_disagreement_page_is_reported_apart_and_leaves_the_aggregates():
    results = {"ok": {"cer": {"0": 0.1}}, "bad": {"disagreement": "true-order control failed"}}
    kept, apart = split_disagreements(results)
    assert list(kept) == ["ok"] and apart == {"bad": "true-order control failed"}


def test_the_root_comes_from_axel_runs_parent_and_the_output_folder_is_mandatory(tmp_path):
    import os
    import subprocess
    env = {k: v for k, v in os.environ.items() if k != "AXEL_RUNS"}
    scripts = Path(__file__).resolve().parents[1] / "scripts/p08_decompose.py"
    res = subprocess.run([sys.executable, str(scripts), "--out-dir", str(tmp_path / "new")], env=env, capture_output=True, text=True)
    assert res.returncode != 0 and "AXEL_RUNS is required" in (res.stderr + res.stdout)


def test_the_disagreement_table_lists_each_page_apart_and_escapes_bars():
    from babaorum.p08_oracles import disagreement_table
    table = disagreement_table({"P2": "ordre | vrai", "P1": "x"})
    assert table.splitlines()[2].startswith("| P1 |") and "ordre / vrai" in table


def test_the_check_pages_are_the_ones_named_by_the_request():
    import random
    from babaorum.p08_oracles import CHECK_PAGES
    tirage = Path.home() / "Documents/reviews/babaorum-pipeline/rappel_pages/tirage_p07b.json"
    if not tirage.exists():
        pytest.skip("tirage_p07b.json absent")
    twenty = sorted(json.loads(tirage.read_text(encoding="utf-8"))["list"])
    assert tuple(random.Random(2).sample(twenty, 2)) == CHECK_PAGES


def test_check_pages_fall_back_to_seed_1_on_the_six_pages():
    import random
    from babaorum.p08_oracles import check_pages_for
    six = [f"P{i}" for i in range(6)]
    assert check_pages_for(six) == tuple(random.Random(1).sample(sorted(six), 2))
