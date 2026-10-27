from __future__ import annotations

import itertools
import math
import random

import pytest

from babaorum.line_geometry import (
    METRIC_ID,
    SENSITIVITY_POLICY_ID,
    GroundTruthLine,
    LineGeometryError,
    PredictedLine,
    evaluate_line_geometry,
    evaluate_line_geometry_sensitivity,
    _minimum_cost_maximum_matching,
)


def gt(line_id, y, *, length=40, height=10, status="available", baseline=None):
    if baseline is None and status == "available":
        baseline = [[0.0, y], [float(length), y]]
    return GroundTruthLine(line_id, baseline, height if status == "available" else None,
                           "ground_truth" if status == "available" else None, status)


def pred(line_id, y, *, length=40, status="available", baseline=None):
    if baseline is None and status == "available":
        baseline = [[0.0, y], [float(length), y]]
    return PredictedLine(line_id, baseline,
                         "predicted" if status == "available" else None, status)


def test_exact_match_has_geometry_only_metrics():
    result = evaluate_line_geometry([gt("g1", 12)], [pred("p1", 12)])
    assert result.metric_id == METRIC_ID
    assert result.status == "evaluated"
    assert (result.precision, result.recall, result.f1) == (1.0, 1.0, 1.0)
    assert result.matches[0].normalized_symmetric_distance == 0.0


def test_one_height_sensitivity_is_separate_and_counts_only_gained_gt():
    result = evaluate_line_geometry([gt("g", 0)], [pred("p", 8)])
    assert result.matched_count == 0
    assert result.precision == result.recall == 0.0
    assert result.sensitivity_policy_id == SENSITIVITY_POLICY_ID
    assert result.sensitivity_1h_matched_count == 1
    assert result.gt_gained_only_at_1h_count == 1
    sensitivity = evaluate_line_geometry_sensitivity([gt("g", 0)], [pred("p", 8)])
    assert sensitivity.metric_id == SENSITIVITY_POLICY_ID
    assert sensitivity.matched_count == 1
    assert sensitivity.matches[0].normalized_symmetric_distance == pytest.approx(0.8)
    assert sensitivity.sensitivity_1h_matched_count is None


def test_split_and_merge_counters_are_diagnostics_not_primary_matches():
    split = evaluate_line_geometry(
        [gt("g", 0, length=100)],
        [pred("p1", 0, length=50), pred("p2", 0, length=50,
                                      baseline=[[50, 0], [100, 0]])],
    )
    assert split.matched_count == 0
    assert split.recall == 0.0 and split.precision == 0.0
    assert split.split_candidate_gt_count == 1
    assert split.split_candidate_pred_count == 2

    merge = evaluate_line_geometry(
        [gt("g1", 0, length=50), gt("g2", 0, length=50,
                                   baseline=[[50, 0], [100, 0]])],
        [pred("p", 0, length=100)],
    )
    assert merge.matched_count == 0
    assert merge.merge_candidate_pred_count == 1
    assert merge.merge_candidate_gt_count == 2


def test_matching_is_invariant_to_input_permutation():
    truths = [gt("g2", 0), gt("g1", 0)]
    predictions = [pred("p2", 0), pred("p1", 0)]
    forward = evaluate_line_geometry(truths, predictions)
    reverse = evaluate_line_geometry(list(reversed(truths)), list(reversed(predictions)))
    assert [(m.gt_id, m.pred_id) for m in forward.matches] == [
        (m.gt_id, m.pred_id) for m in reverse.matches
    ]


def test_arc_midpoint_sampling_is_stable_under_polyline_densification():
    sparse = evaluate_line_geometry([gt("g", 8)], [pred("p", 9)])
    dense_gt = gt("g", 8, baseline=[[0, 8], [10, 8], [20, 8], [30, 8], [40, 8]])
    dense_pred = pred("p", 9, baseline=[[0, 9], [7, 9], [19, 9], [31, 9], [40, 9]])
    dense = evaluate_line_geometry([dense_gt], [dense_pred])
    assert dense.matches[0].gt_to_pred_coverage == sparse.matches[0].gt_to_pred_coverage
    assert dense.matches[0].pred_to_gt_coverage == sparse.matches[0].pred_to_gt_coverage
    assert dense.matches[0].normalized_symmetric_distance == pytest.approx(
        sparse.matches[0].normalized_symmetric_distance, abs=1e-12
    )


def test_short_orthogonal_crossing_does_not_match_at_a_zero_only_sample():
    truth = GroundTruthLine("g", [[0, 1], [2, 1]], 1.0, "ground_truth")
    hypothesis = PredictedLine("p", [[1, 0], [1, 2]], "predicted")
    result = evaluate_line_geometry([truth], [hypothesis])
    assert result.matched_count == 0
    assert result.precision == result.recall == 0.0
    sensitivity = evaluate_line_geometry_sensitivity([truth], [hypothesis])
    assert sensitivity.matches[0].mean_symmetric_distance_px > 0


def test_sampling_domain_limit_rejects_before_large_allocation():
    truth = GroundTruthLine("g", [[0, 0], [10**10, 0]], 1.0, "ground_truth")
    with pytest.raises(LineGeometryError, match="sampling limit"):
        evaluate_line_geometry([truth], [])


def test_unrepresentable_distance_or_subnormal_height_is_a_controlled_error():
    truth = GroundTruthLine("g", [[1e308, 0], [1e308, 1]], 1.0, "ground_truth")
    hypothesis = PredictedLine("p", [[-1e308, 0], [-1e308, 1]], "predicted")
    with pytest.raises(LineGeometryError, match="not representable"):
        evaluate_line_geometry([truth], [hypothesis])

    tiny_height = GroundTruthLine("g", [[0, 0], [1, 0]], 5e-324, "ground_truth")
    with pytest.raises(LineGeometryError, match="cell size"):
        evaluate_line_geometry([tiny_height], [pred("p", 1)])


def test_maximum_cardinality_precedes_local_nearest_choice_and_ids_break_ties():
    truths = [gt("g1", 0), gt("g2", 10)]
    # p1 is eligible for both GT lines, while p2 is eligible only for g1.
    # A local nearest choice for p1 can lose a match; global cardinality may not.
    predictions = [pred("p1", 5), pred("p2", -4)]
    result = evaluate_line_geometry(truths, predictions)
    assert result.matched_count == 2
    assert {(match.gt_id, match.pred_id) for match in result.matches} == {
        ("g1", "p2"), ("g2", "p1")
    }

    tied = evaluate_line_geometry(
        [gt("g2", 0), gt("g1", 0)], [pred("p2", 0), pred("p1", 0)]
    )
    assert [(match.gt_id, match.pred_id) for match in tied.matches] == [
        ("g1", "p1"), ("g2", "p2")
    ]


def test_minimum_cost_matching_distinguishes_sub_femtoscale_costs():
    # The old 1e-15 relaxation treated the direct 1e-15 path as tied with the
    # 0.9e-15 alternating path and returned a strictly more expensive match.
    candidates = [
        (0, 0, 0.0),
        (0, 1, 0.5e-15),
        (1, 0, 0.4e-15),
        (1, 1, 1.0e-15),
    ]
    selected = _minimum_cost_maximum_matching(2, 2, candidates)
    assert selected == {(0, 1), (1, 0)}


def test_minimum_cost_matching_agrees_with_exhaustive_small_graph_oracle():
    rng = random.Random(20261007)

    def exhaustive_score(n_gt, n_pred, candidates):
        costs = {(i, j): cost for i, j, cost in candidates}
        best = None
        for assignment in itertools.product(range(-1, n_pred), repeat=n_gt):
            assigned_predictions = [j for j in assignment if j >= 0]
            if len(set(assigned_predictions)) != len(assigned_predictions):
                continue
            pairs = {(i, j) for i, j in enumerate(assignment) if j >= 0}
            if any(pair not in costs for pair in pairs):
                continue
            score = (len(pairs), math.fsum(costs[pair] for pair in pairs))
            if best is None or score[0] > best[0] or (
                score[0] == best[0] and score[1] < best[1]
            ):
                best = score
        return best

    for trial in range(256):
        n_gt = rng.randint(1, 4)
        n_pred = rng.randint(1, 4)
        candidates = []
        for i in range(n_gt):
            for j in range(n_pred):
                if rng.random() < 0.68:
                    scale = 10.0 ** rng.randint(-16, 16)
                    cost = rng.choice((rng.random() * scale, rng.randrange(100) * scale))
                    candidates.append((i, j, cost))

        selected = _minimum_cost_maximum_matching(n_gt, n_pred, candidates)
        expected_count, expected_cost = exhaustive_score(n_gt, n_pred, candidates)
        actual_cost = math.fsum(
            cost for i, j, cost in candidates if (i, j) in selected
        )
        assert len(selected) == expected_count, f"trial {trial}: {candidates}"
        assert actual_cost == expected_cost, f"trial {trial}: {candidates}"


def test_both_directional_coverages_are_required():
    result = evaluate_line_geometry([gt("g", 0, length=100)], [pred("p", 0, length=50)])
    assert result.matched_count == 0
    assert result.recall == 0.0
    assert result.precision == 0.0


def test_spatial_lower_bound_skips_only_pairs_beyond_one_height(monkeypatch):
    import babaorum.line_geometry as geometry

    original = geometry._pair_measure
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(geometry, "_pair_measure", counted)
    far = evaluate_line_geometry([gt("g", 0, height=10)], [pred("p", 10.0001)])
    assert far.matched_count == 0
    assert far.sensitivity_1h_matched_count == 0
    assert calls == []

    # At the inclusive 1 h boundary, retain the exact pair calculation.
    boundary = evaluate_line_geometry([gt("g", 0, height=10)], [pred("p", 10)])
    assert boundary.matched_count == 0
    assert boundary.sensitivity_1h_matched_count == 1
    assert len(calls) == 2  # primary and separate 1 h sensitivity


def test_spatial_lower_bound_preserves_primary_split_adjacency_at_half_height():
    result = evaluate_line_geometry([gt("g", 0, height=10)], [pred("p", 5)])
    assert result.matched_count == 1
    assert result.matches[0].normalized_symmetric_distance == pytest.approx(0.5)


def test_spatial_pruning_is_exactly_equivalent_to_exhaustive_pairs(monkeypatch):
    import dataclasses
    import babaorum.line_geometry as geometry

    truths = [gt("g0", 0), gt("g1", 10), gt("g2", 100),
              gt("g3", 200, baseline=[[0, 200], [20, 202], [40, 200]])]
    predictions = [pred("p0", 1), pred("p1", 16), pred("p2", 250),
                   pred("p3", 100, baseline=[[0, 100], [20, 101], [40, 100]])]
    pruned = evaluate_line_geometry(truths, predictions)
    monkeypatch.setattr(geometry, "_bounds_distance_lower_bound", lambda *args, **kwargs: 0.0)
    # Second pass G1: also disable the vertical pre-filter window.
    monkeypatch.setattr(geometry._PredictionWindow, "nearby", lambda self, *args: range(self.count))
    exhaustive = evaluate_line_geometry(truths, predictions)
    assert dataclasses.asdict(pruned) == dataclasses.asdict(exhaustive)


def test_prediction_window_never_drops_a_pair_the_exact_bound_keeps():
    import babaorum.line_geometry as geometry

    rng = random.Random(2026100806)
    for _ in range(60):
        scale = rng.choice([1.0, 1e3, 1e6, 1e11, 1e13])
        boxes = []
        for _ in range(rng.randint(1, 40)):
            x0, y0 = rng.uniform(0, 1) * scale, rng.uniform(0, 1) * scale
            boxes.append((x0, y0, x0 + rng.uniform(0, 0.3) * scale, y0 + rng.uniform(0, 0.05) * scale))
        window = geometry._PredictionWindow(boxes, 0.0)
        for _ in range(20):
            gx0, gy0 = rng.uniform(0, 1) * scale, rng.uniform(0, 1) * scale
            gt_box = (gx0, gy0, gx0 + rng.uniform(0, 0.3) * scale, gy0 + rng.uniform(0, 0.01) * scale)
            height = rng.choice([1.0, 10.0, 0.02 * scale])
            exact = [j for j, box in enumerate(boxes)
                     if not geometry._bounds_distance_lower_bound(gt_box, box, height=height) > height]
            near = list(window.nearby(gt_box, height))
            assert near == sorted(near)
            assert [j for j in near if j in exact] == exact


def test_incomplete_gt_nulls_main_metrics_and_keeps_page_counts():
    result = evaluate_line_geometry(
        [gt("g1", 0), gt("g2", 0, status="missing")],
        [pred("p1", 0), pred("p2", 30)],
    )
    assert result.status == "non_evaluable"
    assert result.precision is result.recall is result.f1 is None
    assert result.n_gt == 2 and result.gt_evaluable_count == 1
    assert result.gt_non_evaluable_count == 1
    assert result.matched_count == 1
    assert result.unmatched_pred_count is None


def test_missing_prediction_geometry_counts_against_precision_on_complete_gt():
    result = evaluate_line_geometry(
        [gt("g", 0)], [pred("p1", 0), pred("p2", 0, status="error")]
    )
    assert result.pred_geometry_unavailable_count == 1
    assert result.matched_count == 1
    assert result.unmatched_pred_count == 1
    assert result.precision == 0.5
    assert result.recall == 1.0


def test_empty_ground_truth_and_predictions_have_explicit_null_semantics():
    with_predictions = evaluate_line_geometry([], [pred("p", 0)])
    assert with_predictions.status == "evaluated"
    assert with_predictions.precision == 0.0
    assert with_predictions.recall is None
    without_predictions = evaluate_line_geometry([], [])
    assert without_predictions.precision is None
    assert without_predictions.recall is None


def test_bad_geometry_height_status_and_duplicate_ids_are_rejected():
    with pytest.raises(LineGeometryError, match="height_px must be positive"):
        evaluate_line_geometry([gt("g", 0, height=0)], [])
    with pytest.raises(LineGeometryError, match="requires a baseline"):
        evaluate_line_geometry([GroundTruthLine("g", None, 10, "ground_truth")], [])
    with pytest.raises(LineGeometryError, match="duplicate ground-truth"):
        evaluate_line_geometry([gt("g", 0), gt("g", 2)], [])
    with pytest.raises(LineGeometryError, match="cannot claim ground_truth"):
        evaluate_line_geometry([gt("g", 0)], [PredictedLine("p", [[0, 0], [10, 0]], "ground_truth")])
    with pytest.raises(LineGeometryError, match="invalid geometry_status"):
        evaluate_line_geometry([GroundTruthLine("g", None, None, None, [])], [])


def test_text_is_not_part_of_the_api_or_matching():
    # The public input types contain only IDs and geometry; no OCR transcript can
    # influence eligibility, assignment, or distance.
    result = evaluate_line_geometry([gt("g", 3)], [pred("p", 3)])
    assert result.matches[0].gt_id == "g"
    assert result.matches[0].pred_id == "p"


@pytest.mark.parametrize('scorer', [evaluate_line_geometry, evaluate_line_geometry_sensitivity])
def test_reversed_baseline_has_orientation_diagnostic_without_distance_penalty(scorer):
    truth = gt('g', 8, baseline=[[0, 8], [20, 10], [40, 8]])
    correct = pred('p', 8, baseline=[[0, 8], [20, 10], [40, 8]])
    reversed_line = pred('p', 8, baseline=list(reversed(correct.baseline)))
    normal = scorer([truth], [correct])
    reverse = scorer([truth], [reversed_line])
    assert reverse.matched_count == normal.matched_count == 1
    assert reverse.matches[0].normalized_symmetric_distance == pytest.approx(
        normal.matches[0].normalized_symmetric_distance, abs=1e-12)
    assert reverse.orientation_disagreement_count == 1
    assert normal.orientation_disagreement_count == 0
    assert reverse.matches[0].orientation_dot_sign == -1
    assert reverse.orientation_unresolved_count == normal.orientation_unresolved_count == 0


def test_orientation_with_orthogonal_or_zero_endpoint_vector_is_explicitly_unresolved():
    horizontal = gt('g', 1, height=10, baseline=[[0, 1], [2, 1]])
    vertical = pred('p', 1, baseline=[[1, 0], [1, 2]])
    result = evaluate_line_geometry([horizontal], [vertical])
    assert result.matched_count == 1
    assert result.orientation_disagreement_count == 0
    assert result.orientation_unresolved_count == 1
    assert result.matches[0].orientation_dot_sign == 0
    # The low-level distance accepts an out-and-back curve. Its global endpoint
    # direction is undefined, even if its geometric support matches perfectly.
    loop = pred('p', 1, baseline=[[0, 1], [2, 1], [0, 1]])
    result = evaluate_line_geometry([horizontal], [loop])
    assert result.orientation_unresolved_count == 1
    assert result.matches[0].orientation_dot_sign is None


# --- G1 B1 (2026-10-08, second pass): exact integer solver, global tie rule ---

_TIE527 = (5, 4, [
    (1, 1, 0.7514773410545575), (3, 3, 0.7514773410545575), (1, 0, 0.18394921758972405),
    (1, 2, 3.077603593419141e-09), (4, 3, 0.7514773410545575), (4, 0, 0.7514773410545575),
    (2, 0, 3.077603593419141e-09), (4, 2, 3.077603593419141e-09), (0, 0, 3.077603593419141e-09),
    (0, 1, 0.18394921758972405), (2, 2, 0.18394921758972405), (3, 1, 0.7514773410545575),
    (2, 1, 0.18394921758972405),
])


def _exact_oracle(n_gt, n_pred, candidates):
    """Brute force: max cardinality, exact (Fraction) min sum, then least sorted pair list."""
    from fractions import Fraction

    costs = {(i, j): Fraction(cost) for i, j, cost in candidates}
    best = None
    for assignment in itertools.product(range(-1, n_pred), repeat=n_gt):
        used = [j for j in assignment if j >= 0]
        if len(set(used)) != len(used):
            continue
        pairs = sorted((i, j) for i, j in enumerate(assignment) if j >= 0)
        if any(pair not in costs for pair in pairs):
            continue
        key = (-len(pairs), sum((costs[pair] for pair in pairs), Fraction(0)), pairs)
        if best is None or key < best:
            best = key
    return -best[0], best[1], set(best[2])


def _exact_sum(candidates, selected):
    from fractions import Fraction

    costs = {(i, j): Fraction(cost) for i, j, cost in candidates}
    return sum((costs[pair] for pair in selected), Fraction(0))


class _CountingHeap:
    """Stand-in for heapq inside line_geometry: counts heap operations (N5)."""

    def __init__(self):
        import heapq

        self._heapq = heapq
        self.operations = 0

    def heappush(self, heap, item):
        self.operations += 1
        return self._heapq.heappush(heap, item)

    def heappop(self, heap):
        self.operations += 1
        return self._heapq.heappop(heap)


def _run_with_deadline(seconds, function, *args):
    """Hang guard only (very wide margin); speed is asserted by operation counts."""
    import signal

    class _Deadline(Exception):
        pass

    def on_alarm(*_):
        raise _Deadline()

    previous = signal.signal(signal.SIGALRM, on_alarm)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        return function(*args)
    except _Deadline:
        pytest.fail(f"no return within {seconds} s (non-terminating solver)")
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _tie_ulp_mixed_cost(rng, family, pool):
    if family == "tie":
        return rng.choice(pool)
    if family == "ulp":
        base = rng.choice(pool)
        for _ in range(abs(step := rng.randint(-2, 2))):
            base = math.nextafter(base, math.inf if step > 0 else 0.0)
        return base
    return rng.choice([0.0, 1e-16, 5e-16, 1e-15, 1.0, 1e16, rng.uniform(0, 1),
                       10 ** rng.uniform(-16, 16), 5e-324, 2.2250738585072014e-308])


@pytest.mark.parametrize("family,seed", [("tie", 2026100801), ("ulp", 2026100802), ("mixed", 2026100803)])
def test_exact_solver_matches_brute_force_including_the_tie_rule(family, seed):
    rng = random.Random(seed)
    for trial in range(170):
        n_gt, n_pred = rng.randint(1, 4), rng.randint(1, 4)
        density = rng.uniform(0.2, 1.0)
        pool = [10 ** rng.uniform(-16, 16) for _ in range(2)] + [rng.uniform(0, 1)]
        candidates = [(i, j, _tie_ulp_mixed_cost(rng, family, pool))
                      for i in range(n_gt) for j in range(n_pred) if rng.random() < density]
        rng.shuffle(candidates)
        selected = _run_with_deadline(30.0, _minimum_cost_maximum_matching, n_gt, n_pred, list(candidates))
        expected_count, expected_sum, expected_set = _exact_oracle(n_gt, n_pred, candidates)
        context = f"{family} trial {trial}: {candidates}"
        assert len(selected) == expected_count, context
        assert _exact_sum(candidates, selected) == expected_sum, context  # exact, not fsum (m1)
        assert selected == expected_set, context  # lexicographic tie rule
        assert selected == _minimum_cost_maximum_matching(n_gt, n_pred, list(reversed(candidates))), context
        assert selected == _minimum_cost_maximum_matching(n_gt, n_pred, candidates, _decompose=False), context


def test_tie_rule_is_independent_of_component_decomposition():
    # Many components, many exact ties: whole-graph and per-component solves
    # must return the same set (the float/Fraction SSP did not guarantee this).
    rng = random.Random(2026100804)
    for trial in range(40):
        n = rng.randint(20, 60)
        pool = [rng.uniform(0, 0.5) for _ in range(3)]
        candidates = {}
        for j in range(n):
            base = rng.randrange(n)
            for i in range(max(0, base - rng.randint(0, 2)), min(n, base + rng.randint(1, 3))):
                candidates[(i, j)] = rng.choice(pool)
        rows = [(i, j, c) for (i, j), c in candidates.items()]
        whole = _minimum_cost_maximum_matching(n, n, rows, _decompose=False)
        split = _minimum_cost_maximum_matching(n, n, rows)
        assert whole == split, f"trial {trial}"


def test_scaled_integer_costs_preserve_exact_order_including_subnormals():
    from fractions import Fraction

    from babaorum.line_geometry import _scaled_integer_costs, _validated_candidates

    values = [0.0, 5e-324, 1e-323, 2.2250738585072014e-308, 1e-16, 0.1, 1.0, 1e16, 2**60, 3]
    rows = [(0, j, v) for j, v in enumerate(values)]
    scaled = _scaled_integer_costs(_validated_candidates(1, len(values), rows))
    assert all(isinstance(cost, int) for _, _, cost in scaled)
    ratios = {Fraction(cost) / Fraction(value) for (_, _, cost), value in zip(scaled, values) if value}
    assert len(ratios) == 1  # one common positive scale: order and ties unchanged


def test_exact_solver_terminates_on_float_zero_cost_cycle_counterexample(monkeypatch):
    import babaorum.line_geometry as geometry

    counter = _CountingHeap()
    monkeypatch.setattr(geometry, "heapq", counter)
    n_gt, n_pred, candidates = _TIE527
    selected = _run_with_deadline(30.0, _minimum_cost_maximum_matching, n_gt, n_pred, candidates)
    expected_count, expected_sum, expected_set = _exact_oracle(n_gt, n_pred, candidates)
    assert len(selected) == expected_count == 4
    assert _exact_sum(candidates, selected) == expected_sum
    assert selected == expected_set
    assert 0 < counter.operations <= 400  # bounded work instead of a wall-clock threshold


def test_float_reduced_costs_fail_closed_instead_of_looping():
    # Feed float costs straight into the generic core: the invariant/budget
    # guard must turn the historical infinite loop into a controlled error.
    import babaorum.line_geometry as geometry

    n_gt, n_pred, candidates = _TIE527
    with pytest.raises(LineGeometryError, match="matching"):
        _run_with_deadline(30.0, geometry._successive_shortest_paths, n_gt, n_pred,
                           sorted(candidates, key=lambda row: (row[0], row[1])))


def test_sub_femtoscale_counterexample_keeps_exact_minimum_sum():
    candidates = [(0, 0, 0.0), (0, 1, 0.5e-15), (1, 0, 0.4e-15), (1, 1, 1.0e-15)]
    selected = _minimum_cost_maximum_matching(2, 2, candidates)
    assert selected == {(0, 1), (1, 0)}
    assert _exact_sum(candidates, selected) == _exact_oracle(2, 2, candidates)[1]


def test_solver_rejects_invalid_candidate_costs_and_pairs():
    for bad in (-1e-300, math.nan, math.inf, True, "0", 10**400, -(10**400)):
        with pytest.raises(LineGeometryError, match="cost"):
            _minimum_cost_maximum_matching(1, 1, [(0, 0, bad)])
    assert _minimum_cost_maximum_matching(1, 1, [(0, 0, 2**1000)]) == {(0, 0)}
    with pytest.raises(LineGeometryError, match="unique in-range"):
        _minimum_cost_maximum_matching(1, 1, [(0, 1, 0.0)])
    with pytest.raises(LineGeometryError, match="unique in-range"):
        _minimum_cost_maximum_matching(1, 1, [(0, 0, 0.0), (0, 0, 1.0)])
    for bad_pair in ((True, 0), (0, False), (0.0, 0), (0, 1.0)):
        with pytest.raises(LineGeometryError, match="indices must be integers"):
            _minimum_cost_maximum_matching(2, 2, [(*bad_pair, 0.0)])
    with pytest.raises(LineGeometryError, match="nonnegative integer"):
        _minimum_cost_maximum_matching(True, 1, [])


def test_duplicate_predictions_geometry_case_terminates_with_optimal_matching(monkeypatch):
    import babaorum.line_geometry as geometry

    calls = []
    solver = geometry._minimum_cost_maximum_matching

    def recorded(n_gt, n_pred, candidates):
        selected = solver(n_gt, n_pred, candidates)
        calls.append((n_gt, n_pred, list(candidates), selected))
        return selected

    monkeypatch.setattr(geometry, "_minimum_cost_maximum_matching", recorded)
    counter = _CountingHeap()
    monkeypatch.setattr(geometry, "heapq", counter)
    # Five GT lines, six predictions with p3 == p4: this page hung the float solver.
    truths = [GroundTruthLine(f"g{i}", [[10, 20 * i + 20], [210, 20 * i + 20]], 20.0, "ground_truth")
              for i in range(5)]
    predictions = [
        PredictedLine("p0", [[10, 90], [190, 90]], "predicted"),
        PredictedLine("p1", [[10, 80], [210, 80]], "predicted"),
        PredictedLine("p2", [[10, 45], [190, 45]], "predicted"),
        PredictedLine("p3", [[10, 100], [190, 100]], "predicted"),
        PredictedLine("p4", [[10, 100], [190, 100]], "predicted"),
        PredictedLine("p5", [[30, 40], [190, 40]], "predicted"),
    ]
    primary = _run_with_deadline(30.0, evaluate_line_geometry, truths, predictions)
    assert [(m.gt_id, m.pred_id) for m in primary.matches] == [("g1", "p5"), ("g3", "p1"), ("g4", "p3")]
    assert primary.sensitivity_1h_matched_count == 5
    sensitivity = _run_with_deadline(30.0, evaluate_line_geometry_sensitivity, truths, predictions)
    assert sensitivity.matched_count == 5
    assert 0 < counter.operations <= 1000  # bounded solver work (N5), not wall-clock
    # Every solver call (primary, its 1 h diagnostic, the sensitivity pass)
    # reaches the brute-force optimum, including the p3/p4 tie rule.
    assert len(calls) == 3
    for n_gt, n_pred, candidates, selected in calls:
        expected_count, expected_sum, expected_set = _exact_oracle(n_gt, n_pred, candidates)
        assert len(selected) == expected_count
        assert _exact_sum(candidates, selected) == expected_sum
        assert selected == expected_set


def test_fast_distance_path_is_bit_identical_to_checked_path():
    import babaorum.line_geometry as geometry

    rng = random.Random(2026100805)
    for _ in range(300):
        scale = 10 ** rng.uniform(-3, 6)
        line = [(rng.uniform(-1, 1) * scale, rng.uniform(-1, 1) * scale) for _ in range(rng.randint(2, 6))]
        if rng.random() < 0.3:
            line.insert(1, line[0])  # zero-length segment
        table = geometry._segment_table(tuple(line))
        for _ in range(20):
            p = (rng.uniform(-2, 2) * scale, rng.uniform(-2, 2) * scale)
            assert geometry._point_table_distance(p, table) == geometry._point_polyline_distance(p, tuple(line))
    assert geometry._segment_table(((0.0, 0.0), (1e200, 0.0))) is None  # falls back to checked path
