"""One-step CPU causality checks on a fixed synthetic fixture, without OCR claims."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from babaorum.experimental.joint_causal import discrete_stability, run_one_step_ablation  # noqa: E402
from babaorum.experimental.joint_curved_page import CurvedPageTargets, TinyCurvedJointPage  # noqa: E402


CONFIG = json.loads((Path(__file__).resolve().parents[1] / "config" / "joint_causal_probe.json").read_text())
MINIMUM_CHANGE_NORM = CONFIG["minimum_change_norm"]
MAXIMUM_RELATIVE_RESIDUAL = CONFIG["maximum_relative_residual"]
RESIDUAL_FLOOR = CONFIG["residual_floor"]
FULL_STEP = CONFIG["learning_rate"]
HALF_STEP = CONFIG["half_learning_rate"]


@pytest.fixture(autouse=True, scope="module")
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(CONFIG["threads"])
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def causal_runs():
    # Read the predeclared fixture; do not search seeds, learning rates or images.
    assert CONFIG["device"] == "cpu" and CONFIG["dtype"] == "float64"
    assert CONFIG["image_formula"] == "sin(0.21*x)+cos(0.37*y)"
    torch.manual_seed(CONFIG["seed"])
    names = ("baseline_fraction", "num_queries", "control_points", "channels", "alphabet_size",
             "band_rows", "spacing_px", "max_samples", "penalty_scale_px")
    model = TinyCurvedJointPage(**{name: CONFIG[name] for name in names}).double()
    width, height = CONFIG["image_size_wh"]
    yy, xx = torch.meshgrid(torch.arange(height, dtype=torch.float64), torch.arange(width, dtype=torch.float64), indexing="ij")
    image = (torch.sin(xx * 0.21) + torch.cos(yy * 0.37))[None, None]
    targets = CurvedPageTargets(
        baseline_xy=[torch.tensor(points, dtype=torch.float64) for points in CONFIG["target_baselines_xy"]],
        height_px=torch.tensor(CONFIG["target_heights_px"], dtype=torch.float64),
        texts=copy.deepcopy(CONFIG["target_token_ids"]),
        order=torch.tensor(CONFIG["target_order"]),
    )
    original = {name: value.detach().clone() for name, value in model.state_dict().items()}
    results = {}
    for learning_rate in (FULL_STEP, HALF_STEP):
        results[learning_rate] = run_one_step_ablation(
            model, image, targets, learning_rate=learning_rate,
            minimum_change_norm=MINIMUM_CHANGE_NORM,
            maximum_relative_residual=MAXIMUM_RELATIVE_RESIDUAL,
            residual_floor=RESIDUAL_FLOOR,
        )
    return model, original, results


def test_ablations_leave_base_model_unchanged_and_start_with_identical_forward_values(causal_runs):
    model, original, results = causal_runs
    for name, value in model.state_dict().items():
        assert torch.equal(value, original[name])
    for result in results.values():
        for block in ("E", "G", "R"):
            assert len({result.initial_hashes[arm][block] for arm in ("S2", "B3", "B4")}) == 1
        assert result.initial_hashes["S2"]["geometry_encoder"] == result.initial_hashes["S2"]["E"]
        reference = result.arms["B3"]
        for arm in result.arms.values():
            assert torch.equal(arm.initial_z, reference.initial_z)
            assert float(arm.initial_ctc) == float(reference.initial_ctc)
            assert arm.counts_before == reference.counts_before
            assert arm.counts_before["included_ctc"] == CONFIG["num_queries"]
            for first, second in zip(arm.initial_prediction.queries, reference.initial_prediction.queries):
                assert first.status == second.status
                assert torch.equal(first.ctc_logits, second.ctc_logits)
                assert torch.equal(first.grid.sample_valid, second.grid.sample_valid)
                assert torch.equal(first.grid.cell_valid, second.grid.cell_valid)
        report = result.to_report()
        json.dumps(report, allow_nan=False)
        for name, arm in result.arms.items():
            assert report["arms"][name]["signature_before"] == arm.signature_before
            assert report["arms"][name]["signature_after"] == arm.signature_after


def test_separate_geometry_stays_exact_while_shared_detached_geometry_changes(causal_runs):
    _, _, results = causal_runs
    for result in results.values():
        separate, detached, coupled = (result.arms[name] for name in ("S2", "B3", "B4"))
        assert torch.equal(separate.initial_z, separate.final_z)
        assert torch.count_nonzero(separate.auxiliary["geometry_encoder_delta"]) == 0
        assert set(separate.auxiliary["geometry_encoder_storage_pointers"]).isdisjoint(
            separate.auxiliary["reader_encoder_storage_pointers"]
        )
        assert torch.linalg.vector_norm(separate.parameter_deltas["E"]) > MINIMUM_CHANGE_NORM
        assert torch.linalg.vector_norm(separate.parameter_deltas["R"]) > MINIMUM_CHANGE_NORM
        assert torch.linalg.vector_norm(detached.final_z - detached.initial_z) > MINIMUM_CHANGE_NORM
        assert torch.linalg.vector_norm(coupled.final_z - detached.final_z) > MINIMUM_CHANGE_NORM


def test_gradient_paths_and_sgd_blocks_include_query_embeddings(causal_runs):
    model, _, results = causal_runs
    geometry_parameter_count = model.queries.numel() + sum(parameter.numel() for parameter in model.geometry_head.parameters())
    for learning_rate, result in results.items():
        for name, arm in result.arms.items():
            assert arm.gradient_blocks["G"].numel() == geometry_parameter_count
            for block in ("E", "G", "R"):
                torch.testing.assert_close(
                    arm.parameter_deltas[block], -learning_rate * arm.gradient_blocks[block],
                    atol=1e-13, rtol=1e-9,
                )
            for block in ("E", "R"):
                assert torch.linalg.vector_norm(arm.gradient_blocks[block]) > MINIMUM_CHANGE_NORM
            if name in ("S2", "B3"):
                assert arm.coordinate_gradient is None or torch.count_nonzero(arm.coordinate_gradient) == 0
                assert torch.count_nonzero(arm.gradient_blocks["G"]) == 0
            else:
                assert arm.coordinate_gradient is not None
                assert torch.linalg.vector_norm(arm.coordinate_gradient) > MINIMUM_CHANGE_NORM
                assert torch.linalg.vector_norm(arm.gradient_blocks["G"]) > MINIMUM_CHANGE_NORM
        torch.testing.assert_close(result.arms["S2"].parameter_deltas["E"], result.arms["B3"].parameter_deltas["E"], atol=0, rtol=0)
        torch.testing.assert_close(result.arms["B3"].gradient_blocks["R"], result.arms["B4"].gradient_blocks["R"], atol=0, rtol=0)


def test_first_order_decomposition_reconstructs_observed_difference_and_reports_ratios(causal_runs):
    _, _, results = causal_runs
    for result in results.values():
        assert result.checks["discrete_stable"]
        decomposition = result.decomposition
        observed = result.arms["B4"].final_z - result.arms["B3"].final_z
        encoder = decomposition["encoder_contribution"]
        geometry = decomposition["geometry_head_contribution"]
        predicted = encoder + geometry
        residual = observed - predicted
        torch.testing.assert_close(decomposition["observed_delta43"], observed)
        torch.testing.assert_close(decomposition["predicted_delta43"], predicted)
        torch.testing.assert_close(decomposition["residual_delta43"], residual)
        assert torch.linalg.vector_norm(encoder) > MINIMUM_CHANGE_NORM
        assert torch.linalg.vector_norm(geometry) > MINIMUM_CHANGE_NORM
        observed_norm = torch.linalg.vector_norm(observed)
        relative = torch.linalg.vector_norm(residual).item() / max(observed_norm.item(), RESIDUAL_FLOOR)
        assert decomposition["relative_residual_delta43"] == pytest.approx(relative)
        assert relative <= MAXIMUM_RELATIVE_RESIDUAL
        b3_observed = result.arms["B3"].final_z - result.arms["B3"].initial_z
        b3_residual = b3_observed - decomposition["b3_encoder_prediction"]
        torch.testing.assert_close(decomposition["b3_observed"], b3_observed)
        torch.testing.assert_close(decomposition["b3_residual"], b3_residual)
        b3_relative = torch.linalg.vector_norm(b3_residual).item() / max(torch.linalg.vector_norm(b3_observed).item(), RESIDUAL_FLOOR)
        assert decomposition["relative_residual_b3"] == pytest.approx(b3_relative)
        assert b3_relative <= MAXIMUM_RELATIVE_RESIDUAL
        b4_observed = result.arms["B4"].final_z - result.arms["B4"].initial_z
        torch.testing.assert_close(decomposition["b4_observed"], b4_observed)
        drift_ratio = torch.linalg.vector_norm(b3_observed).item() / max(torch.linalg.vector_norm(b4_observed).item(), RESIDUAL_FLOOR)
        contribution_ratio = torch.linalg.vector_norm(geometry).item() / max(torch.linalg.vector_norm(encoder).item(), RESIDUAL_FLOOR)
        assert decomposition["report_drift_ratio_b3_over_b4"] == pytest.approx(drift_ratio)
        assert decomposition["contribution_norm_ratio_geometry_over_encoder"] == pytest.approx(contribution_ratio)
        assert decomposition["drift_ratio_is_not_a_causal_share"] is True
        assert decomposition["signed_ratios_are_not_causal_percentages"] is True
        for field, component in (("signed_projection_encoder", encoder), ("signed_projection_geometry_head", geometry)):
            expected = torch.dot(component, observed).item() / observed_norm.square().item()
            assert decomposition[field] == pytest.approx(expected)
        assert result.checks["valid"], {name: valid for name, valid in result.checks.items() if not valid}


def test_halving_the_fixed_step_reduces_the_first_order_remainder_quadratically(causal_runs):
    _, _, results = causal_runs
    for field in ("b3_residual", "residual_delta43"):
        full = torch.linalg.vector_norm(results[FULL_STEP].decomposition[field]).item()
        half = torch.linalg.vector_norm(results[HALF_STEP].decomposition[field]).item()
        # A second-order remainder should shrink by about 1/4. The prechosen
        # 0.35 ceiling separates it from an omitted first-order term (~1/2).
        assert half <= CONFIG["maximum_absolute_half_step_factor"] * full + RESIDUAL_FLOOR


@pytest.mark.parametrize(
    "field,check",
    [("time_steps", "time_steps_unchanged"), ("matching_recomputed", "matching_unchanged"), ("masks", "masks_unchanged")],
)
def test_discrete_guard_invalidates_attribution_without_erasing_observations(causal_runs, field, check):
    _, _, results = causal_runs
    before = copy.deepcopy(results[FULL_STEP].arms["B3"].signature_before)
    after = copy.deepcopy(before)
    assert discrete_stability(before, after)["valid"]
    if field == "time_steps":
        after[field][0] += 1
    elif field == "matching_recomputed":
        after[field] = list(reversed(after[field]))
    else:
        after[field][0]["line_valid"] = not after[field][0]["line_valid"]
    before_snapshot, after_snapshot = copy.deepcopy(before), copy.deepcopy(after)
    checks = discrete_stability(before, after)
    assert not checks["valid"] and not checks[check]
    assert before == before_snapshot and after == after_snapshot
