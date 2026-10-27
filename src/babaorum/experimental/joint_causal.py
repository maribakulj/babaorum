"""One CTC-only SGD step isolating separate/shared/coupled geometry paths.

S2 has distinct geometry and reader encoders. B3 shares an encoder but detaches
sampling coordinates; its next geometry can still move through the encoder.
B4 also permits the direct CTC-to-geometry path. These are mechanism ablations,
not trained OCR systems or a reproduction of a full geometry+CTC objective.

The source model is never modified. Directions are detached before geometry
JVPs: no CTC Hessian is taken. Discrete changes invalidate attribution while
preserving observations. The separate arm has different allocation and runtime
costs, which are reported rather than presented as equal-budget performance.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import hashlib
import math
import time
from typing import Any

from .joint_curved_page import CurvedPagePrediction, CurvedPageTargets, TinyCurvedJointPage, match_curved_lines
from .joint_curve_sampling import masked_curve_ctc_loss, sample_curved_band
from .joint_geometry import require_torch, torch


def _z(prediction: CurvedPagePrediction) -> Any:
    return torch.cat((prediction.baseline_xy.reshape(-1), prediction.height_px.reshape(-1)))


def _flat(parameters: list[tuple[str, Any]]) -> Any:
    return torch.cat([parameter.detach().reshape(-1) for _, parameter in parameters]).clone()


def _tensor_hash(value: Any) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(str(tuple(tensor.shape)).encode())
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _group_hash(parameters: list[tuple[str, Any]]) -> str:
    digest = hashlib.sha256()
    for name, parameter in parameters:
        digest.update(name.encode() + b"\0" + _tensor_hash(parameter).encode())
    return digest.hexdigest()


def _norm(value: Any) -> float:
    return float(torch.linalg.vector_norm(value.detach()))


def _summary(value: Any) -> dict[str, Any]:
    return {"norm": _norm(value), "sha256": _tensor_hash(value), "numel": value.numel()}


def _safe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: _safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


@dataclass
class ArmResult:
    initial_prediction: CurvedPagePrediction
    final_prediction: CurvedPagePrediction
    initial_ctc: float
    final_ctc: float
    initial_z: Any
    final_z: Any
    gradient_blocks: dict[str, Any]
    parameter_deltas: dict[str, Any]
    coordinate_gradient: Any
    counts_before: dict[str, int]
    counts_after: dict[str, int]
    signature_before: dict[str, Any]
    signature_after: dict[str, Any]
    parameter_counts: dict[str, int]
    hashes_before: dict[str, str]
    hashes_after: dict[str, str]
    elapsed_seconds: float
    auxiliary: dict[str, Any]


@dataclass
class CausalAblationResult:
    arms: dict[str, ArmResult]
    decomposition: dict[str, Any]
    checks: dict[str, bool]
    initial_hashes: dict[str, dict[str, str]]
    matching: tuple[int, ...]
    thresholds: dict[str, float]
    source_hash_before: str
    source_hash_after: str
    elapsed_seconds: float

    def to_report(self) -> dict[str, Any]:
        """Curated JSON-safe diagnostics; no model, feature map or band grids."""
        arms = {}
        for name, arm in self.arms.items():
            auxiliary = {key: value for key, value in arm.auxiliary.items() if "storage_pointers" not in key}
            if "geometry_encoder_delta" in auxiliary:
                auxiliary["geometry_encoder_delta"] = _summary(auxiliary["geometry_encoder_delta"])
            arms[name] = {
                "initial_ctc": arm.initial_ctc, "final_ctc": arm.final_ctc,
                "initial_z": arm.initial_z, "final_z": arm.final_z,
                "gradient_blocks": {key: _summary(value) for key, value in arm.gradient_blocks.items()},
                "parameter_deltas": {key: _summary(value) for key, value in arm.parameter_deltas.items()},
                "coordinate_gradient": None if arm.coordinate_gradient is None else _summary(arm.coordinate_gradient),
                "signature_before": arm.signature_before, "signature_after": arm.signature_after,
                "counts_before": arm.counts_before, "counts_after": arm.counts_after,
                "parameter_counts": arm.parameter_counts, "hashes_before": arm.hashes_before,
                "hashes_after": arm.hashes_after, "elapsed_seconds": arm.elapsed_seconds,
                "auxiliary": auxiliary,
            }
        return _safe({
            "experiment": "one_ctc_sgd_step", "scope": "SYNTHETIC_CAUSAL_MECHANISM_ONLY",
            "matching": self.matching, "initial_hashes": self.initial_hashes,
            "arms": arms, "decomposition": self.decomposition, "checks": self.checks,
            "thresholds": self.thresholds, "source_hash_before": self.source_hash_before,
            "source_hash_after": self.source_hash_after,
            "source_unchanged": self.source_hash_before == self.source_hash_after,
            "elapsed_seconds": self.elapsed_seconds,
            "optimizer": {"name": "SGD", "steps": 1, "momentum": 0.0, "weight_decay": 0.0},
            "groups": {"E": "reader/shared encoder", "G": "queries plus geometry_head", "R": "ctc_head"},
            "equal_compute_budget_claimed": False,
        })


def geometry_jvp(prediction: CurvedPagePrediction, parameters: tuple[Any, ...], direction: tuple[Any, ...]) -> Any:
    """J_z(theta) times a detached direction, via reverse-over-reverse.

    Only the geometry output graph is differentiated. The CTC loss gradient is
    a fixed supplied direction, so unsupported CTC second derivatives and
    forward-mode traversal of discrete sampling decisions are not involved.
    """
    require_torch()
    if len(parameters) != len(direction):
        raise ValueError("JVP parameter and direction tuples must have equal length")
    output = _z(prediction)
    cotangent = torch.zeros_like(output, requires_grad=True)
    vjp = torch.autograd.grad(output, parameters, grad_outputs=cotangent,
                              create_graph=True, retain_graph=True, allow_unused=True)
    terms = [(derivative * vector.detach()).sum() for derivative, vector in zip(vjp, direction)
             if derivative is not None]
    if not terms:
        return torch.zeros_like(output).detach()
    product = torch.autograd.grad(sum(terms), cotangent, retain_graph=True, allow_unused=True)[0]
    return torch.zeros_like(output).detach() if product is None else product.detach()


def _fixed_ctc(prediction: CurvedPagePrediction, targets: CurvedPageTargets, matching: tuple[int, ...]) -> tuple[Any, dict[str, int], list[str]]:
    # The matcher validates GT geometry; this helper validates all target text,
    # including the labels of any geometrically excluded query.
    if not isinstance(targets.texts, (list, tuple)) or len(targets.texts) != len(matching):
        raise ValueError("one target text sequence is required per fixed match")
    n = len(matching)
    loss = prediction.feature_map.reshape(-1)[0] * 0
    counts = {"matched": n, "included_ctc": 0, "excluded_ctc": 0}
    statuses = ["unmatched:" + query.status for query in prediction.queries]
    for target_index, query_index in enumerate(matching):
        supplied = targets.texts[target_index]
        if isinstance(supplied, (list, tuple)) and not supplied:
            labels = torch.empty(0, dtype=torch.long)
        else:
            labels = torch.as_tensor(supplied)
        if labels.ndim != 1 or labels.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError("CTC targets must be integer sequences")
        if bool(((labels <= 0) | (labels >= prediction.alphabet_size)).any()):
            raise ValueError("CTC targets must exclude blank and lie within the alphabet")
        query = prediction.queries[query_index]
        if query.grid is None or query.ctc_logits is None:
            status = "excluded_" + query.status
        else:
            result = masked_curve_ctc_loss(query.ctc_logits, query.grid, labels)
            status = result.status
            loss = loss + result.loss
        statuses[query_index] = status
        if status == "included":
            counts["included_ctc"] += 1
        else:
            counts["excluded_ctc"] += 1
    charge = prediction.max_samples * math.log(prediction.alphabet_size) * counts["excluded_ctc"]
    return (loss + charge) / max(n, 1), counts, statuses


def _signature(prediction: CurvedPagePrediction, targets: CurvedPageTargets, matching: tuple[int, ...], statuses: list[str]) -> dict[str, Any]:
    masks = []
    for query in prediction.queries:
        grid = query.grid
        if grid is None:
            masks.append(None)
        else:
            masks.append({name: getattr(grid, name).detach().tolist() for name in (
                "sample_valid", "cell_valid", "projection_valid", "jacobian_valid", "line_valid",
                "orientation_valid", "contour_valid", "sharp_turn_valid", "tangent_valid",
            )})
    return {
        "matching_fixed": list(matching), "matching_recomputed": list(match_curved_lines(prediction, targets)),
        "query_statuses": [query.status for query in prediction.queries],
        "time_steps": [None if query.ctc_logits is None else query.ctc_logits.shape[0] for query in prediction.queries],
        "masks": masks, "ctc_statuses": list(statuses),
    }


def discrete_stability(before: dict[str, Any], after: dict[str, Any]) -> dict[str, bool]:
    """Compare discrete signatures without mutating or discarding either one."""
    checks = {
        "time_steps_unchanged": before["time_steps"] == after["time_steps"],
        "masks_unchanged": before["masks"] == after["masks"],
        "query_statuses_unchanged": before["query_statuses"] == after["query_statuses"],
        "matching_unchanged": (before["matching_recomputed"] == after["matching_recomputed"]
                               == before["matching_fixed"] == after["matching_fixed"]),
        "ctc_statuses_unchanged": before["ctc_statuses"] == after["ctc_statuses"],
        "all_ctc_included": all(signature["ctc_statuses"][query] == "included"
                                for signature in (before, after) for query in signature["matching_fixed"]),
    }
    checks["valid"] = all(checks.values())
    return checks


def _groups(model: TinyCurvedJointPage, reader_encoder: Any = None) -> dict[str, list[tuple[str, Any]]]:
    encoder = model.encoder if reader_encoder is None else reader_encoder
    return {
        "E": [("encoder." + name, parameter) for name, parameter in encoder.named_parameters()],
        "G": [("queries", model.queries)] + [("geometry_head." + name, parameter) for name, parameter in model.geometry_head.named_parameters()],
        "R": [("ctc_head." + name, parameter) for name, parameter in model.ctc_head.named_parameters()],
        "other": [(name, parameter) for name, parameter in model.named_parameters()
                  if name.startswith("presence_head.") or name.startswith("order_head.")],
    }


def _separate_forward(model: TinyCurvedJointPage, reader_encoder: Any, image: Any) -> CurvedPagePrediction:
    # The geometry encoder has distinct parameter storage and is never updated.
    # Its ordinary model forward also computes unused reader logits; report that
    # overhead rather than presenting this instrumentation as an efficient S2.
    with torch.no_grad():
        geometry_prediction = model(image, detach_geometry=True)
    reader_features = reader_encoder(image)
    queries = []
    for query in geometry_prediction.queries:
        if query.grid is None:
            queries.append(replace(query, ctc_logits=None))
        else:
            sampled = sample_curved_band(reader_features, query.grid, geometry_prediction.frame, detach_geometry=True)
            queries.append(replace(query, ctc_logits=model.ctc_head(sampled.as_ctc_sequence(require_valid=False))))
    return replace(geometry_prediction, feature_map=reader_features, queries=tuple(queries))


def _directions(flat: Any, parameters: list[tuple[str, Any]]) -> tuple[Any, ...]:
    pieces, offset = [], 0
    for _, parameter in parameters:
        pieces.append(flat[offset:offset + parameter.numel()].reshape_as(parameter).detach())
        offset += parameter.numel()
    if offset != flat.numel():
        raise ValueError("flat direction does not match parameter block")
    return tuple(pieces)


def _gradient_data(loss: Any, prediction: CurvedPagePrediction, groups: dict[str, list[tuple[str, Any]]]) -> tuple[dict[str, Any], dict[int, Any], Any, bool]:
    trainable = [parameter for block in groups.values() for _, parameter in block if parameter.requires_grad]
    derivatives = torch.autograd.grad(loss, trainable, retain_graph=True, allow_unused=True)
    by_id = {id(parameter): None if gradient is None else gradient.detach().clone()
             for parameter, gradient in zip(trainable, derivatives)}
    blocks = {}
    for name in ("E", "G", "R"):
        blocks[name] = torch.cat([
            (torch.zeros_like(parameter) if by_id.get(id(parameter)) is None else by_id[id(parameter)]).reshape(-1)
            for _, parameter in groups[name]
        ]).detach()
    geometry = (prediction.baseline_xy, prediction.height_px)
    active = [value for value in geometry if value.requires_grad]
    coordinate = None
    if active:
        values = torch.autograd.grad(loss, active, retain_graph=True, allow_unused=True)
        supplied = dict(zip((id(value) for value in active), values))
        if any(value is not None for value in values):
            coordinate = torch.cat([(torch.zeros_like(value) if supplied.get(id(value)) is None else supplied[id(value)]).reshape(-1)
                                    for value in geometry]).detach()
    other_absent = all(by_id.get(id(parameter)) is None for _, parameter in groups["other"])
    return blocks, by_id, coordinate, other_absent


def _cosine(first: Any, second: Any, floor: float) -> float | None:
    a, b = _norm(first), _norm(second)
    return None if min(a, b) <= floor else float(torch.dot(first, second)) / (a * b)


def run_one_step_ablation(
    base_model: TinyCurvedJointPage, image: Any, targets: CurvedPageTargets, *,
    learning_rate: float, minimum_change_norm: float, maximum_relative_residual: float, residual_floor: float,
) -> CausalAblationResult:
    """Run exactly one plain-SGD CTC step per arm, without tuning or source edits."""
    require_torch()
    started = time.perf_counter()
    for name, value, positive in (("learning_rate", learning_rate, True), ("minimum_change_norm", minimum_change_norm, False),
                                  ("maximum_relative_residual", maximum_relative_residual, False), ("residual_floor", residual_floor, True)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or (positive and value == 0):
            raise ValueError(f"{name} must be an explicit finite {'positive' if positive else 'nonnegative'} number")
    if not isinstance(base_model, TinyCurvedJointPage) or image.dtype != torch.float64 or image.device.type != "cpu":
        raise ValueError("this causal probe requires TinyCurvedJointPage and CPU float64 input")
    if any(parameter.dtype != torch.float64 or parameter.device.type != "cpu" for parameter in base_model.parameters()):
        raise ValueError("all source-model parameters must be CPU float64")
    if not targets.baseline_xy:
        raise ValueError("the causal probe requires at least one matched line")
    source_before = _group_hash(list(base_model.named_parameters()))
    models = {name: deepcopy(base_model) for name in ("S2", "B3", "B4")}
    separate_reader = deepcopy(base_model.encoder)
    for parameter in models["S2"].parameters():
        parameter.requires_grad_(False)
    for parameter in models["S2"].ctc_head.parameters():
        parameter.requires_grad_(True)
    groups = {name: _groups(model, separate_reader if name == "S2" else None) for name, model in models.items()}
    geometry_encoder = [("encoder." + name, parameter) for name, parameter in models["S2"].encoder.named_parameters()]
    frozen_encoder_before = _flat(geometry_encoder)
    storage_sets = [{parameter.untyped_storage().data_ptr() for parameter in base_model.parameters()}]
    for name in models:
        owned = [parameter for block in groups[name].values() for _, parameter in block]
        if name == "S2":
            owned += [parameter for _, parameter in geometry_encoder]
        storage_sets.append({parameter.untyped_storage().data_ptr() for parameter in owned})
    independent = all(storage_sets[i].isdisjoint(storage_sets[j]) for i in range(len(storage_sets)) for j in range(i + 1, len(storage_sets)))
    separate_geometry_pointers = tuple(parameter.untyped_storage().data_ptr() for _, parameter in geometry_encoder)
    separate_reader_pointers = tuple(parameter.untyped_storage().data_ptr() for _, parameter in groups["S2"]["E"])
    independent &= set(separate_geometry_pointers).isdisjoint(separate_reader_pointers)
    initial_hashes = {name: {block: _group_hash(groups[name][block]) for block in ("E", "G", "R")} for name in models}
    initial_hashes["S2"]["geometry_encoder"] = _group_hash(geometry_encoder)
    initial_flat = {name: {block: _flat(groups[name][block]) for block in ("E", "G", "R")} for name in models}
    initial_predictions, initial_losses, initial_counts, initial_signatures = {}, {}, {}, {}
    gradients, parameter_gradients, coordinate_gradients, other_absent, arm_times = {}, {}, {}, {}, {}
    matching = None
    for name, model in models.items():
        tick = time.perf_counter()
        prediction = _separate_forward(model, separate_reader, image) if name == "S2" else model(image, detach_geometry=name == "B3")
        if matching is None:
            matching = match_curved_lines(prediction, targets)
        loss, counts, statuses = _fixed_ctc(prediction, targets, matching)
        initial_predictions[name], initial_losses[name], initial_counts[name] = prediction, loss, counts
        initial_signatures[name] = _signature(prediction, targets, matching, statuses)
        gradients[name], parameter_gradients[name], coordinate_gradients[name], other_absent[name] = _gradient_data(loss, prediction, groups[name])
        arm_times[name] = time.perf_counter() - tick
    # Evaluate all geometry JVPs before any in-place optimizer update. Directions
    # are detached snapshots, so no CTC gradient is differentiated here.
    b3_direction = -learning_rate * gradients["B3"]["E"]
    extra_encoder_direction = -learning_rate * (gradients["B4"]["E"] - gradients["B3"]["E"])
    geometry_direction = -learning_rate * gradients["B4"]["G"]
    b3_prediction = geometry_jvp(initial_predictions["B3"], tuple(p for _, p in groups["B3"]["E"]), _directions(b3_direction, groups["B3"]["E"]))
    encoder_contribution = geometry_jvp(initial_predictions["B4"], tuple(p for _, p in groups["B4"]["E"]), _directions(extra_encoder_direction, groups["B4"]["E"]))
    geometry_contribution = geometry_jvp(initial_predictions["B4"], tuple(p for _, p in groups["B4"]["G"]), _directions(geometry_direction, groups["B4"]["G"]))
    arms = {}
    for name, model in models.items():
        tick = time.perf_counter()
        optimizer_parameters = [parameter for block in ("E", "G", "R") for _, parameter in groups[name][block] if parameter.requires_grad]
        optimizer = torch.optim.SGD(optimizer_parameters, lr=learning_rate, momentum=0.0, weight_decay=0.0)
        optimizer.zero_grad(set_to_none=True)
        for parameter in optimizer_parameters:
            gradient = parameter_gradients[name].get(id(parameter))
            parameter.grad = None if gradient is None else gradient.clone()
        optimizer.step()
        with torch.no_grad():
            final_prediction = _separate_forward(model, separate_reader, image) if name == "S2" else model(image, detach_geometry=name == "B3")
            final_loss, final_counts, final_statuses = _fixed_ctc(final_prediction, targets, matching)
            final_signature = _signature(final_prediction, targets, matching, final_statuses)
        final_hashes = {block: _group_hash(groups[name][block]) for block in ("E", "G", "R")}
        counts = {block: sum(parameter.numel() for _, parameter in groups[name][block]) for block in groups[name]}
        frozen_count = sum(parameter.numel() for _, parameter in geometry_encoder) if name == "S2" else 0
        counts.update(allocated_total=sum(counts.values()) + frozen_count,
                      optimizer_parameters=sum(parameter.numel() for parameter in optimizer_parameters),
                      geometry_encoder_frozen_parameters=frozen_count,
                      excluded_auxiliary_parameters=counts["other"])
        auxiliary = {"other_parameters_received_ctc_gradient": not other_absent[name],
                     "encoder_forward_passes_per_prediction": 2 if name == "S2" else 1,
                     "discarded_geometry_reader_forward": name == "S2"}
        if name == "S2":
            final_hashes["geometry_encoder"] = _group_hash(geometry_encoder)
            auxiliary.update(geometry_encoder_delta=_flat(geometry_encoder) - frozen_encoder_before,
                             geometry_encoder_storage_pointers=separate_geometry_pointers,
                             reader_encoder_storage_pointers=separate_reader_pointers)
        arms[name] = ArmResult(
            initial_prediction=initial_predictions[name], final_prediction=final_prediction,
            initial_ctc=float(initial_losses[name].detach()), final_ctc=float(final_loss),
            initial_z=_z(initial_predictions[name]).detach().clone(), final_z=_z(final_prediction).detach().clone(),
            gradient_blocks=gradients[name],
            parameter_deltas={block: _flat(groups[name][block]) - initial_flat[name][block] for block in ("E", "G", "R")},
            coordinate_gradient=coordinate_gradients[name], counts_before=initial_counts[name], counts_after=final_counts,
            signature_before=initial_signatures[name], signature_after=final_signature,
            parameter_counts=counts, hashes_before=initial_hashes[name].copy(), hashes_after=final_hashes,
            elapsed_seconds=arm_times[name] + time.perf_counter() - tick, auxiliary=auxiliary,
        )
    b3_observed = arms["B3"].final_z - arms["B3"].initial_z
    b4_observed = arms["B4"].final_z - arms["B4"].initial_z
    observed43 = arms["B4"].final_z - arms["B3"].final_z
    predicted43 = encoder_contribution + geometry_contribution
    residual3, residual43 = b3_observed - b3_prediction, observed43 - predicted43
    relative3 = _norm(residual3) / max(_norm(b3_observed), residual_floor)
    relative43 = _norm(residual43) / max(_norm(observed43), residual_floor)
    projection_denominator = float(torch.dot(observed43, observed43))
    decomposition = {
        "b3_encoder_prediction": b3_prediction, "b3_observed": b3_observed, "b4_observed": b4_observed,
        "b3_residual": residual3,
        "encoder_contribution": encoder_contribution, "geometry_head_contribution": geometry_contribution,
        "predicted_delta43": predicted43, "observed_delta43": observed43, "residual_delta43": residual43,
        "absolute_residual_b3": _norm(residual3), "absolute_residual_delta43": _norm(residual43),
        "relative_residual_b3": relative3, "relative_residual_delta43": relative43,
        "signed_projection_encoder": None if _norm(observed43) <= residual_floor else float(torch.dot(encoder_contribution, observed43)) / projection_denominator,
        "signed_projection_geometry_head": None if _norm(observed43) <= residual_floor else float(torch.dot(geometry_contribution, observed43)) / projection_denominator,
        "cosine_components": _cosine(encoder_contribution, geometry_contribution, residual_floor),
        "cosine_prediction_observed": _cosine(predicted43, observed43, residual_floor),
        "report_drift_ratio_b3_over_b4": _norm(b3_observed) / max(_norm(b4_observed), residual_floor),
        "contribution_norm_ratio_geometry_over_encoder": _norm(geometry_contribution) / max(_norm(encoder_contribution), residual_floor),
        "drift_ratio_is_not_a_causal_share": True,
        "signed_ratios_are_not_causal_percentages": True,
        "directions_detached_before_jvp": True,
    }
    source_after = _group_hash(list(base_model.named_parameters()))
    reference = arms["B3"]
    forward_equal = all(
        torch.equal(arm.initial_z, reference.initial_z)
        and torch.equal(arm.initial_prediction.feature_map, reference.initial_prediction.feature_map)
        and torch.equal(arm.initial_prediction.presence_logits, reference.initial_prediction.presence_logits)
        and torch.equal(arm.initial_prediction.order_logits, reference.initial_prediction.order_logits)
        and arm.initial_ctc == reference.initial_ctc and arm.signature_before == reference.signature_before
        and all((first.ctc_logits is None and second.ctc_logits is None)
                or (first.ctc_logits is not None and second.ctc_logits is not None and torch.equal(first.ctc_logits, second.ctc_logits))
                for first, second in zip(arm.initial_prediction.queries, reference.initial_prediction.queries))
        for arm in arms.values()
    )
    checks = {
        "initial_hashes_equal": all(len({initial_hashes[name][block] for name in models}) == 1 for block in ("E", "G", "R"))
                                 and initial_hashes["S2"]["E"] == initial_hashes["S2"]["geometry_encoder"],
        "initial_forward_equal": forward_equal, "storage_independent": independent,
        "source_model_unchanged": source_before == source_after,
        "s2_geometry_bit_identical": torch.equal(arms["S2"].initial_z, arms["S2"].final_z)
                                     and torch.count_nonzero(arms["S2"].auxiliary["geometry_encoder_delta"]).item() == 0,
        "b3_geometry_changed": _norm(b3_observed) > minimum_change_norm,
        "b3_direct_coordinate_gradient_absent": arms["B3"].coordinate_gradient is None or _norm(arms["B3"].coordinate_gradient) == 0,
        "b3_G_gradient_zero": _norm(gradients["B3"]["G"]) == 0,
        "b4_direct_coordinate_gradient_nonzero": arms["B4"].coordinate_gradient is not None and _norm(arms["B4"].coordinate_gradient) > minimum_change_norm,
        "b4_G_gradient_nonzero": _norm(gradients["B4"]["G"]) > minimum_change_norm,
        "s2_reader_step_equals_b3": torch.equal(arms["S2"].parameter_deltas["E"], arms["B3"].parameter_deltas["E"]),
        "b3_b4_R_gradient_equal": torch.equal(gradients["B3"]["R"], gradients["B4"]["R"]),
        "other_parameter_gradients_absent": all(other_absent.values()),
        "discrete_stable": all(discrete_stability(arm.signature_before, arm.signature_after)["valid"] for arm in arms.values()),
        "linearization_b3_within_tolerance": relative3 <= maximum_relative_residual,
        "linearization_delta43_within_tolerance": relative43 <= maximum_relative_residual,
    }
    checks["valid"] = all(checks.values())
    decomposition["attribution_valid"] = checks["valid"]
    return CausalAblationResult(
        arms=arms, decomposition=decomposition, checks=checks, initial_hashes=initial_hashes, matching=matching,
        thresholds={"learning_rate": learning_rate, "minimum_change_norm": minimum_change_norm,
                    "maximum_relative_residual": maximum_relative_residual, "residual_floor": residual_floor},
        source_hash_before=source_before, source_hash_after=source_after, elapsed_seconds=time.perf_counter() - started,
    )
