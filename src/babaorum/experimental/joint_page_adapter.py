"""Prediction-only bridge from the tiny curved-page probe to page_graph.v1.

This adapter has no GT or training-loss argument. Every presence-selected query
is emitted, including failed geometry; the metric bridge preserves those failed
lines in its precision denominator. Sampling bands never become composition
polygons or word boxes. Unresolved order logits never become invented edges.

The PNG is actually decoded and hashed. Its dimensions are checked against the
prediction frame, but the model prediction does not contain its input pixels or
digest: linkage to those bytes is declared by the caller, not independently
verified here. Coordinates move from pixel centers to continuous pixel edges
by adding (.5,.5), without clipping, rounding or changing band distances.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import io
import math
from typing import Any

from ..line_geometry import PredictedLine
from ..page_graph import (
    COORDINATE_CONVENTION, COORDINATE_FRAME, NORMAL_CONVENTION, SCHEMA_VERSION,
    validate_page_graph,
)
from .joint_curved_page import CurvedPagePrediction
from .joint_geometry import require_torch, torch


@dataclass
class PageGraphAdaptation:
    graph: dict[str, Any]
    diagnostics: dict[str, Any]


def _decode_png(png_bytes: bytes, expected_size: tuple[int, int]) -> tuple[int, int]:
    if not isinstance(png_bytes, bytes):
        raise ValueError("png_bytes must be the actual encoded PNG bytes")
    try:
        from PIL import Image
    except ImportError as exc:  # optional dependency, no automatic installation
        raise RuntimeError("PNG verification requires Pillow in the optional experiment environment") from exc
    try:
        with Image.open(io.BytesIO(png_bytes)) as encoded:
            if encoded.format != "PNG":
                raise ValueError("png_bytes must encode a PNG image")
            if encoded.size != expected_size:
                raise ValueError("PNG dimensions do not match the prediction's original image frame")
            if getattr(encoded, "n_frames", 1) != 1:
                raise ValueError("the adapter requires a single-frame PNG page")
            encoded.verify()
        with Image.open(io.BytesIO(png_bytes)) as decoded:
            decoded.load()  # decode pixel data; do not apply EXIF orientation
            return decoded.size
    except (OSError, SyntaxError) as exc:
        raise ValueError("png_bytes must contain a complete decodable PNG") from exc


def _json_safe(value: Any) -> Any:
    """Keep diagnostics JSON-safe while retaining explicit failure reasons."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _empty_geometry(status: str) -> dict[str, Any]:
    return {"status": status, "points": None, "source": None}


def _empty_band(status: str) -> dict[str, Any]:
    return {"status": status, "normal_negative_px": None, "normal_positive_px": None, "source": None}


def _baseline(points: Any, orientation: str, width: int, height: int) -> tuple[dict[str, Any], str | None]:
    if not bool(torch.isfinite(points).all()):
        return _empty_geometry("error"), "nonfinite_baseline"
    converted = points.detach().double() + 0.5
    if bool(((converted[:, 0] < 0) | (converted[:, 0] > width)
             | (converted[:, 1] < 0) | (converted[:, 1] > height)).any()):
        return _empty_geometry("error"), "baseline_outside_continuous_image_bounds"
    if not bool((converted[1:] != converted[:-1]).any()):
        return _empty_geometry("error"), "entirely_collapsed_baseline"
    if orientation == "unknown":
        return _empty_geometry("error"), "baseline_orientation_unknown"
    axis, sign = {"ltr": (0, 1), "rtl": (0, -1), "ttb": (1, 1), "btt": (1, -1)}[orientation]
    if not bool(sign * (converted[-1, axis] - converted[0, axis]) > 0):
        return _empty_geometry("error"), "baseline_endpoint_orientation_conflict"
    return {"status": "available", "points": converted.tolist(), "source": "predicted"}, None


def _grid_diagnostics(grid: Any) -> dict[str, Any] | None:
    if grid is None:
        return None
    return {
        "pixel_xy": _json_safe(grid.pixel_xy), "shape": list(grid.pixel_xy.shape),
        "sample_count": grid.sample_valid.numel(), "valid_sample_count": int(grid.sample_valid.sum()),
        "cell_count": grid.cell_valid.numel(), "valid_cell_count": int(grid.cell_valid.sum()),
        "line_valid": bool(grid.line_valid), "orientation_valid": bool(grid.orientation_valid),
        "contour_valid": bool(grid.contour_valid), "sharp_turn_valid": bool(grid.sharp_turn_valid),
        "tangent_valid": bool(grid.tangent_valid),
        "boundary_loss": _json_safe(grid.boundary_loss), "fold_loss": _json_safe(grid.fold_loss),
        "coordinate_convention": "pixel_centers_0_width_minus_1_height_minus_1",
    }


def _tokens(logits: Any, alphabet_size: int) -> tuple[list[int] | None, str | None]:
    if logits is None:
        return None, "ctc_logits_unavailable"
    if (not isinstance(logits, torch.Tensor) or logits.ndim != 3 or logits.shape[0] < 2
            or logits.shape[1:] != (1, alphabet_size) or logits.dtype not in (torch.float32, torch.float64)
            or logits.device.type != "cpu" or not bool(torch.isfinite(logits).all())):
        return None, "invalid_ctc_logits"
    result, previous = [], None
    for label in logits.detach().argmax(-1)[:, 0].tolist():
        if label != 0 and label != previous:
            result.append(label)
        previous = label
    return result, None


def adapt_curved_page(
    prediction: CurvedPagePrediction,
    *,
    png_bytes: bytes,
    page_key: str,
    document_key: str | None,
    title_key: str | None,
    run_id: str,
    components: list[dict[str, Any]],
    presence_threshold: float,
    alphabet: tuple[str, ...] | None = None,
) -> PageGraphAdaptation:
    """Create and validate a partial page graph from actual predictions only.

    A cap-excluded query can retain an evaluable baseline while its band is
    missing and its text is in error. A folded/out-of-support sampling band is
    in error independently of baseline availability. All selected slots remain
    graph nodes. Counts describe emitted predictions, never unknown GT recall.
    """
    require_torch()
    if not isinstance(prediction, CurvedPagePrediction):
        raise ValueError("prediction must be a CurvedPagePrediction")
    prediction.frame.validate()
    points = prediction.baseline_xy
    if (not isinstance(points, torch.Tensor) or points.ndim != 3 or points.shape[-1] != 2
            or points.shape[1] < 2 or not 1 <= points.shape[0] <= 6
            or points.dtype not in (torch.float32, torch.float64) or points.device.type != "cpu"):
        raise ValueError("baselines must have shape [1<=Q<=6,K>=2,2] in CPU float32/float64")
    q = len(points)
    for value, shape in ((prediction.height_px, (q,)), (prediction.presence_logits, (q,)),
                          (prediction.order_logits, (q, q))):
        if (not isinstance(value, torch.Tensor) or value.shape != shape
                or value.dtype not in (torch.float32, torch.float64) or value.device.type != "cpu"):
            raise ValueError("prediction height/presence/order tensors have invalid shapes or dtypes")
    if not bool(torch.isfinite(prediction.presence_logits).all()):
        raise ValueError("presence logits must be finite to make explicit selection decisions")
    if (isinstance(presence_threshold, bool) or not isinstance(presence_threshold, (int, float))
            or not math.isfinite(presence_threshold) or not 0 <= presence_threshold <= 1):
        raise ValueError("presence_threshold must be finite and in [0,1]")
    if not math.isfinite(prediction.baseline_fraction) or not 0 <= prediction.baseline_fraction <= 1:
        raise ValueError("prediction baseline_fraction must lie in [0,1]")
    if prediction.orientation not in ("ltr", "rtl", "ttb", "btt", "unknown"):
        raise ValueError("prediction orientation is not a supported contract value")
    intrinsic_states = {"sampled", "invalid_geometry", "degenerate_curve", "sampling_cap_exceeded",
                        "invalid_height", "numerical_geometry_failure"}
    if len(prediction.queries) != q or any(query.status not in intrinsic_states for query in prediction.queries):
        raise ValueError("queries must contain intrinsic inference statuses; GT-dependent loss statuses are forbidden")
    if not isinstance(prediction.alphabet_size, int) or prediction.alphabet_size < 2:
        raise ValueError("prediction alphabet_size must include blank plus at least one token")
    if alphabet is not None:
        if (not isinstance(alphabet, tuple) or len(alphabet) != prediction.alphabet_size or alphabet[0] != ""
                or any(not isinstance(symbol, str) or not symbol for symbol in alphabet[1:])):
            raise ValueError("alphabet must be a tuple of A strings with blank '' at index 0 and nonempty other symbols")
    width, height = _decode_png(png_bytes, prediction.frame.original_size_wh)
    image_sha256 = hashlib.sha256(png_bytes).hexdigest()
    probabilities = prediction.presence_logits.detach().sigmoid().tolist()
    lines, query_diagnostics = [], []
    for index, query in enumerate(prediction.queries):
        selected = probabilities[index] >= presence_threshold
        line_id = f"line_q{index:03d}"
        baseline, baseline_reason = _baseline(points[index], prediction.orientation, width, height)
        h = float(prediction.height_px[index].detach())
        negative, positive = prediction.baseline_fraction * h, (1 - prediction.baseline_fraction) * h
        if baseline["status"] != "available":
            band, band_reason = _empty_band("error"), "baseline_unavailable"
        elif not math.isfinite(h) or h <= 0:
            band, band_reason = _empty_band("error"), "invalid_predicted_height"
        elif query.grid is None:
            if query.status in {"sampling_cap_exceeded", "sampled"}:
                band, band_reason = _empty_band("missing"), "sampling_validity_unavailable"
            else:
                band, band_reason = _empty_band("error"), "intrinsic_sampling_failure"
        elif not bool(query.grid.line_valid):
            band, band_reason = _empty_band("error"), "invalid_sampling_band"
        else:
            band = {"status": "available", "normal_negative_px": negative,
                    "normal_positive_px": positive, "source": "predicted"}
            band_reason = None
        tokens, token_error = _tokens(query.ctc_logits, prediction.alphabet_size)
        if baseline["status"] != "available":
            text, text_reason = {"status": "error", "value": None}, "baseline_unavailable"
        elif query.status != "sampled":
            text, text_reason = {"status": "error", "value": None}, "intrinsic_" + query.status
        elif query.grid is None or not bool(query.grid.line_valid):
            text, text_reason = {"status": "error", "value": None}, "sampling_invalid_or_unavailable"
        elif band["status"] != "available":
            text, text_reason = {"status": "error", "value": None}, "baseline_band_unavailable"
        elif token_error is not None:
            text, text_reason = {"status": "error", "value": None}, token_error
        elif alphabet is None:
            text, text_reason = {"status": "missing", "value": None}, "alphabet_unavailable"
        else:
            text = {"status": "available", "value": "".join(alphabet[token] for token in tokens)}
            text_reason = None
        if selected:
            lines.append({
                "id": line_id, "status": "partial" if baseline["status"] == "available" else "error",
                "polygon": _empty_geometry("missing"), "baseline": baseline,
                "baseline_semantics": "writing_line_foot_of_body", "orientation": prediction.orientation,
                "baseline_band": band, "text": text,
            })
        query_diagnostics.append({
            "query_index": index, "line_id": line_id, "selected": selected,
            "presence_probability": probabilities[index], "intrinsic_status": query.status,
            "baseline_reason": baseline_reason, "band_reason": band_reason, "text_reason": text_reason,
            "baseline_status": baseline["status"], "band_status": band["status"], "text_status": text["status"],
            "raw_baseline_xy": _json_safe(points[index]), "raw_height_px": _json_safe(h),
            "raw_band_distances": {"normal_negative_px": _json_safe(negative), "normal_positive_px": _json_safe(positive)},
            "ctc_token_ids": tokens, "ctc_tokens_usable": text["status"] == "available",
            "required_steps": query.required_steps, "failure_reason": query.failure_reason,
            "grid": _grid_diagnostics(query.grid),
        })
    line_available = sum(line["status"] != "error" for line in lines)
    line_errors = sum(line["status"] == "error" for line in lines)
    text_available = sum(line["text"]["status"] == "available" for line in lines)
    text_errors = sum(line["text"]["status"] == "error" for line in lines)

    def coverage(status: str, available: int = 0, errors: int = 0) -> dict[str, Any]:
        return {"status": status, "expected_count": None, "available_count": available, "error_count": errors}

    graph = {
        "schema_version": SCHEMA_VERSION,
        "identity": {"page_key": page_key, "document_key": document_key, "title_key": title_key},
        "coordinate_frame": COORDINATE_FRAME, "coordinate_convention": COORDINATE_CONVENTION,
        "normal_convention": NORMAL_CONVENTION,
        "image": {"sha256": image_sha256, "width": width, "height": height},
        "provenance": {"run_id": run_id, "components": deepcopy(components)},
        "run_status": "error" if lines and line_errors == len(lines) else "partial",
        "coverage": {
            "blocks": coverage("unavailable"), "words": coverage("unavailable"),
            "lines": coverage("error" if line_errors else "unknown", line_available, line_errors),
            "transcription": coverage("error" if text_errors else "unknown" if text_available else "unavailable", text_available, text_errors),
            "reading_order": coverage("unknown"),
        },
        "articles": [], "blocks": [], "lines": lines, "words": [], "relations": [],
    }
    validate_page_graph(graph)
    diagnostics = {
        "schema_version": "babaorum.joint_page_adapter.diagnostic.v1",
        "status": "UNTRAINED_SYNTHETIC_MECHANISM_ONLY", "image_sha256": image_sha256,
        "png_decode_verified": True, "prediction_input_content_verified": False,
        "image_binding": "caller_declared_prediction_link",
        "presence_threshold": float(presence_threshold),
        "selected_count": len(lines), "query_count": q,
        "queries": query_diagnostics,
        "coordinate_conversion": {"translation_xy": [0.5, 0.5],
                                  "source": "pixel_centers_0_width_minus_1_height_minus_1",
                                  "destination": COORDINATE_CONVENTION, "clamped": False},
        "order_logits": _json_safe(prediction.order_logits),
        "order_logits_finite": bool(torch.isfinite(prediction.order_logits).all()),
        "order_status": "unresolved_logits_only", "emitted_order_edges": 0,
        "word_localization_available": False, "composition_polygon_available": False,
    }
    return PageGraphAdaptation(graph=graph, diagnostics=diagnostics)


def geometry_predictions_from_graph(graph: dict[str, Any]) -> tuple[PredictedLine, ...]:
    """Bridge every emitted line to the metric, including missing/error ones.

    Baseline status governs evaluability, independently of a line's text, band
    or sampling status. Filtering failed baselines here would inflate precision.
    This function itself needs neither PyTorch nor Pillow.
    """
    validate_page_graph(graph)
    if any(line["baseline"]["source"] == "ground_truth" for line in graph["lines"]):
        raise ValueError("prediction metric bridge cannot accept ground_truth baseline provenance")
    return tuple(
        PredictedLine(
            id=line["id"], baseline=line["baseline"]["points"],
            baseline_source=line["baseline"]["source"], geometry_status=line["baseline"]["status"],
        )
        for line in graph["lines"]
    )
