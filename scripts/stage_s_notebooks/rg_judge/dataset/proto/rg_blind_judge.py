"""Blind mfn judge: frozen RG-17/19 metric, page aggregates only, stdlib.

No inference. GT stays at p6. Use only with independently pinned input SHAs.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import signal
import sys

PAGES = tuple(f"mfn_{i:04d}" for i in range(20))
STREAMS = ("detector_B3", "detector_S2")
METHODS = ("B1-g0.5", "C2-fixed", "G2-fixed")
METRIC_PATH = "src/babaorum/line_geometry.py"
BAND_PATH = "src/babaorum/experimental/stage_s_data.py"
SOURCE_SHA = {METRIC_PATH: "b2a36dd9ec36d3edb083d0c47d9d9e9a05f8963610ecac8c962345fb8284c948",
              BAND_PATH: "d61e9a8da772fa647197257b78659de9ec360c9b6b975bf73546a668f513c6d6"}


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def require_sha(value):
    if (not isinstance(value, str) or len(value) != 64 or
            any(c not in "0123456789abcdef" for c in value)):
        raise ValueError("invalid SHA256")
    return value


def read_pinned(path, expected):
    data = Path(path).read_bytes()
    if sha_bytes(data) != require_sha(expected):
        raise ValueError("pinned input mismatch")
    return data


def load_judge(root):
    """Import metric unchanged; compile the exact pinned _band function AST.

    _band has math as its only global dependency; the rest of its original
    module (numpy/Pillow/image loaders) is deliberately not executed.
    """
    root = Path(root)
    metric_bytes = read_pinned(root/METRIC_PATH, SOURCE_SHA[METRIC_PATH])
    band_bytes = read_pinned(root/BAND_PATH, SOURCE_SHA[BAND_PATH])
    name = "_rg_frozen_line_geometry"
    spec = importlib.util.spec_from_file_location(name, root/METRIC_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    # Execute the bytes whose SHA was checked, without a second file read.
    exec(compile(metric_bytes, str(root/METRIC_PATH), "exec"), module.__dict__)
    nodes = [node for node in ast.parse(band_bytes).body
             if isinstance(node, ast.FunctionDef) and node.name == "_band"]
    if len(nodes) != 1:
        raise ValueError("exactly one frozen _band required")
    namespace = {"math": math}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(root/BAND_PATH), "exec"), namespace)
    return module, namespace["_band"]


def reject_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def parse_json(data):
    return json.loads(data, object_pairs_hook=reject_duplicates)


def validate_predictions(data, freeze_sha):
    lines = data.decode("utf-8").splitlines()
    if not data.endswith(b"\n") or len(lines) != 121 or any(not line for line in lines):
        raise ValueError("exactly 121 newline-terminated prediction records required")
    provenance, *rows = [parse_json(line) for line in lines]
    if (set(provenance) != {"type", "schema", "scope", "coordinates", "freeze_sha256", "data_manifest_sha256"} or
            provenance["type"] != "provenance" or provenance["schema"] != "rg.v2.predictions.v1" or
            provenance["scope"] != "CONFIRMATION_MFN" or provenance["coordinates"] != "native_page_px" or
            provenance["freeze_sha256"] != freeze_sha):
        raise ValueError("wrong prediction provenance")
    require_sha(provenance["data_manifest_sha256"])
    expected = [(p, s, m) for p in PAGES for s in STREAMS for m in METHODS]
    seen_origins = {}
    for row, key in zip(rows, expected):
        if (set(row) != {"type", "page_id", "stream", "method", "native_wh", "fragment_sha256", "curves"} or
                row["type"] != "prediction" or (row["page_id"], row["stream"], row["method"]) != key):
            raise ValueError("wrong prediction panel/order/schema")
        dims = row["native_wh"]
        if not isinstance(dims, list) or len(dims) != 2 or any(type(v) is not int or v <= 0 for v in dims):
            raise ValueError("positive integer native dimensions required")
        require_sha(row["fragment_sha256"])
        origin = (tuple(dims), row["fragment_sha256"])
        if key[:2] in seen_origins and seen_origins[key[:2]] != origin:
            raise ValueError("methods used different page/fragment inputs")
        seen_origins[key[:2]] = origin
        if not isinstance(row["curves"], list):
            raise ValueError("curve list required")
        for curve in row["curves"]:
            if not isinstance(curve, list) or len(curve) < 2:
                raise ValueError("at least two curve points required")
            for point in curve:
                if (not isinstance(point, list) or len(point) != 2 or
                        any(type(v) not in (int, float) or not math.isfinite(v) for v in point)):
                    raise ValueError("finite XY points required")
            if any(b[0] <= a[0] for a, b in zip(curve, curve[1:])):
                raise ValueError("strictly increasing curve x required")
    return provenance, rows


def score_page(metric, band, page, curves):
    gts = []
    for line in page["lines"]:
        neg, pos = band(line["baseline"], line["polygon"])
        gts.append(metric.GroundTruthLine(id=line["id"], baseline=line["baseline"],
                                         height_px=neg+pos, baseline_source="ground_truth"))
    predictions = [metric.PredictedLine(id=f"p{k:05d}", baseline=curve, baseline_source="predicted")
                   for k, curve in enumerate(curves)]
    result = metric.evaluate_line_geometry(gts, predictions)
    if (result.status != "evaluated" or result.gt_non_evaluable_count or
            result.recall is None or (predictions and result.precision is None)):
        raise ValueError("entire GT page must be evaluable")
    return {"precision": result.precision if predictions else 0., "recall": result.recall,
            "n_gt": len(gts), "n_refined": len(predictions), "n_matches": len(result.matches)}


def run(args):
    freeze_data = read_pinned(args.freeze, args.freeze_sha)
    frozen = parse_json(freeze_data)
    wrapper_sha = sha_bytes(Path(__file__).read_bytes())
    if (frozen["schema"] != "rg.v2.freeze.v1" or
            frozen["files_sha256"]["proto/rg_blind_judge.py"] != wrapper_sha or
            any(frozen["upstream_sha256"][name] != value for name, value in SOURCE_SHA.items())):
        raise ValueError("wrong frozen judge bundle")
    pred_bytes = read_pinned(args.predictions, args.predictions_sha)
    provenance, rows = validate_predictions(pred_bytes, args.freeze_sha)
    generation = read_pinned(args.generation_manifest, args.generation_sha)
    index = parse_json(read_pinned(args.gt_index, args.gt_index_sha))
    if (set(index) != {"schema", "generation_manifest_sha256", "pages"} or
            index["schema"] != "rg.blind.gt-index.v1" or
            index["generation_manifest_sha256"] != sha_bytes(generation) or
            set(index["pages"]) != set(PAGES)):
        raise ValueError("wrong GT index or source manifest")
    for value in index["pages"].values():
        require_sha(value)
    metric, band = load_judge(args.judge_root)
    scores = []
    for page_id in PAGES:
        # The explicit whitelist forbids opening any image or other page.
        page = parse_json(read_pinned(Path(args.gt_root)/"pages"/f"{page_id}.json", index["pages"][page_id]))
        if page["page_id"] != page_id:
            raise ValueError("GT page identity mismatch")
        dims = [page["image"]["width"], page["image"]["height"]]
        if any(type(v) is not int or v <= 0 for v in dims):
            raise ValueError("invalid GT dimensions")
        for row in rows[len(scores):len(scores)+6]:
            if row["page_id"] != page_id or row["native_wh"] != dims:
                raise ValueError("native prediction/GT dimensions differ")
            scores.append({k: row[k] for k in ("page_id", "stream", "method", "fragment_sha256")} |
                          {"gt_sha256": index["pages"][page_id]} |
                          score_page(metric, band, page, row["curves"]))
    if len(scores) != 120:
        raise ValueError("complete 120 score panel required")
    return {"schema": "rg.blind.scores.v1", "status": "complete", "scope": "CONFIRMATION_MFN",
            "predictions_sha256": args.predictions_sha, "freeze_sha256": args.freeze_sha,
            "wrapper_sha256": wrapper_sha, "judge_sha256": SOURCE_SHA,
            "generation_manifest_sha256": args.generation_sha, "gt_index_sha256": args.gt_index_sha,
            "data_manifest_sha256": provenance["data_manifest_sha256"],
            "python": platform.python_version(), "scores": scores}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("judge-root", "freeze", "freeze-sha", "predictions", "predictions-sha",
                 "generation-manifest", "generation-sha", "gt-index", "gt-index-sha", "gt-root", "output"):
        parser.add_argument("--"+name, required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError("exclusive fresh score output required")
    def expire(signum, frame):
        raise TimeoutError("blind judge deadline")
    signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, 3600)
    try:
        report = run(args)
    except Exception as error:
        # No GT values, IDs, coordinates, traceback or partial rows returned.
        report = {"schema": "rg.blind.scores.v1", "status": "failed", "error_type": type(error).__name__}
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    with Path(args.output).open("x") as out:
        out.write(json.dumps(report, allow_nan=False, indent=2)+"\n")
    print(json.dumps({"status": report["status"], "output_sha256": sha_bytes(Path(args.output).read_bytes())}))
    if report["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
