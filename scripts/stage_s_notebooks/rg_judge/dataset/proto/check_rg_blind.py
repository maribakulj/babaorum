"""Synthetic compatibility and privacy checks for the frozen blind judge."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import platform
from types import SimpleNamespace
from unittest.mock import patch

import rg_blind_judge as blind


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge-root", type=Path, default=Path("/Users/marcel/babaorum"))
    parser.add_argument("--scratch-dir", type=Path,
                        default=Path(__file__).resolve().parents[1]/".tmp/rg-24-blind-synthetic")
    parser.add_argument("--stdlib-only", action="store_true")
    args = parser.parse_args()
    root = args.scratch_dir
    root.mkdir(parents=True, exist_ok=False)
    (root/"pages").mkdir()
    written = {}
    def write(name, value):
        data = (json.dumps(value, allow_nan=False)+"\n").encode()
        (root/name).write_bytes(data)
        written[name] = data
        return blind.sha_bytes(data)

    freeze_sha = write("freeze.json", {
        "schema": "rg.v2.freeze.v1", "files_sha256": {
            "proto/rg_blind_judge.py": blind.sha_bytes(Path(blind.__file__).read_bytes())},
        "upstream_sha256": blind.SOURCE_SHA})
    generation_sha = write("generation.json", {"synthetic": True})
    gt_shas = {}
    curves = [[[0., 10.], [20., 10.]], [[0., 40.], [20., 40.]]]
    page_template = {"image": {"width": 100, "height": 200}, "lines": [
        {"id": f"secret_line{i}", "baseline": c,
         "polygon": [[0, y-5], [20, y-5], [20, y+5], [0, y+5]],
         "text": "PRIVATE_TEXT_NEVER_RETURN", "block_id": "PRIVATE_BLOCK"}
        for i, (c, y) in enumerate(zip(curves, (10, 40)))]}
    rows = []
    for pid in blind.PAGES:
        page = {"page_id": pid, **page_template}
        gt_shas[pid] = write(f"pages/{pid}.json", page)
        for stream in blind.STREAMS:
            for method in blind.METHODS:
                predictions = curves if method == "B1-g0.5" else (
                    curves[:1] if method == "C2-fixed" else [curves[0], curves[0]])
                rows.append({"type": "prediction", "page_id": pid, "stream": stream, "method": method,
                             "native_wh": [100, 200], "fragment_sha256": "1"*64, "curves": predictions})
    index = {"schema": "rg.blind.gt-index.v1", "generation_manifest_sha256": generation_sha, "pages": gt_shas}
    index_sha = write("index.json", index)
    provenance = {"type": "provenance", "schema": "rg.v2.predictions.v1", "scope": "CONFIRMATION_MFN",
                  "coordinates": "native_page_px", "freeze_sha256": freeze_sha, "data_manifest_sha256": "2"*64}
    def encode(prov, records):
        return ("\n".join(json.dumps(r, allow_nan=False) for r in [prov]+records)+"\n").encode()
    pred_bytes = encode(provenance, rows)
    (root/"predictions.jsonl").write_bytes(pred_bytes)
    call = SimpleNamespace(judge_root=args.judge_root, freeze=root/"freeze.json", freeze_sha=freeze_sha,
                           predictions=root/"predictions.jsonl", predictions_sha=blind.sha_bytes(pred_bytes),
                           generation_manifest=root/"generation.json", generation_sha=generation_sha,
                           gt_index=root/"index.json", gt_index_sha=index_sha, gt_root=root)

    # A read spy forbids any page beyond the 20 named fixtures, or any image.
    original_read = Path.read_bytes
    gt_reads = []
    def guard(path):
        if path.parent == root/"pages":
            assert path.name in {f"{p}.json" for p in blind.PAGES}
            gt_reads.append(path.name)
        assert "images" not in path.parts
        return original_read(path)
    with patch.object(Path, "read_bytes", guard):
        report = blind.run(call)
    assert len(gt_reads) == 20 and len(set(gt_reads)) == 20
    assert report["status"] == "complete" and len(report["scores"]) == 120
    expected = {"B1-g0.5": (1., 1., 2, 2), "C2-fixed": (1., .5, 1, 1), "G2-fixed": (.5, .5, 2, 1)}
    for row in report["scores"]:
        assert (row["precision"], row["recall"], row["n_refined"], row["n_matches"]) == expected[row["method"]]
        assert row["n_gt"] == 2
        assert set(row) == {"page_id", "stream", "method", "fragment_sha256", "gt_sha256",
                            "precision", "recall", "n_gt", "n_refined", "n_matches"}
    serialized = json.dumps(report)
    for secret in ("PRIVATE", "secret_line", "baseline", "polygon", "block_id", "text", "curves"):
        assert secret not in serialized

    # Full local compatibility with RG-17/19, without rewriting gt_list/score.
    exact_rows = 0
    if not args.stdlib_only:
        from rg_j1 import gt_list, score
        gts = gt_list(page_template)
        for row, prediction in zip(report["scores"], rows):
            old = score(gts, prediction["curves"])
            assert all(row[key] == value for key, value in old.items())
            exact_rows += 1

    rejected = 0
    def reject(prov, records, expected_exception=ValueError):
        nonlocal rejected
        try:
            blind.validate_predictions(encode(prov, records), freeze_sha)
        except expected_exception:
            rejected += 1
        else:
            raise AssertionError("invalid prediction accepted")
    for field, value in (("coordinates", "scaled_page_px"), ("scope", "SECOND_REGARD"),
                         ("freeze_sha256", "0"*64), ("data_manifest_sha256", "bad")):
        prov = dict(provenance, **{field: value})
        reject(prov, rows)
    for field, value in (("page_id", "mfn_0020"), ("stream", "detector_B4"), ("method", "retuned"),
                         ("native_wh", [True, 200]), ("fragment_sha256", "bad"),
                         ("curves", [[[10, 10], [0, 10]]]), ("curves", [[[0, 10]]])):
        bad = deepcopy(rows)
        bad[0][field] = value
        reject(provenance, bad)
    reject(provenance, rows[:-1])
    reject(provenance, rows+[rows[0]])
    bad = deepcopy(rows)
    bad[1]["fragment_sha256"] = "3"*64
    reject(provenance, bad)
    bad = deepcopy(rows)
    bad[0], bad[1] = bad[1], bad[0]
    reject(provenance, bad)

    # Empty predictions follow the old precision0 convention.
    metric, band = blind.load_judge(args.judge_root)
    empty = blind.score_page(metric, band, page_template, [])
    assert empty == {"precision": 0., "recall": 0., "n_gt": 2, "n_refined": 0, "n_matches": 0}
    try:
        blind.read_pinned(root/"generation.json", "0"*64)
    except ValueError:
        rejected += 1
    else:
        raise AssertionError("bad SHA accepted")
    original = written["pages/mfn_0000.json"]
    (root/"pages/mfn_0000.json").write_bytes(original+b" ")
    try:
        blind.run(call)
    except ValueError:
        rejected += 1
    else:
        raise AssertionError("tampered GT accepted")
    (root/"pages/mfn_0000.json").write_bytes(original)
    print(json.dumps({"status": "passed", "scope": "synthetic blind-judge compatibility only",
                      "stdlib_only": args.stdlib_only, "synthetic_scores": len(report["scores"]),
                      "exact_compatibility_rows_RG17_19": exact_rows, "invalid_cases_rejected": rejected,
                      "whitelisted_GT_reads": len(gt_reads), "no_line_details_returned": True,
                      "source_sha256": blind.SOURCE_SHA, "python": platform.python_version()}))


if __name__ == "__main__":
    main()
