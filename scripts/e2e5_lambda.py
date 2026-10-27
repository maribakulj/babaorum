#!/usr/bin/env python3
"""E2E-5 pilot: B3 with the L_ctc gradient entering enc_det multiplied by lambda = 1/6 (docs/END_TO_END_E2E5_LOT.txt).

Order of the run (amendment 1 §6 of the lot prevails over §0 to §4):
  1. guards: --confirmed-go, paths, output folder under /kaggle/working (or the smoke folder with --smoke);
  2. pages: the 60 pages of the E2E-3 run 3 output are checked by sha256 against its report (not in smoke);
     training pages mf_0000..0049 and evaluation pages mf_0050..0059, disjoint;
  3. step-0 control, before any optimiser step: L_ctc alone, same weights, same batch, flattened enc_det grad
     norm with s = 1/6 divided by the one with s = 1 must be 1/6 within 1e-4, else IMPLEMENTATION_MISMATCH;
  4. training of B3 alone for 2000 steps (cap 35 min), evaluation of the 10 pages at step 2000 with the
     ORIGINAL extraction (train_stage_s_v2.run_v2);
  5. status and report. R7 is written in the report only; the printed summary is status, sha and step-0 ratio.
R7 (B3, lambda): mean recall >= 0.5 and mean precision >= 0.5 over the 10 evaluation pages at step 2000 only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import traceback
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import train_stage_s_v2 as T  # noqa: E402
from train_stage_s import encode_text, lines_in_tile, load_scaled_page, perturb_baseline, sigma_for_step  # noqa: E402
from babaorum.experimental.stage_s_arms import StageSModel  # noqa: E402
from babaorum.experimental.stage_s_data import build_alphabet  # noqa: E402

CONFIG = ROOT / "config/e2e5_lambda.json"
KAGGLE_OUT = Path("/kaggle/working")
SMOKE_OUT = Path("/private/tmp/e2e5_smoke")
SUMMARY_KEYS = ("status", "report_sha256", "step0_ratio")
STATUSES = ("COMPLETE", "INCOMPLETE", "FINAL_EVAL_SKIPPED", "MISMATCH", "IMPLEMENTATION_MISMATCH", "FAILED")
LAMBDA = float(Fraction(1, 6))


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def flat_enc_det_grad(model) -> torch.Tensor:
    parts = [(p.grad if p.grad is not None else torch.zeros_like(p)).detach().reshape(-1)
             for p in model.enc_det.parameters()]
    return torch.cat(parts)


def first_batch(pages, rng, cfg, char_to_idx):
    """The batch that train_v2 draws at step 0 (same draws, no training)."""
    tile, centred = cfg["tiles"]["size_px"], cfg["tiles"]["line_centered_windows"]
    page = pages[int(rng.integers(len(pages)))]
    x0, y0 = T.draw_window(page, rng, tile, centred)
    image = np.ascontiguousarray(page["image"][y0:y0 + tile, x0:x0 + tile])
    inside = lines_in_tile(page["lines"], x0, y0, tile)
    readable = [ln for ln in inside if ln["legibility"] == "readable" and ln["text"]]
    take = min(cfg["ctc_lines_per_step_max"], len(readable))
    chosen = [readable[i] for i in rng.permutation(len(readable))[:take]] if take else []
    polylines = [perturb_baseline(ln["baseline"], ln["normal_negative_px"], ln["normal_positive_px"], rng)
                 for ln in chosen]
    labels = [encode_text(ln["text"], char_to_idx) for ln in chosen]
    return image, polylines, labels


def step0_control(model, batch, device, sigma_px: float, tolerance: float) -> dict:
    """Same weights and batch; two backward passes of L_ctc alone; ratio of flattened enc_det grad norms."""
    image, polylines, labels = batch
    image = torch.as_tensor(image, device=device)[None, None]
    norms = {}
    for s in (1.0, LAMBDA):
        model.ctc_det_grad_scale = s
        model.zero_grad(set_to_none=True)
        out = model.forward_lines(image, polylines, labels, sigma_px=sigma_px)
        included = [x for x in out["ctc_losses"] if x is not None]
        if not included:
            return {"evaluable": False, "ratio": None, "ok": False}
        (sum(included) / len(included)).backward()
        norms[s] = float(flat_enc_det_grad(model).norm())
    model.ctc_det_grad_scale = LAMBDA
    ratio = norms[LAMBDA] / norms[1.0] if norms[1.0] > 0 else None
    ok = ratio is not None and abs(ratio - LAMBDA) <= tolerance
    return {"evaluable": True, "ratio": ratio, "norm_s1": norms[1.0], "norm_lambda": norms[LAMBDA],
            "expected": LAMBDA, "tolerance": tolerance, "ok": bool(ok)}


def page_sha_check(root: Path, page_ids, expected: dict) -> dict:
    files = {}
    for pid in page_ids:
        for rel in (f"pages/{pid}.json", f"images/{pid}.png"):
            got = sha256(root / rel)
            files[rel] = {"sha256": got, "match": got == expected.get(rel)}
    return {"ok": all(v["match"] for v in files.values()), "files": len(files), "detail": files}


def r7_reading(eval_block, n_pages: int = 10) -> dict:
    """R7: mean recall >= 0.5 and mean precision >= 0.5 over the evaluation pages; precision 0 if n_pred = 0."""
    if len(eval_block) != n_pages:
        return {"evaluable": False, "pass": None, "n_pages": len(eval_block)}
    recalls, precisions = [], []
    for e in eval_block:
        after = e["metrics"]["after"]
        recalls.append(after["recall"])
        precisions.append(after["precision"] if (after["n_pred"] > 0 and after["precision"] is not None) else 0.0)
    rec, prec = sum(recalls) / n_pages, sum(precisions) / n_pages
    return {"evaluable": True, "recall_mean": rec, "precision_mean": prec, "pass": bool(rec >= 0.5 and prec >= 0.5),
            "n_pages": n_pages}


def public_summary(pilot_report_path: Path, report: dict) -> dict:
    """What the notebook prints: status, sha of the report, and the step-0 ratio. Nothing else."""
    return {"status": report["status"], "report_sha256": sha256(pilot_report_path),
            "step0_ratio": report.get("step0", {}).get("ratio")}


def check_output_dir(out_dir: Path, root: Path, smoke: bool) -> None:
    out = out_dir.resolve()
    if out == root.resolve() or root.resolve() in out.parents or out in root.resolve().parents:
        raise SystemExit("refused: the output must not be the pages folder or inside it")
    allowed = [SMOKE_OUT.resolve()] if smoke else [KAGGLE_OUT.resolve()]
    if not any(out == a or a in out.parents for a in allowed):
        raise SystemExit(f"refused: output {out} outside {[str(a) for a in allowed]}")


def run_pilot(args, cfg_e2e5: dict) -> dict:
    base_cfg_path = ROOT / cfg_e2e5["base_config"]["path"]
    if not sha256(base_cfg_path).startswith(cfg_e2e5["base_config"]["sha256_prefix"]):
        raise SystemExit("refused: base config v2 does not match the pre-registered sha prefix")
    cfg = json.loads(base_cfg_path.read_text())
    cfg["_config_sha256"] = sha256(base_cfg_path)
    target_h = cfg["scale"]["target_median_height_px"]
    root = args.tmp / "mf"
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    train_ids = args.train_ids or cfg_e2e5["pages"]["train"]
    eval_ids = args.eval_ids or cfg_e2e5["pages"]["eval"]
    if set(train_ids) & set(eval_ids):
        raise SystemExit("refused: an evaluation page is also a training page")
    steps = args.steps or cfg_e2e5["steps"]
    device = torch.device(args.device)
    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    torch.set_num_threads(cfg["threads"])

    report = {"schema_version": "e2e5_lambda_report_v1", "lambda": LAMBDA, "steps_requested": steps,
              "train_pages": list(train_ids), "eval_pages": list(eval_ids), "smoke": bool(args.smoke),
              "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
              "base_config_sha256": cfg["_config_sha256"],
              "script_sha256": sha256(Path(__file__).resolve())}

    expected = json.loads((args.run3_report).read_text())["steps"]["page_cache"]["sha256"]
    if args.smoke:
        report["page_sha_check"] = {"status": "NOT_CHECKED_SMOKE",
                                    "reason": "the Mac cache differs from the run 3 pages (see journal)"}
    else:
        check = page_sha_check(root, train_ids + eval_ids, expected)
        report["page_sha_check"] = {"status": "OK" if check["ok"] else "MISMATCH", "files": check["files"]}
        if not check["ok"]:
            report["status"] = "MISMATCH"
            return report

    train_loaded = [load_scaled_page(root, pid, target_h) for pid in train_ids]
    alphabet = build_alphabet(train_loaded)
    char_to_idx = {c: i for i, c in enumerate(alphabet) if i > 0}
    model = StageSModel("B3", len(alphabet), channels=cfg["model"]["channels"], K=cfg["model"]["K"],
                        hidden=cfg["model"].get("hidden", 128), ctc_det_grad_scale=LAMBDA).to(device)
    sigma0 = sigma_for_step(0, steps, target_h)
    batch = first_batch(train_loaded, rng, cfg, char_to_idx)
    report["step0"] = step0_control(model, batch, device, sigma0, cfg_e2e5["step0"]["tolerance"])
    if not report["step0"]["ok"]:
        report["status"] = "IMPLEMENTATION_MISMATCH"
        return report

    run_out = out_dir / "e2e5_lambda_run_report.json"
    run = T.run_v2(cfg, "B3", run_out, train_pages=train_ids, eval_pages=eval_ids, steps=steps, root=root,
                   device_name=args.device, time_cap_min=cfg_e2e5["time_cap_min"],
                   eval_every=cfg_e2e5["eval_every"], log_dir=out_dir / "logs", ctc_det_grad_scale=LAMBDA)
    report["run_report_sha256"] = sha256(run_out)
    report["run_steps_completed"] = run["steps_completed"]
    report["run_elapsed_seconds"] = run["elapsed_seconds"]
    report["criterion_eval_step"] = run["criterion_eval_step"]
    report["final_eval"] = run["final_eval"]
    report["step_seconds_mean"] = run["seconds_per_step_mean"]

    if run["steps_completed"] < steps:
        report["status"] = "INCOMPLETE"
    elif run["final_eval"] != "RUN":
        report["status"] = "FINAL_EVAL_SKIPPED"
    elif run["criterion_eval_step"] != steps:
        report["status"] = "INCOMPLETE"
    else:
        report["r7"] = r7_reading(run["eval"]) if not args.smoke else {"evaluable": False, "reason": "smoke"}
        report["status"] = "COMPLETE" if report["r7"]["evaluable"] or args.smoke else "INCOMPLETE"
    if args.smoke:
        report["status"] = "SMOKE"
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--confirmed-go", action="store_true")
    p.add_argument("--tmp", type=Path, required=False, help="folder holding mf/ (E2E-3 run 3 pages)")
    p.add_argument("--run3-report", type=Path, required=False, help="e2e3_fp_classes_run3.json (sha of the pages)")
    p.add_argument("--out-dir", type=Path, required=False)
    p.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    p.add_argument("--steps", type=int)
    p.add_argument("--train-ids", nargs="+")
    p.add_argument("--eval-ids", nargs="+")
    p.add_argument("--smoke", action="store_true", help="Mac smoke test only: output in /private/tmp/e2e5_smoke")
    args = p.parse_args(argv)
    if not args.confirmed_go:
        raise SystemExit("refused: E2E-5 runs only with --confirmed-go after the GO of w2:p6 and pK")
    for name in ("tmp", "run3_report", "out_dir"):
        if getattr(args, name) is None:
            raise SystemExit(f"refused: --{name.replace('_', '-')} is required")
    check_output_dir(args.out_dir, args.tmp / "mf", args.smoke)
    cfg_e2e5 = json.loads(CONFIG.read_text())
    out_path = args.out_dir / "e2e5_lambda_report.json"
    if out_path.exists():
        raise SystemExit("refused: the report already exists; a report is never overwritten")
    try:
        report = run_pilot(args, cfg_e2e5)
    except SystemExit:
        raise
    except Exception:  # the report is written even on failure
        report = {"status": "FAILED", "traceback": traceback.format_exc()[-2000:]}
    out_path.write_text(json.dumps(report, indent=2, allow_nan=False, default=float) + "\n")
    print(json.dumps(public_summary(out_path, report)))
    return report


if __name__ == "__main__":
    main()
