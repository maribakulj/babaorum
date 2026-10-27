"""T2, dernière partie : marges et additivité sur les négatifs perturbés (w2:pD)."""
import gzip
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
from reader_t2_margins import additivity, bounds, margins  # noqa: E402

TIMING = Path.home() / "axel/runs/e17/kaggle/a1/timing.json"
la_labels = ("",) + tuple(json.load(open(TIMING))["chars"])
rep = {}
for name in ("a1", "kraken"):
    for mode in ("blank", "scramble", "blur_strong", "noise_strong"):
        f = HERE / f"reports/reader_pert_{name}_{mode}.jsonl.gz"
        labels, n, none, S, add, mins = la_labels, 0, 0, [], [], []
        for line in gzip.open(f, "rt"):
            r = json.loads(line)
            if "labels" in r:
                labels = tuple(r["labels"])
                continue
            n += 1
            if r.get("logp") is None:
                none += 1
                continue
            res = margins(np.asarray(r["logp"], np.float64), labels)
            if res is None:
                none += 1
                continue
            b = bounds(res["units"], res["words"], res["m"])
            if b["S_w"] is not None:
                S.append(b["S_w"])
            mins.append(min(res["m"]))
            a = additivity(res)
            if a and a["sum"] > 0:
                add.append(a["joint"] / a["sum"])
        rep[f"{name}/{mode}"] = {"lines": n, "no_output": none, "ge3_words": len(S),
                                 "S_w_median": float(np.median(S)) if S else None,
                                 "frac_S_w_gt_20": float(np.mean([s > 20 for s in S])) if S else None,
                                 "min_margin_median": float(np.median(mins)) if mins else None,
                                 "additivity_n": len(add),
                                 "additivity_ratio_q05_q50_q95": [float(np.quantile(add, q)) for q in (0.05, 0.5, 0.95)] if add else None}
(HERE / "reports/reader_t2_pert_report.json").write_text(json.dumps(rep, indent=1))
print(json.dumps(rep, indent=1))
