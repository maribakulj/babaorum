"""R3 (Ancre) — IC par page des hypothèses H-R3a et H-R3b sur la moitié eval.

Entrées : reports/reader_r3_dev_eval.jsonl.gz (sortie Kaggle 358707154, carnet
64a67508…, décodeur non corrigé) et reports/reader_r1d_eval_eval_c8.json (textes
gloutons A1 des mêmes lignes). IC : apparié par page, 10 000 tirages, graine 1,
percentiles 2,5 / 97,5, BnL et NewsEye séparés (comme le lot 1).
H-R3a : CER(glouton CTC) − CER(joint) ; revendication si l'IC est entièrement > 0.
H-R3b : CER(glouton CTC multitâche) − CER(A1) ; tenue si la borne haute < 0,1 point.
Sortie : reports/reader_r3_report.json.
"""
import gzip
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path.home() / "axel/src"))
from axel.reader.metric import line_errors  # noqa: E402

R = HERE / "reports"
a1 = {r["stem"]: r["a1"] for r in json.load(open(R / "reader_r1d_eval_eval_c8.json"))["rows"]}
rows = [json.loads(line) for line in gzip.open(R / "reader_r3_dev_eval.jsonl.gz", "rt")]
assert len(rows) == 2405 and all(r["stem"] in a1 for r in rows), "lignes eval différentes de R1d"
page = lambda s: s.split("__")[1]  # noqa: E731
ARMS = ("ctc", "att", "joint", "a1")
E: dict = {k: {} for k in ARMS}
for r in rows:
    for arm in ARMS:
        e, n = line_errors(a1[r["stem"]] if arm == "a1" else r[arm], r["truth"], "d150")
        acc = E[arm].setdefault(page(r["stem"]), [0, 0])
        acc[0] += e
        acc[1] += n


def cer(arm, ps):
    return 100 * sum(E[arm][p][0] for p in ps) / max(1, sum(E[arm][p][1] for p in ps))


def paired_ci(x, y, ps_pool, draws=10000, seed=1):
    rng = random.Random(seed)
    ds = []
    for _ in range(draws):
        ps = [rng.choice(ps_pool) for _ in ps_pool]
        ds.append(cer(x, ps) - cer(y, ps))
    ds.sort()
    return [ds[int(0.025 * draws)], ds[int(0.975 * draws) - 1]]


pages = sorted(E["ctc"])
strata: dict = {}
for p in pages:
    strata.setdefault("bnl" if p.isdigit() else "newseye", []).append(p)
ci_a = paired_ci("ctc", "joint", pages)
ci_b = paired_ci("ctc", "a1", pages)
out = {
    "lines": len(rows), "pages": len(pages),
    "cer_d150": {arm: cer(arm, pages) for arm in ARMS},
    "by_stratum": {k: {arm: cer(arm, v) for arm in ARMS} for k, v in strata.items()},
    "H_R3a": {"diff": "CER(ctc) - CER(joint)", "ci95": ci_a,
              "by_stratum": {k: paired_ci("ctc", "joint", v) for k, v in strata.items()},
              "claim": ci_a[0] > 0},
    "H_R3b": {"diff": "CER(ctc multitache) - CER(A1)", "ci95": ci_b,
              "by_stratum": {k: paired_ci("ctc", "a1", v) for k, v in strata.items()},
              "holds_non_inferiority_0.1": ci_b[1] < 0.1},
    "ci_spec": "apparié par page, 10 000 tirages, graine 1, percentiles ; BnL et NewsEye séparés",
    "decoder": "joint_decode NON corrigé (couloir aux fins), carnet 64a67508…, Kaggle 358707154",
    "provenance": {"r3": "reports/reader_r3_dev_eval.jsonl.gz", "a1": "reports/reader_r1d_eval_eval_c8.json"},
}
(R / "reader_r3_report.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
print(json.dumps(out, indent=1, ensure_ascii=False))
