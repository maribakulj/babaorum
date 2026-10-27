"""Lot 1c — R1d (faisceau joint, eval) contre R1b (re-notation) sur les mêmes lignes.

IC bootstrap 95 % par page (strates NewsEye / BnL), par corpus, et résumé de la
porte (reports/reader_gate_calib.json s'il existe). Sortie :
reports/reader_lot1_report.json.
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
d = json.load(open(R / "reader_r1d_eval_eval_c8.json"))
rows = {r["stem"]: r for r in d["rows"]}
rep1b = json.load(open(R / "reader_r1b.json"))
ch = rep1b["chosen"]
a, lam, beta = ch["a"], ch["lm"], ch["beta"]
cons = {}
for line in gzip.open(R / "reader_r1b_cands.jsonl.gz", "rt"):
    r = json.loads(line)
    if r["stem"] in rows:
        cons[r["stem"]] = max(r["cands"], key=lambda c: a * c["a"] + (1 - a) * c["e"] + lam * c["lm"] + beta * len(c["t"]))["t"]
stems = sorted(set(rows) & set(cons))
page = lambda s: s.split("__")[1]  # noqa: E731
E: dict = {"r1d": {}, "r1b": {}, "a1": {}}
for s in stems:
    t = rows[s]["truth"]
    for arm, h in (("r1d", rows[s]["joint"]), ("r1b", cons[s]), ("a1", rows[s]["a1"])):
        e, n = line_errors(h, t, "d150")
        acc = E[arm].setdefault(page(s), [0, 0])
        acc[0] += e
        acc[1] += n


def cer(arm, ps):
    return 100 * sum(E[arm][p][0] for p in ps) / max(1, sum(E[arm][p][1] for p in ps))


pages = sorted(E["r1d"])
strata: dict = {}
for p in pages:
    strata.setdefault("bnl" if p.isdigit() else "newseye", []).append(p)


def paired_ci(ps_pool, draws=10000, seed=1):
    """IC accepté par pM : différence appariée CER(R1d) − CER(R1b), rééchantillonnage
    des pages avec remise, 10 000 tirages, graine 1, percentiles 2,5 / 97,5."""
    rng = random.Random(seed)
    ds = []
    for _ in range(draws):
        ps = [rng.choice(ps_pool) for _ in ps_pool]
        ds.append(cer("r1d", ps) - cer("r1b", ps))
    ds.sort()
    return [ds[int(0.025 * draws)], ds[int(0.975 * draws) - 1]]


out = {
    "lines": len(stems), "pages": len(pages), "cfg_r1d": d["cfg"], "cfg_r1b": rep1b["chosen"],
    "cer_d150": {arm: cer(arm, pages) for arm in E},
    "by_stratum": {k: {arm: cer(arm, v) for arm in E} for k, v in strata.items()},
    "ci95_r1d_minus_r1b": paired_ci(pages),
    "ci95_r1d_minus_r1b_by_stratum": {k: paired_ci(v) for k, v in strata.items()},
    "ci_spec": "apparié par page, 10 000 tirages, graine 1, percentiles ; BnL et NewsEye séparés (NewsEye eval = 2 pages)",
    "target_r1b": {"cer_d150": 1.09863, "lines": 2405, "chosen": {"a": 0.8, "lm": 0.35, "beta": 2.0}},
    "newseye_pages": strata.get("newseye", []),
    "provenance": {"r1d": "reports/reader_r1d_eval_eval_c8.json", "r1b": "reports/reader_r1b_cands.jsonl.gz + reader_r1b.json",
                   "split_sha256_eval": "996ab0cdb199101463c87d458cd8ed8a2c34509a1e2c5fa93ef23e8de255c9bd"},
}
g = R / "reader_gate_calib.json"
if g.exists():
    gc = json.load(open(g))
    out["gate"] = {"theta": gc["theta"], "rule": gc["rule"], "n_true": gc["n_true"], "n_perturbed": gc["n_perturbed"],
                   "row_at_theta": next((r for r in gc["table"] if r["theta"] == gc["theta"]), None)}
(R / "reader_lot1_report.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
print(json.dumps(out, indent=1, ensure_ascii=False))
