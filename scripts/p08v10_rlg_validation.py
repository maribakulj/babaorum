"""NOTE-B-P08-oracle-E27-correction v2.3 (GO pQ, full double judging), §§ 2-3: validation of rule R-LG (p1).

draw  : validation observations, in this fixed order: (a) the stage-2 draws of draws.json, in order, skipping those that fall on a tuning group
        (the distinct groups of stage 1), never re-drawn; (b) 40 PPS draws with replacement from the restricted population (the 855 suspect
        groups minus the tuning groups, p = chars / sum), numpy default_rng(20261031). Writes validation_draws.json (refused if it exists) with
        the same population and the sequence under the key "stage1", so that p08v9_instrument_planches.py renders it unchanged (--stage 1).
score : the judges' statuses, e-/e+, the agreement and the stops of p08v9_instrument_score.py (functions imported unchanged), then the INVERSE
        decision of § 3 (never `decide`): validated iff lower(e-) - 0.0005 >= 0.85; rejected iff upper(e+) + 0.0005 < 0.85; an empty sequence
        or both -> note B; otherwise "non conclu". m = 1 - e is reported (m- = 1 - e+, m+ = 1 - e-). Rule VT-002 (pA, before the judges):
        numbered V lines with an identical polygon count as one line (smallest number) in the statuses and the characters.
    PYTHONDONTWRITEBYTECODE=1 ~/axel/.venv/bin/python scripts/p08v10_rlg_validation.py draw --draws runs/p08v9_instrument/draws.json \
        --out-dir runs/p08v10_rlg
    AXEL_RUNS=... ~/axel/.venv/bin/python scripts/p08v9_instrument_planches.py --draws runs/p08v10_rlg/validation_draws.json --stage 1 \
        --out-dir runs/p08v10_rlg
    AXEL_RUNS=... ~/axel/.venv/bin/python scripts/p08v10_rlg_validation.py score --dir runs/p08v10_rlg
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("p08v9_instrument_score", ROOT / "scripts/p08v9_instrument_score.py")
sc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sc)

SEED_EXTRA = 20261031
N_EXTRA = 40
THRESHOLD_E = 0.85                       # m+ <= 0.15
GRID_STEP = sc.GRID_STEP


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def validation_sequence(draws: dict, seed: int = SEED_EXTRA, n: int = N_EXTRA) -> dict:
    tuning = set(draws["stage1"])
    kept = [g for g in draws["stage2"] if g not in tuning]
    restricted = [i for i in range(len(draws["population"])) if i not in tuning]
    w = np.array([draws["population"][i]["chars"] for i in restricted], dtype=float)
    extra = [restricted[int(j)] for j in np.random.default_rng(seed).choice(len(restricted), size=n, replace=True, p=w / w.sum())]
    return {"sequence": kept + extra, "stage2_kept": kept, "stage2_skipped": [g for g in draws["stage2"] if g in tuning], "extra": extra,
            "tuning_groups": sorted(tuning), "restricted_size": len(restricted)}


def canonical_v(polys: dict[int, tuple]) -> dict[int, int]:
    """VT-002 rule (pA, 24/10, before the judges): numbered V lines with an IDENTICAL polygon (same point list) count as ONE line, the
    smallest number. Returns {V number: canonical number}."""
    first: dict[tuple, int] = {}
    for v in sorted(polys):
        first.setdefault(tuple(map(tuple, polys[v])), v)
    return {v: first[tuple(map(tuple, polys[v]))] for v in polys}


def status_dedup(ans: dict, group_v: list[int], canon: dict[int, int]) -> dict[int, bool | None]:
    """judge_status of the instrument on canonical V numbers: answers are mapped to canonical numbers, the FULL attribution is the set of
    canonical numbers; split_status is the instrument's, unchanged. Without duplicates this equals sc.judge_status."""
    mapped = {p: (frozenset(canon[x] for x in s_) if s_ is not None else None) for p, s_ in ans.items()}
    allv = frozenset(canon.values())
    empty = {p: (s_ if s_ is not None else frozenset()) for p, s_ in mapped.items()}
    full = {p: (s_ if s_ is not None else allv) for p, s_ in mapped.items()}
    out = {}
    for v in sorted({canon[g] for g in group_v}):
        a, b = sc.split_status(empty, v), sc.split_status(full, v)
        out[v] = a if a == b else None
    return out


def decide_inverse(e_minus: list[float], e_plus: list[float]) -> dict:
    lo_m, hi_m = sc.cs(e_minus)
    lo_p, hi_p = sc.cs(e_plus)
    if np.isnan(lo_m) or np.isnan(hi_p):
        return {"cs_e_minus": [lo_m, hi_m], "cs_e_plus": [lo_p, hi_p], "validated": False, "rejected": False,
                "outcome": "séquence vide : note B"}
    validated = lo_m - GRID_STEP >= THRESHOLD_E
    rejected = hi_p + GRID_STEP < THRESHOLD_E
    if validated and rejected:
        outcome, validated, rejected = "anomalie (les deux) : arrêt, note B", False, False
    elif validated:
        outcome = "R-LG validé"
    elif rejected:
        outcome = "R-LG rejeté : note B"
    else:
        outcome = "non conclu : note B"
    return {"cs_e_minus": [lo_m, hi_m], "cs_e_plus": [lo_p, hi_p], "cs_m_minus": [1 - hi_p, 1 - lo_p], "cs_m_plus": [1 - hi_m, 1 - lo_m],
            "validated": validated, "rejected": rejected, "outcome": outcome}


def cmd_draw(a) -> int:
    out_dir = Path(a.out_dir)
    out = out_dir / "validation_draws.json"
    if out.exists():
        raise SystemExit(f"{out} exists, refused")
    draws_p = Path(a.draws)
    draws = json.loads(draws_p.read_text())
    v = validation_sequence(draws)
    report = {"note": "NOTE-B-P08-oracle-E27-correction.md v2.3 18bcc7f9 (GO pQ)", "inputs": {"draws": sha(draws_p), "script": sha(Path(__file__))},
              "population": draws["population"], "stage1": v["sequence"], "seed_extra": SEED_EXTRA,
              **{k: v[k] for k in ("stage2_kept", "stage2_skipped", "extra", "tuning_groups", "restricted_size")}}
    out_dir.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"observations": len(v["sequence"]), "stage2_kept": len(v["stage2_kept"]), "stage2_skipped": v["stage2_skipped"],
                      "distinct_groups": len(set(v["sequence"])), "sha": sha(out)}))
    return 0


def cmd_score(a) -> int:
    d = Path(a.dir)
    out_p = d / "scores_validation.json"
    if out_p.exists():
        raise SystemExit(f"{out_p} exists, refused")
    vd = json.loads((d / "validation_draws.json").read_text())
    pop, seq = vd["population"], vd["stage1"]
    man = json.loads((d / "planches_manifest_stage1.json").read_text())["groups"]
    sys.path.insert(0, str(ROOT / "scripts"))
    importlib.import_module("p08_decompose")                                     # puts axel on sys.path
    from axel.layout.gt import load_page
    RUNS = Path(os.environ["AXEL_RUNS"]).resolve()
    pages = {p["page"]: p for p in json.loads((RUNS / "e12/pages.json").read_text())}
    texts: dict[str, dict[str, str]] = {}
    per_group, ignored_log, duplicates = {}, {}, {}
    for gi in sorted(set(seq)):
        g, m = pop[gi], man[str(gi)]
        if g["page"] not in texts:
            texts[g["page"]] = {ln.id: (ln.text, ln.polygon) for ln in load_page(Path(pages[g["page"]]["xml"])).lines()}
        vnum = {gid: int(n) for n, gid in m["V"].items()}
        canon = canonical_v({int(n): texts[g["page"]][gid][1] for n, gid in m["V"].items()})
        if any(k != c for k, c in canon.items()):
            duplicates[str(gi)] = {str(k): c for k, c in canon.items() if k != c}
        group_v = [vnum[i] for i in g["gt_ids"]]
        chars = {}
        for i in g["gt_ids"]:                                                     # a duplicated true line counts once (VT-002)
            chars.setdefault(canon[vnum[i]], len(texts[g["page"]][i][0]))
        statuses = []
        for judge in ("J1", "J2"):
            f = d / "answers" / judge / f"{m['opaque']}.txt"
            if not f.exists():
                raise SystemExit(f"missing answer {f}")
            ans, ignored = sc.parse_answer(f.read_text(encoding="utf-8"), len(m["P"]), len(m["V"]))
            ignored_log[f"{judge}:{m['opaque']}"] = ignored
            statuses.append(status_dedup(ans, group_v, canon))
        per_group[gi] = {**sc.group_scores(statuses[0], statuses[1], chars), "s1": statuses[0], "s2": statuses[1]}
    pairs = [(per_group[gi]["s1"][v], per_group[gi]["s2"][v]) for gi in seq for v in per_group[gi]["s1"]
             if per_group[gi]["s1"][v] is not None and per_group[gi]["s2"][v] is not None]
    agr = sc.agreement(pairs)
    e_m = [per_group[gi]["e_minus"] for gi in seq]
    e_p = [per_group[gi]["e_plus"] for gi in seq]
    u_hat = float(np.mean([per_group[gi]["u"] for gi in seq]))
    stop = u_hat > sc.STOP_U or (agr["kappa"] is not None and agr["kappa"] < 0.4 and agr["raw"] < 0.85)
    report = {"observations": len(seq), "u_hat": u_hat, "agreement": agr, "stop": stop,
              "decision": None if stop else decide_inverse(e_m, e_p),
              "descriptive": {"e_minus": sc.descriptive(e_m), "e_plus": sc.descriptive(e_p)},
              "per_group": {str(gi): {k: v for k, v in per_group[gi].items() if k in ("e_minus", "e_plus", "u", "verdict")} for gi in per_group},
              "ignored_keys": ignored_log, "vt002_duplicates": duplicates,
              "inputs_sha": {p.name: sha(p) for p in [d / "validation_draws.json", d / "planches_manifest_stage1.json"]}
              | {"p08v9_instrument_score.py": sha(ROOT / "scripts/p08v9_instrument_score.py"), "script": sha(Path(__file__))}}
    if stop:
        report["outcome"] = "ARRÊT : instrument de jugement faible (indéterminé > 25 % ou accord faible) ; note B"
    out_p.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("observations", "u_hat", "agreement", "stop", "decision")}, default=str))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("draw")
    p.add_argument("--draws", required=True)
    p.add_argument("--out-dir", required=True)
    p = sub.add_parser("score")
    p.add_argument("--dir", required=True)
    a = ap.parse_args()
    return cmd_draw(a) if a.cmd == "draw" else cmd_score(a)


if __name__ == "__main__":
    raise SystemExit(main())
