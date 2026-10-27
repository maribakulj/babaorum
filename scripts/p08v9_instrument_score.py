"""NOTE-B-P08-instrument-E27 v2.8 (GO pT/pQ), §§ 2-4: scores of the blind annotations and the decision (p1).

Inputs: draws.json (population, stage1, stage2), planches_manifest_stage{1,2}.json, answers/<J1|J2>/<k>.txt (raw text of each judge's answer,
one per opaque planche k). Rules, all fixed by the note:
- answer (§ 2, pT C2): not an interpretable JSON object -> every P unreadable; a repeated P key -> those P unreadable; a P absent, or a value that
  is neither "illisible" nor a list of DISTINCT integers among the V numbers of the planche -> that P unreadable; other keys ignored and logged;
- status of a true line V OF THE GROUP under an attribution A: really split iff >= 2 P contain V, or V is contained by a P that contains >= 2
  numbered V (merge, inside or outside the group). For a judge it is computed under the EMPTY (unreadable P contains nothing) and the FULL
  (unreadable P contains every numbered V) attributions; determined iff both agree (the status is monotone in the content);
- V indeterminate iff undetermined for a judge, or both determined and different; e_g- counts indeterminate characters as right for E27,
  e_g+ as errors; u_g = e_g+ - e_g-; characters = len(true text), no normalisation;
- agreement (§ 2): one observation per group V per draw (duplicates included), V undetermined for a judge excluded; raw agreement and Cohen kappa;
- stop (§ 4.1): mean(u_g) > 0.25, or (kappa < 0.4 and raw agreement < 0.85);
- decision: hedged-capital betting confidence sequences (Waudby-Smith & Ramdas 2023, th. 3, predictable plug-in; alpha 0.05, theta 1/2,
  c 1/2, grid step 0.0005) on e_g- and e_g+ in DRAW order (stage 1 draws, then stage 2 draws); accept iff upper(e+) + 0.0005 <= 0.15;
  reject iff lower(e-) - 0.0005 > 0.15; empty set or both -> neither; neither -> stage 2 (after stage 1) or "instrument non validé" (after 84).
Descriptive only: mean, percentile bootstrap (default_rng(20261024), 10 000), Student; line_geometry accuracy on its 1:1 pairs.
    PYTHONDONTWRITEBYTECODE=1 ~/axel/.venv/bin/python scripts/p08v9_instrument_score.py --dir runs/p08v9_instrument --stage 1
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np

X = 0.15
GRID_STEP = 0.0005
ALPHA = 0.05
STOP_U = 0.25
UNREADABLE = None


def parse_answer(text: str, n_p: int, n_v: int) -> tuple[dict[int, frozenset | None], list[str]]:
    """{P number: frozenset of V numbers, or None if unreadable}, and the ignored keys (rule C2, no repair)."""
    pairs_seen: list[tuple[str, object]] = []
    try:
        obj = json.loads(text, object_pairs_hook=lambda pairs: (pairs_seen.extend(pairs), dict(pairs))[1])
    except (json.JSONDecodeError, TypeError, ValueError):
        obj = None
    if not isinstance(obj, dict):
        return {p: UNREADABLE for p in range(1, n_p + 1)}, []
    keys = [k for k, _ in pairs_seen if isinstance(k, str)]
    repeated = {k for k in keys if keys.count(k) > 1}
    out, ignored = {}, []
    valid_keys = {f"P{p}" for p in range(1, n_p + 1)}
    for k in obj:
        if k not in valid_keys:
            ignored.append(k)
    for p in range(1, n_p + 1):
        key = f"P{p}"
        v = obj.get(key, "absent")
        if key in repeated or key not in obj:
            out[p] = UNREADABLE
        elif v == "illisible":
            out[p] = UNREADABLE
        elif (isinstance(v, list) and all(isinstance(x, int) and not isinstance(x, bool) and 1 <= x <= n_v for x in v)
              and len(set(v)) == len(v)):
            out[p] = frozenset(v)
        else:
            out[p] = UNREADABLE
    return out, ignored


def split_status(attr: dict[int, frozenset], v: int) -> bool:
    holders = [p for p, s in attr.items() if v in s]
    return len(holders) >= 2 or any(len(attr[p]) >= 2 for p in holders)


def judge_status(ans: dict[int, frozenset | None], group_v: list[int], n_v: int) -> dict[int, bool | None]:
    empty = {p: (s if s is not None else frozenset()) for p, s in ans.items()}
    full = {p: (s if s is not None else frozenset(range(1, n_v + 1))) for p, s in ans.items()}
    out = {}
    for v in group_v:
        a, b = split_status(empty, v), split_status(full, v)
        out[v] = a if a == b else None
    return out


def group_scores(s1: dict, s2: dict, chars: dict[int, int]) -> dict:
    """e-, e+ and the per-V verdicts for one group, from the two judges' statuses (V numbers -> True/False/None)."""
    G = sum(chars.values())
    split = indet = 0
    verdict = {}
    for v, c in chars.items():
        a, b = s1[v], s2[v]
        if a is None or b is None or a != b:
            indet += c
            verdict[v] = "indeterminate"
        else:
            verdict[v] = "split" if a else "not_split"
            split += c if a else 0
    return {"e_minus": 1 - (split + indet) / G, "e_plus": 1 - split / G, "u": indet / G, "verdict": verdict}


def agreement(pairs: list[tuple[bool, bool]]) -> dict:
    n = len(pairs)
    if not n:
        return {"n": 0, "raw": None, "kappa": None}
    po = sum(a == b for a, b in pairs) / n
    p1, p2 = sum(a for a, _ in pairs) / n, sum(b for _, b in pairs) / n
    pe = p1 * p2 + (1 - p1) * (1 - p2)
    return {"n": n, "raw": po, "kappa": (po - pe) / (1 - pe) if pe < 1 else (1.0 if po == 1 else 0.0)}


def cs(x, alpha=ALPHA, theta=0.5, c=0.5, grid=None):
    """Hedged-capital betting confidence sequence (identical to the audited prototype instrument_E27_betting_cs_prototype.py, 4e1c7abd)."""
    grid = np.linspace(0, 1, 2001) if grid is None else grid
    x = np.asarray(x, float)
    n = len(x)
    i = np.arange(1, n + 1)
    sq = np.concatenate([[0], np.cumsum((x - (0.5 + np.cumsum(x)) / (i + 1)) ** 2)[:-1]])
    sig2 = (0.25 + sq) / i
    lam = np.sqrt(2 * np.log(2 / alpha) / (sig2 * i * np.log(i + 1)))
    kp = np.ones_like(grid)
    km = np.ones_like(grid)
    keep = np.ones_like(grid, bool)
    for t in range(n):
        lp = np.minimum(lam[t], c / np.maximum(grid, 1e-12))
        lm = np.minimum(lam[t], c / np.maximum(1 - grid, 1e-12))
        kp *= 1 + lp * (x[t] - grid)
        km *= 1 - lm * (x[t] - grid)
        keep &= np.maximum(theta * kp, (1 - theta) * km) < 1 / alpha
    g = grid[keep]
    return (float(g.min()), float(g.max())) if len(g) else (float("nan"), float("nan"))


def decide(e_minus: list[float], e_plus: list[float], stage: int) -> dict:
    lo_m, hi_m = cs(e_minus)
    lo_p, hi_p = cs(e_plus)
    accept = not np.isnan(hi_p) and hi_p + GRID_STEP <= X
    reject = not np.isnan(lo_m) and lo_m - GRID_STEP > X
    if accept and reject:
        accept = reject = False                                                 # only if a sequence missed: neither
    if accept:
        outcome = "E27 accepté POUR CET INDICATEUR (statut des lignes dans les 855 groupes suspects)"
    elif reject:
        outcome = "correction de l'oracle à décider par une note B"
    else:
        outcome = "étape 2" if stage == 1 else "instrument non validé : découpe à mesurer avec line_geometry seul, en sensibilité (lot relu)"
    return {"cs_e_minus": [lo_m, hi_m], "cs_e_plus": [lo_p, hi_p], "accept": accept, "reject": reject, "outcome": outcome}


def descriptive(x: list[float]) -> dict:
    a = np.asarray(x, float)
    rng = np.random.default_rng(20261024)
    boots = [a[rng.integers(0, len(a), len(a))].mean() for _ in range(10000)]
    sd = a.std(ddof=1) if len(a) > 1 else 0.0
    return {"mean": float(a.mean()), "bootstrap_percentile": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "student_approx_2sd": [float(a.mean() - 2.0 * sd / np.sqrt(len(a))), float(a.mean() + 2.0 * sd / np.sqrt(len(a)))]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--stage", type=int, choices=(1, 2), required=True)
    a = ap.parse_args()
    d = Path(a.dir)
    out_p = d / f"scores_stage{a.stage}.json"
    if out_p.exists():
        raise SystemExit(f"{out_p} exists, refused")
    draws = json.loads((d / "draws.json").read_text())
    pop = draws["population"]
    man = json.loads((d / "planches_manifest_stage1.json").read_text())["groups"]
    if a.stage == 2:
        man |= json.loads((d / "planches_manifest_stage2.json").read_text())["groups"]
    seq = draws["stage1"] + (draws["stage2"] if a.stage == 2 else [])
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    importlib.import_module("p08_decompose")                                     # puts axel on sys.path
    from axel.layout.gt import load_page
    texts: dict[str, dict[str, str]] = {}
    RUNS = Path(os.environ["AXEL_RUNS"]).resolve()
    pages = {p["page"]: p for p in json.loads((RUNS / "e12/pages.json").read_text())}
    per_group, ignored_log = {}, {}
    for gi in sorted(set(seq)):
        g, m = pop[gi], man[str(gi)]
        if g["page"] not in texts:
            texts[g["page"]] = {ln.id: ln.text for ln in load_page(Path(pages[g["page"]]["xml"])).lines()}
        vnum = {gid: int(n) for n, gid in m["V"].items()}
        group_v = [vnum[i] for i in g["gt_ids"]]
        chars = {vnum[i]: len(texts[g["page"]][i]) for i in g["gt_ids"]}
        n_p, n_v = len(m["P"]), len(m["V"])
        statuses, lg_ok = [], {}
        for judge in ("J1", "J2"):
            f = d / "answers" / judge / f"{m['opaque']}.txt"
            if not f.exists():                                                  # a missing file is an error, never "all unreadable" (F7)
                raise SystemExit(f"missing answer {f}")
            ans, ignored = parse_answer(f.read_text(encoding="utf-8"), n_p, n_v)
            ignored_log[f"{judge}:{m['opaque']}"] = ignored
            statuses.append(judge_status(ans, group_v, n_v))
            lg_ok[judge] = [ans[int(p)] == frozenset({v}) for p, v in m.get("LG", {}).items()]   # reported only (F6)
        per_group[gi] = {**group_scores(statuses[0], statuses[1], chars), "s1": statuses[0], "s2": statuses[1], "lg_ok": lg_ok}
    pairs = [(per_group[gi]["s1"][v], per_group[gi]["s2"][v]) for gi in seq for v in per_group[gi]["s1"]
             if per_group[gi]["s1"][v] is not None and per_group[gi]["s2"][v] is not None]
    agr = agreement(pairs)
    e_m = [per_group[gi]["e_minus"] for gi in seq]
    e_p = [per_group[gi]["e_plus"] for gi in seq]
    u_hat = float(np.mean([per_group[gi]["u"] for gi in seq]))
    stop = u_hat > STOP_U or (agr["kappa"] is not None and agr["kappa"] < 0.4 and agr["raw"] < 0.85)
    lg_acc = {j: (lambda xs: (sum(xs) / len(xs)) if xs else None)([ok for gi in seq for ok in per_group[gi]["lg_ok"][j]]) for j in ("J1", "J2")}
    report = {"stage": a.stage, "draws": len(seq), "u_hat": u_hat, "agreement": agr, "stop": stop, "line_geometry_accuracy_reported": lg_acc,
              "decision": None if stop else decide(e_m, e_p, a.stage),
              "descriptive": {"e_minus": descriptive(e_m), "e_plus": descriptive(e_p)},
              "per_group": {str(gi): {k: v for k, v in per_group[gi].items() if k in ("e_minus", "e_plus", "u", "verdict")} for gi in per_group},
              "ignored_keys": ignored_log,
              "inputs_sha": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [d / "draws.json", d / "planches_manifest_stage1.json"]}}
    if stop:
        report["outcome"] = "ARRÊT : instrument de jugement faible (indéterminé > 25 % ou accord faible) ; note B"
    out_p.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("stage", "draws", "u_hat", "agreement", "stop")} | {"decision": report["decision"]}, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
