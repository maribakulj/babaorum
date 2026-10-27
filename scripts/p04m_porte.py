"""AV-P04 v1.2 §§ 2 and 5 (GO pQ 6e221a1b, then v1.3): the PAIRING GATE, page by page, pass 1 (stock blla, Kaggle) against the historic
reference runs_26. No truth is read. Runs on Kaggle at the end of pass 1; writes ONE small JSON, porte_pairage.json (refused if it exists).

- compare() of p07b_compare_runs.py (v2, 76369901) is imported UNCHANGED and called ONCE PER PAGE on two single-page wrapper roots
  (<root>/s/runs/<folder>/<page>.json, links to the real files).
- Placeholders (v1.2 § 5): ONLY for the two e32 'extra' readers, and ONLY when the file is absent on BOTH sides: the same empty file
  {"texts":{},"ctc":{},"fallback":[]} on both sides, logged per page. Any other folder absent on one side or both: no placeholder (the
  inventory of compare() makes it an anomaly). A file present on one side only is never filled.
- The 6 P-07 pages (computed on the Mac) must ALSO hold the 4 portability thresholds of LOT-P07b v2 point 2, computed by the same code as the
  P-07b notebook (detections >= 0.3; blla lines per zone and total <= 1 %; c1 lines found with symmetric baseline coverage >= 0.95 for
  >= 99 %; c1 reading CER on found lines <= 0.1 %). c1 lines come from p07b_chain_units.py (ee2b0c62), unchanged.
- Rule: 20 P-07b pages "identique"; 6 P-07 pages "identique" or "équivalent" AND the 4 thresholds; any anomaly, missing page or failed
  threshold closes the gate. Exit code 0 only if the gate is OPEN.
    HOME=<dir holding axel/> python p04m_porte.py --ref <runs_26/runs> --pass1 <runs> --axel <axel dir> --python-axel <axel-venv python> \
        --out porte_pairage.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
P07 = ("L_OEuvre_4613537_T0000001", "L_OEuvre_4613814_T0000002", "L_OEuvre_4621234_T0000001", "La_Presse_0545076_T0000002",
       "Le_Matin_0551841_T0000003", "Marie-Claire_4621234_T0000008")
P07B = ("L_OEuvre_4615451_T0000001", "L_OEuvre_4616168_T0000002", "L_OEuvre_4617562_T0000007", "L_OEuvre_4620081_T0000004",
        "L_OEuvre_4621234_T0000010", "La_Fronde_6704254_T0000002", "La_Presse_0476474_T0000001", "La_Presse_0477885_T0000002",
        "La_Presse_0514199_T0000004", "La_Presse_0542128_T0000001", "La_Presse_0543209_T0000001", "Le_Matin_0554721_T0000004",
        "Le_Matin_0558051_T0000005", "Le_Matin_0569308_T0000002", "Le_Matin_0574578_T0000002", "Le_Matin_0576768_T0000002",
        "Le_Matin_0576768_T0000006", "Le_Matin_0586574_T0000003", "Marie-Claire_4613173_T0000004", "Marie-Claire_4621234_T0000002")
PLACEHOLDER_OK = {"e32/dev/read/german_print_s/extra", "e32/dev/read/mccatmus_press/extra"}
PLACEHOLDER = {"texts": {}, "ctc": {}, "fallback": []}
COMPARE_SHA = "76369901"
CHAIN_UNITS_SHA = "ee2b0c62"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def folders(cr) -> list[str]:
    return [cr.BLOCKS] + [f for p, s in cr.LINE_PRODUCERS.items() for f in [p, *s["readers"], *s["selections"]]]


def wrap(cr, page: str, ref: Path, p1: Path, tmp: Path) -> tuple[Path, Path, list[str]]:
    """Two single-page roots; placeholders only for PLACEHOLDER_OK folders absent on BOTH sides."""
    a, b = tmp / page / "A", tmp / page / "B"
    filled = []
    for f in folders(cr):
        fa, fb = ref / f / f"{page}.json", p1 / f / f"{page}.json"
        if not fa.exists() and not fb.exists():
            if f in PLACEHOLDER_OK:
                for root in (a, b):
                    d = root / "s/runs" / f
                    d.mkdir(parents=True, exist_ok=True)
                    (d / f"{page}.json").write_text(json.dumps(PLACEHOLDER), encoding="utf-8")
                filled.append(f)
            continue
        for src, root in ((fa, a), (fb, b)):
            if src.exists():
                d = root / "s/runs" / f
                d.mkdir(parents=True, exist_ok=True)
                os.symlink(src.resolve(), d / f"{page}.json")
    return a, b, filled


def _ints(s_) -> tuple:
    return tuple(int(round(v)) for q in s_ for v in q)


def biject(mu: list, ku: list, sym_cov) -> dict:
    """AV-P04 v1.4, a BIJECTION (pQ B3): identity first (integer baseline and polygon; each pass-1 line used once), then the remaining lines
    by symmetric coverage >= 0.95, best first, deterministic ties, each pass-1 line used at most once (as compare().pair). No geometric
    pre-filter (pQ minor, 2nd audit: a 100 px box filter could drop a real coverage of 1): every remaining pair is scored.
    Returns {reference index: pass-1 index}."""
    by_key: dict = {}
    for j, k in enumerate(ku):
        by_key.setdefault((_ints(k["baseline"]), _ints(k["polygon"])), []).append(j)
    match, rest = {}, []
    for i, m in enumerate(mu):
        js = by_key.get((_ints(m["baseline"]), _ints(m["polygon"])))
        if js:
            match[i] = js.pop(0)
        else:
            rest.append(i)
    used = set(match.values())
    left = [j for j in range(len(ku)) if j not in used]
    cand = []
    for i in rest:
        for j in left:
            c = sym_cov(mu[i], ku[j])
            if c >= 0.95:
                cand.append((c, i, j))
    for c, i, j in sorted(cand, key=lambda t: (-t[0], t[1], t[2])):
        if i in match or j in used:
            continue
        match[i] = j
        used.add(j)
    return match


def chain_units(root: Path, page: str, axel: Path, py: str, out: Path) -> dict:
    r = subprocess.run([py, str(HERE / "p07b_chain_units.py"), "--axel", str(axel), "--page", page, "--out", str(out)],
                       env=dict(os.environ, AXEL_RUNS=str(root.resolve()), PYTHONDONTWRITEBYTECODE="1"), capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"p07b_chain_units failed on {page}: {r.stderr[-400:]}")
    return json.loads(out.read_text())


def portability(page: str, ref: Path, p1: Path, axel: Path, py: str, tmp: Path) -> dict:
    """The 4 thresholds of LOT-P07b v2 point 2, same code as the P-07b notebook (cell 4), reference playing the 'Mac' role."""
    from axel.layout.fill import baseline_cover, line_height
    tp = lambda s_: tuple((float(x), float(y)) for x, y in s_)            # noqa: E731

    def cov(u, v):
        tu = tp(u["baseline"])
        return baseline_cover(tu, tp(v["baseline"]), 0.5 * line_height(tp(u["polygon"]), tu))

    def lev(u, v):
        prev = list(range(len(v) + 1))
        for i, cu in enumerate(u, 1):
            cur = [i]
            for j, cv in enumerate(v, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (cu != cv)))
            prev = cur
        return prev[-1]

    def dets(path):
        return sorted((x.get("label"), round(x["score"], 3)) for x in json.loads(path.read_text())["blocks"] if x["score"] >= 0.3)

    out: dict = {}
    md, kd = dets(ref / f"e12/blocks/ppdl3/{page}.json"), dets(p1 / f"e12/blocks/ppdl3/{page}.json")
    out["detections"] = {"ref": len(md), "pass1": len(kd), "same_labels_in_order": [x[0] for x in md] == [x[0] for x in kd]}
    ml = json.loads((ref / f"e12/lines/ppdl3_z/{page}.json").read_text())["blocks"]
    kl = json.loads((p1 / f"e12/lines/ppdl3_z/{page}.json").read_text())["blocks"]
    nm, nk = sum(len(x["lines"]) for x in ml), sum(len(x["lines"]) for x in kl)
    kz = {x["id"]: len(x["lines"]) for x in kl}
    out["blla_lines"] = {"ref": nm, "pass1": nk, "gap_total": abs(nm - nk) / max(nm, 1),
                         "gap_per_zone_max": max((abs(len(x["lines"]) - kz.get(x["id"], 0)) / max(len(x["lines"]), 1) for x in ml), default=0)}
    mu = chain_units(ref, page, axel, py, tmp / f"{page}_ref_units.json")["c1"]
    ku = chain_units(p1, page, axel, py, tmp / f"{page}_p1_units.json")["c1"]
    match = biject(mu, ku, lambda u, v: min(cov(u, v), cov(v, u)))
    found = len(match)
    edits = sum(lev(mu[i]["text"], ku[j]["text"]) for i, j in match.items())
    chars = sum(max(len(mu[i]["text"]), 1) for i in match)
    out["c1"] = {"lines_ref": len(mu), "lines_pass1": len(ku), "found_share": found / max(len(mu), 1), "reading_cer": edits / max(chars, 1)}
    out["passed"] = bool(out["detections"]["ref"] == out["detections"]["pass1"] and out["detections"]["same_labels_in_order"]
                         and out["blla_lines"]["gap_total"] <= 0.01 and out["blla_lines"]["gap_per_zone_max"] <= 0.01
                         and out["c1"]["found_share"] >= 0.99 and out["c1"]["reading_cer"] <= 0.001)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--pass1", required=True)
    ap.add_argument("--axel", required=True)
    ap.add_argument("--python-axel", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"--out exists, refused: {out}")
    for name, want in (("p07b_compare_runs.py", COMPARE_SHA), ("p07b_chain_units.py", CHAIN_UNITS_SHA)):
        if not sha(HERE / name).startswith(want):
            raise SystemExit(f"{name}: sha differs from {want}")
    ref, p1, axel = Path(a.ref), Path(a.pass1), Path(a.axel)
    sys.path[:0] = [str(HERE), str(axel / "src")]
    import p07b_compare_runs as cr                                                  # functions unchanged (sha checked above)
    pages_out, closed = {}, []
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for page in P07 + P07B:
            wa, wb, filled = wrap(cr, page, ref, p1, tmp)
            rep = cr.compare(wa, wb)
            entry = {"origin": "P-07 (Mac)" if page in P07 else "P-07b (Kaggle)", "verdict": rep["verdict"], "placeholders": filled,
                     "anomalies": rep["anomalies"], "totals": rep["totals"]}
            ok = rep["verdict"] == "identique" if page in P07B else rep["verdict"] in ("identique", "équivalent")
            if page in P07:
                try:
                    entry["portability"] = portability(page, ref, p1, axel, a.python_axel, tmp)
                except (FileNotFoundError, KeyError, RuntimeError, json.JSONDecodeError) as e:
                    entry["portability"] = {"passed": False, "error": f"{type(e).__name__}: {str(e)[:200]}"}
                ok = ok and entry["portability"]["passed"]
            entry["page_ok"] = bool(ok)
            if not ok:
                closed.append(page)
            pages_out[page] = entry
    report = {"note": "AV-P04 v1.2 §§ 2 et 5 : porte de pairage par page ; aucune vérité lue",
              "inputs": {"p07b_compare_runs.py": sha(HERE / "p07b_compare_runs.py"), "p07b_chain_units.py": sha(HERE / "p07b_chain_units.py"),
                         "script": sha(Path(__file__))},
              "pages_expected": len(P07) + len(P07B), "pages": pages_out, "pages_closing": closed, "porte_ouverte": not closed}
    with open(out, "x", encoding="utf-8") as f:
        f.write(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps({"porte_ouverte": report["porte_ouverte"], "pages_closing": closed}, ensure_ascii=False))
    return 0 if not closed else 1


if __name__ == "__main__":
    raise SystemExit(main())
