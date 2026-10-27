"""LOT-P08 reproduction control on the 5 pages of the E12 dev (c1+ma). Run WITHOUT AXEL_RUNS: ~/axel/runs is read only.

    env -u AXEL_RUNS PYTHONDONTWRITEBYTECODE=1 ~/axel/.venv/bin/python scripts/p08_e12_control.py
CER_0 (micro, D-217 reordered) must equal 0.05188611919580012 (published, runs/e32/dev.json), else the script stops.
The listing of ~/axel/runs (path, size, mtime) is taken before and after and written next to the result.
"""
from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

assert not os.environ.get("AXEL_RUNS", "").strip(), "run without AXEL_RUNS"
AX = Path.home() / "axel"
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(AX / "src"), str(AX / "scripts"), str(ROOT / "src"), str(ROOT / "scripts")]
EXPECTED = 0.05188611919580012

from p08_decompose import analyse, summarize, truth_polys  # noqa: E402
from axel.eval.doc_bootstrap import DocScore, micro  # noqa: E402
from axel.layout.gt import load_page  # noqa: E402
from axel.layout.pagetext import score_page  # noqa: E402
from babaorum.p08_oracles import parts_sum_error  # noqa: E402


def listing(dest: Path) -> str:
    out = subprocess.run("find runs -type f -exec stat -f '%m %z %N' {} + | sort -k3", shell=True, cwd=AX, capture_output=True, text=True, check=True).stdout
    dest.write_text(out, encoding="utf-8")
    import hashlib
    return hashlib.sha256(out.encode()).hexdigest()


def run(out_dir: Path) -> int:
    """LOT-P08 v9: the listings and the result go to a NEW folder (out_dir), never to etude/."""
    ETUDE = out_dir
    before = listing(ETUDE / "axel_runs_E12_AVANT.txt")
    e28 = importlib.import_module("e28_fill")
    e32 = importlib.import_module("e32")
    bench = e32.BENCHES["dev"]
    rows, docs, failures = [], [], []
    for p in e28.dev_pages():
        name = p["page"]
        gt = load_page(Path(p["xml"]))
        glines = gt.lines()
        arms, zones = e32.build_arms(bench, "c1", name)
        units = arms["ma"].units
        ps = score_page([u.text for u in units], [g.text for g in glines])
        docs.append(DocScore(name, ps.cer_reordered * ps.chars, ps.chars))
        try:
            dec, _, info = analyse(units, glines, gt.text_regions, zones, truth_polys(gt))
        except ValueError as exc:  # LOT-P08 v8: reported as a finding, no workaround (E27 and line_geometry disagree on one pair)
            failures.append({"page": name, "error": str(exc)})
            continue
        assert abs(parts_sum_error(dec)) < 1e-9 and abs(dec.cer["0"] - ps.cer_reordered) < 1e-12
        rows.append({"page": name, "chars": ps.chars, "cer": dec.cer, "parts": dec.parts})
    cer0 = micro(docs)
    value = cer0["value"] if isinstance(cer0, dict) else cer0
    ok = abs(value - EXPECTED) < 1e-12
    after = listing(ETUDE / "axel_runs_E12_APRES.txt")
    res = {"chain": "c1+ma", "pages": [r["page"] for r in rows], "cer_0_micro": value, "expected": EXPECTED, "exact": ok,
           "decomposition_exposed_pages_for_information": summarize(rows, with_ci=False),
           "decomposition_failures_reported_as_findings": failures,
           "finding": "La_Presse_0512130 : une prédite est E27 appariée à tl_484 et line_geometry l'apparie à tl_485 ; le contrôle ordre vrai échoue (CER 0,00226) ; pas de contournement (v8)",
           "axel_runs_listing_sha256": {"before": before, "after": after, "identical": before == after}}
    (ETUDE / "e12_control.json").write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"cer_0_micro": value, "expected": EXPECTED, "exact": ok, "axel_runs_identical": before == after}))
    return 0 if ok and before == after else 1




if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, help="NEW folder (refused if it exists)")
    a = ap.parse_args()
    out = Path(a.out_dir).expanduser().resolve()
    if out.exists():
        raise SystemExit(f"--out-dir exists, refused: {out}")
    out.mkdir(parents=True)
    raise SystemExit(run(out))
