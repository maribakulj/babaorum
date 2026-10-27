"""AV-P04 (avenants v1.1 to v1.4): the SMALL private Kaggle input dataset ``babaorum-p04m``, complementary to ``babaorum-p07b`` (reused as is).
Built on the Mac, light CPU, NOTHING uploaded by this script. NO truth (no XML anywhere). Contents:
- axel at the branch commit (git archive of p04-blla-modele, c62abf9, minus data/contamination/gallica as in P-07b), checked free of XML;
- the 5 P-07 images absent from babaorum-p07b (La_Presse_0545076 is there as the old control page);
- pages_26.json: the 26 pages in a FROZEN order (the 21 P-07b pages' order, then the 5 others), ``xml`` pointing to a path that does not exist;
- runs_26_reference_tar_gz: the historic reference (rappel_pages/runs_26/runs, manifest 77ee5723…) for the gate, minus e12/pages.json's xml;
- the gate scripts (p04m_porte.py, p07b_compare_runs.py 76369901, p07b_chain_units.py ee2b0c62);
- pins_p04m.json: exact versions per environment = Mac freezes of babaorum-p07b with the session-1 diffs of P-07b applied (the versions P-07b
  actually ran on Kaggle), so that pass 1 reruns the P-07b environment;
- MANIFEST_SHA256.txt over every file.
The two gzip tars carry NO archive extension (axel_tar_gz, runs_26_reference_tar_gz): Kaggle unpacks *.tar.gz on upload (seen on
the first upload, 25/10) but leaves extension-less files as they are (the part_* of babaorum-p07b).
    PYTHONDONTWRITEBYTECODE=1 ~/axel/.venv/bin/python scripts/p04m_build_dataset.py --out <new dir> [--dry-run]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RP = Path.home() / "Documents/reviews/babaorum-pipeline/rappel_pages"
P07B_DS = RP / "kaggle_dataset/babaorum-p07b"
REF_RUNS = RP / "runs_26/runs"
AXEL = Path.home() / "axel"
AXEL_REF = "p04-blla-modele"
AXEL_COMMIT = "c62abf9"
SESSION1_REPORT = ROOT / "runs/p07b_out_s1_v4/p07b_session1_report.json"
GATE_SCRIPTS = {"p04m_porte.py": None, "p07b_compare_runs.py": "76369901", "p07b_chain_units.py": "ee2b0c62"}
SLUG = "babaorum-p04m"


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def pins() -> dict:
    diffs = json.loads(SESSION1_REPORT.read_text())["env_freeze_diff_vs_mac"]
    out = {}
    for env in ("layout-env", "kraken-env", "axel-venv"):
        mac = {}
        for line in (P07B_DS / f"mac_freeze_{env}.txt").read_text().splitlines():
            if "==" in line:
                k, v = line.split("==", 1)
                mac[k.lower().replace("_", "-")] = v
        kag = dict(mac)
        for k, (_, v_kaggle) in diffs.get(env, {}).items():
            kag[k] = v_kaggle
        out[env] = sorted(f"{k}=={v}" for k, v in kag.items())
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"--out exists, refused: {out}")
    ref_pages = json.loads((REF_RUNS / "e12/pages.json").read_text())
    names = [p["page"] for p in ref_pages]
    assert len(names) == 26 == len(set(names))
    have = {p.stem for p in (P07B_DS / "images").glob("*.jpg")}
    extra_imgs = [n for n in names if n not in have]
    assert len(extra_imgs) == 5, extra_imgs
    p07b_order = [e["page"] for s in ("pages_session1.json", "pages_session2.json") for e in json.loads((P07B_DS / s).read_text())]
    order = p07b_order + [n for n in names if n not in p07b_order]
    assert sorted(order) == sorted(names)
    head = subprocess.run(["git", "-C", str(AXEL), "rev-parse", AXEL_REF], capture_output=True, text=True, check=True).stdout.strip()
    assert head.startswith(AXEL_COMMIT), head
    for name, want in GATE_SCRIPTS.items():
        if want:
            assert sha(ROOT / "scripts" / name).startswith(want), name
    plan = {"axel_commit": head, "images": extra_imgs, "order": order, "pins": {k: len(v) for k, v in pins().items()}}
    if a.dry_run:
        print(json.dumps(plan, indent=1))
        return 0
    out.mkdir(parents=True)
    with open(out / "axel_tar_gz", "wb") as f:                                    # git archive: tracked files of the commit only, minus the
        subprocess.run(["git", "-C", str(AXEL), "archive", "--format=tar.gz", head, "--", ".",   # Gallica OAI/SRU metadata XML (not truth), as
                        ":(exclude)data/contamination/gallica"], stdout=f, check=True)        # p07b_build_dataset.py does
    with tarfile.open(out / "axel_tar_gz") as t:
        assert not [m for m in t.getnames() if m.lower().endswith(".xml")], "axel archive holds an XML"
    (out / "AXEL_COMMIT.txt").write_text(f"{head}\n{AXEL_REF}\n")
    (out / "images").mkdir()
    for n in extra_imgs:
        src = next(Path(p["image"]) for p in ref_pages if p["page"] == n)
        shutil.copy2(src, out / "images" / f"{n}.jpg")
    by_name = {p["page"]: p for p in ref_pages}                                # keys read by axel (split, group); counts of truth zeroed as in P-07b
    entries = [{"page": n, "split": by_name[n]["split"], "group": by_name[n]["group"], "image": f"images/{n}.jpg",
                "xml": f"/nonexistent/truth/{n}.xml", "width": by_name[n]["width"], "height": by_name[n]["height"],
                "lines": 0, "regions": 0, "line_geometry": False} for n in order]
    (out / "pages_26.json").write_text(json.dumps(entries, indent=1) + "\n")
    with tarfile.open(out / "runs_26_reference_tar_gz", "w:gz") as t:          # the reference, minus its pages.json (truth paths)
        for p in sorted(REF_RUNS.rglob("*")):
            if p.is_file() and p.relative_to(REF_RUNS).as_posix() != "e12/pages.json":
                assert not p.name.lower().endswith(".xml")
                t.add(p, arcname=f"runs/{p.relative_to(REF_RUNS).as_posix()}")
    (out / "scripts").mkdir()
    for name in GATE_SCRIPTS:
        shutil.copy2(ROOT / "scripts" / name, out / "scripts" / name)
    (out / "pins_p04m.json").write_text(json.dumps(pins(), indent=1) + "\n")
    (out / "dataset-metadata.json").write_text(json.dumps({"title": SLUG, "id": f"mabaku/{SLUG}", "licenses": [{"name": "other"}]}, indent=1))
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name not in ("MANIFEST_SHA256.txt", "dataset-metadata.json"))
    assert not [p for p in files if p.suffix.lower() == ".xml"]
    (out / "MANIFEST_SHA256.txt").write_text("".join(f"{sha(p)}  {p.relative_to(out).as_posix()}\n" for p in files))
    print(json.dumps({**plan, "manifest_sha256": sha(out / "MANIFEST_SHA256.txt"),
                      "size_mb": round(sum(p.stat().st_size for p in files) / 1e6, 1)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
