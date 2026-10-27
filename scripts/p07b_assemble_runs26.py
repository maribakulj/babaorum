"""LOT-P07b point 6: assemble rappel_pages/runs_26 (26 pages) by REAL copies, never symlinks, never touching the sources.

    PYTHONDONTWRITEBYTECODE=1 ~/axel/.venv/bin/python scripts/p07b_assemble_runs26.py

Sources (read-only): rappel_pages/runs + rappel_pages/read_rest_all (P-07, 6 pages, Mac arm64) and the 20 Kaggle archives of
runs/p07b_out_s1_v4 + runs/p07b_out_s2_v1 (checked against their SHA256SUMS and the session reports). Destination:
rappel_pages/runs_26/{runs,read_rest_all} so that pB's scripts run unchanged with AXEL_RUNS=<abs>/rappel_pages/runs_26/runs
(their asserts: name == "runs", "rappel_pages" in the path; outputs go to RUNS.parent). Refuses an existing destination.
runs/e12/pages.json = the 6 P-07 entries unchanged + 20 entries built like axel scripts/e12_pages.py (truth XML on the Mac).
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

AX = Path.home() / "axel"
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(AX / "src")]
from axel.layout.gt import load_page  # noqa: E402

RAP = Path.home() / "Documents/reviews/babaorum-pipeline/rappel_pages"
DEST = RAP / "runs_26"
SESSIONS = [(1, ROOT / "runs/p07b_out_s1_v4"), (2, ROOT / "runs/p07b_out_s2_v1")]
STEP_DIRS = ["e12/blocks/ppdl3", "e12/lines/ppdl3_z", "e12/read/ppdl3_z", "e22/read/ppdl3_z", "e28/lines/rest_z", "e28/add/a_z",
             "e28/read/german_print_s/a_z", "e28/read/mccatmus_press/a_z", "e32/dev/add/a_m"]
EXTRA_DIRS = ["e32/dev/read/german_print_s/extra", "e32/dev/read/mccatmus_press/extra"]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def tree_digest(root: Path) -> tuple[int, str]:
    """(file count, sha256 of the sorted 'relpath size sha' listing): proves a source tree is unchanged."""
    rows = sorted(f"{p.relative_to(root)} {p.stat().st_size} {sha(p)}" for p in root.rglob("*") if p.is_file())
    return len(rows), hashlib.sha256("\n".join(rows).encode()).hexdigest()


def axel_runs_listing() -> str:
    rows = sorted(f"{p} {p.stat().st_size} {p.stat().st_mtime_ns}" for p in (AX / "runs").rglob("*") if p.is_file())
    return hashlib.sha256("\n".join(rows).encode()).hexdigest()


def image_of(name: str) -> Path:
    for line in (AX / "data/prepared/press/train.jsonl").open(encoding="utf-8"):
        if f'"page": "{name}"' in line:
            return Path(json.loads(line)["image"])
    raise SystemExit(f"no image for {name}")


def entry(name: str) -> dict:
    image = image_of(name)
    xml = image.with_suffix(".xml")
    gt = load_page(xml)
    return {"page": name, "split": "dev", "group": "corrected_T", "image": str(image), "xml": str(xml), "width": gt.width,
            "height": gt.height, "lines": len(gt.lines()), "regions": len(gt.text_regions), "line_geometry": gt.has_line_geometry}


def copy_new(src: Path, dst: Path, log: list) -> None:
    if dst.exists():
        raise SystemExit(f"collision, refused: {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)  # real copy, content only
    log.append(str(dst.relative_to(DEST)))


def main() -> int:
    if DEST.exists():
        raise SystemExit(f"{DEST} exists: refused")
    tir = json.loads((RAP / "tirage_p07b.json").read_text(encoding="utf-8"))
    p07_pages = json.loads((RAP / "runs/e12/pages.json").read_text(encoding="utf-8"))
    names6 = [p["page"] for p in p07_pages]
    names20 = list(tir["list"])
    assert len(names6) == 6 and len(names20) == 20 and not set(names6) & set(names20)

    axel_before = axel_runs_listing()
    src_before = {d: tree_digest(RAP / d) for d in ("runs", "read_rest_all")}

    # 1. archive integrity: SHA256SUMS and the session report
    archives = []
    for s, folder in SESSIONS:
        sums = dict(reversed(line.split()) for line in (folder / "SHA256SUMS").read_text().splitlines() if line.strip())
        rep = json.loads((folder / f"p07b_session{s}_report.json").read_text())
        assert set(rep["pages"]) == set(tir["sessions"][str(s)]), f"session {s}: page list differs from tirage_p07b.json"
        for step, h in rep["archives_sha256"].items():
            [tar] = list(folder.glob(f"p07b_s{s}_*_{step}.tar.gz"))
            assert sha(tar) == h == sums[tar.name], f"SHA mismatch {tar}"
            archives.append(tar)
    assert len(archives) == 20

    copied: list[str] = []
    DEST.mkdir()
    try:
        # 2. P-07 (6 pages), every file of runs/ and read_rest_all/
        for d in ("runs", "read_rest_all"):
            for p in sorted((RAP / d).rglob("*")):
                if p.is_file() and p.relative_to(RAP / d).as_posix() != "e12/pages.json":
                    copy_new(p, DEST / d / p.relative_to(RAP / d), copied)
        # 3. P-07b (20 pages), from the archives (safe extraction into a temp dir, json members only)
        with tempfile.TemporaryDirectory() as tmp:
            for tar in archives:
                with tarfile.open(tar) as t:
                    members = [m for m in t.getmembers() if m.isfile()]
                    for m in members:
                        assert m.name.endswith(".json") and ".." not in m.name and not m.name.startswith("/"), m.name
                    t.extractall(tmp, members=members, filter="data")
            for p in sorted(Path(tmp).rglob("*.json")):
                rel = p.relative_to(tmp).as_posix()
                if rel.startswith("runs/"):
                    copy_new(p, DEST / rel, copied)
                elif rel.startswith("working/read_rest_all/"):
                    copy_new(p, DEST / "read_rest_all" / p.name, copied)
                else:
                    raise SystemExit(f"unexpected member {rel}")
        # 4. pages.json: 6 P-07 entries unchanged, then the 20 in tirage order
        pages = p07_pages + [entry(n) for n in names20]
        (DEST / "runs/e12/pages.json").write_text(json.dumps(pages, indent=1) + "\n", encoding="utf-8")
    except BaseException:
        shutil.rmtree(DEST)
        raise

    # 5. completeness: 26 files per step dir, extras listed apart
    allnames = set(names6 + names20)
    counts = {}
    for d in STEP_DIRS + ["../read_rest_all"]:
        got = {p.stem for p in (DEST / "runs" / d).resolve().glob("*.json")}
        counts[d] = len(got)
        assert got == allnames, f"{d}: {len(got)} pages, missing {sorted(allnames - got)[:5]}"
    extras = {d: len(list((DEST / "runs" / d).glob("*.json"))) for d in EXTRA_DIRS}
    nomb = len(list((DEST / "runs/e32/dev/add/a_m_nomb").glob("*.json")))

    axel_after = axel_runs_listing()
    src_after = {d: tree_digest(RAP / d) for d in ("runs", "read_rest_all")}
    assert axel_before == axel_after, "~/axel/runs changed"
    assert src_before == src_after, "P-07 sources changed"

    files = sorted(p for p in DEST.rglob("*") if p.is_file())
    report = {
        "label": "LOT-P07b point 6: runs_26 assembled by real copies",
        "written_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": sha(Path(__file__)),
        "pages": {"p07_mac_arm64": names6, "p07b_kaggle_x86": names20},
        "archives": {a.name: sha(a) for a in archives},
        "files_copied": len(copied), "files_total": len(files),
        "per_step_pages": counts, "e32_extra_files": extras,
        "a_m_nomb_files": nomb,
        "a_m_nomb_note": "only P-07 has e32/dev/add/a_m_nomb (6); the Kaggle notebook does not produce it; read only by p07_read_rest_all.py (already run)",
        "pages_json_sha256": sha(DEST / "runs/e12/pages.json"),
        "axel_runs_listing_sha256": {"before": axel_before, "after": axel_after},
        "p07_sources_digest": {"before": src_before, "after": src_after},
        "manifest_sha256": hashlib.sha256("\n".join(f"{p.relative_to(DEST)} {sha(p)}" for p in files).encode()).hexdigest(),
    }
    (DEST / "ASSEMBLY_REPORT.json").write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (DEST / "MANIFEST_SHA256.txt").write_text("".join(f"{sha(p)}  {p.relative_to(DEST)}\n" for p in files), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("files_copied", "per_step_pages", "e32_extra_files", "a_m_nomb_files", "manifest_sha256")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
