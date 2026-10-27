"""DATA-2 — 20 pages neuves mfn_0000..mfn_0019 (génération, renommage, contrôle de recouvrement, index VT).

Lot : docs/END_TO_END_DATA2_LOT.txt (p6, 21/10/2026, amendements 1 à 3). Exécutant : pC.
RÉSERVE : ces pages sont NEUVES mais tirées de la MÊME distribution que mf_0000-0059 (même générateur, commit,
profil, sources de texte, polices). Ce n'est pas un corpus indépendant.

Sous-commandes (chacune est une étape du carnet, avec son délai) :
  generate  : mille-feuilles au commit 23875ab, profil pilot-v0.2 (rendu de la configuration), graine 20261021,
              20 pages, dans un dossier brut (noms mf_XXXX : intermédiaires, jamais des sorties).
  rename    : mf_XXXX -> mfn_XXXX. Seuls page_id et image.path changent dans le JSON ; les octets du PNG
              ne changent pas. Écrit <out>/mfn/pages, <out>/mfn/images et <out>/data2_generation_manifest.json.
  overlap   : contrôle de recouvrement avec mf_0000-0049 SEULEMENT (chemins explicites), seuil 5 % sur le texte.
              Écrit <out>/data2_overlap_report.json. Statuts : OK, OVERLAP_PAGE, OVERLAP_TEXT_FLAGGED.
  index     : <out>/rg-v2-mfn-gt-index.json au format rg.blind.gt-index.v1 (sha du manifeste et des 20 pages).

Le code commun (découverte, groupes de processus, battement, pipeline) est dans scripts/data2_common.py,
copié de LD-0 gelé (provenance dans son en-tête). Aucune image ni aucune vérité n'est transmise à pR par ce script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import shutil
import subprocess
import sys
import time
import traceback
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

CONFIG = ROOT / "config" / "data2_pages.json"
KIT_FILES = ["scripts/data2_generate.py", "scripts/data2_common.py", "config/data2_pages.json",
             "tests/test_data2_generate.py"]
MF_PAGES = [f"mf_{i:04d}" for i in range(50)]
RAW_PAGES = [f"mf_{i:04d}" for i in range(20)]
OUT_PAGES = [f"mfn_{i:04d}" for i in range(20)]
MF_FORBIDDEN = {f"mf_{i:04d}" for i in range(50, 60)}


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_config() -> dict:
    return json.loads(CONFIG.read_text())


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# ---------- generate (a subprocess step of the carnet: its group is killed at the step timeout) ----------

def generate_command(src_root: Path, raw: Path, cfg: dict, pages: int = None, seed: int = None) -> list:
    g = cfg["generator"]
    r = g["render"]
    return [sys.executable, "-c", "from mille_feuilles.cli import main; raise SystemExit(main())",
            "generate", "--output", str(raw), "--pages", str(pages or g["pages"]),
            "--seed", str(seed if seed is not None else g["seed"]),
            "--width", str(r["width"]), "--height", str(r["height"]), "--dpi", str(r["dpi"]),
            "--degradation", r["degradation"], "--jobs", "1"]


def generate(src_root: Path, raw: Path, cfg: dict, pages: int = None, seed: int = None) -> int:
    """Runs the generator from the extracted commit (PYTHONPATH on its src). Returns the exit code."""
    import os
    env = dict(os.environ, PYTHONPATH=str(Path(src_root) / "src"))
    raw.mkdir(parents=True, exist_ok=True)
    with open(raw.parent / "generate.stdout", "w") as out_f, open(raw.parent / "generate.stderr", "w") as err_f:
        proc = subprocess.run(generate_command(src_root, raw, cfg, pages, seed), cwd=str(src_root), env=env,
                              stdout=out_f, stderr=err_f)
    return proc.returncode


# ---------- rename (mf_XXXX -> mfn_XXXX; only page_id and image.path change) ----------

def rename_page(raw: Path, out_mfn: Path, mf_name: str, mfn_name: str) -> dict:
    page = json.loads((raw / "pages" / f"{mf_name}.json").read_text())
    if page.get("page_id") != mf_name:
        raise ValueError(f"{mf_name}: page_id is {page.get('page_id')!r}")
    original = json.loads(json.dumps(page))
    page["page_id"] = mfn_name
    page["image"]["path"] = f"images/{mfn_name}.png"
    changed = [k for k in original if k != "image" and page[k] != original[k]]
    changed += ["image." + k for k in original["image"] if page["image"].get(k) != original["image"][k]]
    (out_mfn / "pages").mkdir(parents=True, exist_ok=True)
    (out_mfn / "images").mkdir(parents=True, exist_ok=True)
    out_json = out_mfn / "pages" / f"{mfn_name}.json"
    out_json.write_text(json.dumps(page, ensure_ascii=False, indent=1))
    src_png = raw / "images" / f"{mf_name}.png"
    dst_png = out_mfn / "images" / f"{mfn_name}.png"
    shutil.copyfile(src_png, dst_png)
    if sha256(src_png) != sha256(dst_png):
        raise ValueError(f"{mfn_name}: PNG bytes changed by the copy")
    return {"mf_name": mf_name, "page_id": mfn_name, "json_sha256": sha256(out_json),
            "png_sha256": sha256(dst_png), "png_source_sha256": sha256(src_png), "fields_changed": changed,
            "orig_image_path": original["image"].get("path")}


def rename_all(raw: Path, out: Path, cfg: dict, meta: dict, n: int = 20) -> dict:
    out_mfn = out / "mfn"
    rows = [rename_page(raw, out_mfn, mf, mfn) for mf, mfn in zip(RAW_PAGES[:n], OUT_PAGES[:n])]
    files = {}
    for row in rows:
        files[f"pages/{row['page_id']}.json"] = row["json_sha256"]
        files[f"images/{row['page_id']}.png"] = row["png_sha256"]
    manifest = {
        "schema_version": "data2_generation_manifest_v1", "pages": rows, "files_sha256": files,
        "generator": {"commit": cfg["generator"]["commit"], "profile": cfg["generator"]["pilot_profile"],
                      "seed": cfg["generator"]["seed"], "pages": cfg["generator"]["pages"],
                      "render": cfg["generator"]["render"], "command": cfg["generator"]["command"],
                      "pilot_config_sha256": cfg["pilot_config_sha256"]},
        "names": {"raw": RAW_PAGES[:n], "output": OUT_PAGES[:n], "note": "raw names are intermediate, never outputs"},
        "versions": versions(), "meta": meta,
    }
    (out / "data2_generation_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest


def versions() -> dict:
    out = {"python": platform.python_version()}
    for mod in ("numpy", "PIL", "lxml", "fontTools", "jsonschema", "shapely"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            out[mod] = None
    return out


# ---------- overlap (reads mf_0000-0049 only, by explicit paths; §11 (3) of the lot and §2) ----------

def normalise_text(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()


def overlap_check(out: Path, mf_dir: Path, reference: dict, cfg: dict, n: int = 20) -> dict:
    """mf_dir is the folder containing pages/ of the E2E-3 run 3 output (e2e3/mf). Only the JSON pages of
    mf_0000-0049 are read, and only the images sha of the reference are used (no image of mf is opened)."""
    mf_list = list(cfg["overlap"]["mf_pages"])
    assert mf_list == MF_PAGES and not set(mf_list) & MF_FORBIDDEN
    ref_sha = reference["steps"]["page_cache"]["sha256"]
    mf_image_sha = {ref_sha[f"images/{p}.png"] for p in mf_list}
    mf_texts = set()
    for p in mf_list:                                   # explicit path, one file at a time
        page = json.loads((mf_dir / "pages" / f"{p}.json").read_text())
        for ln in page["lines"]:
            if ln.get("text"):
                mf_texts.add(normalise_text(ln["text"]))
    per_page, total, hit = [], 0, 0
    image_hits = []
    for mfn in OUT_PAGES[:n]:
        png_sha = sha256(out / "mfn" / "images" / f"{mfn}.png")
        if png_sha in mf_image_sha:
            image_hits.append(mfn)
        page = json.loads((out / "mfn" / "pages" / f"{mfn}.json").read_text())
        lines = [normalise_text(ln["text"]) for ln in page["lines"] if ln.get("text")]
        n_hit = sum(1 for t in lines if t in mf_texts)
        per_page.append({"page_id": mfn, "lines": len(lines), "lines_in_mf_text": n_hit,
                         "share": (n_hit / len(lines)) if lines else None})
        total += len(lines)
        hit += n_hit
    share = hit / total if total else 0.0
    threshold = cfg["overlap"]["text_share_threshold"]
    if image_hits:
        status = "OVERLAP_PAGE"
    elif share > threshold:
        status = "OVERLAP_TEXT_FLAGGED"
    else:
        status = "OK"
    report = {"schema_version": "data2_overlap_report_v1", "status": status,
              "image_sha_matches": image_hits, "text_share_global": share, "text_share_threshold": threshold,
              "lines_total": total, "lines_in_mf_text": hit, "per_page": per_page,
              "mf_pages_read": mf_list, "note": "a flagged status is reported to pA before any transmission"}
    (out / "data2_overlap_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


# ---------- index (rg.blind.gt-index.v1; hashes only, no ground truth content) ----------

def build_index(out: Path, n: int = 20) -> dict:
    manifest_path = out / "data2_generation_manifest.json"
    index = {"schema": "rg.blind.gt-index.v1",
             "generation_manifest_sha256": sha256(manifest_path),
             "pages": {mfn: sha256(out / "mfn" / "pages" / f"{mfn}.json") for mfn in OUT_PAGES[:n]}}
    (out / "rg-v2-mfn-gt-index.json").write_text(json.dumps(index, indent=2, ensure_ascii=False))
    return index


# ---------- command line ----------

def build_parser():
    p = argparse.ArgumentParser(description="DATA-2 : génération, renommage, recouvrement, index")
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--src", type=Path, required=True, help="generator source tree (commit 23875ab)")
    g.add_argument("--raw", type=Path, required=True)
    g.add_argument("--pages", type=int, default=None, help="smoke only (default: the lot's 20)")
    g.add_argument("--seed", type=int, default=None, help="smoke only (default: the lot's 20261021)")
    r = sub.add_parser("rename")
    r.add_argument("--raw", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--pages", type=int, default=20)
    o = sub.add_parser("overlap")
    o.add_argument("--out", type=Path, required=True)
    o.add_argument("--mf-dir", type=Path, required=True, help="folder with pages/ of the run 3 output")
    o.add_argument("--reference-report", type=Path, required=True)
    o.add_argument("--pages", type=int, default=20)
    i = sub.add_parser("index")
    i.add_argument("--out", type=Path, required=True)
    i.add_argument("--pages", type=int, default=20)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config()
    t0 = time.monotonic()
    try:
        if args.cmd == "generate":
            rc = generate(args.src, args.raw, cfg, args.pages, args.seed)
            print(json.dumps({"generate_exit": rc, "seconds": round(time.monotonic() - t0, 1)}), flush=True)
            return rc
        if args.cmd == "rename":
            manifest = rename_all(args.raw, args.out, cfg, {"seconds_rename": None}, args.pages)
            print(json.dumps({"pages": len(manifest["pages"]), "status": "RENAMED"}), flush=True)
            return 0
        if args.cmd == "overlap":
            ref = json.loads(args.reference_report.read_text())
            report = overlap_check(args.out, args.mf_dir, ref, cfg, args.pages)
            print(json.dumps({"status": report["status"], "text_share_global": report["text_share_global"]}), flush=True)
            return 0
        if args.cmd == "index":
            idx = build_index(args.out, args.pages)
            print(json.dumps({"pages": len(idx["pages"]), "schema": idx["schema"]}), flush=True)
            return 0
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
