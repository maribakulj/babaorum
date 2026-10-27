"""LOT-P07b: tirage_p07b.json and the private Kaggle input dataset ``babaorum-p07b`` (built on the Mac, light CPU, NOTHING uploaded).

    PYTHONDONTWRITEBYTECODE=1 ~/axel/.venv/bin/python scripts/p07b_build_dataset.py --stage tirage       # writes rappel_pages/tirage_p07b.json
    PYTHONDONTWRITEBYTECODE=1 ~/axel/.venv/bin/python scripts/p07b_build_dataset.py --stage build --dry-run
    PYTHONDONTWRITEBYTECODE=1 ~/axel/.venv/bin/python scripts/p07b_build_dataset.py --stage build --axel-ref p07b-annees

The dataset holds NO truth (no XML), no test file. ``pages.json`` points ``xml`` to a path that does not exist. Contents: axel at the branch
commit (git archive), the two reader models, the Hugging Face snapshot of PP-DocLayoutV3_safetensors (HF_HUB_OFFLINE=1 on Kaggle), the 20 images
(+ the portability-control page with the Mac outputs of P-07), pages lists for the two sessions, the Mac freezes of the environments, and a
SHA-256 manifest of every file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]
AX = Path.home() / "axel"
R = Path.home() / "Documents/reviews/babaorum-pipeline/rappel_pages"
OUT_DEFAULT = R / "kaggle_dataset" / "babaorum-p07b"
CONTROL_PAGE = "La_Presse_0545076_T0000002"
SLUG = "babaorum-p07b"
HF_REPO = "models--PaddlePaddle--PP-DocLayoutV3_safetensors"
LIST_SHA_EXPECTED = "b1476f8d5e7c82f7"  # sorted names joined by \n, no trailing newline (LOT-P07b)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def remaining_pages() -> list[str]:
    t = json.loads((R / "tirage.json").read_text(encoding="utf-8"))
    return sorted(set(t["pool"]) - set(t["draw"]))


def stage_tirage() -> int:
    names = remaining_pages()
    assert len(names) == 20
    list_sha = hashlib.sha256("\n".join(names).encode()).hexdigest()
    assert list_sha.startswith(LIST_SHA_EXPECTED), f"the list SHA {list_sha[:16]} is not the one p1 wrote ({LIST_SHA_EXPECTED})"
    src = json.loads((R / "tirage.json").read_text(encoding="utf-8"))
    doc = {"label": "LOT-P07b : les 20 pages restantes du vivier, sans tirage ni remplacement", "written_utc": datetime.now(timezone.utc).isoformat(),
           "rule": "vivier b791edb5 (rappel_pages/tirage.json, 26 pages) moins le tirage de P-07 (6 pages)", "pool_tirage_sha256": sha(R / "tirage.json"),
           "list": names, "list_sha256": list_sha, "sessions": {"1": names[:10], "2": names[10:]}, "no_replacement": True,
           "ic_groups_rule": "pages regroupées seulement si le numéro ET le titre sont communs (P-07 v3)",
           "ic_groups": [["Le_Matin_0576768_T0000002", "Le_Matin_0576768_T0000006"]],
           "ic_groups_note": "les groupes 4621234 (L'Œuvre T1 avec T10 ; Marie-Claire T2 avec T8) traversent le tirage de P-07 et la liste : établis sur les 26 pages à P-08 v9",
           "p07_draw": src["draw"]}
    path = R / "tirage_p07b.json"
    if path.exists():
        raise SystemExit(f"{path} exists: the list is written once, before any inference")
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"written": str(path), "sha256": sha(path), "list_sha256": list_sha[:16]}))
    return 0


def check_eligibility(names: list[str]) -> dict:
    from babaorum.g0_collections import eligibility
    from babaorum.g0_keys import ids_of, title_of_prefix

    registry = json.loads((ROOT / "config/g0_collections_v2_2.json").read_text(encoding="utf-8"))
    out = {}
    for n in names:
        page_key, issue_key = ids_of(n)
        r = eligibility(registry, "dev", page_key=page_key, issue_key=issue_key, title_key=title_of_prefix(n))
        out[n] = {"allowed": r["allowed"], "refusals": r["refusals"]}
    return out


def image_of(name: str) -> Path:
    for line in (AX / "data/prepared/press/train.jsonl").open(encoding="utf-8"):
        if f'"page": "{name}"' in line:
            return Path(json.loads(line)["image"])
    raise SystemExit(f"no image for {name}")


def page_entry(name: str, image: Path, mount: str) -> dict:
    from lxml import etree

    root = etree.parse(str(image.with_suffix(".xml"))).getroot()  # size only; the truth stays on the Mac
    pg = next(e for e in root.iter() if etree.QName(e.tag).localname == "Page")
    return {"page": name, "split": "dev", "group": "corrected_T", "image": f"{mount}/images/{image.name}",
            "xml": f"/nonexistent/no_truth_on_kaggle/{name}.xml", "width": int(pg.get("imageWidth")), "height": int(pg.get("imageHeight")),
            "lines": 0, "regions": 0, "line_geometry": False}


def stage_build(args) -> int:
    tir = json.loads((R / "tirage_p07b.json").read_text(encoding="utf-8"))
    names = tir["list"]
    assert hashlib.sha256("\n".join(names).encode()).hexdigest() == tir["list_sha256"]
    elig = check_eligibility(names)
    bad = {n: v for n, v in elig.items() if not v["allowed"]}
    if bad:
        raise SystemExit(f"eligibility(dev) refuses: {bad}")
    ref = subprocess.run(["git", "-C", str(AX), "rev-parse", args.axel_ref], capture_output=True, text=True)
    if ref.returncode:
        raise SystemExit(f"axel ref {args.axel_ref!r} not found")
    commit = ref.stdout.strip()
    route = subprocess.run(["git", "-C", str(AX), "show", f"{commit}:src/axel/pipeline/route.py"], capture_output=True, text=True, check=True).stdout
    missing = [n for n in names + [CONTROL_PAGE] if f'"{n}"' not in route]
    if missing:
        raise SystemExit(f"route._YEARS at {commit[:8]} lacks: {missing}")
    mount = f"/kaggle/input/{SLUG}"
    hf_root = Path.home() / ".cache/huggingface/hub" / HF_REPO
    snaps = sorted((hf_root / "snapshots").iterdir())
    assert len(snaps) == 1, "one HF snapshot expected"
    plan = [("axel.tar.gz", "git archive", None),
            ("models/mccatmus_press.mlmodel", "runs/e8/models/mccatmus_press.mlmodel", AX / "runs/e8/models/mccatmus_press.mlmodel"),
            ("models/german_print_s.mlmodel", "runs/e5/models/german_print_s.mlmodel", AX / "runs/e5/models/german_print_s.mlmodel")]
    for f in sorted(snaps[0].iterdir()):
        plan.append((f"hf/hub/{HF_REPO}/snapshots/{snaps[0].name}/{f.name}", "HF snapshot", f.resolve()))
    plan.append((f"hf/hub/{HF_REPO}/refs/main", "HF ref", None))
    for n in names + [CONTROL_PAGE]:
        im = image_of(n)
        plan.append((f"images/{im.name}", "image", im))
    for tag in ("blocks/ppdl3", "lines/ppdl3_z"):
        plan.append((f"reference_mac/e12/{tag}/{CONTROL_PAGE}.json", "Mac output P-07", R / "runs/e12" / tag / f"{CONTROL_PAGE}.json"))
    plan.append((f"reference_mac/e22/read/ppdl3_z/{CONTROL_PAGE}.json", "Mac output P-07", R / "runs/e22/read/ppdl3_z" / f"{CONTROL_PAGE}.json"))
    plan.append(("scripts/p07_read_rest_all.py", "babaorum script", ROOT / "scripts/p07_read_rest_all.py"))
    plan.append(("scripts/p07b_chain_units.py", "babaorum script", ROOT / "scripts/p07b_chain_units.py"))
    plan.append(("reference_mac/chain_units.json", "Mac chains c1 and c1+ma (computed at build)", None))
    total = sum(p[2].stat().st_size for p in plan if p[2] is not None)
    print(json.dumps({"axel_ref": args.axel_ref, "commit": commit, "files_planned": len(plan), "bytes_without_axel_archive": total, "dry_run": args.dry_run,
                      "eligibility_dev_all_allowed": True}))
    for dest, what, src in plan:
        print(f"  {dest:95s} {what:22s} {src.stat().st_size if src else '-'}")
    if args.dry_run:
        return 0
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists: remove it yourself, nothing is overwritten")
    out.mkdir(parents=True)
    for dest, what, src in plan:
        d = out / dest
        d.parent.mkdir(parents=True, exist_ok=True)
        if what == "git archive":
            # zero XML anywhere (p1): the Gallica OAI notices under data/contamination/gallica are left out; no chain script imports them
            subprocess.run(["git", "-C", str(AX), "archive", "--format=tar.gz", "-o", str(d), commit, "--", ".", ":(exclude)data/contamination/gallica"], check=True)
            listing = subprocess.run(["tar", "tzf", str(d)], capture_output=True, text=True, check=True).stdout.lower().splitlines()
            assert not [x for x in listing if x.endswith(".xml")], "the axel archive still holds an XML file"
        elif what == "HF ref":
            d.write_text(snaps[0].name, encoding="utf-8")
        elif dest == "reference_mac/chain_units.json":
            subprocess.run([str(AX / ".venv/bin/python"), str(ROOT / "scripts/p07b_chain_units.py"), "--axel", str(AX), "--page", CONTROL_PAGE, "--out", str(d)],
                           env={**os.environ, "AXEL_RUNS": str(R / "runs"), "PYTHONDONTWRITEBYTECODE": "1"}, check=True)
        else:
            shutil.copy2(src, d)  # real files, no symlink
    entries = {n: page_entry(n, image_of(n), mount) for n in names + [CONTROL_PAGE]}
    (out / "pages_session1.json").write_text(json.dumps([entries[n] for n in tir["sessions"]["1"]], indent=1) + "\n")
    (out / "pages_session2.json").write_text(json.dumps([entries[n] for n in tir["sessions"]["2"]], indent=1) + "\n")
    (out / "pages_control.json").write_text(json.dumps([entries[CONTROL_PAGE]], indent=1) + "\n")
    (out / "tirage_p07b.json").write_text((R / "tirage_p07b.json").read_text(encoding="utf-8"))
    for env, py in (("layout-env", "~/outils-seg/layout-env/bin/python"), ("kraken-env", "~/outils-seg/kraken-env/bin/python"), ("axel-venv", "~/axel/.venv/bin/python")):
        fr = subprocess.run(["uv", "pip", "freeze", "--python", os.path.expanduser(py)], capture_output=True, text=True)
        (out / f"mac_freeze_{env}.txt").write_text(fr.stdout, encoding="utf-8")
    (out / "dataset-metadata.json").write_text(json.dumps({"title": SLUG, "id": f"<kaggle-username>/{SLUG}", "licenses": [{"name": "other"}],
                                                          "note": "PRIVATE dataset ; to be uploaded by p1 (LOT-P07b v2, option b)"}, indent=1) + "\n")
    (out / "AXEL_COMMIT.txt").write_text(f"{commit}\n{args.axel_ref}\n")
    lines = [f"{sha(p)}  {p.relative_to(out)}" for p in sorted(out.rglob("*")) if p.is_file() and p.name != "MANIFEST_SHA256.txt"]
    (out / "MANIFEST_SHA256.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"built": str(out), "files": len(lines), "manifest_sha256": sha(out / "MANIFEST_SHA256.txt"),
                      "hf_model_sha256": next(l.split()[0] for l in lines if l.endswith("model.safetensors")),
                      "no_xml": not any(p.suffix == ".xml" for p in out.rglob("*"))}))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("tirage", "build"), required=True)
    ap.add_argument("--axel-ref", default="p07b-annees")
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    return stage_tirage() if args.stage == "tirage" else stage_build(args)


if __name__ == "__main__":
    raise SystemExit(main())
