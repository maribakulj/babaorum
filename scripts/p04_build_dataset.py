"""LOT-P04 preparation (no GPU, nothing uploaded): validation.json, train.json and the private Kaggle input dataset ``babaorum-p04``.

    PYTHONDONTWRITEBYTECODE=1 ~/axel/.venv/bin/python scripts/p04_build_dataset.py

Validation (fixed by pJ): 6390977-001, 438752-001, 752214-001..004. Training: the 12 other remaining pages (311 zones). The dataset holds ONLY the
zones of these 18 pages (jpg + xml, from seg_lines.tar), the lists train.txt / validation.txt, the exclusion list (26 pages, 897 zones) and a
SHA-256 manifest. No excluded page, no E12/E13/E31 zone, no dev, no test.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import tarfile
from pathlib import Path

R = Path.home() / "Documents/reviews/babaorum-pipeline/rappel_pages"
E = R / "etude/p04"
TAR = Path.home() / "colab-staging/seg_c7/seg_lines.tar"
OUT = R / "kaggle_dataset/babaorum-p04"
VALIDATION = ["6390977-001", "438752-001", "752214-001", "752214-002", "752214-003", "752214-004"]


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main() -> int:
    rem = {r["page"]: r for r in json.loads((E / "remaining_pages.json").read_text())}
    exc = json.loads((E / "excluded_pages.json").read_text())
    assert set(VALIDATION) <= set(rem), "a validation page is not among the remaining pages"
    train_pages = sorted(set(rem) - set(VALIDATION))
    val = [{k: rem[p][k] for k in ("page", "number", "zones", "lines")} for p in sorted(VALIDATION)]
    trn = [{k: rem[p][k] for k in ("page", "number", "zones", "lines")} for p in train_pages]
    assert sum(r["zones"] for r in trn) == 311
    (E / "validation.json").write_text(json.dumps(val, indent=1) + "\n")
    (E / "train.json").write_text(json.dumps(trn, indent=1) + "\n")
    tar = tarfile.open(TAR)
    names = tar.extractfile("lines/read_train.txt").read().decode().split()
    by_page: dict[str, list[str]] = {}
    for n in names:
        by_page.setdefault(re.match(r"lines/read/read_(.+)_z\d+_\d+\.xml$", n).group(1), []).append(n)
    excluded_zone_files = sorted(z for p in (e["page"] for e in exc) for z in by_page[p])
    exclusion = {"pages": sorted(e["page"] for e in exc), "numbers": sorted({e["number"] for e in exc}), "zone_xml_files": excluded_zone_files,
                 "rule": "pages of the dev n° 2 (26) and their numbers; E12, E13, seals v1/v2, e22 bench/indep, E31 lists were checked and add nothing (inventory.json)",
                 "inventory_sha256": sha(E / "inventory.json")}
    if OUT.exists():
        raise SystemExit(f"{OUT} exists: remove it first")
    (OUT / "lines/read").mkdir(parents=True)
    (OUT / "lists").mkdir()
    lists = {"train": [], "validation": []}
    for kind, pages in (("train", train_pages), ("validation", sorted(VALIDATION))):
        for p in pages:
            for xml in by_page[p]:
                jpg = xml[:-4] + ".jpg"
                for member in (xml, jpg):
                    dest = OUT / member
                    dest.write_bytes(tar.extractfile(member).read())
                lists[kind].append(xml.replace("lines/", "", 1))   # relative to lines/, as ketos expects
    (OUT / "lists/train.txt").write_text("\n".join(lists["train"]) + "\n")
    (OUT / "lists/validation.txt").write_text("\n".join(lists["validation"]) + "\n")
    (OUT / "exclusion_list.json").write_text(json.dumps(exclusion, ensure_ascii=False, indent=1) + "\n")
    shutil.copy2(E / "train.json", OUT / "train.json")
    shutil.copy2(E / "validation.json", OUT / "validation.json")
    (OUT / "dataset-metadata.json").write_text(json.dumps({"title": "babaorum-p04", "id": "<kaggle-username>/babaorum-p04", "licenses": [{"name": "other"}],
                                                          "note": "PRIVATE ; to be uploaded by p1 with the CLI after pJ"}, indent=1) + "\n")
    files = sorted(p for p in OUT.rglob("*") if p.is_file() and p.name != "MANIFEST_SHA256.txt")
    (OUT / "MANIFEST_SHA256.txt").write_text("\n".join(f"{sha(p)}  {p.relative_to(OUT)}" for p in files) + "\n")
    leaked = [p for p in files if any(e["page"] in p.name for e in exc)]
    assert not leaked, f"an excluded page is in the dataset: {leaked[:3]}"
    print(json.dumps({"dataset": str(OUT), "files": len(files) + 1, "train_zones": len(lists["train"]), "validation_zones": len(lists["validation"]),
                      "excluded_zone_files_listed": len(excluded_zone_files), "manifest_sha256": sha(OUT / "MANIFEST_SHA256.txt"),
                      "exclusion_list_sha256": sha(OUT / "exclusion_list.json"), "train_json_sha256": sha(E / "train.json"),
                      "validation_json_sha256": sha(E / "validation.json"), "bytes": sum(p.stat().st_size for p in files)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
