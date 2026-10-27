"""LOT-P04 step 0 (class A, CPU, read only): inventory of the E21 READ train (44 pages) against the exclusion lists. No GPU, no model, no test truth.

    PYTHONDONTWRITEBYTECODE=1 ~/axel/.venv/bin/python scripts/p04_inventory.py

Reads: lines/read_train.txt (names of the 1 335 train zones) inside seg_lines.tar; the zone XML only of the pages that REMAIN (train data);
names (never contents) of the files of runs/e22/bench, runs/e22/indep and gallica-banc-e31; keys of the registry. A page is excluded when its
NUMBER (the digits taken from the file name) is in any exclusion list; titles that share an excluded number go out together.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]
from babaorum.g0_keys import title_of_prefix  # noqa: E402

AX = Path.home() / "axel"
R = Path.home() / "Documents/reviews/babaorum-pipeline/rappel_pages"
OUT = R / "etude/p04"
TAR = Path.home() / "colab-staging/seg_c7/seg_lines.tar"
RESERVED_TITLES = ["L'Intransigeant", "L'Écho de Paris", "Le Petit Marseillais", "La Petite Gironde", "Le Petit Troyen", "L'Action Pyrénéenne",
                   "Le Progrès de la Côte-d'Or"]
NUM = re.compile(r"(?<!\d)(\d{6,8})(?!\d)")


def number_of(stem: str) -> str | None:
    m = NUM.search(stem)
    return str(int(m.group(1))) if m else None   # leading zeros stripped: gallica:512130 == 0512130


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(name, obj):
    path = OUT / name
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return {"file": str(path), "sha256": sha(path)}


def keyfile_numbers(collection):
    nums = set()
    for f in (ROOT / "config/g0_keys").glob(f"{collection}.*.json"):
        for k in json.loads(f.read_text(encoding="utf-8")):
            m = re.match(r"gallica:(\d+)", str(k))
            if m:
                nums.add(str(int(m.group(1))))
    return nums


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    tar = tarfile.open(TAR)
    names = tar.extractfile("lines/read_train.txt").read().decode().split()
    zones_by_page = defaultdict(list)
    for n in names:
        m = re.match(r"lines/read/read_(.+)_z(\d+)_(\d+)\.xml$", n)
        zones_by_page[m.group(1)].append(n)
    pages = {}
    for stem, zs in sorted(zones_by_page.items()):
        pages[stem] = {"page": stem, "number": number_of(stem), "title": title_of_prefix(stem), "zones": len(zs)}
    lists = {}
    tirage = json.loads((R / "tirage.json").read_text(encoding="utf-8"))
    lists["dev2_26_pages"] = {number_of(p) for p in tirage["pool"]}
    lists["e12_dev_and_test"] = {number_of(p["page"]) for p in json.loads((AX / "runs/e12/pages.json").read_text()) if number_of(p["page"])}
    lists["e13"] = keyfile_numbers("gallica_e13")
    seals = set()
    for tag in ("v1", "v2"):
        seal = json.loads((ROOT / f"config/sealed/presse_e7_test_{tag}.json").read_text(encoding="utf-8"))
        for pg in seal["pages"] + seal["refused"]:
            for k in pg["issue_keys"]:
                m = re.match(r"gallica:(\d+)", k)
                if m:
                    seals.add(str(int(m.group(1))))
    lists["seals_v1_v2"] = seals
    names_listing = {}
    for sub in ("bench", "indep"):
        root = AX / "runs/e22" / sub
        names_listing[sub] = sorted({q.stem.split(".")[0] for q in root.rglob("*") if q.is_file()})
        lists[f"e22_{sub}_file_names"] = {number_of(s) for s in names_listing[sub] if number_of(s)}
    e31_names = sorted(q.name for q in (Path.home() / "corpus-vt/gallica-banc-e31/pages").iterdir()) if (Path.home() / "corpus-vt/gallica-banc-e31/pages").exists() else []
    lists["e31_reserve_keys_and_page_names"] = keyfile_numbers("gallica_e31_reserve") | {number_of(n) for n in e31_names if number_of(n)}
    lists = {k: sorted(v) for k, v in lists.items()}
    reserved_keys = {re.sub(r"[^a-z0-9 ]", "", t.lower().replace("é", "e").replace("'", " ")) for t in RESERVED_TITLES}
    excluded, remaining, title_unknown = {}, {}, 0
    for stem, info in pages.items():
        why = [k for k, v in lists.items() if info["number"] in v]
        tk = info["title"]
        if tk is None:
            title_unknown += 1
        elif any(tk.replace("'", " ").strip() == rk.strip() for rk in reserved_keys):
            why.append("e31_reserved_title")
        (excluded if why else remaining)[stem] = {**info, "excluded_by": why}
    # a number excluded for one page removes every page that has it: check that the rule closes (no remaining page shares an excluded number)
    ex_numbers = {i["number"] for i in excluded.values()}
    assert not [s for s, i in remaining.items() if i["number"] in ex_numbers]
    lines_by_page = {}
    for stem in remaining:            # train XML of the REMAINING pages only
        total = 0
        for zn in zones_by_page[stem]:
            total += len(re.findall(rb"<TextLine\b", tar.extractfile(zn).read()))
        lines_by_page[stem] = total
    for stem in remaining:
        remaining[stem]["lines"] = lines_by_page[stem]
    n_pages, n_zones, n_lines = len(remaining), sum(i["zones"] for i in remaining.values()), sum(lines_by_page.values())
    verdict = {"pages_min": 10, "zones_min": 300, "pages": n_pages, "zones": n_zones, "lines": n_lines, "pages_ok": n_pages >= 10, "zones_ok": n_zones >= 300,
               "gpu_may_start": n_pages >= 10 and n_zones >= 300}
    files = {
        "train_pages": write("train_pages.json", list(pages.values())),
        "exclusion_lists": write("exclusion_lists.json", {"numbers_normalised_without_leading_zeros": lists, "reserved_titles": RESERVED_TITLES,
                                                          "e22_file_names": names_listing, "e31_page_names": e31_names}),
        "excluded_pages": write("excluded_pages.json", list(excluded.values())),
        "remaining_pages": write("remaining_pages.json", list(remaining.values())),
    }
    summary = {"seg_lines_tar_sha256": "2a692082da14acad76f561a0787181248038ac4ac9baa1a706ae028410d790b4 (SHA256SUMS of seg_c7)",
               "train_pages": len(pages), "train_zones": sum(i["zones"] for i in pages.values()),
               "excluded_pages": len(excluded), "excluded_zones": sum(i["zones"] for i in excluded.values()),
               "excluded_by_list": dict(Counter(w for i in excluded.values() for w in i["excluded_by"])),
               "excluded_numbers": sorted(ex_numbers), "pages_with_unknown_title": title_unknown, "verdict": verdict, "files": files,
               "list_sizes": {k: len(v) for k, v in lists.items()}}
    files["summary"] = write("inventory.json", summary)
    print(json.dumps({k: summary[k] for k in ("train_pages", "train_zones", "excluded_pages", "excluded_zones", "excluded_by_list", "pages_with_unknown_title", "verdict", "list_sizes")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
