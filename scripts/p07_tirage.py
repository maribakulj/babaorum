"""LOT-P07: the draw, written BEFORE any computation. Reads names and identifiers only; no inference.

    PYTHONDONTWRITEBYTECODE=1 ~/axel/.venv/bin/python scripts/p07_tirage.py

Pool = corrected_T pages of press/train admitted as ``dev`` by eligibility (registry v2), with line truth and baselines, and whose NUMBER
(issue_key from the file name) is in none of: sealed presse_e7 v1/v2, runs/e22/bench and indep, E31, E12, E13.
Draw = random.Random(20261015).sample(sorted names, 6). No replacement after seeing any output.
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

from lxml import etree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from babaorum.g0_collections import eligibility  # noqa: E402
from babaorum.g0_keys import ids_of, title_of_prefix  # noqa: E402

AX = Path.home() / "axel"
OUT = Path.home() / "Documents/reviews/babaorum-pipeline/rappel_pages"
SEED = 20261015
REGISTRY = json.loads((ROOT / "config/g0_collections_v2_2.json").read_text(encoding="utf-8"))


def issue_of(name):
    return ids_of(name)[1]


def keyset(collection):
    entry, keys = REGISTRY["collections"][collection], set()
    for level in ("page", "issue"):
        spec = entry["match_keys"][level]
        if spec != "aucune clé à ce niveau":
            keys |= set(json.loads(Path(spec["values_file"]["path"]).read_text(encoding="utf-8"))) if "values_file" in spec else set(spec["values"])
    return keys


def line_truth(xml):
    root = etree.parse(str(xml)).getroot()
    lines = [l for l in root.iter() if etree.QName(l.tag).localname == "TextLine"]
    with_base = [l for l in lines if any(etree.QName(c.tag).localname == "Baseline" and c.get("points") for c in l)]
    return len(lines), len(with_base)


def main() -> int:
    pages = {}
    for line in (AX / "data/prepared/press/train.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        if r["corpus"] == "newseye_read" and r["group"] == "corrected_T":
            pages[r["page"]] = r["image"]
    sources = {}
    sources["presse_e7_seal_v1_v2"] = set()
    for tag in ("v1", "v2"):
        seal = json.loads((ROOT / f"config/sealed/presse_e7_test_{tag}.json").read_text(encoding="utf-8"))
        sources["presse_e7_seal_v1_v2"] |= {k for pg in seal["pages"] + seal["refused"] for k in pg["issue_keys"]}
    for sub in ("bench", "indep"):
        root = AX / "runs/e22" / sub
        sources[f"e22_{sub}"] = {i for q in root.rglob("*") if q.is_file() for i in [issue_of(q.stem.split(".")[0])] if i} if root.exists() else set()
    sources["e31"] = {k for k in keyset("gallica_e31_reserve") if k.count("/") == 0}
    sources["e12"] = {i for p in json.loads((AX / "runs/e12/pages.json").read_text()) for i in [issue_of(p["page"])] if i}
    sources["e13"] = {k for k in keyset("gallica_e13") if k.count("/") == 0}
    removed, pool, detail = [], [], {}
    for name in sorted(pages):
        page_key, issue_key = ids_of(name)
        reasons = []
        res = eligibility(REGISTRY, "dev", page_key=page_key, issue_key=issue_key, title_key=title_of_prefix(name))
        if not res["allowed"]:
            reasons.append("eligibility(dev): " + "; ".join(res["refusals"])[:200])
        for label, keys in sources.items():
            if issue_key in keys:
                reasons.append(f"numéro {issue_key} dans {label}")
        n_lines, n_base = line_truth(Path(pages[name]).with_suffix(".xml"))
        if n_lines == 0 or n_base == 0:
            reasons.append("pas de vérité de lignes de base")
        detail[name] = {"issue_key": issue_key, "page_key": page_key, "xml_lines": n_lines, "xml_lines_with_baseline": n_base}
        (removed.append({"page": name, "reasons": reasons}) if reasons else pool.append(name))
    draw = random.Random(SEED).sample(sorted(pool), 6)
    doc = {"label": "LOT-P07 tirage, écrit avant tout calcul (dev n° 2, issu du train)", "written_utc": datetime.now(timezone.utc).isoformat(),
           "seed": SEED, "rule": "random.Random(20261015).sample(sorted(pool), 6)", "pool_size": len(pool), "pool": sorted(pool), "draw": draw,
           "removed": removed, "sources_checked": {k: len(v) for k, v in sources.items()}, "pages_detail": {p: detail[p] for p in sorted(pool)},
           "no_replacement": "aucun remplacement après avoir regardé une sortie ; une page qui plante est consignée",
           "reserve_reader": "press/train a servi à entraîner le lecteur mccatmus_press : le CER est biaisé à la baisse",
           "reserve_segmentation": "aucun entraînement connu du détecteur ni du découpeur sur ces pages ; non vérifiable pour les préentraînements"}
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "tirage.json"
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    (ROOT / "reports/lotP07_tirage_pb.json").write_text(json.dumps({"tirage": str(path), "sha256": sha, "written_utc": doc["written_utc"], "seed": SEED, "pool_size": len(pool), "draw": draw}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"pool": len(pool), "draw": draw, "removed": removed, "sha": sha[:12]}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
