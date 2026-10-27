"""LOT-D01: word-geometry manifests for pR (read-only on axel and corpus-vt; writes only the output folder).

    PYTHONDONTWRITEBYTECODE=1 ~/axel/.venv/bin/python scripts/d01_words_manifests.py

Outputs in ~/Documents/reviews/babaorum-pipeline/geometrie_mots/ : press_train_words.jsonl, press_dev_words.jsonl,
press_dev_c1ma_correspondance.jsonl, refused_pages.json, MANIFESTE.md (written by hand afterwards from summary.json).
No test page is read: the pages come from axel/data/prepared/press/{train,dev}.jsonl and every page goes through
babaorum.g0_collections.eligibility (registry v2) with its role. The dev is exposed (D-600).
"""
from __future__ import annotations

import hashlib
import importlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from lxml import etree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from babaorum.g0_collections import eligibility  # noqa: E402
from babaorum.g0_keys import ids_of, title_of_prefix  # noqa: E402
from babaorum.line_geometry import GroundTruthLine, PredictedLine, evaluate_line_geometry  # noqa: E402

AX = Path.home() / "axel"
PRESS = AX / "data/prepared/press"
OUT = Path.home() / "Documents/reviews/babaorum-pipeline/geometrie_mots"
REGISTRY = json.loads((ROOT / "config/g0_collections_v2_2.json").read_text(encoding="utf-8"))
ROLE = {"train": "train", "dev": "dev"}
#: LOT-D01 v3: exception written by p1 for the three dev pages that E13 used as a historical diagnostic (gallica_e13, test_used)
OVERRIDE_PAGES = {"La_Presse_0512130_T0000003", "La_Fronde_6703127_T0000003", "La_Fronde_6703523_T0000006"}
OVERRIDE = {"eligibility_override": True, "override_by": "p1, LOT-D01 v3, 15 octobre 2026",
            "override_reason": "dev d'E12, exposé (D-600) ; les mêmes numéros ont servi au diagnostic historique E13 (test_used)",
            "allowed_use": "lecture de géométrie et mesure de développement par pR",
            "forbidden_use": "entraînement, réglage d'un modèle livré, toute revendication de test"}


def local(tag):
    return etree.QName(tag).localname


def pts(el, child):
    node = next((c for c in el if local(c.tag) == child), None)
    if node is None or not node.get("points"):
        return None
    return [[float(a), float(b)] for a, b in (p.split(",") for p in node.get("points").split())]


def bbox(poly):
    xs, ys = [p[0] for p in poly], [p[1] for p in poly]
    return {"x": min(xs), "y": min(ys), "w": max(xs) - min(xs), "h": max(ys) - min(ys)}


def reading_order(el):
    m = re.search(r"index:(\d+)", el.get("custom") or "")
    return int(m.group(1)) if m else None


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def no_test_crossing(pages_and_keys):
    """Cross every exported page with every test/reserve source: names and identifiers only, nothing is read.

    Sources: E12 test pages (runs/e12/pages.json, split=test), the presse_e7 seal v1 and v2 (issue and page keys), the pinned
    registry collections e12_test / gallica_e13 / e22_bench / gallica_e31_reserve (keys and tree file names), and the file
    names under runs/e22/bench and runs/e22/indep (listing only).
    """
    e12_test = {p["page"] for p in json.loads((AX / "runs/e12/pages.json").read_text()) if p["split"] == "test"}
    seals = {}
    for tag in ("v1", "v2"):
        seal = json.loads((ROOT / f"config/sealed/presse_e7_test_{tag}.json").read_text(encoding="utf-8"))
        seals[tag] = {k for pg in seal["pages"] + seal["refused"] for k in [*pg["issue_keys"], pg["scan_page_key"]]}
    listed = {}
    for sub in ("bench", "indep"):
        root = AX / "runs/e22" / sub
        listed[sub] = {q.stem.split(".")[0] for q in root.rglob("*") if q.is_file()} if root.exists() else set()
    locked = {}
    for name in ("e12_test", "gallica_e13", "e22_bench", "gallica_e31_reserve"):
        entry = REGISTRY["collections"][name]
        keys = set()
        for level in ("page", "issue"):
            spec = entry["match_keys"][level]
            if spec != "aucune clé à ce niveau":
                keys |= set(json.loads(Path(spec["values_file"]["path"]).read_text(encoding="utf-8"))) if "values_file" in spec else set(spec["values"])
        locked[name] = keys
    hits = []
    for page, page_key, issue_key in pages_and_keys:
        for label, names in (("e12_test_names", e12_test), ("e22_bench_listing", listed["bench"]), ("e22_indep_listing", listed["indep"])):
            if page in names:
                hits.append({"page": page, "in": label})
        for label, keys in (*seals.items(), *locked.items()):
            hit = {k for k in (page_key, issue_key) if k and k in keys}
            if hit:
                hits.append({"page": page, "in": label, "keys": sorted(hit)})
    return hits, {"e12_test_pages": len(e12_test), "seal_keys": {k: len(v) for k, v in seals.items()},
                  "listing_files": {k: len(v) for k, v in listed.items()}, "registry_keys": {k: len(v) for k, v in locked.items()}}


def manifest_rows():
    """newseye_read corrected_T rows of press/train and press/dev, by (split, page)."""
    rows = defaultdict(dict)
    pages = {}
    for split in ("train", "dev"):
        for line in (PRESS / f"{split}.jsonl").open(encoding="utf-8"):
            r = json.loads(line)
            if r["corpus"] == "newseye_read":
                rows[(split, r["page"])][r["line_id"]] = r
                pages[(split, r["page"])] = r["group"]
    return rows, pages


def parse_page(xml_path: Path):
    root = etree.parse(str(xml_path)).getroot()
    lines = []
    for ln in root.iter():
        if local(ln.tag) != "TextLine":
            continue
        words = []
        for w in ln:
            if local(w.tag) != "Word":
                continue
            poly = pts(w, "Coords")
            text = next((u.text or "" for u in w.iter() if local(u.tag) == "Unicode"), "")
            words.append({"word_id": w.get("id"), "text": text, "polygon": poly, "reading_order": reading_order(w)})
        lines.append({"line_id": ln.get("id"), "polygon": pts(ln, "Coords"), "baseline": pts(ln, "Baseline"),
                      "reading_order": reading_order(ln), "words": words})
    return lines


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, groups = manifest_rows()
    crossing_input = []
    refused, summary = [], {"pages": {}, "refused_pages": 0, "groups_without_words": {}}
    files = {"train": (OUT / "press_train_words.jsonl").open("w", encoding="utf-8"),
             "override": (OUT / "press_dev_words_override.jsonl").open("w", encoding="utf-8")}
    overridden = set()
    for (split, page), group in sorted(groups.items()):
        manifest_lines = rows[(split, page)]
        first = next(iter(manifest_lines.values()))
        xml_path = Path(first["image"]).with_suffix(".xml")
        parsed = parse_page(xml_path)
        n_words = sum(len(l["words"]) for l in parsed)
        if n_words == 0:
            c = summary["groups_without_words"].setdefault(f"{split}/{group}", {"pages": 0})
            c["pages"] += 1
            continue
        page_key, issue_key = ids_of(page)
        result = eligibility(REGISTRY, ROLE[split], page_key=page_key, issue_key=issue_key, title_key=title_of_prefix(page))
        is_override = not result["allowed"] and split == "dev" and page in OVERRIDE_PAGES
        if not result["allowed"]:
            refused.append({"split": split, "page": page, "group": group, "words": n_words, "refusals": result["refusals"],
                            "keys": result["supplied"], "override_granted": is_override})
            if not is_override:
                continue
            overridden.add(page)
        if not is_override:
            crossing_input.append((page, page_key, issue_key))
        in_manifest = [l for l in parsed if l["line_id"] in manifest_lines]
        written = 0
        for l in parsed:
            m = manifest_lines.get(l["line_id"])
            for w in l["words"]:
                if w["polygon"] is None:
                    continue
                rec = {"split": split, "role": ROLE[split], "page": page, "group": group, "image": first["image"],
                       "line_id": l["line_id"], "line_in_manifest": m is not None,
                       "word_id": w["word_id"], "text": w["text"], "polygon": w["polygon"], **bbox(w["polygon"]),
                       "word_reading_order": w["reading_order"],
                       "line_baseline": l["baseline"], "line_polygon": l["polygon"], "line_reading_order": l["reading_order"],
                       "manifest_line_polygon": m["polygon"] if m else None, "manifest_line_truth": m["truth"] if m else None}
                files["override" if is_override else split].write(json.dumps({**rec, **OVERRIDE} if is_override else rec, ensure_ascii=False) + "\n")
                written += 1
        summary["pages"][f"{split}/{page}"] = {
            "xml_words": n_words, "words_written": written, "xml_lines": len(parsed),
            "xml_lines_in_manifest": len(in_manifest), "manifest_lines": len(manifest_lines),
            "xml_lines_not_in_manifest": sorted(l["line_id"] for l in parsed if l["line_id"] not in manifest_lines)[:20],
            "manifest_lines_not_in_xml": sorted(set(manifest_lines) - {l["line_id"] for l in parsed})[:20],
            "xml_sha256": sha(xml_path)}
    for f in files.values():
        f.close()
    summary["refused_pages"] = len(refused)
    summary["dev_manifest_pages"] = len({pg for (sp, pg, g) in [(k[0], k[1], v) for k, v in groups.items()] if sp == "dev"}) + len(
        {json.loads(l)["page"] for l in (PRESS / "dev.jsonl").open(encoding="utf-8") if json.loads(l)["corpus"] != "newseye_read"})
    hits, crossing_sources = no_test_crossing(crossing_input)
    summary["no_test_crossing"] = {"exported_pages_checked": len(crossing_input), "hits": hits, "sources": crossing_sources}
    ov_hits, _ = no_test_crossing([(pg, *ids_of(pg)) for pg in sorted(overridden)])
    unexpected = [h for h in ov_hits if h["in"] not in ("gallica_e13", "e12_test_names") or h["in"] == "e12_test_names"]
    summary["override_crossing"] = {"pages": sorted(overridden), "hits": ov_hits, "unexpected": unexpected,
                                    "expected": "seulement gallica_e13 (c'est le motif de l'exception)"}
    if unexpected:
        raise SystemExit(f"an override page is also in another test source: {unexpected[:3]}")
    if hits:
        raise SystemExit(f"a test or reserve page is among the exported pages: {hits[:3]}")
    (OUT / "refused_pages.json").write_text(json.dumps(refused, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    # --- part 3: the dev pages with true words, matched to the c1+M+A lines (line_geometry.v1, 0.5 h) ---
    sys.path[:0] = [str(AX / "src"), str(AX / "scripts")]
    from axel.layout.gt import load_page
    from axel.pipeline.chain import reading_of
    e28 = importlib.import_module("e28_fill")
    e32 = importlib.import_module("e32")
    import bench_pipeline as BP  # scripts/bench_pipeline.py: read_dirs, read_index, unit_spans (same adapter as the bench)

    dev_words = defaultdict(lambda: defaultdict(list))
    for line in (OUT / "press_dev_words_override.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        dev_words[r["page"]][r["line_id"]].append({k: r[k] for k in ("word_id", "text", "polygon", "x", "y", "w", "h", "word_reading_order")})
    bench = e32.BENCHES["dev"]
    unmatched_all, matched_total, pred_total = {}, 0, 0
    corr = (OUT / "press_dev_c1ma_correspondance_override.jsonl").open("w", encoding="utf-8")
    per_page = {}
    for p in e28.dev_pages():
        name = p["page"]
        if name not in dev_words:
            continue
        arms = e32.build_arms(bench, "c1", name)
        chain = arms[0].get("ma") if arms else None
        if chain is None:
            per_page[name] = {"status": "no_c1ma_chain"}
            continue
        texts, ctc = BP.read_index(BP.read_dirs(AX, "c1+ma"), name)
        gt = load_page(Path(p["xml"]))
        gt_lines = gt.lines()
        gt_geo = [GroundTruthLine(g.id, g.baseline, (max(q[1] for q in g.polygon) - min(q[1] for q in g.polygon)), "ground_truth")
                  if g.baseline and len(g.baseline) >= 2 and g.polygon else GroundTruthLine(g.id, None, None, None, "missing")
                  for g in gt_lines]
        pred_geo = [PredictedLine(u.id, [list(q) for q in u.baseline], "predicted") if u.baseline and len(u.baseline) >= 2
                    else PredictedLine(u.id, None, None, "missing") for u in chain.units]
        lg = evaluate_line_geometry(gt_geo, pred_geo)
        units = {u.id: u for u in chain.units}
        gts = {g.id: g for g in gt_lines}
        matched = set()
        for mt in lg.matches:
            u, g = units[mt.pred_id], gts[mt.gt_id]
            spans, confs, status = BP.unit_spans(u.text, u.id, texts, ctc, reading_of)
            corr.write(json.dumps({
                "page": name, "gt_line_id": mt.gt_id, "pred_line_id": mt.pred_id, "predicted_text": u.text,
                "cuts": [list(s) for s in spans] if spans else None, "conf": confs, "cuts_status": status,
                "pred_baseline": [list(q) for q in u.baseline], "pred_polygon": [list(q) for q in u.polygon],
                "gt_baseline": [list(q) for q in g.baseline], "gt_polygon": [list(q) for q in g.polygon], "gt_text": g.text,
                "gt_words": dev_words[name].get(mt.gt_id, []), **OVERRIDE}, ensure_ascii=False) + "\n")
            matched.add(mt.gt_id)
        assert len({mt.pred_id for mt in lg.matches}) == len(lg.matches), "principal matching: one predicted line per true line"
        unmatched = [g.id for g in gt_lines if g.id not in matched]
        unmatched_all[name] = {"gt_lines_unmatched": unmatched, "gt_words_in_unmatched": sum(len(dev_words[name].get(i, [])) for i in unmatched)}
        matched_total += len(matched)
        pred_total += len(chain.units)
        per_page[name] = {"status": lg.status, "n_gt": lg.n_gt, "n_pred": lg.n_pred, "matched": lg.matched_count,
                          "gt_words": sum(len(v) for v in dev_words[name].values()),
                          "gt_words_in_matched_lines": sum(len(dev_words[name].get(i, [])) for i in matched)}
    corr.close()
    summary["correspondence"] = {"pages": per_page, "unmatched": unmatched_all, "matching": "line_geometry.v1, tolérance 0,5 h (valeur par défaut), c1+ma"}
    summary["files"] = {f.name: sha(f) for f in sorted(OUT.glob("*.jsonl")) + [OUT / "refused_pages.json"]}
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"pages_ok": len(summary["pages"]), "refused": len(refused), "no_words": summary["groups_without_words"],
                      "correspondence": per_page}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
