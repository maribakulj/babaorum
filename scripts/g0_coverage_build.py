"""Build config/g0_coverage.json: which collection(s) every data entry touches (G0 v2, step A2).

Covers the 23 entries of ~/corpus-vt, every entry of ~/axel/runs, and every file under
~/axel/data/prepared. Evidence comes from file NAMES and from axel's own docs/*.md; no
image, ground truth or transcription is opened. An entry that carries page data but
matches no collection is left "unclassified" (the check then fails) instead of guessed.
An entry is "no_page_data" only on a recorded proof (extensions found, none of them a page list).

    PYTHONDONTWRITEBYTECODE=1 ~/axel/.venv/bin/python scripts/g0_coverage_build.py
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from babaorum.g0_collections import REQUIRED_COLLECTIONS, verify_coverage  # noqa: E402

H = Path.home()
CV, AX = H / "corpus-vt", H / "axel"
OUT = ROOT / "config" / "g0_coverage.json"
PREP_DIR = AX / "data" / "prepared"
PAGE_LIST_SUFFIXES = (".jsonl", "pages.json", "manifest.json")

#: tokens found in file or folder names -> collections touched
NAME_TOKENS = [
    ("bnl", ["bnl_pack"]), ("gallicorpora", ["gallicorpora"]), ("ocr17", ["gallicorpora"]),
    ("icdar2017", ["icdar2017_2019"]), ("icdar2019", ["icdar2017_2019"]), ("newseye", ["newseye"]),
    ("impresso", ["impresso"]), ("nzz", ["nzz_impresso"]), ("hipe", ["hipe_ocrepair"]),
    ("press.dev", ["newseye", "bnl_pack"]), ("press.train", ["newseye", "bnl_pack"]),
    ("presse_e7.test", ["presse_e7_test"]), ("presse_e7.dev", ["presse_e7_train_dev"]), ("presse_e7.train", ["presse_e7_train_dev"]),
    ("impact", ["kb_impact"]), ("french-pd", ["pleias_french_pd_322", "reader_lm_corpus"]), ("pseudo", ["gallica_pseudo"]),
    ("e31", ["gallica_e31_reserve"]), ("gardee", ["gallicorpora"]),
    ("fetch_kb", ["kb_impact"]), ("fetch_bnf", ["bnf_ocr_corrige_presse"]), ("e12_dev", ["e12_dev"]),
    ("indep", ["e22_bench"]), ("sak_newseye", ["newseye"]),
]
#: words in axel/docs/<exp>.md that signal a consumed collection (over-inclusion is the safe direction)
DOC_TOKENS = [
    (r"\bbnl\b", ["bnl_pack"]), (r"gallicorpora|ocr17", ["gallicorpora"]), (r"icdar", ["icdar2017_2019"]),
    (r"newseye", ["newseye"]), (r"impresso", ["impresso"]), (r"\bnzz\b", ["nzz_impresso"]), (r"hipe", ["hipe_ocrepair"]),
    (r"presse_e7", ["presse_e7_train_dev", "presse_e7_test"]), (r"impact|\bkb\b", ["kb_impact"]),
    (r"\bAS\b|newseye_as", ["as_anno_183"]), (r"dev d.E12|E12 dev|5 pages de dev|pages dev d.E12", ["e12_dev"]),
    (r"test d.E12|E12 test|12 pages de test", ["e12_test"]), (r"e22[ /_-]*bench|banc e22", ["e22_bench"]), (r"\be13\b", ["gallica_e13"]),
]
CORPUS_VT = {
    "bnf-presse-seg": ["bnf_ocr_corrige_presse"], "bnl-gt-pack": ["bnl_pack"],
    "french-pd-sample": ["pleias_french_pd_322", "reader_lm_corpus"], "gallica-alto-e13": ["gallica_e13"],
    "gallica-banc-e31": ["gallica_e31_reserve"], "gardee": ["gallicorpora"], "gardee-med": ["gallicorpora"],
    "gardee-tune": ["gallicorpora"], "hipe-ocrepair": ["hipe_ocrepair"], "icdar2019": ["icdar2017_2019"],
    "impact-kb-kranten": ["kb_impact"], "newseye": ["newseye"], "newseye-as": ["as_anno_183"],
    "newseye-read": ["newseye"], "nzz-gt": ["nzz_impresso"], "ocr17_gallica": ["gallicorpora"],
    "presse-e7": ["presse_e7_train_dev", "presse_e7_test"], "pseudo": ["gallica_pseudo"], "raw": ["gallicorpora"],
    "selection.json": ["gallicorpora"], "vision": ["gallicorpora"], "vision-med": ["gallicorpora"],
}
CORPUS_VT_SOURCE = {
    "gardee": "OCR17+ : mêmes pages que corpus-vt/raw (Balzac1624…, Boyer1697…) ; axel/docs/PROTOCOLE.md, phase1.md",
    "gardee-med": "OCR17+ ; axel/docs/phase1.md", "gardee-tune": "OCR17+ ; axel/docs/phase1.md",
    "selection.json": "liste de paires OCR17+ (document, page), lue par le nom : Balzac1624_Lettres…",
    "ocr17_gallica": "dossier vide ; OCR17+ via corpus-vt/raw (docs/e1.md)", "nzz-gt": "banc NZZ de la mesure unique E30 (DECISIONS.md, E30)",
    "raw": "pages OCR17+ (docs/e1.md, data/prepared/ocr17.test.jsonl)",
    "vision": "traces sur pages OCR17+ (Moliere1663…, LaFayette1678… : mêmes documents que corpus-vt/raw)",
    "vision-med": "traces sur pages OCR17+ (mêmes documents que corpus-vt/raw)",
    "presse-e7": "images des pages de data/prepared/presse_e7/pages.jsonl (train, dev et test mêlés)",
}


def walk_names(root: Path, depth: int = 4, cap: int = 400_000):
    base = len(root.parts)
    n = 0
    for dirpath, dirnames, filenames in os.walk(root):
        if len(Path(dirpath).parts) - base >= depth:
            dirnames[:] = []
        for name in [*dirnames, *filenames]:
            n += 1
            if n > cap:
                return
            yield Path(dirpath, name)


def tokens_from_names(root: Path) -> tuple[list[str], list[str]]:
    found: dict[str, str] = {}
    for p in walk_names(root):
        low = p.name.lower()
        for token, colls in NAME_TOKENS:
            if token in low:
                for c in colls:
                    found.setdefault(c, str(p.relative_to(root.parent.parent if root.parent.name == "runs" else root.parent)))
    return sorted(found), sorted(set(found.values()))[:6]


def doc_collections(exp: str) -> tuple[list[str], str | None]:
    doc = AX / "docs" / f"{exp}.md"
    if not doc.is_file():
        return [], None
    text = doc.read_text(encoding="utf-8", errors="replace")
    found = sorted({c for pattern, colls in DOC_TOKENS if re.search(pattern, text, re.I) for c in colls})
    return found, f"axel/docs/{exp}.md"


def scan_report_text(root: Path) -> tuple[list[str], list[str]]:
    """Fallback: tokens inside axel's own small .md/.log reports (searched, not analysed)."""
    found: set[str] = set()
    cited: list[str] = []
    for p in walk_names(root, depth=2):
        if p.is_file() and p.suffix in (".md", ".log") and p.stat().st_size <= 300_000:
            text = p.read_text(encoding="utf-8", errors="replace")
            hits = {c for pattern, colls in DOC_TOKENS if re.search(pattern, text, re.I) for c in colls}
            if hits:
                found |= hits
                cited.append(f"axel/{p.relative_to(AX)}")
    return sorted(found), cited[:6]


def proof_no_page_data(root: Path) -> dict | None:
    ext = Counter()
    for p in walk_names(root):
        if p.is_file():
            ext[p.suffix.lower() or "(sans extension)"] += 1
            if p.name.endswith(PAGE_LIST_SUFFIXES):
                return None
    return {"command": f"os.walk({root.name}, profondeur 4) : aucun fichier *.jsonl, *pages.json ni *manifest.json",
            "result_empty": True, "extensions_found": dict(ext.most_common(8))}


SIX_SOURCES = ["bnl_pack", "gallicorpora", "icdar2017_2019", "impresso"]
RUN_OVERRIDES = {
    "e12": (["e12_dev", "e12_test"], ["axel/runs/e12/pages.json"]),
    "tune": (SIX_SOURCES, ["axel/docs/phase2.md", "axel/runs/tune/s2.md"]),
    "e21": (["as_anno_183", "bnf_ocr_corrige_presse", "kb_impact", "newseye"],
            ["axel/docs/e21_proposition.md", "axel/runs/e21/logs/fetch_kb.log", "axel/runs/e21/logs/fetch_bnf.log"]),
    # S2 : « 21 278 lignes (six sources) » ; l'estimateur liste bnl, gallicorpora17/18, icdar2017/2019, impresso
    "train": (SIX_SOURCES, ["axel/docs/phase2.md", "axel/runs/train/s2_seed1_3sources.jsonl"]),
    "estimate": (SIX_SOURCES, ["axel/runs/estimate/dev.md", "axel/docs/phase2.md"]),
}


def classify_run(d: Path) -> dict:
    if d.name in RUN_OVERRIDES:
        colls, cited = RUN_OVERRIDES[d.name]
        return {"collections": sorted(colls), "method": "explicit_override_from_axel_docs", "source_files": cited}
    colls, cited = tokens_from_names(d)
    method = "file_names"
    stem = re.search(r"e\d+", d.name)
    doc_name = d.name if (AX / "docs" / f"{d.name}.md").is_file() else (stem.group(0) if stem else d.name)
    doc_colls, doc = doc_collections(doc_name)
    if doc_colls:
        colls = sorted({*colls, *doc_colls})
        cited = [*cited, doc] if doc else cited
        method = "file_names+axel_doc_keywords" if method == "file_names" and cited[:-1] else "axel_doc_keywords"
    if not colls:
        colls, cited = scan_report_text(d)
        method = "axel_report_keywords"
    if d.name == "e22":
        colls = sorted({*colls, "e22_bench"}); cited = [*cited, "axel/runs/e22/bench/runs/e12/pages.json"]
    if d.name == "e13":
        colls = sorted({*colls, "gallica_e13"}); cited = [*cited, "axel/runs/e13/scores.json"]
    if colls:
        return {"collections": colls, "method": method, "source_files": cited}
    proof = proof_no_page_data(d)
    if proof is not None:
        return {"collections": [], "no_page_data": proof, "method": "no_page_data_proof"}
    return {"collections": [], "unclassified": True, "method": "needs_review"}


def main() -> int:
    entries: dict[str, dict] = {}
    for p in sorted(CV.iterdir()):
        key = f"corpus-vt/{p.name}"
        if p.name in CORPUS_VT:
            entries[key] = {"collections": CORPUS_VT[p.name], "method": "explicit_table",
                            "source_files": [CORPUS_VT_SOURCE.get(p.name, f"nom du dossier ; {CORPUS_VT[p.name][0]} (registre G0)")]}
        else:
            names, cited_names = tokens_from_names(p) if p.is_dir() else ([], [])
            if names:
                entries[key] = {"collections": names, "method": "file_names", "source_files": [f"corpus-vt/{p.name}/{c.split('/')[-1]}" for c in cited_names][:3] or [key]}
                continue
            proof = proof_no_page_data(p) if p.is_dir() else None
            entries[key] = ({"collections": [], "no_page_data": proof, "method": "no_page_data_proof"} if proof
                            else {"collections": [], "unclassified": True, "method": "needs_review"})
    for p in sorted((AX / "runs").iterdir()):
        entries[f"axel/runs/{p.name}"] = classify_run(p) if p.is_dir() else {"collections": [], "unclassified": True, "method": "needs_review"}
    for p in sorted((AX / "data" / "prepared").rglob("*")):
        if not p.is_file():
            continue
        rel = str(p.relative_to(AX / "data" / "prepared"))
        key = f"axel/data/prepared/{rel}"
        low = rel.lower()
        aggregate = low.startswith(("stats", "confusion")) or low in ("seg/coco_stats.json", "seg/lines_stats.json")
        if aggregate:
            # pJ (A2): statistics and confusion tables are aggregates, not page lists
            if p.suffix == ".md":
                shown = next((ln.strip() for ln in p.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()), "")
                evidence = f"première ligne : {shown[:80]}"
            else:
                obj = json.loads(p.read_text(encoding="utf-8"))
                evidence = "clés : " + ", ".join(list(obj)[:8]) if isinstance(obj, dict) else f"{type(obj).__name__} de {len(obj)} éléments"
            entries[key] = {"collections": [], "method": "aggregate_not_page_list",
                            "no_page_data": {"command": f"lecture de l'en-tête de {rel} ({evidence}) : tableau d'effectifs ou d'erreurs agrégés, aucune liste de pages",
                                             "result_empty": True, "kind": "aggregate_not_page_list"}}
            continue
        if low in ("olr/newseye_as.json", "olr/newseye_as.cross.json"):
            # pJ (A2): newseye_as is NOT forced to AS only; its own `source` field names NewsEye/READ AS (Zenodo 5654841)
            origin = json.loads((PREP_DIR / "olr/newseye_as.json").read_text(encoding="utf-8")).get("source", "")
            entries[key] = {"collections": ["as_anno_183", "newseye"], "method": "source_field_of_the_file",
                            "source_files": [f"axel/data/prepared/{rel}", f"champ source : {origin}"]}
            continue
        colls = sorted({c for token, cs in NAME_TOKENS if token.replace(".", "") in low.replace(".", "") or token in low for c in cs})
        if low.startswith("press/"):
            colls = ["bnl_pack", "newseye"]
        elif low.startswith("presse_e7/") or low.startswith("presse_test_manifest"):
            colls = ["presse_e7_test"] if "test" in low else ["presse_e7_train_dev"]
            if low.endswith(("pages.jsonl", "excluded.json")):
                colls = ["presse_e7_train_dev", "presse_e7_test", "presse_e7_test_refused"]
            if "test" in low and low.endswith("test.jsonl"):
                colls = ["presse_e7_test", "presse_e7_test_refused", "kb_impact", "bnf_ocr_corrige_presse"]
            if "impact" in low:
                colls = ["kb_impact"]
            if "pages_bnf" in low:
                colls = ["bnf_ocr_corrige_presse"]
        elif low.startswith("seg/"):
            colls = ["as_anno_183", "kb_impact", "newseye", "bnf_ocr_corrige_presse"]
        elif low.startswith("olr/"):
            colls = ["as_anno_183", "newseye"]
        entries[key] = ({"collections": colls, "method": "file_names", "source_files": [f"axel/data/prepared/{rel}"]} if colls
                        else {"collections": [], "unclassified": True, "method": "needs_review"})
    unknown = sorted({c for e in entries.values() for c in e["collections"]} - REQUIRED_COLLECTIONS)
    doc = {
        "schema_version": "babaorum.g0_coverage.v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "basis": "noms de fichiers et docs d'axel ; aucun contenu de page, image ou vérité lu. Sur-inclusion = sens prudent.",
        "roots": {"corpus_vt": str(CV), "axel_runs": str(AX / "runs"), "axel_prepared": str(AX / "data/prepared")},
        "entries": entries,
    }
    OUT.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    counts = Counter("unclassified" if e.get("unclassified") else "no_page_data" if "no_page_data" in e else "classified" for e in entries.values())
    print(json.dumps({"entries": len(entries), **counts, "unknown_collection_names": unknown}, ensure_ascii=False))
    for k, e in entries.items():
        if e.get("unclassified"):
            print("UNCLASSIFIED:", k)
    try:
        print(json.dumps(verify_coverage(OUT, ROOT / "config" / "g0_collections.json"), ensure_ascii=False))
    except ValueError as exc:
        print("coverage check:", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
