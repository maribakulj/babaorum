"""Match keys of every G0 collection, per level (G0 v2). Read-only on every source.

Lives in src/ (moved from scripts/g0_keys_build.py in LOT-G0-02) so that verify_collections can RECOMPUTE the
declared keys from the pinned sources and demand exact equality, for every collection.

Levels and the key used to match a document across collections:
  page  : canonical page key (identity.canonical_page, e.g. gallica:512130/f3, kb:00531071) or, for presse_e7,
          the Wikimedia file view (scan_page_key)
  issue : canonical issue key (identity.canonical_issue, e.g. gallica:512130); outside the bpt6k namespace
          the full ARK is kept with the prefix "ark:" (it can only match another ARK of the same namespace)
  title : normalized title (g0_inventory.title_key)
A level where a collection has no identity in these namespaces is "aucune clé à ce niveau", with its reason.
Only names, identifiers and metadata fields are read; no image, no transcription is analysed.
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
from babaorum.identity import IdentityError, canonical_issue, canonical_page  # noqa: E402

# Set by compute_keys() from the DECLARED roots (LOT G0 v2.2): no default under the home directory, ever.
AX = CV = PREP = E7 = None
LEVELS = ("page", "issue", "title")

_spec = importlib.util.spec_from_file_location("g0_inventory_for_keys", ROOT / "scripts/g0_inventory.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]
title_key = _mod.title_key

NAMES = {"page": "doc_key (canonical_page)", "issue": "issue_key (canonical_issue)", "title": "titre normalisé (title_key)"}


def ids_of(name: str):
    page = issue = None
    try:
        page = canonical_page(name)
    except IdentityError:
        pass
    try:
        issue = canonical_issue(name)
    except IdentityError:
        pass
    return page, issue


def title_of_prefix(name: str) -> str | None:
    m = re.match(r"^(.+?)_\d{5,8}_[A-Z]\d{7}$", name)
    return title_key(m.group(1).replace("_", " ").replace("-", " ")) if m else None


def ark_issue(text: str) -> str | None:
    m = re.search(r"(bpt6k[0-9a-z]+)", text)
    if m:
        try:
            return canonical_issue(m.group(1))
        except IdentityError:
            return None
    m = re.search(r"(btv1b[0-9a-z]+|cb[0-9]{6,}[0-9a-z])", text)
    return f"ark:{m.group(1)}" if m else None


def make(units, names=None, notes=None):
    """units: iterable of dicts with page/issue/title (a value, a list of values, or None). Returns level -> spec|None, reasons."""
    units = list(units)
    keys, reasons = {}, {}
    for level in LEVELS:
        values, keyed = set(), 0
        for u in units:
            v = u.get(level)
            vs = [x for x in (v if isinstance(v, (list, tuple)) else [v]) if x]
            if vs:
                keyed += 1
                values.update(vs)
        if values:
            keys[level] = {"name": (names or NAMES)[level] + ((" — " + notes[level]) if notes and level in notes else ""),
                           "values": sorted(values), "coverage": {"keyed": keyed, "total": len(units)}}
        else:
            keys[level] = None
            reasons[level] = (notes or {}).get(level) or f"aucune des {len(units)} unités n'a d'identité à ce niveau"
    return keys, reasons


def no_key(reason: str):
    return {level: None for level in LEVELS}, {level: reason for level in LEVELS}


def jsonl_fields(path: Path, fields):
    for line in path.open(encoding="utf-8", errors="replace"):
        if line.strip():
            r = json.loads(line)
            yield {f: r.get(f) for f in fields}


def e7_pages():
    """presse_e7 pages.jsonl joined with the Wikisource crosswalk: split, title, scan_page_key, credited documents."""
    inv = json.loads((ROOT / "reports/g0_wikisource_crosswalk_investigation.json").read_text(encoding="utf-8"))
    scan, docs = {}, {}
    for g in inv["groups"]:
        credited = [d["document_key"] for d in (g.get("source_documents") or [])]
        for p in g["pages"]:
            scan[p["page_id"]], docs[p["page_id"]] = p["scan_page_key"], credited
    out = {}
    for r in jsonl_fields(E7 / "pages.jsonl", ("page_id", "index", "title", "date", "split", "image")):
        out[r["page_id"]] = {**r, "scan_page_key": scan[r["page_id"]], "issue_keys": docs[r["page_id"]],
                             "title_key": title_key(r["title"])}
    return out


def inv_units(source: str):
    rows = json.loads((ROOT / "reports/g0_inventory.json").read_text(encoding="utf-8"))["pages"]
    return [{"page": r["page_key"], "issue": r["issue_key"], "title": r["title_key"]} for r in rows if r["source"] == source]


def name_units(names):
    out = []
    for n in names:
        page, issue = ids_of(n)
        out.append({"page": page, "issue": issue, "title": title_of_prefix(n)})
    return out


def compute_keys():
    global AX, CV, PREP, E7
    from babaorum.g0_collections import declared_root  # lazy: the registry module imports this one lazily too
    AX, CV = declared_root("axel_root"), declared_root("corpus_root")
    PREP = AX / "data/prepared"
    E7 = PREP / "presse_e7"
    K = {}
    K["bnl_pack"] = no_key("identifiants locaux du pack BnL (0018, TL_0001…) : ni page, ni numéro, ni titre Gallica")
    units = inv_units("as")
    K["as_anno_183"] = make(units, notes={"page": "obtenu seulement par la concordance READ (8 pages sur 183)",
                                          "issue": "obtenu seulement par la concordance READ (8 numéros sur 183)"})
    kb = inv_units("kb")
    for r in jsonl_fields(E7 / "pages_impact.jsonl", ("page_id",)):
        pk, _ = ids_of(r["page_id"])
        if pk and not any(u["page"] == pk for u in kb):
            kb.append({"page": pk, "issue": None, "title": None})
    K["kb_impact"] = make(kb, notes={"issue": "une page KB n'identifie aucun numéro historique (B1)",
                                     "title": "concordance titre/date KB absente (impact_concord.xlsx introuvable)"})
    bnf = inv_units("bnf")
    for r in jsonl_fields(E7 / "pages_bnf.jsonl", ("page_id", "ark", "title")):
        bnf.append({"page": None, "issue": ark_issue(r["ark"] or ""), "title": title_key(r["title"])})
    K["bnf_ocr_corrige_presse"] = make(bnf, notes={"page": "folio absent pour une partie des pages (paquet BnF par numéro)"})
    news = inv_units("read")
    seen = set()
    for split in ("train", "dev"):
        for r in jsonl_fields(PREP / f"press/{split}.jsonl", ("corpus", "page")):
            if r["corpus"] == "newseye_read" and r["page"] not in seen:
                seen.add(r["page"])
                news += name_units([r["page"]])
    val = CV / "newseye/val/ATR_ValidationSet_BnF_Newseye_M2+"
    news += name_units(sorted({p.stem for p in val.glob("*.xml")}))
    K["newseye"] = make(news, notes={"title": "titre pris dans le préfixe du nom de fichier quand il existe"})
    gc = []
    for stem in ("gallicorpora17", "gallicorpora18"):
        for split in ("train", "dev", "test"):
            f = PREP / f"{stem}.{split}.jsonl"
            if f.exists():
                for r in jsonl_fields(f, ("doc_id", "unit_id")):
                    doc = r["doc_id"] or ""
                    page = None
                    m = re.match(r"f(\d+):", str(r["unit_id"] or ""))
                    if m and doc.startswith("bpt6k"):
                        try:
                            page = canonical_page(f"{doc}.f{m.group(1)}")
                        except IdentityError:
                            page = None
                    gc.append({"page": page, "issue": ark_issue(doc), "title": None})
    for r in jsonl_fields(PREP / "ocr17.test.jsonl", ("doc_id",)):
        gc.append({"page": None, "issue": ark_issue(r["doc_id"] or ""), "title": None})
    K["gallicorpora"] = make(gc, notes={"title": "aucun titre dans les manifestes Gallicorpora/OCR17+",
                                        "page": "OCR17+ (ark btv1b/cb…) n'a pas de clé de page comparable"})
    K["hipe_ocrepair"] = no_key("jeux hétérogènes (ICDAR-2017, DTA19, Overproof…) à identifiants propres, sans identité Gallica")
    pleias = json.loads((CV / "french-pd-sample/manifest.json").read_text(encoding="utf-8"))["kept_file_ids"]
    K["pleias_french_pd_322"] = make([{"issue": ark_issue(i)} for i in pleias],
                                     notes={"page": "corpus de numéros entiers, sans folio", "title": "aucun titre dans le manifeste"})
    K["reader_lm_corpus"] = K["pleias_french_pd_322"]
    pages = e7_pages()
    for name, splits in (("presse_e7_train_dev", ("train", "dev")),):
        K[name] = make([{"page": p["scan_page_key"], "issue": p["issue_keys"], "title": p["title_key"]}
                        for p in pages.values() if p["split"] in splits],
                       names={"page": "scan_page_key (fichier Wikimedia SHA-1 + vue)", "issue": NAMES["issue"] + " du document Gallica crédité", "title": NAMES["title"]},
                       notes={"issue": "seulement les pages dont la notice crédite un document Gallica"})
    K["mille_feuilles_sources"] = no_key("textes originaux synthétiques CC0 : aucune identité de page, de numéro ni de titre historique")
    K["icdar2017_2019"] = no_key("identifiants locaux ICDAR (fr_periodical-77, …) sans identité Gallica ni titre normalisé")
    K["impresso"] = no_key("identifiants impresso (EDA-1843-08-07-a…) : codes de titre et de date, pas de numéro ni de titre Gallica")
    nzz_keys, nzz_reasons = make([{"title": title_key("Neue Zürcher Zeitung")}],
                                 notes={"page": "images et XML NZZ à identifiants propres, hors Gallica",
                                        "issue": "numéros NZZ hors Gallica, aucune identité Gallica",
                                        "title": "titre unique de la collection NZZ (docs/e7.md, DECISIONS.md E30) : 167 pages, tout le même journal"})
    nzz_keys["title"]["complete"] = True
    nzz_keys["title"]["coverage"] = {"keyed": 167, "total": 167}
    K["nzz_impresso"] = (nzz_keys, nzz_reasons)
    K["mccatmus"] = no_key("données d'entraînement du modèle non énumérables (22 jeux CATMuS)")
    K["kraken_blla_2026"] = no_key("données d'entraînement du modèle non énumérables")
    K["dots_ocr"] = no_key("préentraînement non auditable")
    K["gallica_pseudo"] = no_key("liste de pages sur le Drive monté : lecture du contenu en délai dépassé (Errno 60), seuls noms et tailles épinglés")
    p12 = json.loads((AX / "runs/e12/pages.json").read_text())
    for name, split in (("e12_dev", "dev"), ("e12_test", "test")):
        K[name] = make(name_units([p["page"] for p in p12 if p["split"] == split]),
                       notes={"title": "titre pris dans le préfixe du nom de fichier"})
    pb = json.loads((AX / "runs/e22/bench/runs/e12/pages.json").read_text())
    K["e22_bench"] = make(name_units([p["page"] for p in pb]), notes={"title": "titre pris dans le préfixe du nom de fichier"})
    names13 = sorted({re.sub(r"\.(alto|pagination|toc)\.xml(\.err)?$", "", p.name) for p in (CV / "gallica-alto-e13").glob("*.alto.xml")})
    K["gallica_e13"] = make([{"page": ids_of(n)[0], "issue": ids_of(n)[1], "title": None} for n in names13],
                            notes={"title": "aucun titre dans les noms de fichiers"})
    e31 = []
    for p in sorted((CV / "gallica-banc-e31/pages").glob("*.jpg")):
        m = re.match(r"e31_\d+_(\d+)_f(\d+)$", p.stem)
        if m:
            page, issue = ids_of(f"gallica:{m.group(1)}/f{m.group(2)}")
            e31.append({"page": page, "issue": issue, "title": None})
    K["gallica_e31_reserve"] = make(e31, notes={"title": "aucun titre dans les noms de fichiers (titres seulement dans docs/e31.md)"})
    return K


if __name__ == "__main__":
    for name, (keys, reasons) in compute_keys().items():
        print(f"{name:26s}", {lv: (f"{len(keys[lv]['values'])} clés ({keys[lv]['coverage']['keyed']}/{keys[lv]['coverage']['total']})"
                                   if keys[lv] else "AUCUNE") for lv in LEVELS})


# --- keys derived from a seal (shared by the registry builder and by verify_collections) -----------------

SEAL_NAMES = {"page": "scan_page_key (fichier Wikimedia SHA-1 + vue)", "issue": NAMES["issue"] + " du document Gallica crédité",
              "title": NAMES["title"]}


def keys_from_seal(seal: dict, section: str):
    """Keys of the pages listed under ``section`` ('pages' = sealed, 'refused') of a presse_e7 test seal."""
    pages = seal[section]
    return make([{"page": p["scan_page_key"], "issue": p["issue_keys"], "title": p["title_key"]} for p in pages],
                names=SEAL_NAMES, notes={"issue": "seulement les pages dont la notice crédite un document Gallica"})


def headline_keys_from_seal(seal: dict) -> list[str]:
    """Page keys of the sealed pages on which the measure that counts may be computed (headline=true)."""
    return sorted(p["scan_page_key"] for p in seal["pages"] if p["headline"] is True)
