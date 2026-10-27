"""G0-04 role `suivi`, step B: eligibility(registry, "suivi", page_key=..., issue_key=...).

Contract: ~/Documents/reviews/babaorum-pipeline/LOT-G0-04-role-suivi.md, v4 and v4.1 (design GO of the auditor on 912639a1); the final emission
order of v4 is the only one in force. Identity and scope come from g0_suivi_identity (points 1 and 2, GO pT on 431b47f1).

The input interface is frozen (pA, 22/10): the format is config/g0_suivi_inputs_schema_v1.json, pinned below by SHA-256; the paths and anchors of
the real files live in g0_suivi_anchors.py (data only). Fail closed everywhere:
- a snapshot keeps only the RAW BYTES of the files; eligibility re-hashes them against the module anchors and parses them itself, so a document
  edited or forged after loading is never read (pT, 23/10, B1);
- an absent or unpinned file gives SEAL_NOT_PINNED and its content is never read;
- a malformed, partial or duplicate-key file raises SuiviInputError (SUIVI_INPUT_INVALID), so no page is ever admitted from it (pT, B2);
- every page-level constraint follows the class of PROVEN page aliases of the queried key; document-only links never carry page constraints (B3).
Trust boundary: the module anchors (SUIVI_ANCHORS, a module-level dict) are code authority, like the anchors file itself; code running in the same
process can change them, exactly as it could edit that file. They protect against stale or edited DATA, not against hostile code in-process.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
import json
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .g0_collections import NO_KEY, _issue_keys_complete, _key_values, _pin_codes, _scope
from .g0_suivi_anchors import SUIVI_ANCHORS, SUIVI_PATHS
from .g0_suivi_identity import NAMESPACES, Alias, PageIdentity, Usage, _closure, _namespace_of, _whole, applicable_usages, page_identity
from .identity import IdentityError

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "config" / "g0_suivi_inputs_schema_v1.json"
SCHEMA_SHA256 = "554086841993566d45d435c3fac618bbf595bad566852eae9689b5e618139751"
PINNED_FILES = ("usage", "sources", "journals", "select_evidence", "consultations")
SCHEMAS = {"usage": "babaorum.g0_page_usage.v1", "sources": "babaorum.g0_usage_sources.v1",
           "journals": "babaorum.g0_journal_review.v1", "select_evidence": "babaorum.g0_select_evidence.v1",
           "consultations": "babaorum.g0_suivi_consultations.v1"}
SEALED = "presse_e7_test"
USAGES = ("train", "dev", "select", "eval")
USAGE_STATUS = ("consumed", "allocated_only")
SCOPES = ("page", "document", "fichier")
SOURCE_STATUS = ("consumed", "allocated_only", "non_resolue", "non_joignable", "modele_tiers_preentraine")
MODEL = "modele_tiers_preentraine"            # AV-G0-SUIVI-MODELES (pA, 24/10): a pretrained third-party model, not a source of pages
JOINING = ("consumed", "allocated_only")       # only these rows of the table can cover a space (v4 point 7)
CODE_OF = {"train": "SUIVI_TRAINED", "dev": "SUIVI_DEV", "select": "SUIVI_SELECTED"}
FINDINGS = ("train_overlap", "eval_overlap", "third_party_proven", "third_party_unresolved", "third_party_eval", "lm_overlap", "content_overlap")
CONTENT_CHECK = "ngram8_content"                # AV-G0-SUIVI-CONTENU (pA, 25/10): the page text against ALL the texts of a collection
PAGE_TEXT_KINDS = ("vt_bbvlm",)                 # the text of the page compared: the clean VT of BBVLM, never an OCR
TEXT_SPACES = (["unknown"], ["gallica"])       # rows whose texts on disk may replace the refusal by ngram8 (AV-G0-SUIVI-LM ; AV-G0-SUIVI-TEXTES, pA 24/10)
LM_CHECK = "ngram8"                            # amendment AV-G0-SUIVI-LM: the check every page owes to an LM source whose texts are on disk
TABLE_SPACES = ("gallica", "kb", "wikimedia")  # covered only by a joining row of the sources table; gallica also needs complete number keys
NZZ_CHECKS = ("catmus", "ngram8")              # the controls of v4 point 6 that must be finished before nzz is covered (v4 point 4)
FORCED_LIMITS = {sp: ("LIMIT:mccatmus_non_french_uncovered",) for sp in ("nzz", "kb", "bnl")}   # v4 point 5 for nzz (pT, C2); kb and bnl
# added by pA (24/10, AV-G0-SUIVI-LIMITES): non-French or mixed pages that the French-printed CATMuS index does not cover
# AV-G0-SUIVI-LIMITES (pA, 24/10), v1 lines 41-43 and v2 point 4 of the contract: the third-party pretrainings that cannot be audited are
# rendered on every page as labels, never as a refusal and never as an absence; dots.mocr is pA's addition (the model of R7(a))
PRETRAINING_LIMITS = ("LIMIT:PP-DocLayoutV3", "LIMIT:blla", "LIMIT:dots.mocr", "LIMIT:dots.ocr", "LIMIT:german_print")
# AV-G0-SUIVI-MODELES: the only names a model row may take, each bound to its contract limit (Sonnet: a row named mccatmus, or dots_ocr
# with LIMIT:blla, must not pass); McCATMuS is resolved by E6 and is never a model row
MODEL_LIMITS = {"dots_ocr": "LIMIT:dots.ocr", "dots_mocr": "LIMIT:dots.mocr", "kraken_blla_2026": "LIMIT:blla",
                "pp_doclayoutv3": "LIMIT:PP-DocLayoutV3", "german_print": "LIMIT:german_print"}
_HEX64 = frozenset("0123456789abcdef")
# AV-G0-SUIVI-TITRES (pA, 25/10): a BnF title record (notice de titre) "cb" + 8 digits + a check character of the ARK alphabet
CB = re.compile(r"cb[0-9]{8}[0-9bcdfghjkmnpqrstvwxz]")
NONE = "__none__"                              # concerns ["__none__"]: the journal concerns no bench (amendment AV-G0-SUIVI-JOURNAUX)


class SuiviInputError(ValueError):
    code = "SUIVI_INPUT_INVALID"


def _invalid(ok: bool, message: str) -> None:
    if not ok:
        raise SuiviInputError(f"SUIVI_INPUT_INVALID: {message}")


def _text(v: Any) -> bool:
    return isinstance(v, str) and bool(v.strip())


def _texts(v: Any, *, empty: bool = False) -> bool:
    return isinstance(v, list) and (empty or bool(v)) and all(_text(x) for x in v)


def _fold(v: str) -> str:
    """Name comparison key: NFKC, casefold, without format (Cf) characters such as zero-width spaces nor any whitespace (pQ, B2)."""
    return "".join(ch for ch in unicodedata.normalize("NFKC", v).casefold() if unicodedata.category(ch) != "Cf" and not ch.isspace())


def lm_fingerprint(files: list[Mapping[str, Any]]) -> str:
    """Fingerprint of an lm_texts manifest (AV-G0-SUIVI-LM v2, pQ B1): SHA-256 of the compact JSON of the sorted [root, path, sha256] triples.
    A check must carry, for each LM source it was run against, the fingerprint of the manifest it actually compared."""
    rows = sorted([f["path"]["root"], f["path"]["path"], f["sha256"]] for f in files)
    return hashlib.sha256(json.dumps(rows, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def _sha64(v: Any) -> bool:
    return isinstance(v, str) and len(v) == 64 and set(v) <= _HEX64


def _ident(v: Any) -> bool:
    """A join key: a non-empty string with no surrounding whitespace (pT, 23/10: "kb:00531074 " must not slip past a join)."""
    return isinstance(v, str) and bool(v) and v == v.strip()


def _idents(v: Any, *, empty: bool = False) -> bool:
    return isinstance(v, list) and (empty or bool(v)) and all(_ident(x) for x in v)


def _canonical_page(v: Any) -> bool:
    """The exact canonical page key of page_identity (no surrounding space, no equivalent spelling)."""
    if not _ident(v):
        return False
    ns = _namespace_of(v)
    try:
        return ns is not None and page_identity(v, ns).page == v
    except IdentityError:
        return False


DECLARED_SPACES = ("impresso", "icdar", "hipe")   # AV-G0-SUIVI-ESPACES (pA, 24/10): known spaces of sources, without a page-key format
SPACES = frozenset(NAMESPACES) | frozenset(DECLARED_SPACES) | {"unknown"}
OTHER_SPACES = (frozenset(NAMESPACES) | frozenset(DECLARED_SPACES)) - {"gallica"}   # AV-G0-SUIVI-TITRES-ESPACES: never a Gallica page
# AV-G0-SUIVI-TITRES-NOMS (pA, 25/10): the names of the 6 REF titles and their known variants, pinned here; any overlap of a significant word
# of a title name of the collection with the REF title of the page refuses that page, "Petit" or "Écho" alone included
REF_TITLES = {
    "L'Écho de Paris": ("Écho de Paris", "L'Echo de Paris"),
    "Le Petit Marseillais": ("Petit Marseillais",),
    "La Petite Gironde": ("Petite Gironde",),
    "Le Petit Troyen": ("Petit Troyen",),
    "L'Action pyrénéenne": ("Action pyrénéenne",),
    "Le Progrès de la Côte-d'Or": ("Progrès de la Côte d'Or", "Le Progrès de la Côte d'Or"),
}
# AV-G0-SUIVI-TITRES-NOMS v2 (pA, 25/10): the CLOSED list of the generic words of newspaper titles, pinned before any table, never adjusted
# after seeing one; a word is generic only if it is exactly one of them once normalized (echos, parisien, francaise, prog are distinctive)
GENERIC_TITLE_WORDS = ("petit", "petite", "journal", "écho", "gazette", "courrier", "progrès", "action", "nouvelliste", "républicain", "moniteur",
                       "soir", "matin", "nouvelles", "dépêche", "presse", "paris", "france",
                       "petits", "petites", "journaux", "échos", "gazettes", "courriers", "dépêches")   # pA (25/10): plurals, a tightening
# AV-G0-SUIVI-TITRES-NOMS v2 (pA, 25/10, after Sonnet M1): the CLOSED list of the FORM words of a title, fixed now and never adjusted after the
# tables; with the numbers, the one-letter words and the isolated signs, they are left out when deciding whether a name is made only of generic
# words (« Le Petit & Cie », « Le Petit 1900 », « Le Petit (supplément) » are « Le Petit »), and are never compared; exact words only. pA (25/10):
# the ordinals are numbers, the abbreviations and variants join the list; the lists may be completed after seeing tables ONLY in the sense that
# adds refusals (never an admission, so no bias for the test), and may never be lightened
FORM_TITLE_WORDS = ("illustré", "supplément", "édition", "quotidien", "hebdomadaire", "cie",
                    "suppl", "hebdo", "co", "éditions", "illustrée", "illustrés", "illustrées")
ORDINAL_ENDS = ("e", "er", "ere", "eme")   # 2e, 1er, 1ère, 3eme, 3ème once the accents are dropped
NAME_METHODS = ("file_names", "mastheads_read_twice", "workbook")   # how the complete set of the names is proven
NOT_SIGNIFICANT = frozenset({"le", "la", "les", "l", "un", "une", "des", "du", "d", "de", "a", "au", "aux", "et", "en"})   # articles, de, à, et, en


def _list_of_dicts(v: Any) -> bool:
    return isinstance(v, list) and all(isinstance(x, dict) for x in v)


@dataclass(frozen=True)
class SuiviInputs:
    """A snapshot: the raw bytes of the five files (None = absent), their names for SEAL_NOT_PINNED codes. Never trusted by itself:
    eligibility re-hashes every byte string against the module anchors before parsing it. ``anchors`` only records what the loader used."""
    raw: Mapping[str, bytes | None]
    names: Mapping[str, str]
    anchors: Mapping[str, str | None] = field(default_factory=dict)


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    keys = [k for k, _ in pairs]
    _invalid(len(keys) == len(set(keys)), f"duplicate JSON key in {sorted(k for k in keys if keys.count(k) > 1)}")
    return dict(pairs)


def _parse(name: str, raw: bytes) -> dict[str, Any]:
    try:
        doc = json.loads(raw.decode("utf-8"), object_pairs_hook=_no_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SuiviInputError(f"SUIVI_INPUT_INVALID: {name}: not valid JSON: {exc}") from exc
    _validate(name, doc)
    return doc


def _validate(name: str, doc: Any) -> None:
    _invalid(isinstance(doc, dict) and doc.get("schema") == SCHEMAS[name], f"{name}: schema must be {SCHEMAS[name]}")
    if name == "usage":
        for k in ("version", "created", "consultations_seq_max", "usages", "aliases", "checks", "limits"):
            _invalid(k in doc, f"usage: missing field {k}")
        _invalid(isinstance(doc["version"], int) and not isinstance(doc["version"], bool) and doc["version"] >= 1, "usage: version")
        _invalid(_text(doc["created"]), "usage: created")
        _invalid(isinstance(doc["consultations_seq_max"], int) and not isinstance(doc["consultations_seq_max"], bool)
                 and doc["consultations_seq_max"] >= 0, "usage: consultations_seq_max")
        for k in ("usages", "aliases", "checks"):
            _invalid(_list_of_dicts(doc[k]), f"usage: {k} must be a list of objects")
        for u in doc["usages"]:
            _invalid(u.get("scope") in SCOPES and _ident(u.get("key")) and u.get("usage") in USAGES and u.get("status") in USAGE_STATUS
                     and _ident(u.get("source")) and _text(u.get("proof")), f"usage: bad usage entry {u!r}")
        keys = [(u["scope"], u["key"], u["usage"], u["source"]) for u in doc["usages"]]
        _invalid(len(keys) == len(set(keys)), "usage: duplicate usage entry (same scope, key, usage, source)")
        for a in doc["aliases"]:
            _invalid(isinstance(a.get("proven_page"), bool) and _text(a.get("source"))
                     and all(isinstance(a.get(s), dict) and a[s].get("namespace") in NAMESPACES and _canonical_page(a[s].get("value"))
                             and _namespace_of(a[s]["value"]) == a[s]["namespace"] for s in ("left", "right")),
                     f"usage: bad alias (canonical page keys of their own namespace required) {a!r}")
        for c in doc["checks"]:
            _invalid(_canonical_page(c.get("page")) and _ident(c.get("check")) and c.get("status") in ("complete", "incomplete")
                     and _list_of_dicts(c.get("findings")) and all(f.get("kind") in FINDINGS and _ident(f.get("source")) for f in c["findings"])
                     and ("against" not in c or _idents(c["against"]))    # against: the sources the check compared the page with
                     and ("against_manifests" not in c or (isinstance(c["against_manifests"], dict) and bool(c["against_manifests"])
                          and set(c["against_manifests"]) <= set(c.get("against", [])) and all(_sha64(v) for v in c["against_manifests"].values()))),
                     f"usage: bad check (canonical page key required) {c!r}")
        for c in doc["checks"]:   # AV-G0-SUIVI-CONTENU: a content check names the text of the page it compared, and only it may
            if c["check"] == CONTENT_CHECK:
                pt = c.get("page_text")
                _invalid(isinstance(pt, dict) and set(pt) == {"kind", "sha256"} and pt["kind"] in PAGE_TEXT_KINDS and _sha64(pt["sha256"])
                         and _idents(c.get("against")) and set(c.get("against_manifests", {})) == set(c["against"])
                         and all(f["kind"] == "content_overlap" for f in c["findings"]),
                         f"usage: bad {CONTENT_CHECK} check (page_text {{kind: vt_bbvlm, sha256}}, against with a manifest each, "
                         f"content_overlap findings only) {c!r}")
            else:
                _invalid("page_text" not in c and all(f["kind"] != "content_overlap" for f in c["findings"]),
                         f"usage: page_text and content_overlap belong to the {CONTENT_CHECK} check only {c!r}")
        pairs = [(c["page"], c["check"]) for c in doc["checks"]]
        _invalid(len(pairs) == len(set(pairs)), "usage: duplicate (page, check) entry")
        _invalid(isinstance(doc["limits"], dict) and all(k in NAMESPACES for k in doc["limits"])
                 and all(_idents(v, empty=True) and all(x.startswith("LIMIT:") for x in v) for v in doc["limits"].values()), "usage: limits")
    elif name == "sources":
        for k in ("sources", "required_checks", "nzz_search"):
            _invalid(k in doc, f"sources: missing field {k}")
        _invalid(_list_of_dicts(doc["sources"]), "sources: sources must be a list of objects")
        names = [s.get("source") for s in doc["sources"]]
        _invalid(all(_ident(n) for n in names), "sources: every source needs a name (a string without surrounding space)")
        _invalid(len(names) == len(set(names)), "sources: source names must be unique")
        for s in doc["sources"]:
            files_ok = _list_of_dicts(s.get("files")) and all(
                isinstance(f.get("path"), dict) and _text(f["path"].get("root")) and _text(f["path"].get("path"))
                and isinstance(f.get("sha256"), str) and len(f["sha256"]) == 64 and set(f["sha256"]) <= _HEX64 for f in s["files"])
            _invalid(s.get("status") in SOURCE_STATUS and _text(s.get("proof")) and _idents(s.get("key_spaces"))
                     and all(k in SPACES for k in s["key_spaces"]) and files_ok, f"sources: bad source {s.get('source')!r}")
            if s["status"] == MODEL:   # no page: unknown space only, no file, and one of the contract's pretraining limits
                _invalid(s["key_spaces"] == ["unknown"] and s["files"] == [] and "lm_texts" not in s
                         and s["source"] in MODEL_LIMITS and s.get("limit") == MODEL_LIMITS[s["source"]],
                         f"sources: bad {MODEL} {s.get('source')!r} (a name of MODEL_LIMITS with its limit, key_spaces [\"unknown\"], no files)")
            else:
                _invalid("limit" not in s, f"sources: limit is only for a {MODEL} ({s.get('source')!r})")
            if "lm_texts" in s:   # AV-G0-SUIVI-LM: only an unresolved row of unknown space, with its texts pinned file by file
                t = s["lm_texts"]
                _invalid(s["status"] == "non_resolue" and s["key_spaces"] in TEXT_SPACES and isinstance(t, dict)
                         and isinstance(t.get("all_on_disk"), bool) and isinstance(t.get("documents"), int) and not isinstance(t["documents"], bool)
                         and t["documents"] >= 1 and _list_of_dicts(t.get("files")) and bool(t["files"])
                         and all(isinstance(f.get("path"), dict) and _text(f["path"].get("root")) and _text(f["path"].get("path"))
                                 and _sha64(f.get("sha256")) for f in t["files"]),
                         f"sources: bad lm_texts on {s.get('source')!r} (non_resolue, key_spaces [\"unknown\"] or [\"gallica\"], files with sha256, documents, all_on_disk)")
        if "title_disjunction" in doc:   # AV-G0-SUIVI-TITRES: validated whenever present
            _validate_title_disjunction(doc["title_disjunction"])
        if "content_texts" in doc:   # AV-G0-SUIVI-CONTENU: per collection, ALL its texts pinned file by file, its unit count and the proof of it
            ct = doc["content_texts"]
            _invalid(isinstance(ct, dict) and bool(ct) and all(
                _ident(n) and isinstance(c, dict) and set(c) == {"files", "units", "proof"} and _proof(c["proof"])
                and isinstance(c["units"], int) and not isinstance(c["units"], bool) and c["units"] >= 1
                and _list_of_dicts(c["files"]) and bool(c["files"])
                and all(set(f) == {"path", "sha256"} and isinstance(f["path"], dict) and set(f["path"]) == {"root", "path"}
                        and _text(f["path"]["root"]) and _text(f["path"]["path"]) and _sha64(f["sha256"]) for f in c["files"])
                and len({(f["path"]["root"], f["path"]["path"]) for f in c["files"]}) == len(c["files"]) for n, c in ct.items()),
                "sources: content_texts must be {<collection>: {files: [{path: {root, path}, sha256}] non-empty and unique, units >= 1, proof}}")
        _invalid(isinstance(doc["required_checks"], dict) and all(k in NAMESPACES for k in doc["required_checks"])
                 and all(_idents(v) for v in doc["required_checks"].values()),
                 "sources: required_checks must map a known space to a non-empty list of check names")
        n = doc["nzz_search"]
        _invalid(isinstance(n, dict) and isinstance(n.get("done"), bool) and isinstance(n.get("hits"), int) and not isinstance(n["hits"], bool)
                 and n["hits"] >= 0 and _texts(n.get("files"), empty=True), "sources: nzz_search")
    elif name == "journals":
        _invalid(_list_of_dicts(doc.get("journals")), "journals: journals must be a list of objects")
        # amendment AV-G0-SUIVI-JOURNAUX (24/10): benches without a source line are declared, ["__none__"] needs the archived search
        declared = doc.get("benches_without_source", [])
        _invalid(_list_of_dicts(declared) and all(_ident(b.get("bench")) and _text(b.get("proof")) and b["bench"] != NONE
                                                  and b["bench"] not in SPACES for b in declared)
                 and len({b["bench"] for b in declared}) == len(declared), "journals: bad benches_without_source")
        archive = doc.get("search_archive")
        if "search_archive" in doc or any(j.get("concerns") == [NONE] for j in doc["journals"]):   # validated whenever present (null too)
            _invalid(isinstance(archive, dict) and isinstance(archive.get("path"), dict) and _text(archive["path"].get("root"))
                     and _text(archive["path"].get("path")) and _sha64(archive.get("sha256")),
                     "journals: concerns [\"__none__\"] needs search_archive {path: {root, path}, sha256}")
        for j in doc["journals"]:
            _invalid(_ident(j.get("journal")) and j.get("status") in ("reviewed", "pending")
                     and ("concerns" not in j or _idents(j["concerns"])),   # absent = everything; an empty list is refused
                     f"journals: bad entry {j!r}")
            c, proof = j.get("concerns"), j.get("none_proof")
            if "none_proof" in j:
                _invalid(isinstance(proof, dict), f"journals: {j['journal']}: none_proof must be an object")
            if c is not None and NONE in c:
                _invalid(c == [NONE] and isinstance(proof, dict) and proof.get("inventory_sha256") == archive["sha256"]
                         and _sha64(proof.get("file_sha256")) and proof.get("collections") == [] and proof.get("benches") == [],
                         f"journals: {j['journal']}: [\"__none__\"] alone, with none_proof (inventory_sha256 = search_archive, "
                         "file_sha256, empty collections and benches)")
            else:
                _invalid("none_proof" not in j, f"journals: {j['journal']}: none_proof without concerns [\"__none__\"]")
    elif name == "consultations":  # an empty journal is a DECLARATION, never an absence (pA, 22/10)
        for k in ("consultations", "auteur", "date", "periode_couverte"):
            _invalid(k in doc, f"consultations: missing field {k}")
        _invalid(_list_of_dicts(doc["consultations"]) and all(_text(doc[k]) for k in ("auteur", "date", "periode_couverte")),
                 "consultations: a list of objects, and non-empty auteur, date and periode_couverte")
        lines = doc["consultations"]
        _invalid(bool(lines) or _text(doc.get("declaration")), "consultations: an empty journal needs a declaration")
        for line in lines:   # every field of the frozen format (pT, B2 b)
            _invalid(isinstance(line.get("seq"), int) and not isinstance(line["seq"], bool) and _text(line.get("date"))
                     and _text(line.get("agent")) and isinstance(line.get("pages"), list) and line["pages"]
                     and all(_canonical_page(p) for p in line["pages"]) and line.get("purpose") in ("monitor", "choose"),
                     f"consultations: bad entry (canonical page keys required) {line!r}")
        seqs = [line["seq"] for line in lines]
        _invalid(all(a < b for a, b in zip(seqs, seqs[1:])), "consultations: seq must increase strictly")
    else:
        for k in ("evidence", "documented_without_proof"):
            _invalid(_list_of_dicts(doc.get(k)), f"select_evidence: {k} must be a list of objects")
        for e in doc["evidence"]:
            target = e.get("target")
            _invalid(_ident(target) and (_canonical_page(target) or _namespace_of(target) is None) and _text(e.get("file"))
                     and isinstance(e.get("line"), int) and not isinstance(e["line"], bool) and e["line"] >= 1,
                     f"select_evidence: bad evidence (a canonical page key or a source name) {e!r}")
        for e in doc["documented_without_proof"]:
            _invalid(_ident(e.get("bench")) and _ident(e.get("journal")), f"select_evidence: bad entry {e!r}")


def _cb(v: Any) -> bool:
    return isinstance(v, str) and CB.fullmatch(v) is not None


def _proof(v: Any) -> bool:
    return isinstance(v, dict) and isinstance(v.get("path"), dict) and _text(v["path"].get("root")) and _text(v["path"].get("path")) \
        and _sha64(v.get("sha256"))


def _gallica_document(v: Any) -> bool:
    """The exact document key that page_identity gives to the pages of a Gallica document (gallica:<id>, no folio)."""
    if not _ident(v) or _namespace_of(v) != "gallica" or "/" in v:
        return False
    try:
        return page_identity(f"{v}/f1", "gallica").document == v
    except IdentityError:
        return False


LATIN_NAME = re.compile(r"[A-Za-z0-9œŒæÆ '’‘\-.,:;!?()&/«»\"]+")   # after NFKD and the accents dropped: nothing invisible, no lookalike script


def _plain(v: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKD", v) if not unicodedata.combining(ch))


def _latin_name(v: Any) -> bool:
    """A title name written in Latin letters, digits and usual punctuation only (Sonnet B1, M1): an invisible character (soft hyphen, zero
    width) would cut a word in two and a Cyrillic or Greek lookalike would hide it, so such a name is INVALID, never compared."""
    return _text(v) and LATIN_NAME.fullmatch(_plain(v)) is not None


def _folded(v: str) -> str:
    return _plain(v).casefold().replace("œ", "oe").replace("æ", "ae")


def _squeezed(v: str) -> str:
    """The name with every non-letter non-digit taken out: « Pe-tit », « Pe'tit », « Pe tit » give back petit (Sonnet v2, B1)."""
    return re.sub(r"[\W_]+", "", _folded(v))


def _name_words(v: str) -> frozenset[str]:
    """The significant words of a title name: accents dropped, œ and æ spelt out, case folded, cut on every non-letter non-digit (a letter is
    never dropped), articles and de, à, et, en out."""
    plain = _folded(v)
    return frozenset(w for w in re.split(r"[\W_]+", plain) if w and w not in NOT_SIGNIFICANT)


def _words_overlap(a: str, b: str) -> bool:
    """Two words meet if equal, if one of at least 4 letters begins the other (petit, petite), if one of at least 5 letters is inside the
    other (lepetittroyen, Sonnet M2), or if they share their first 5 letters (pyrénées, pyrénéenne): by prudence, a near form is the same word."""
    short, long_ = sorted((a, b), key=len)
    return a == b or (len(short) >= 4 and long_.startswith(short)) or (len(short) >= 5 and (short in long_ or a[:5] == b[:5]))


REF_WORDS = {k: frozenset().union(*(_name_words(n) for n in (k, *vs))) for k, vs in REF_TITLES.items()}   # the words of each REF title
GENERIC = frozenset().union(*(_name_words(w) for w in GENERIC_TITLE_WORDS))
FORM = frozenset().union(*(_name_words(w) for w in FORM_TITLE_WORDS))
REF_FORMS = {k: tuple(dict.fromkeys(_name_words(n) for n in (k, *vs))) for k, vs in REF_TITLES.items()}   # the word set of each pinned spelling


SPLITS_MAX = 4096


def _generic_splits(q: str, singles: frozenset[int] = frozenset()) -> set[frozenset[str]] | None:
    """Every way to cut the separator-free name q exactly into generic words and words that are not significant, as the sets of its
    generic words: « Pe-tit » (petit) and « LePetit » (le + petit) are names made only of generic words once their cuts are glued back.
    None when more than SPLITS_MAX sets reach one place: the caller then refuses (Sonnet M2: a bound never fails towards admission). Form words,
    any run of digits (with an ordinal ending) and, at the places `singles` (where the name had a one-letter word), that letter are read and
    left out."""
    vocab = GENERIC | NOT_SIGNIFICANT | FORM
    ends: list[set[frozenset[str]]] = [set() for _ in range(len(q) + 1)]
    ends[0].add(frozenset())
    for i in range(len(q)):
        if not ends[i]:
            continue
        steps = [(w, len(w)) for w in vocab if q.startswith(w, i)]
        j = i
        while j < len(q) and q[j].isdigit():
            j += 1
            steps.append(("", j - i))   # a number
            steps += [("", j - i + len(e)) for e in ORDINAL_ENDS if q.startswith(e, j)]   # an ordinal is a number
        if i in singles:
            steps.append(("", 1))   # a one-letter word of the name
        for f in ends[i]:
            for w, k in steps:
                ends[i + k].add(f | ({w} & GENERIC))
                if len(ends[i + k]) > SPLITS_MAX:
                    return None
    return {f for f in ends[len(q)] if f}


def _names_meet(n: str, ref: str) -> bool:
    """AV-G0-SUIVI-TITRES-NOMS v2 (pA, 25/10): does the title name n of a collection meet the REF title ref? A distinctive (not generic) word of
    the REF title met with the near forms and cuts of NOMS; a REF title made only of generic words (L'Écho de Paris) compared in full, all its
    words inside the separator-free name; a name made only of generic words, cuts glued back, compared in full: each of its words meets a word of
    the REF title, the ambiguity refuses (« Le Petit », « L'Écho »). « Paris-Soir » does not meet L'Écho de Paris, « Le Petit Parisien » does not
    meet Le Petit Marseillais."""
    words, q = _name_words(n), _squeezed(n)
    tokens = [w for w in re.split(r"[\W_]+", _folded(n)) if w]
    starts = [sum(len(x) for x in tokens[:k]) for k in range(len(tokens))]
    splits = _generic_splits(q, frozenset(i for i, w in zip(starts, tokens) if len(w) == 1))
    if splits is None:
        return True   # too many ways to read the name: the ambiguity refuses
    alone = sorted(splits, key=sorted)   # every reading made only of generic words: « Le Petit & Cie », « Le Pe-tit 1900 » are « Le Petit »
    for form in REF_FORMS[ref]:
        distinct = form - GENERIC
        if distinct and (any(_words_overlap(w, d) for w in words for d in distinct) or any(p in q for p in _ref_pieces(distinct))):
            return True
        if not distinct and all(r in q for r in form):
            return True
        if any(all(any(_words_overlap(w, r) for r in form) for w in f) for f in alone):
            return True
    return False


def _good_name(v: Any) -> bool:
    return _latin_name(v) and bool(_name_words(v))


def _entry_names(x: Mapping[str, Any]) -> list[str]:
    """The names an entry of the names level holds: its name, every name of a group (AV-G0-SUIVI-TITRES-GROUPES), none for a space."""
    return [x["name"]] if "name" in x else list(x.get("names", []))


def _validate_title_disjunction(t: Any) -> None:
    """AV-G0-SUIVI-TITRES: {collections: {<collection>: {uncovered: [{unit, title_cb}], proof} or {titles: [{title_cb, units, name?}], proof}},
    documents: [{document, title_cb, proof}]}. Unit level lists the uncovered units; collection level (pA, 25/10) gives the COMPLETE set of the
    titles of the collection with the number of units of each. A null title_cb (record unknown) keeps the refusal of its collection.
    AV-G0-SUIVI-TITRES-ESPACES (pA, 25/10): units of a space other than gallica count with their space instead of a cb ({space, units} in the
    titles, {unit, space} among the uncovered, the space being then the namespace of the unit key) and never block a Gallica page."""
    _invalid(isinstance(t, dict) and set(t) == {"collections", "documents"} and isinstance(t["collections"], dict)
             and _list_of_dicts(t["documents"]), "sources: title_disjunction must be {collections: {...}, documents: [...]}")
    for name, c in t["collections"].items():
        if isinstance(c, dict) and "names" in c:   # AV-G0-SUIVI-TITRES-NOMS: names level, the complete set of the title NAMES
            _invalid(_ident(name) and set(c) == {"names", "method", "proof"} and c["method"] in NAME_METHODS and _proof(c["proof"])
                     and _list_of_dicts(c["names"]) and bool(c["names"])
                     and all((set(x) == {"name", "units"} and _good_name(x["name"])
                              or set(x) == {"names", "units"} and isinstance(x["names"], list) and len(x["names"]) >= 2
                              and all(_good_name(n) for n in x["names"]) and len({_name_words(n) for n in x["names"]}) == len(x["names"])
                              or set(x) == {"space", "units"} and isinstance(x["space"], str) and x["space"] in OTHER_SPACES)
                             and isinstance(x["units"], int) and not isinstance(x["units"], bool) and x["units"] >= 1 for x in c["names"]),
                     f"sources: title_disjunction: bad collection {name!r} (names [{{name: Latin letters, with a significant word, units >= 1}}, "
                     f"{{names: 2 or more distinct such names, units >= 1}} or {{space: not gallica, units >= 1}}] non-empty, method in "
                     f"{NAME_METHODS}, proof)")
            keys = [_name_words(x["name"]) for x in c["names"] if "name" in x] + [x["space"] for x in c["names"] if "space" in x]
            keys += [frozenset(_name_words(n) for n in x["names"]) for x in c["names"] if "names" in x]   # two identical groups (Sonnet m1)
            _invalid(len(keys) == len(set(keys)), f"sources: title_disjunction: duplicate name, group or space in {name!r}")   # a group may repeat a name
            continue
        if isinstance(c, dict) and "titles" in c:   # collection level
            _invalid(_ident(name) and set(c) == {"titles", "proof"} and _proof(c["proof"]) and _list_of_dicts(c["titles"]) and bool(c["titles"])
                     and all((set(x) - {"name"} == {"title_cb", "units"} and (x["title_cb"] is None or _cb(x["title_cb"]))
                              or set(x) - {"name"} == {"space", "units"} and isinstance(x["space"], str) and x["space"] in OTHER_SPACES)
                             and isinstance(x["units"], int) and not isinstance(x["units"], bool) and x["units"] >= 1
                             and ("name" not in x or _text(x["name"])) for x in c["titles"]),
                     f"sources: title_disjunction: bad collection {name!r} "
                     f"(titles [{{title_cb: cb or null, units >= 1, name?}} or {{space: not gallica, units >= 1, name?}}] non-empty, proof)")
            cbs = [x["title_cb"] for x in c["titles"] if x.get("title_cb") is not None]
            _invalid(len(cbs) == len(set(cbs)), f"sources: title_disjunction: duplicate title_cb in {name!r}")
            sps = [x["space"] for x in c["titles"] if "space" in x]
            _invalid(len(sps) == len(set(sps)), f"sources: title_disjunction: duplicate space in {name!r}")
            continue
        _invalid(_ident(name) and isinstance(c, dict) and set(c) == {"uncovered", "proof"} and _proof(c["proof"])
                 and _list_of_dicts(c["uncovered"]) and bool(c["uncovered"])
                 and all(set(u) == {"unit", "title_cb"} and _ident(u["unit"]) and (u["title_cb"] is None or _cb(u["title_cb"]))
                         or set(u) == {"unit", "space"} and _canonical_page(u["unit"]) and isinstance(u["space"], str) and u["space"] in OTHER_SPACES
                         and _namespace_of(u["unit"]) == u["space"]   # a real canonical page key, not a prefix only (pQ, B1)
                         for u in c["uncovered"]),
                 f"sources: title_disjunction: bad collection {name!r} "
                 f"(uncovered [{{unit, title_cb: cb or null}} or {{unit: canonical page key, space: its namespace, not gallica}}] non-empty, proof)")
        units = [_fold(u["unit"]) for u in c["uncovered"]]   # "unit" and "UNIT" are one unit: a fake one cannot fill the count (Sonnet, M2)
        _invalid(len(units) == len(set(units)), f"sources: title_disjunction: duplicate unit in {name!r}")
    for d in t["documents"]:   # a document gives its title record, its REF title (AV-G0-SUIVI-TITRES-NOMS), or both
        _invalid({"document", "proof"} <= set(d) <= {"document", "title_cb", "ref_title", "proof"} and ({"title_cb", "ref_title"} & set(d))
                 and _gallica_document(d["document"]) and _proof(d["proof"]) and ("title_cb" not in d or _cb(d["title_cb"]))
                 and ("ref_title" not in d or isinstance(d["ref_title"], str) and d["ref_title"] in REF_TITLES),
                 f"sources: title_disjunction: bad document {d!r} (a Gallica document key, its title_cb and/or its ref_title, proof)")
    docs = [d["document"] for d in t["documents"]]
    _invalid(len(docs) == len(set(docs)), "sources: title_disjunction: duplicate document")


def _count_ok(entry: Mapping[str, Any], count: int, whole: bool) -> bool:
    """The count of a pinned set against the registry: total (whole collection) or total - keyed (uncovered units) of a VALID coverage,
    0 <= keyed <= total, absent or null refuses (pQ, M1). Without a number key (NO_KEY) the count is only declared."""
    spec = entry["match_keys"]["issue"]
    if spec == NO_KEY:
        return True
    cov = spec.get("coverage") if isinstance(spec, dict) else None
    total, keyed = (cov.get("total"), cov.get("keyed")) if isinstance(cov, dict) else (None, None)
    if not all(isinstance(x, int) and not isinstance(x, bool) for x in (total, keyed)) or not 0 <= keyed <= total:
        return False
    return count == (total if whole else total - keyed)


def _title_disjoint(entry: Mapping[str, Any], t: Mapping[str, Any] | None, name: str, members: list[str]) -> bool:
    """AV-G0-SUIVI-TITRES (pA, 25/10): may the collection `name`, whose number keys are incomplete, leave the page alone? Only if the title
    records it pins are all known (none null) and complete by count, and the title record of the document of EVERY Gallica member of the
    page's class is pinned and is none of them. Count: unit level, the uncovered units are total - keyed of the registry's coverage; collection
    level, the units of the titles add up to its total. Without a coverage in the registry (no number key), the count is only declared."""
    if not t or name not in t["collections"]:
        return False
    c = t["collections"][name]
    if "names" in c:
        return False   # a names level proves no title record: only _name_disjoint may read it
    whole = "titles" in c
    titles = {x["title_cb"] for x in (c["titles"] if whole else c["uncovered"]) if "space" not in x}   # another space blocks no Gallica page
    if None in titles:
        return False
    if not _count_ok(entry, sum(x["units"] for x in c["titles"]) if whole else len(c["uncovered"]), whole):
        return False   # the pinned set is not the registry's (stale or partial): the refusal stays
    of_doc = {d["document"]: d.get("title_cb") for d in t["documents"]}
    gallica = [m for m in members if _namespace_of(m) == "gallica"]
    units = set() if whole else {_fold(u["unit"]) for u in c["uncovered"]}
    spaces = {x["space"] for x in c["titles"] if "space" in x} if whole else set()
    for m in members:   # the WHOLE class (pQ, B2): another space is not another physical page when a page alias is proven
        ns = _namespace_of(m)
        doc = page_identity(m, ns).document   # None for a kb, bnl, ... page
        if ns in spaces or {_fold(k) for k in (m, doc) if k is not None} & units:
            return False   # a member listed among the uncovered units, or of a space the titles count without a key
    for m in gallica:
        cb = of_doc.get(page_identity(m, "gallica").document)
        if cb is None or cb in titles:
            return False
    return bool(gallica)


def _ref_pieces(ref_words: set[str]) -> set[str]:
    """What every near form of a REF word of at least 4 letters holds (pQ B1, Sonnet v3): its first 4 letters (a prefix of 4 letters or more,
    the same first 5 letters, the word inside another) or one of its 5-letter windows (a piece of 5 letters or more inside the word). The name
    taken without any separator is searched for them, so near forms and cuts compose: Pe-tit, Py-rénées, Pro-g., LePetit, Seil-lais, Ron-de."""
    return {r[:4] for r in ref_words if len(r) >= 4} | {r[i:i + 5] for r in ref_words for i in range(len(r) - 4)}


def _name_disjoint(entry: Mapping[str, Any], t: Mapping[str, Any] | None, name: str, members: list[str]) -> bool:
    """AV-G0-SUIVI-TITRES-NOMS (pA, 25/10), weaker than _title_disjoint: the complete set of the title NAMES of the collection is pinned and
    complete by count; EVERY Gallica member of the page's class is a page of a REF title (its document pinned with ref_title) and none of the
    significant words of the names meets a word of THAT REF title; no member is of a space the names count. The refusal is per page: a
    collection holding "Paris-Soir" refuses the pages of L'Écho de Paris, not those of Le Petit Marseillais. No cb is deduced from a name."""
    c = (t or {}).get("collections", {}).get(name)
    if not isinstance(c, dict) or "names" not in c:
        return False
    if not _count_ok(entry, sum(x["units"] for x in c["names"]), True):
        return False
    every = [n for x in c["names"] for n in _entry_names(x)]   # a group is compared name by name: one overlap refuses (pA, 25/10)
    spaces = {x["space"] for x in c["names"] if "space" in x}
    if any(_namespace_of(m) in spaces for m in members):
        return False
    ref_of = {d["document"]: d.get("ref_title") for d in t["documents"]}
    gallica = [m for m in members if _namespace_of(m) == "gallica"]
    refs = [ref_of.get(page_identity(m, "gallica").document) for m in gallica]
    return bool(gallica) and None not in refs and not any(_names_meet(n, ref) for ref in set(refs) for n in every)


def _content_disjoint(entry: Mapping[str, Any], src: Mapping[str, Any] | None, name: str, members: list[str],
                      checks: Mapping[tuple[str, str], Mapping[str, Any]]) -> bool:
    """AV-G0-SUIVI-CONTENU (pA, 25/10): may the collection `name` leave the page alone by its CONTENT? Only if ALL its texts are pinned
    (content_texts, their unit count equal to the registry total when it has one, declared otherwise) and EVERY Gallica member of the page's
    class carries a complete ngram8_content check run on the clean VT of BBVLM against exactly these texts (same fingerprint), without one
    shared 8-gram (a single content_overlap refuses)."""
    c = ((src or {}).get("content_texts") or {}).get(name)
    if c is None or not _count_ok(entry, c["units"], True):
        return False
    printed = lm_fingerprint(c["files"])
    gallica = [m for m in members if _namespace_of(m) == "gallica"]
    for m in gallica:
        k = checks.get((m, CONTENT_CHECK))
        if (k is None or k["status"] != "complete" or name not in k["against"] or k["against_manifests"].get(name) != printed
                or any(f["source"] == name for f in k["findings"])):
            return False
    return bool(gallica)


def _check_schema() -> None:
    try:
        ok = hashlib.sha256(SCHEMA_PATH.read_bytes()).hexdigest() == SCHEMA_SHA256
    except OSError:
        ok = False
    _invalid(ok, f"the input schema {SCHEMA_PATH.name} is not the pinned one")


def load_inputs(paths: Mapping[str, Path | str] | None = None, anchors: Mapping[str, str | None] | None = None) -> SuiviInputs:
    """Read the raw bytes of the five files. A file whose SHA-256 is not its anchor is kept as None; a pinned file is validated now, so that a
    partial file fails at load time. ``anchors`` defaults to the module anchors; eligibility always re-checks against the module anchors."""
    _check_schema()
    paths = {k: REPO_ROOT / v for k, v in SUIVI_PATHS.items()} | dict(paths or {})
    anchors = dict(SUIVI_ANCHORS if anchors is None else anchors)
    raw: dict[str, bytes | None] = {}
    for name in PINNED_FILES:
        try:
            b = Path(paths[name]).read_bytes()
        except OSError:
            b = None
        raw[name] = b if b is not None and anchors.get(name) and hashlib.sha256(b).hexdigest() == anchors[name] else None
    _derived({name: _parse(name, raw[name]) if raw[name] is not None else None for name in PINNED_FILES})  # fail now on a partial file
    return SuiviInputs(MappingProxyType(raw), MappingProxyType({k: Path(paths[k]).name for k in PINNED_FILES}), MappingProxyType(anchors))


def _documents(inputs: SuiviInputs) -> dict[str, dict[str, Any] | None]:
    """Parse ONLY byte strings whose SHA-256 equals the MODULE anchor (provenance), whatever the snapshot says (pT, B1)."""
    docs: dict[str, dict[str, Any] | None] = {}
    for name in PINNED_FILES:
        b = inputs.raw.get(name) if isinstance(inputs.raw, Mapping) else None
        anchor = SUIVI_ANCHORS.get(name)
        ok = isinstance(b, (bytes, bytearray)) and bool(anchor) and hashlib.sha256(bytes(b)).hexdigest() == anchor
        docs[name] = _parse(name, bytes(b)) if ok else None
    return docs


def _derived(docs: Mapping[str, Any]) -> tuple[list[Usage], list[Alias], int | None]:
    usages: list[Usage] = []
    aliases: list[Alias] = []
    if docs["usage"] is not None:
        try:
            usages = [Usage(u["scope"], u["key"], u["usage"], u["source"]) for u in docs["usage"]["usages"]]
            aliases = [Alias(page_identity(a["left"]["value"], a["left"]["namespace"]),
                             page_identity(a["right"]["value"], a["right"]["namespace"]), a["proven_page"], a["source"])
                       for a in docs["usage"]["aliases"]]
        except (ValueError, KeyError, IdentityError) as exc:
            raise SuiviInputError(f"SUIVI_INPUT_INVALID: usage: {exc}") from exc
    # closed by default: every name of concerns must be known. Skipped without a pinned sources file: eligibility then refuses every page
    # (SEAL_NOT_PINNED), and load_inputs only loses this typo check
    if docs["usage"] is not None and docs["sources"] is not None:   # AV-G0-SUIVI-LM: names in checks must be sources of the table
        names = {s["source"] for s in docs["sources"]["sources"]}
        used = {u["source"] for u in docs["usage"]["usages"]}
        model_names = {s["source"] for s in docs["sources"]["sources"] if s["status"] == MODEL}
        named = set(used) | {n for c in docs["usage"]["checks"] if c["check"] != CONTENT_CHECK for n in c.get("against", [])} \
            | {f["source"] for c in docs["usage"]["checks"] if c["check"] != CONTENT_CHECK for f in c["findings"]}
        if docs["journals"] is not None:
            named |= {n for j in docs["journals"]["journals"] for n in j.get("concerns", [])}
        if docs["select_evidence"] is not None:
            named |= {e["bench"] for e in docs["select_evidence"]["documented_without_proof"]} | {e["target"] for e in docs["select_evidence"]["evidence"]}
        models = sorted(model_names & named)
        if models:   # a source that has pages cannot be a pretrained model (pA, 24/10); nor can one a journal, check or evidence relies on
            raise SuiviInputError(f"SUIVI_INPUT_INVALID: sources: {MODEL} rows named by usages, checks, journals or evidence: {models}")
        idle = sorted(s["source"] for s in docs["sources"]["sources"] if s["status"] in JOINING and s["source"] not in used)
        if idle:   # pA, 24/10: a joining row without usage refuses nothing yet covers its spaces (reader_lm_corpus)
            raise SuiviInputError(f"SUIVI_INPUT_INVALID: sources: joining rows without any usage line: {idle}")
        lm = {s["source"] for s in docs["sources"]["sources"] if "lm_texts" in s}
        content = set(docs["sources"].get("content_texts", {}))
        for c in docs["usage"]["checks"]:   # AV-G0-SUIVI-CONTENU: a content check is against collections with pinned texts, not sources
            if c["check"] == CONTENT_CHECK and (set(c["against"]) - content or {f["source"] for f in c["findings"]} - set(c["against"])):
                raise SuiviInputError(f"SUIVI_INPUT_INVALID: usage: {CONTENT_CHECK} of {c['page']} names collections without content_texts")
        for c in docs["usage"]["checks"]:
            if c["check"] == CONTENT_CHECK:
                continue
            if set(c.get("against_manifests", {})) - lm:
                raise SuiviInputError(f"SUIVI_INPUT_INVALID: usage: check {c['check']} of {c['page']} has manifests for sources without lm_texts")
        for c in docs["usage"]["checks"]:
            if c["check"] == CONTENT_CHECK:
                continue
            if set(c.get("against", [])) - names:
                raise SuiviInputError(f"SUIVI_INPUT_INVALID: usage: check {c['check']} of {c['page']} is against unknown sources")
            if any(f["kind"] == "lm_overlap" and f["source"] not in lm for f in c["findings"]):
                raise SuiviInputError(f"SUIVI_INPUT_INVALID: usage: an lm_overlap of {c['page']} names a source without lm_texts")
    if docs["journals"] is not None and docs["sources"] is not None:
        sources = {s["source"] for s in docs["sources"]["sources"]}
        declared = {b["bench"] for b in docs["journals"].get("benches_without_source", [])}
        if NONE in sources:
            raise SuiviInputError("SUIVI_INPUT_INVALID: sources: __none__ is reserved")
        if declared & sources:
            raise SuiviInputError(f"SUIVI_INPUT_INVALID: journals: benches_without_source that have a source line: {sorted(declared & sources)}")
        fold = {_fold(n) for n in set(SPACES) | sources | {NONE}}
        if len({_fold(b) for b in declared}) != len(declared):
            raise SuiviInputError("SUIVI_INPUT_INVALID: journals: two benches of benches_without_source fold to the same name")
        if any(_fold(b) in fold for b in declared):           # "Gallica", "E22", "e22\u200b" must not shadow a space or a source
            raise SuiviInputError(f"SUIVI_INPUT_INVALID: journals: a bench of benches_without_source collides with a space or a source")
        known = set(NAMESPACES) | set(DECLARED_SPACES) | sources | declared | {NONE}   # "unknown" is not a space one can concern (pQ, B1)
        for j in docs["journals"]["journals"]:
            unknown = sorted(set(j.get("concerns", [])) - known)
            if unknown:
                raise SuiviInputError(f"SUIVI_INPUT_INVALID: journals: {j['journal']}: unknown names in concerns {unknown}")
    seq = None
    if docs["consultations"] is not None:
        lines = docs["consultations"]["consultations"]
        seq = lines[-1]["seq"] if lines else 0
    return usages, aliases, seq


def _document_key(page_key: str) -> str | None:
    ns = _namespace_of(page_key)
    try:
        return page_identity(page_key, ns).document if ns else None
    except IdentityError:
        return None


def eligibility(registry: Mapping[str, Any], role: str, *, page_key: str | None = None, issue_key: str | None = None,
                title_key: str | None = None, inputs: SuiviInputs | None = None) -> dict[str, Any]:
    """May this page enter the `suivi` corpus? Every refusal is coded (``refusal_codes``, parallel to ``refusals``), in the final order of v4:
    REGISTRY_ANCHOR_CHANGED (raised) ; SEAL_NOT_PINNED ; SUIVI_REGISTRY_STALE ; SUIVI_NO_PAGE_KEY ; SUIVI_SPACE_UNCOVERED then
    INCOMPLETE_NUMBER_KEY ; SUIVI_SOURCE_UNRESOLVED ; SUIVI_CHECK_INCOMPLETE ; SUIVI_RESERVED ; SUIVI_TRAINED ; SUIVI_DEV ; SUIVI_SELECTED ;
    SUIVI_SELECTION_UNREVIEWED ; SUIVI_TEXT_OVERLAP, SUIVI_LM_OVERLAP and SUIVI_CONTENT_OVERLAP (in the order of the checks). Labels (no
    refusal) are sorted.
    INCOMPLETE_NUMBER_KEY:<c> becomes the label TITLE_DISJOINT:<c> when the pinned title records allow it (AV-G0-SUIVI-TITRES). ``page_key`` must already be canonical (page_identity).
    ``inputs`` is a snapshot of raw bytes (load_inputs); only bytes matching the module anchors are read.
    """
    if role != "suivi":
        raise ValueError(f"g0_suivi.eligibility serves role suivi only, not {role!r}")
    _check_schema()
    inputs = load_inputs() if inputs is None else inputs
    docs = _documents(inputs)
    all_usages, aliases, consultations_seq = _derived(docs)
    names = {k: (inputs.names.get(k) if isinstance(inputs.names, Mapping) else None) or Path(SUIVI_PATHS[k]).name for k in PINNED_FILES}
    codes: list[str] = []
    refusals: list[str] = []
    labels: set[str] = set(PRETRAINING_LIMITS)   # rendered in every output, even before the page key is known (AV-G0-SUIVI-LIMITES)

    def refuse(code: str, text: str) -> None:
        if code not in codes:
            codes.append(code)
            refusals.append(text)

    usage, src = docs["usage"], docs["sources"]
    supplied = {k: v for k, v in (("page", page_key), ("issue", issue_key), ("title", title_key)) if v is not None}
    result = lambda: {"role": role, "allowed": not codes, "refusals": refusals, "refusal_codes": codes, "labels": sorted(labels),  # noqa: E731
                      "supplied": supplied, "usage_registry_version": (usage or {}).get("version")}
    # 2 pins: every one of the five documents must be present and pinned by the MODULE anchors (1, REGISTRY_ANCHOR_CHANGED, raised in _pin_codes)
    for name in PINNED_FILES:
        if docs[name] is None:
            refuse(f"SEAL_NOT_PINNED:{names[name]}", f"{names[name]} is absent or not pinned: its content is not read")
    sealed = registry["collections"].get(SEALED)
    if sealed is None:
        refuse(f"SEAL_NOT_PINNED:{SEALED}", "the sealed test is missing from the registry: its reservation cannot be checked")
    else:
        for code in _pin_codes(registry, SEALED, sealed, "page", "test"):
            refuse(code, f"{SEALED}: the in-memory page list is not the pinned file of the anchored registry")
    # 3 stale
    if usage is not None and consultations_seq is not None and consultations_seq > usage["consultations_seq_max"]:
        refuse("SUIVI_REGISTRY_STALE", f"consultation seq {consultations_seq} is after the anchored registry "
                                       f"({usage['consultations_seq_max']}): a new registry version is due")
    # 4 page key
    ns = _namespace_of(page_key) if isinstance(page_key, str) else None
    ident: PageIdentity | None = None
    if ns is not None:
        try:
            ident = page_identity(page_key, ns)
        except IdentityError:
            ident = None
    if ident is None or ident.page != page_key:
        refuse("SUIVI_NO_PAGE_KEY", f"no canonical page key: {page_key!r}")
        return result()
    if issue_key is not None and issue_key != ident.document:
        raise ValueError(f"issue_key {issue_key!r} is not the document of {page_key!r} ({ident.document!r})")
    # the class of the same physical page: proven page aliases only (B3); document links never carry page constraints
    same_page = _closure((a.left.page, a.right.page) for a in aliases if a.proven_page).get(ident.page, {ident.page})
    members = [ident.page] + sorted(same_page - {ident.page})   # the queried key first, everywhere a code lists members
    spaces = sorted({_namespace_of(m) for m in members}, key=lambda s: (s != ns, s))   # the queried space first, then the others
    applied, issue_labels = applicable_usages(ident, all_usages, aliases)
    # 5 space coverage of every space of the class, then the number keys of Gallica (registry order)
    incomplete = [n for n, e in registry["collections"].items() if _scope(e) == "possible" and not _issue_keys_complete(e)]
    table = src["sources"] if src is not None else []
    for sp in spaces:
        if src is None:
            refuse(f"SUIVI_SPACE_UNCOVERED:{sp}", "the sources table is not pinned: no space can be shown covered")
        elif sp in TABLE_SPACES:
            if not any(sp in s["key_spaces"] and s["status"] in JOINING for s in table):
                refuse(f"SUIVI_SPACE_UNCOVERED:{sp}", f"no joining row of the sources table covers the space {sp}")
        elif sp == "as":
            if incomplete or not any("as" in s["key_spaces"] and s["status"] in JOINING for s in table):
                refuse("SUIVI_SPACE_UNCOVERED:as", "the Gallica side is incomplete (G0-03 B1) or no row covers as")
        elif sp == "nzz":
            declared = {x for s in table for x in s["key_spaces"]}
            search = src["nzz_search"]
            required = set(src["required_checks"].get("nzz", []))
            if ("nzz" in declared or "unknown" in declared or not search["done"] or search["hits"] or not search["files"]
                    or not set(NZZ_CHECKS) <= required):
                refuse("SUIVI_SPACE_UNCOVERED:nzz", "nzz is not shown covered (a source declares nzz or unknown, the search is not done, "
                                                     "hit or empty, or the controls of point 6 are not all required)")
        else:
            refuse(f"SUIVI_SPACE_UNCOVERED:{sp}", f"the contract shows no coverage for the space {sp}")
    if "gallica" in spaces:
        disjunction = (src or {}).get("title_disjunction")
        page_checks = {(c["page"], c["check"]): c for c in (usage or {}).get("checks", [])}
        for name in incomplete:   # AV-G0-SUIVI-TITRES: a collection whose uncovered units are all of other titles does not refuse
            if _title_disjoint(registry["collections"][name], disjunction, name, members):
                labels.add(f"TITLE_DISJOINT:{name}")
            elif _name_disjoint(registry["collections"][name], disjunction, name, members):
                labels.add(f"TITLE_DISJOINT_NAME:{name}")   # AV-G0-SUIVI-TITRES-NOMS: a distinct, weaker label, reported apart
            elif _content_disjoint(registry["collections"][name], src, name, members, page_checks):
                labels.add(f"CONTENT_DISJOINT:{name}")   # AV-G0-SUIVI-CONTENU: no shared 8-gram with ALL the texts of the collection
            else:
                refuse(f"INCOMPLETE_NUMBER_KEY:{name}", f"{name} (gallica_scope possible) has no complete number key")
    # 6 unresolved sources: a row of a space of the class that does not join, any row of unknown space, a usage source missing from the table
    lm_checked = [s["source"] for s in table if s.get("lm_texts", {}).get("all_on_disk") is True]   # AV-G0-SUIVI-LM
    lm_print = {s["source"]: lm_fingerprint(s["lm_texts"]["files"]) for s in table if s["source"] in lm_checked}
    lm_spaces = {s["source"]: set(s["key_spaces"]) for s in table if s["source"] in lm_checked}
    for s in table:
        sp_row = set(s["key_spaces"])
        if s["source"] in lm_checked:
            continue   # its texts are all on disk: the page owes it the ngram8 check (rank 7) instead of a refusal
        if s["status"] == MODEL:
            continue   # a pretrained model refuses no page: it is rendered by its LIMIT label (AV-G0-SUIVI-MODELES)
        if (s["status"] not in JOINING and sp_row & set(spaces)) or "unknown" in sp_row:
            refuse(f"SUIVI_SOURCE_UNRESOLVED:{s['source']}", f"source {s['source']} ({s['status']}, spaces {sorted(sp_row)}) does not join")
    if src is not None:
        known = {s["source"] for s in table}
        for u in sorted({u.source for u in applied} - known):
            refuse(f"SUIVI_SOURCE_UNRESOLVED:{u}", f"usage source {u} is not a row of the sources table")
    # 7 checks: every member of the class must pass the checks required for its own space
    checks = {(c["page"], c["check"]): c for c in (usage or {}).get("checks", [])}   # duplicates refused by _validate
    for m in members:
        for chk in (src or {}).get("required_checks", {}).get(_namespace_of(m), []):
            c = checks.get((m, chk))
            if c is None or c["status"] != "complete":
                refuse(f"SUIVI_CHECK_INCOMPLETE:{chk}", f"required check {chk} is missing or incomplete for {m}")
        for name in lm_checked:   # a member owes ngram8 against a source with texts on disk if the source is unknown or of its space
            if "unknown" not in lm_spaces[name] and _namespace_of(m) not in lm_spaces[name]:
                continue
            c = checks.get((m, LM_CHECK))
            if (c is None or c["status"] != "complete" or name not in c.get("against", [])
                    or c.get("against_manifests", {}).get(name) != lm_print[name]):   # a check run on other texts is stale (pQ, B1)
                refuse(f"SUIVI_CHECK_INCOMPLETE:{LM_CHECK}:{name}", f"{LM_CHECK} against the LM texts of {name} is missing or incomplete for {m}")
    # 8 reservation of the sealed test, whole, whatever the alias
    wholes = _closure((_whole(a.left), _whole(a.right)) for a in aliases).get(_whole(ident), {_whole(ident)})
    if sealed is not None:
        sealed_pages = _key_values(sealed["match_keys"]["page"]) or set()
        sealed_issues = _key_values(sealed["match_keys"]["issue"]) or set()
        whole_keys = {k for _, k in wholes}
        if (same_page & sealed_pages or whole_keys & sealed_issues
                or whole_keys & {d for d in map(_document_key, sealed_pages) if d}):
            refuse(f"SUIVI_RESERVED:{SEALED}", f"{SEALED} is reserved whole (page, issue or file, through any alias)")
    # 9-11 usages, sources in the order of the table; a select evidence promotes its bench or page (v3 point 3); a page named by a `choose`
    # consultation already covered by the anchored registry is selected by that consultation (v4 point 8, enforced here: pT, C1)
    labels.update(issue_labels)
    rank = {s["source"]: i for i, s in enumerate(table)}
    order = lambda name: (rank.get(name, len(rank)), name)  # noqa: E731
    status_of = {(u["scope"], u["key"], u["usage"], u["source"]): u["status"] for u in (usage or {}).get("usages", [])}
    promoted = {e["target"] for e in (docs["select_evidence"] or {}).get("evidence", [])} & ({u.source for u in applied} | same_page)
    chosen = {f"suivi_choose_seq{c['seq']}" for c in (docs["consultations"] or {}).get("consultations", [])
              if c["purpose"] == "choose" and set(c["pages"]) & same_page}
    for kind in ("train", "dev", "select"):
        found = {u.source for u in applied if u.usage == kind} | (promoted | chosen if kind == "select" else set())
        for name in sorted(found, key=order):
            refuse(f"{CODE_OF[kind]}:{name}", f"{kind} by {name}")
        for u in applied:
            if u.usage == kind and status_of.get((u.scope, u.key, u.usage, u.source)) == "allocated_only":
                labels.add(f"ALLOCATION_ONLY:{u.source}")
    labels.update(f"EVALUATED_BEFORE:{u.source}" for u in applied if u.usage == "eval")
    # 12 selections not reviewed: the spaces of the class and the applied sources
    concerned = set(spaces) | {u.source for u in applied}
    # pQ B-A (24/10): the declared spaces of the sources applied to the page are concerned too (a journal on hipe reaches a page used by a
    # source declared [hipe, ...]); sources that do not apply to the page add nothing
    applied_names = {u.source for u in applied}
    concerned |= {sp for s_ in table if s_["source"] in applied_names for sp in s_["key_spaces"]} - {"unknown"}
    for j in (docs["journals"] or {}).get("journals", []):
        if j["status"] == "pending" and ("concerns" not in j or set(j["concerns"]) & concerned):
            refuse(f"SUIVI_SELECTION_UNREVIEWED:{j['journal']}", f"journal {j['journal']} is pending and concerns this page")
    for e in (docs["select_evidence"] or {}).get("documented_without_proof", []):
        if e["bench"] in concerned:
            refuse(f"SUIVI_SELECTION_UNREVIEWED:{e['journal']}", f"a choice on {e['bench']} is documented in {e['journal']} without proof")
    # 13 text overlap of any member of the class, and the labels of the checks
    for page, c in ((p_, checks[(p_, k)]) for p_ in members for (q, k) in sorted(checks) if q == p_):
        for f in c["findings"]:
            if f["kind"] == "train_overlap":
                refuse(f"SUIVI_TEXT_OVERLAP:{f['source']}", f"text shared with the training text {f['source']} ({page})")
            elif f["kind"] == "lm_overlap":
                refuse(f"SUIVI_LM_OVERLAP:{f['source']}", f"8-grams shared with the language-model texts {f['source']} ({page})")
            elif f["kind"] == "content_overlap":   # AV-G0-SUIVI-CONTENU (Sonnet M1): a measured shared 8-gram refuses, however C was lifted
                refuse(f"SUIVI_CONTENT_OVERLAP:{f['source']}", f"8-grams shared with the texts of the collection {f['source']} ({page})")
            elif f["kind"] == "eval_overlap":
                labels.add(f"TEXT_OVERLAP_EVAL:{f['source']}")
            elif f["kind"] == "third_party_eval":
                labels.add(f"THIRD_PARTY_EVAL:{f['source']}")
            else:
                labels.add(f"THIRD_PARTY_OVERLAP:{f['source']}:{'proven' if f['kind'] == 'third_party_proven' else 'unresolved'}")
    for sp in spaces:
        labels.update((usage or {}).get("limits", {}).get(sp, []))
        labels.update(FORCED_LIMITS.get(sp, ()))
    return result()
