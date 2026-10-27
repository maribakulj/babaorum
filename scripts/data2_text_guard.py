"""DATA-2 garde-fou « aucun champ texte » (docs/END_TO_END_DATA2_LOT.txt §14 (3), §15, §16 (amendement 7)).

Liste blanche TOTALE, parcours récursif (dicts et listes imbriqués) :
  - toute clé hors liste blanche (exacte ou motif de nom de fichier/page), à n'importe quelle profondeur ;
  - toute chaîne qui n'est ni identifiant (page, ligne, bras), ni sha, ni version, ni statut listé, ni chemin relatif
    de sortie ou d'entrée, ni nom retiré par la sanitisation (valeurs de removed_keys seulement) ;
  - reason : admis seulement si le statut n'est pas DONE / COMPLETE.
Contrôles de fichiers : LISEZMOI.txt et LISEZMOI_DATA2.txt doivent être EXACTEMENT le texte attendu (gabarit et manifeste ;
seule la date d'exécution varie) ; la sortie ne contient que les fichiers listés.
"""
import ast
import hashlib
import json
import os
import re
from pathlib import Path

ARMS = ("B3", "S2")
REMOVED_NAMES = ("alphabet", "alphabet_size")          # §15 (1) : seuls noms admis, uniquement dans removed_keys
RENAME_PAIRS = (("times_s.alphabet_models", "times_s.models_load"),)   # §17 (1) : seul couple admis, dans renamed_keys
DATE_RE = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}"

KEYS = {
    # manifeste de DATA-1 (réel, §16 (1)) et manifeste nettoyé
    "schema_version", "status", "smoke", "shard", "index", "of", "range", "pages", "pages_fragments", "arms",
    "models_load", "renamed_keys", "inputs", "generation_manifest_sha256", "e2e4_report_sha256", "checkpoints_sha256", "pages_sha256",
    "pages_checked_against_e2e4_report", "pages_sha_mismatch", "versions", "python", "torch", "numpy", "kit_sha256",
    "config_sha256", "data1", "e2e4", "pages_written", "outputs_sha256", "times_s", "inputs_check", "page_load",
    "infer_and_score", "refine", "write", "total", "counts", "before_total", "reason", "removed_keys",
    "original_manifest_sha256",
    # fragments d'une page (assemble_payload)
    "page_id", "bras", "échelle", "factor", "native_wh", "scaled_wh", "stride", "seuil", "before",
    "id", "polyline_px", "refined_baseline_px", "score", "n_cells",
}
KEY_PATTERNS = [
    re.compile(r"^(mf|mfn)_\d{4}$"),
    re.compile(r"^(pages|images)/(mf|mfn)_\d{4}\.(json|png)$"),
    re.compile(r"^fragments/(B3|S2)/(mf|mfn)_\d{4}\.json$"),
    re.compile(r"^(scripts|config|tests)/[A-Za-z0-9_./-]+\.(py|json|ipynb|txt)$"),
    re.compile(r"^(B3|S2)$"),
]
STATUS_ENUM = ("OK", "SMOKE", "DONE", "COMPLETE", "INCOMPLETE", "FAILED", "MISMATCH")
VALUE_PATTERNS = [
    re.compile(r"^(mf|mfn)_\d{4}$"),                                              # page
    re.compile(r"^(B3|S2):(mf|mfn)_\d{4}:b\d+$"),                                 # ligne de fragment
    re.compile(r"^[0-9a-f]{64}$"),                                                # sha256
    re.compile(r"^\d+(\.\d+)*(\+[0-9A-Za-z.]+)?$"),                              # version
    re.compile(r"^(B3|S2)$"),                                                     # bras
    re.compile(r"^(BLOCKED_[A-Z_]+|UNEXPECTED_OUTPUT|" + "|".join(STATUS_ENUM) + r")$"),  # statut
    re.compile(r"^data1_manifest_v1$|^rg\.blind\.gt-index\.v1$|^data2_[a-z_]+_v1$"),     # schéma
    re.compile(r"^fragments/(B3|S2)/(mf|mfn)_\d{4}\.json$"),                       # chemin de sortie
    re.compile(r"^(pages|images)/(mf|mfn)_\d{4}\.(json|png)$"),                    # chemin d'entrée
    re.compile(r"^(data1_manifest\.json|LISEZMOI\.txt|LISEZMOI_DATA2\.txt|data2_notebook_manifest\.json)$"),
    re.compile(r"^(scripts|config|tests)/[A-Za-z0-9_./-]+\.(py|json|ipynb|txt)$"),   # chemin du kit
]


class TextFieldFound(RuntimeError):
    """BLOCKED_TEXT_FIELD : une clé ou une valeur hors liste blanche, ou un fichier texte différent du texte attendu."""


def _key_ok(k) -> bool:
    return isinstance(k, str) and (k in KEYS or any(p.match(k) for p in KEY_PATTERNS))


def _value_ok(v) -> bool:
    return isinstance(v, str) and any(p.match(v) for p in VALUE_PATTERNS)


def _walk(obj, where: str, status, problems: list) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if not _key_ok(k):
                problems.append(f"{where}: clé hors liste blanche {k!r}")
                continue
            here = f"{where}.{k}"
            if k == "reason":
                if status in ("DONE", "COMPLETE"):
                    problems.append(f"{here}: reason interdit avec le statut {status}")
                elif not isinstance(v, str):
                    problems.append(f"{here}: reason doit être une chaîne")
                continue
            if k == "renamed_keys":
                if not (isinstance(v, list) and all(tuple(x) in RENAME_PAIRS for x in v)):
                    problems.append(f"{here}: renamed_keys hors du couple admis")
                continue
            if k == "removed_keys":
                if not (isinstance(v, list) and all(x in REMOVED_NAMES for x in v)):
                    problems.append(f"{here}: removed_keys hors des noms retirés")
                continue
            if k == "status" and isinstance(v, str) and not _value_ok(v):
                problems.append(f"{here}: statut {v!r} hors liste")
                continue
            _walk(v, here, status, problems)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            _walk(item, f"{where}[{i}]", status, problems)
    elif isinstance(obj, str):
        if not _value_ok(obj):
            problems.append(f"{where}: chaîne {obj!r} hors liste blanche")
    elif obj is None or isinstance(obj, (bool, int, float)):
        return
    else:
        problems.append(f"{where}: type {type(obj).__name__} interdit")


def check_payload(payload: dict, where: str = "payload") -> list:
    """Fragments d'une page (assemble_payload) : liste blanche totale."""
    problems: list = []
    _walk(payload, where, None, problems)
    return problems


def check_manifest(manifest: dict, where: str = "manifest") -> list:
    """Manifeste de DATA-1 (réel ou factice) : liste blanche totale, y compris les clés imbriquées."""
    problems: list = []
    _walk(manifest, where, manifest.get("status"), problems)
    return problems


def check_all(payloads: list, manifest: dict) -> list:
    problems: list = []
    for i, p in enumerate(payloads):
        problems.extend(check_payload(p, where=f"payload[{i}]"))
    problems.extend(check_manifest(manifest))
    return problems


# ---- §15 : sanitisation of the DATA-1 manifest before pR (the DATA-1 code stays frozen)

def sanitize_manifest(manifest: dict, original_sha: str) -> tuple:
    """§15 (1) et §17 (1) : retire alphabet / alphabet_size ; renomme times_s.alphabet_models en times_s.models_load."""
    removed = [k for k in REMOVED_NAMES if k in manifest]
    clean = {k: v for k, v in manifest.items() if k not in REMOVED_NAMES}
    renamed = []
    times = clean.get("times_s")
    if isinstance(times, dict) and "alphabet_models" in times:
        times = dict(times)
        times["models_load"] = times.pop("alphabet_models")
        clean["times_s"] = times
        renamed.append(list(RENAME_PAIRS[0]))
    clean["removed_keys"] = removed
    clean["renamed_keys"] = renamed
    clean["original_manifest_sha256"] = original_sha
    return clean, removed


def sanitize_file(path) -> dict:
    """Remplace le manifeste d'origine par sa version nettoyée : l'original est SUPPRIMÉ (pR voit toute la sortie)."""
    path = Path(path)
    raw = path.read_bytes()
    original_sha = hashlib.sha256(raw).hexdigest()
    clean, removed = sanitize_manifest(json.loads(raw), original_sha)
    path.unlink()
    path.write_text(json.dumps(clean, indent=1, ensure_ascii=False), encoding="utf-8")
    return {"original_sha256": original_sha, "removed_keys": removed,
            "clean_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


# ---- §16 (2) : LISEZMOI.txt exact, reconstruit à partir du gabarit de data1_fragments.py et du manifeste

def template_from_source(source_text: str) -> str:
    """Le gabarit README = chaîne littérale de data1_fragments.py, lue sans importer le module (torch)."""
    for node in ast.parse(source_text).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "README" for t in node.targets):
            return node.value.value
    raise ValueError("README template not found")


def expected_readme(manifest: dict, template: str, cfg: dict) -> str:
    """Même règle que data1_fragments.readme_text ; seule la date d'exécution est laissée à DATE_RE."""
    shard = manifest.get("shard")
    pages = manifest.get("pages_fragments", [])
    values = {
        "<shard>": f"{shard['index']} (--shard {shard['index']}/{shard['of']})" if shard else "aucun (run complet)",
        "<pages>": f"{pages[0]} à {pages[-1]} ({len(pages)} pages)" if pages else "aucune",
        "<arms>": ", ".join(ARMS),
        "<status>": str(manifest.get("status")),
        "<date>": "\x00DATE\x00",
        "<target>": f"{cfg['scale']['target_median_height_px']:g}",
        "<stride>": str(cfg["extraction"]["stride"]),
        "<half>": f"{cfg['extraction']['stride'] / 2:g}",
        "<K>": str(cfg["model"]["K"]),
    }
    text = template
    for key, val in values.items():
        text = text.replace(key, val)
    return text


def readme_matches(text: str, manifest: dict, template: str, cfg: dict) -> bool:
    parts = expected_readme(manifest, template, cfg).split("\x00DATE\x00")
    return re.fullmatch(DATE_RE.join(re.escape(part) for part in parts), text) is not None


def exact_text_matches(text: str, expected: str) -> bool:
    return text == expected


# ---- §16 (3) : la sortie ne contient que les fichiers listés

def expected_output(manifest: dict, overlap_text_accepted: bool) -> set:
    rel = {"data1_manifest.json", "LISEZMOI.txt", "data2_notebook_manifest.json"}
    for arm in ARMS:
        for pid in manifest.get("pages_fragments", []):
            rel.add(f"fragments/{arm}/{pid}.json")
    if overlap_text_accepted:
        rel.add("LISEZMOI_DATA2.txt")
    return rel


def unexpected_output(root, expected_rel: set) -> list:
    found = []
    root = Path(root)
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            rel = os.path.relpath(os.path.join(dirpath, name), root)
            if rel not in expected_rel:
                found.append(rel)
    return sorted(found)
