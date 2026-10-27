"""config/g0_collections_v2.json from the frozen v1 registry (LOT-G0-02). The v1 file is read, never changed.

Adds to every collection a reasoned ``gallica_scope`` (p1's table of the 25 collections, section v4 of LOT-G0-02.md):
possible | excluded (identifiers are not Gallica identifiers; says nothing about shared text) | not_auditable (models,
kept as declared limits) | not_applicable (the seal itself and its sibling). Pins the v2 seal and lists its headline pages.

    PYTHONDONTWRITEBYTECODE=1 ~/axel/.venv/bin/python scripts/g0_registry_v2_build.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from babaorum.g0_collections import SCHEMA_VERSION_V4, verify_collections  # noqa: E402

V1, V2 = ROOT / "config/g0_collections.json", ROOT / "config/g0_collections_v2.json"
SEAL_V2 = ROOT / "config/sealed/presse_e7_test_v2.json"
P, X, M, N = "possible", "excluded", "not_auditable", "not_applicable"
SCOPE = {
    "as_anno_183": (P, "Pages AS en partie raccordées à Gallica (8 sur 183)"),
    "newseye": (P, "NewsEye/READ, numéros BnF et Gallica"),
    "gallicorpora": (P, "Sources Gallica"),
    "bnf_ocr_corrige_presse": (P, "Paquet BnF"),
    "gallica_pseudo": (P, "Pseudo-étiquettes Gallica"),
    "icdar2017_2019": (P, "fr_periodical cite Europeana Newspapers et la BnF"),
    "hipe_ocrepair": (P, "Provenance française non établie"),
    "presse_e7_train_dev": (P, "Wikisource crédité Gallica"),
    "pleias_french_pd_322": (P, "322 numéros Gallica (modèle de langue R2)"),
    "reader_lm_corpus": (P, "Corpus Gallica du modèle de langue du lecteur"),
    "e12_dev": (P, "Pages NewsEye 1937 / Gallica"),
    "e12_test": (P, "Pages NewsEye 1937 / Gallica"),
    "e22_bench": (P, "Pages Gallica"),
    "gallica_e13": (P, "ALTO Gallica"),
    "gallica_e31_reserve": (P, "Pages Gallica"),
    "bnl_pack": (X, "Identifiants BnL (Luxembourg), pas des identifiants Gallica"),
    "kb_impact": (X, "Identifiants KB (Pays-Bas), pas des identifiants Gallica"),
    "impresso": (X, "Identifiants impresso (Suisse), pas des identifiants Gallica"),
    "nzz_impresso": (X, "Identifiants NZZ / impresso, pas des identifiants Gallica"),
    "mille_feuilles_sources": (X, "Textes synthétiques CC0, sans identifiant de document réel"),
    "mccatmus": (M, "Préentraînement non auditable : limite déclarée"),
    "dots_ocr": (M, "Préentraînement non auditable : limite déclarée"),
    "kraken_blla_2026": (M, "Préentraînement non auditable : limite déclarée"),
    "presse_e7_test": (N, "C'est le scellé lui-même : les règles 1 et 2 ne s'y appliquent pas"),
    "presse_e7_test_refused": (N, "La sœur du scellé : seule la règle de conflit s'applique"),
}


def main() -> int:
    registry = json.loads(V1.read_text(encoding="utf-8"))
    assert set(SCOPE) == set(registry["collections"]), "the table must classify exactly the 25 collections"
    seal = json.loads(SEAL_V2.read_text(encoding="utf-8"))
    registry["schema_version"] = SCHEMA_VERSION_V4
    for name, (value, reason) in SCOPE.items():
        registry["collections"][name]["gallica_scope"] = {"value": value, "reason": reason}
    sealed = registry["collections"]["presse_e7_test"]
    raw = SEAL_V2.read_bytes()
    sealed["sealed_manifest"] = {"path": str(SEAL_V2), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    sealed["headline_page_keys"] = {"name": "scan_page_key des pages headline=true", "values": sorted(p["scan_page_key"] for p in seal["pages"] if p["headline"] is True)}
    registry["limits"] = [*registry["limits"],
                          "v2 : german_print_s n'est pas une collection du registre ; son préentraînement est une limite listée dans le scellé v2 (limits).",
                          "v2 : tant qu'une collection gallica_scope=possible n'a pas de clé de numéro complète, aucune page n'est admise pour le rôle test (G0-03 complètera les clés)."]
    V2.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = verify_collections(V2)
    print(json.dumps({"registry_v2_sha256": hashlib.sha256(V2.read_bytes()).hexdigest()[:12], **summary}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
