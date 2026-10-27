"""Rebuild config/g0_collections.json (G0 v2 registry, schema v3).

Read-only on every source: it hashes manifest files and fingerprints directories
by names and sizes only. No image, ground truth or transcription is opened, and
nothing outside this repository's config/ is written. Match keys are declared
"aucune clé à ce niveau" until step B fills them (the refusal is then cautious).

    PYTHONDONTWRITEBYTECODE=1 ~/axel/.venv/bin/python scripts/g0_collections_build.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from babaorum.g0_collections import NO_KEY, SCHEMA_VERSION, verify_collections  # noqa: E402
import g0_keys_build as KEYS  # noqa: E402

H = Path.home()
AX, CV, HS, OS = H / "axel", H / "corpus-vt", H / "heritage-synth", H / "outils-seg"
OUT = ROOT / "config" / "g0_collections.json"


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def manifest(p: Path) -> dict:
    return {"path": str(p), "sha256": sha(p), "bytes": p.stat().st_size}


def tree(root: Path) -> dict:
    items = sorted((str(p.relative_to(root)), p.stat().st_size) for p in root.rglob("*") if p.is_file())
    h = hashlib.sha256()
    for n, s in items:
        h.update(f"{n}\0{s}\n".encode())
    return {"root": str(root), "file_count": len(items), "tree_sha256_names_and_sizes": h.hexdigest(),
            "tree_basis": "names_and_sizes_only_no_file_content_read"}


KEY_DIR = ROOT / "config" / "g0_keys"
SEAL = ROOT / "config" / "sealed" / "presse_e7_test_v1.json"


def attach_keys(registry: dict) -> None:
    """Fill match_keys from scripts/g0_keys_build.py (and from the seal for the two presse_e7 test collections)."""
    keys = KEYS.compute_keys()
    seal = json.loads(SEAL.read_text(encoding="utf-8"))
    for name, pages in (("presse_e7_test", seal["pages"]), ("presse_e7_test_refused", seal["refused"])):
        keys[name] = KEYS.make([{"page": p["scan_page_key"], "issue": p["issue_keys"], "title": p["title_key"]} for p in pages],
                               names={"page": "scan_page_key (fichier Wikimedia SHA-1 + vue)", "issue": KEYS.NAMES["issue"] + " du document Gallica crédité",
                                      "title": KEYS.NAMES["title"]},
                               notes={"issue": "seulement les pages dont la notice crédite un document Gallica"})
    KEY_DIR.mkdir(parents=True, exist_ok=True)
    for name, entry in registry["collections"].items():
        spec, reasons = keys[name]
        entry["match_keys_reasons"] = {}
        for level in KEYS.LEVELS:
            if spec[level] is None:
                entry["match_keys"][level] = NO_KEY
                entry["match_keys_reasons"][level] = reasons[level]
                continue
            file = KEY_DIR / f"{name}.{level}.json"
            file.write_text(json.dumps(spec[level]["values"], ensure_ascii=False) + "\n", encoding="utf-8")
            entry["match_keys"][level] = {"name": spec[level]["name"], "values_file": manifest(file), "coverage": spec[level]["coverage"],
                                          **({"complete": True} if spec[level].get("complete") else {})}
        entry["match_keys_status"] = "renseigné à l'étape B : une clé par niveau (nom + fichier épinglé + couverture) ou « aucune clé à ce niveau » avec sa raison"
    registry["collections"]["presse_e7_test"]["sealed_manifest"] = manifest(SEAL)


def coll(status: str, why: str, manifests=(), trees=(), **extra) -> dict:
    return {"status": status, "scope": extra.pop("scope", "collection_entiere"), "why": why,
            "manifests": [manifest(p) for p in manifests], "trees": [tree(p) for p in trees],
            "match_keys": {level: NO_KEY for level in ("page", "issue", "title")},
            "match_keys_status": "à renseigner à l'étape B (clé par niveau : page doc_key, numéro issue_key, titre normalisé)",
            **extra}


def build() -> dict:
    E1M = AX / "runs/e1/manifest"
    press = [AX / "data/prepared/press/train.jsonl", AX / "data/prepared/press/dev.jsonl"]
    pe7 = AX / "data/prepared/presse_e7"
    blla = OS / "modeles/blla-2026"
    dots = OS / "modeles/DotsMOCR"
    c = {
        "bnl_pack": coll("exposed", "Pas de crosswalk page/numéro ; crops de lignes en entraînement/dev press, 6 pages E1 dev/test.",
                         [E1M / "bnl.dev.jsonl", E1M / "bnl.test.jsonl", *press], [CV / "bnl-gt-pack"]),
        "as_anno_183": coll("exposed", "Manifeste AS 183 pages (123/20/40) : train en segmentation, dev/test en évaluation.",
                            [AX / "data/prepared/olr/newseye_as.json", AX / "data/prepared/seg/manifest.json"], [CV / "newseye-as"]),
        "kb_impact": coll("exposed", "1024 pages KB/IMPACT du manifeste de segmentation, sans table titre/date.",
                          [pe7 / "pages_impact.jsonl", AX / "data/prepared/seg/manifest.json"], [CV / "impact-kb-kranten"]),
        "newseye": coll("exposed", "Toutes pages NewsEye (validation ATR, READ, AS) ; E1 dev/test, press train/dev.",
                        [E1M / "newseye.test.jsonl", E1M / "newseye_e9.dev.jsonl", *press], [CV / "newseye", CV / "newseye-read"]),
        "mccatmus": coll("exposed_not_auditable", "McCATMuS entraîné sur 22 jeux CATMuS ; préentraînement non auditable.",
                         [AX / "runs/e1/models/McCATMuS_nfd_nofix_V1.mlmodel", AX / "runs/e1/models/McCATMuS.metadata.json",
                          AX / "data/contamination.json"], scope="donnees_entrainement_modele"),
        "gallicorpora": coll("exposed", "Gallicorpora 17e/18e et OCR17+ ; Gallicorpora+ entraîné dessus.",
                             [E1M / n for n in ("gallicorpora17.dev.jsonl", "gallicorpora17.test.jsonl", "gallicorpora18.dev.jsonl",
                                                "gallicorpora18.test.jsonl", "ocr17.test.jsonl")]
                             + [AX / "runs/e1/models/Gallicorpora+_best.mlmodel", AX / "runs/e1/models/Gallicorpora+.metadata.json"],
                             [CV / "raw"], local_image_roots_absent=[str(CV / "gallicorpora"), str(CV / "ocr17_gallica")],
                             note="Images Gallicorpora absentes de cette machine ; l'exclusion repose sur les doc_id des manifestes."),
        "hipe_ocrepair": coll("exposed", "HIPE-OCRepair utilisé en post-correction (docs axel).", [], [CV / "hipe-ocrepair"]),
        "pleias_french_pd_322": coll("exposed", "322 numéros Gallica du LM R2 (58 032 354 caractères).", [CV / "french-pd-sample/manifest.json"]),
        "presse_e7_train_dev": coll("exposed", "presse_e7 train et dev ; le test est une collection à part (presse_e7_test).",
                                    [pe7 / "train.jsonl", pe7 / "dev.jsonl", pe7 / "excluded.json", pe7 / "pages.jsonl"]),
        "mille_feuilles_sources": coll("disjoint_by_construction",
                                       "Textes originaux CC0 de démonstration, sans journal, corpus OCR, Wikisource ni Gallica (ACTIFS.md) ; à exclure des LM.",
                                       [HS / "docs/ACTIFS.md"], scope="textes_sources_synthetiques"),
        "icdar2017_2019": coll("exposed", "Jeux ICDAR 2017 et 2019 : manifestes préparés d'axel (train, dev, test) ; contacts ICDAR2017 consignés dans G0_AUDIT.",
                               [AX / f"data/prepared/{n}.{sp}.jsonl" for n in ("icdar2017", "icdar2019") for sp in ("train", "dev", "test")],
                               [CV / "icdar2019"], note="Seul le dossier ICDAR2019 est présent localement en images ; ICDAR2017 n'est épinglé que par ses manifestes."),
        "impresso": coll("exposed", "Jeu impresso (manifestes préparés train, dev, test) lu par les expériences de lecture ; référence texte auditée (audit-references.md).",
                         [AX / f"data/prepared/impresso.{sp}.jsonl" for sp in ("train", "dev", "test")]),
        "e12_dev": coll("exposed", "E12 : 5 pages dev NewsEye (1937) du banc de dev du pipeline ; liste commune aux 17 pages (pages.json, champ split).",
                        [AX / "runs/e12/pages.json"], note="Le fichier liste aussi les 12 pages de test (collection e12_test)."),
        "e12_test": coll("test_used", "E12 : 12 pages de test NewsEye déjà mesurées comme test (D-217 sur 12 pages).",
                         [AX / "runs/e12/pages.json"], note="Même fichier que e12_dev ; le champ split sépare les pages."),
        "nzz_impresso": coll("test_used", "Banc NZZ black-letter (impresso, allemand) : mesure de test unique d'E30 (DECISIONS.md).",
                             [], [CV / "nzz-gt"]),
        "bnf_ocr_corrige_presse": coll("exposed", "Paquet BnF « OCR corrigé presse » (part française d'Europeana Newspapers), utilisé en apprentissage E21 (D-300).",
                                       [], [CV / "bnf-presse-seg"]),
        "gallica_pseudo": coll("exposed", "Pseudo-vérité Gallica (P3) : dossier lié au Drive monté ; épinglé par noms et tailles seulement.",
                               [], [CV / "pseudo"],
                               note="La lecture du contenu via le Drive monté a dépassé le délai (Errno 60) : aucun contenu lu. Le registre devient invalide si le Drive est démonté."),
        "gallica_e13": coll("test_used", "E13 : 7 pages historiques déjà mesurées comme test (diagnostic GT-assisté).",
                            [AX / "runs/e13/scores.json"], [CV / "gallica-alto-e13"]),
        "e22_bench": coll("test_used", "Banc indépendant E22 (13 pages 1937), déjà mesuré comme test.",
                          [AX / "runs/e22/bench/runs/e12/pages.json"], note="Seule la liste de pages est épinglée."),
        "gallica_e31_reserve": coll("reserved_test", "Banc E31 réservé au test unique du pipeline (D-608) : refuse train, dev et test tant qu'il n'est pas scellé.",
                                    [CV / "gallica-banc-e31/exposition.json"], [CV / "gallica-banc-e31/pages"]),
        "reader_lm_corpus": coll("exposed", "Corpus du LM de R2 : même manifeste que pleias_french_pd_322.",
                                 [CV / "french-pd-sample/manifest.json"]),
        "kraken_blla_2026": coll("exposed_not_auditable", "Modèle blla-2026 (Apache-2.0) : données d'entraînement non auditables.",
                                 [blla / "blla.mlmodel", blla / "record.json"], scope="donnees_entrainement_modele"),
        "dots_ocr": coll("exposed_not_auditable", "dots.ocr / dots.mocr : préentraînement non auditable.",
                         [dots / "config.json", dots / "dots.mocr LICENSE AGREEMENT"], scope="donnees_entrainement_modele"),
        "presse_e7_test": coll("sealed_test", "presse_e7 split test : les pages scellées (410 avec vérité Wikisource) ; test réutilisé, déjà mesuré (reports/test_ledger.jsonl).",
                               [pe7 / "pages.jsonl"], note="Seul sealed_test du registre. Le scellé est config/sealed/presse_e7_test_v1.json (par copie et empreintes)."),
        "presse_e7_test_refused": coll("reserved_test", "presse_e7 split test : les pages non scellées (sans ligne de vérité ou en recouvrement) ; restent verrouillées pour train, dev et test.",
                                       [pe7 / "pages.jsonl"], note="Liste dans config/sealed/presse_e7_test_v1.json, champ refused.", sibling_of="presse_e7_test"),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "rule": "Seul un sealed_test rend une page éligible au test (pour les pages de son manifeste). exposed et exposed_not_auditable: test refusé, train/dev permis. test_used et reserved_test: train, dev et test refusés. sealed_test: test permis pour son manifeste, train/dev refusés. disjoint_by_construction et not_exposed: train/dev permis, test refusé hors scellé.",
        "decided_by": "Marcel, 8 octobre 2026 (G0_PLAN_DEBLOCAGE.md D1, D2)",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "limits": [
            "Préentraînement de dots.ocr non auditable.", "Préentraînement de kraken McCATMuS non auditable (22 jeux CATMuS).",
            "Les arbres de fichiers hachent noms et tailles, pas le contenu.",
            "Clés de rapprochement (page, numéro, titre) non renseignées : refus prudent tant qu'elles ne le sont pas (étape B).",
            "ICDAR2017 n'est épinglé que par ses manifestes préparés (pas d'images locales).",
            "config/g0_release.json n'existe pas et n'est pas créé."],
        "collections": c,
    }


def main() -> int:
    previous = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    registry = build()
    attach_keys(registry)
    changed = []
    for name, entry in registry["collections"].items():
        old = {m["path"]: m["sha256"] for m in previous.get("collections", {}).get(name, {}).get("manifests", [])}
        for m in entry["manifests"]:
            if m["path"] in old and old[m["path"]] != m["sha256"]:
                changed.append(f"{name}: {m['path']} {old[m['path']][:12]} -> {m['sha256'][:12]}")
    OUT.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = verify_collections(OUT)
    print(json.dumps(summary, ensure_ascii=False))
    for line in changed:
        print("re-pinned:", line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
