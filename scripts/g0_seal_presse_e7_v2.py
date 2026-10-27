"""presse_e7 test, seal v2 and usage ledger v2 (LOT-G0-02, lots 2 and 3). The v1 files are read, never changed.

    PYTHONDONTWRITEBYTECODE=1 ~/axel/.venv/bin/python scripts/g0_seal_presse_e7_v2.py

Writes config/sealed/presse_e7_test_v2.json and reports/test_ledger_v2.jsonl. The v2 copies the v1 pages and adds:
the written limits (lot 2), the past use of the test (lot 3), the per-page ``sibling_exception`` label and the list of
differences from the v1. The status stays pending_independent_audit; config/g0_release.json is never created.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "config/sealed/presse_e7_test_v1.json"
V2 = ROOT / "config/sealed/presse_e7_test_v2.json"
LEDGER_V1 = ROOT / "reports/test_ledger.jsonl"
LEDGER_V2 = ROOT / "reports/test_ledger_v2.jsonl"
REGISTRY_V1 = ROOT / "config/g0_collections.json"
DECISIONS = Path.home() / "axel/docs/DECISIONS.md"
V1_SHA256_PREFIX = "f9552a4d"  # the v1 reviewed by pJ and audited by pQ and pM
LEDGER_V1_SHA256_PREFIX = "d605ba93"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    if not sha(V1).startswith(V1_SHA256_PREFIX) or not sha(LEDGER_V1).startswith(LEDGER_V1_SHA256_PREFIX):
        raise SystemExit("the v1 seal or its usage ledger is not the audited one: nothing written")
    v1_before = sha(V1)
    seal = json.loads(V1.read_text(encoding="utf-8"))
    registry = json.loads(REGISTRY_V1.read_text(encoding="utf-8"))["collections"]
    refused_issues = {k for p in seal["refused"] for k in p["issue_keys"]}
    pages = []
    for p in seal["pages"]:
        # lot 1 rule 4: a number shared with the unsealed sibling stays labelled, never presented as a clean page
        pages.append({**p, "sibling_exception": bool(set(p["issue_keys"]) & refused_issues)})
    exceptions = sum(p["sibling_exception"] for p in pages)
    trees = {n: {Path(t["root"]).name: t["file_count"] for t in e["trees"]} for n, e in registry.items()}
    limits = [
        {"id": "icdar_2017", "text": "ICDAR 2017 : les images sont absentes de cette machine ; seuls les manifestes préparés (icdar2017.train, dev, test) sont épinglés ; "
                                       "aucune concordance Gallica ni titre n'est disponible."},
        {"id": "gallica_pseudo", "text": f"gallica_pseudo : dossier lié au Drive monté, épinglé par les noms et les tailles de {trees['gallica_pseudo']['pseudo']} fichiers seulement ; "
                                         "la lecture du contenu a dépassé le délai (Errno 60)."},
        {"id": "nzz", "text": f"NZZ (nzz_impresso) : le dossier local corpus-vt/nzz-gt est vide ({trees['nzz_impresso']['nzz-gt']} fichier) ; le compte de 167 pages est hérité des "
                              "documents d'axel (docs/e7.md, DECISIONS.md E30), pas d'un fichier lisible ; le titre « neue zurcher zeitung » est déclaré complet sur cette foi."},
        {"id": "collections_without_keys", "text": "Collections sans clé, ou à clé partielle, au niveau page ou numéro (recopié de headline_limits du scellé v1) : "
                                                    + ", ".join(sorted(seal["headline_limits"]["collections_with_no_or_partial_page_or_issue_keys"])) + "."},
        {"id": "non_auditable_pretraining", "text": "Le préentraînement n'est pas auditable pour : mccatmus, mccatmus_press, german_print_s, dots_ocr et kraken_blla_2026. "
                                                     "Un document de test a pu être vu pendant ce préentraînement."},
        {"id": "unresolved_gallica_ark", "text": "Gallica : deux ARK hors du préfixe bpt6k (bd6t5700010r, bd6t511774123) n'ont pas pu être rapprochés d'un alias bpt6k ; "
                                                  "la sonde de l'audit (audit_codex/gallica_alias_probe) a reçu HTTP 403. Ils sont conservés tels quels, non résolus."},
        {"id": "incomparable_namespaces", "text": "Les espaces de noms d'identifiants ne sont pas comparables entre eux : scan_page_key (fichier Wikimedia), folios Gallica "
                                                   "(gallica:N/fK), kb:ID, identifiants locaux (BnL, ICDAR, impresso, AS date_numéro). Un recouvrement de page entre ces espaces "
                                                   "n'est pas détectable (correspondances folio Wikisource vers Gallica prouvées : zéro) ; le rapprochement se fait surtout au niveau du numéro."},
    ]
    usage_explanation = [
        "Ce test a servi au CHOIX DES LECTEURS PAR ÉPOQUE : D-196 (E8-17, 2026-10-06, docs/DECISIONS.md) compare les lecteurs sur le manifeste E7 ; "
        "D-294 (E20-5, 2026-10-06) en reprend le résultat (german_print_s avant 1800, mccatmus_press après).",
        "Les 410 pages scellées, dont les 176 headline, ont DÉJÀ ÉTÉ LUES (E1, E2, E3, E8, E11 ; voir reports/test_ledger_v2.jsonl).",
        "Il ne permet donc AUCUNE REVENDICATION sur les lecteurs choisis grâce à lui.",
        "Il ne sert qu'à comparer des modèles NOUVEAUX entre eux, ou contre ces lecteurs avec cette réserve écrite.",
    ]
    differences = [
        "Ajout de limits (lot 2) et de usage_status / usage_explanation (lot 3).",
        f"Ajout, sur chaque page scellée, de sibling_exception (numéro partagé avec une page non scellée du même split) : {exceptions} pages sur {len(pages)}.",
        "Les pages, les refus, les 176 headline et les clés sont ceux de la v1, sans réglage : les règles d'éligibilité plus strictes s'appliquent par eligibility() "
        "et leur recomptage est publié à part (reports/lot02_recount.json).",
        "Le registre des usages passe en v2 (reports/test_ledger_v2.jsonl), avec D-196 et D-294.",
    ]
    v2 = {k: v for k, v in seal.items() if k not in ("pages", "generated_utc", "schema_version", "name", "decision", "totals")}
    v2.update({
        "schema_version": "babaorum.sealed_test.v2", "name": "presse_e7_test_v2", "generated_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "pending_independent_audit",
        "derived_from": {"path": str(V1), "sha256": v1_before, "ledger_v1_sha256": sha(LEDGER_V1)},
        "usage_status": "comparison_of_new_models_only", "usage_explanation": usage_explanation,
        "limits": limits, "differences_from_v1": differences,
        "totals": {**seal["totals"], "sibling_exception_pages": exceptions},
        "pages": pages,
    })
    V2.write_text(json.dumps(v2, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    old = LEDGER_V1.read_text(encoding="utf-8").splitlines()
    decisions_sha = sha(DECISIONS)
    new = [
        {"kind": "selection_decision", "decision": "D-196", "experiment": "E8-17", "date": "2026-10-06", "collections": ["presse_e7_test"],
         "nature": "sélection : CER par époque des lecteurs sur le manifeste E7 ; conséquence : german_print_s avant 1800, mccatmus_press après",
         "source": "axel/docs/DECISIONS.md", "source_line": 2695, "source_sha256": decisions_sha},
        {"kind": "selection_decision", "decision": "D-294", "experiment": "E20-5", "date": "2026-10-06", "collections": ["presse_e7_test"],
         "nature": "reprise de la sélection de D-196 comme second nœud du consensus ; pas de nouvelle lecture du test",
         "source": "axel/docs/DECISIONS.md", "source_line": 2930, "source_sha256": decisions_sha},
    ]
    header = {"kind": "header_v2", "generated_utc": v2["generated_utc"], "v1_ledger": "reports/test_ledger.jsonl", "v1_ledger_sha256": sha(LEDGER_V1),
              "v1_lines_copied_verbatim": len(old), "added": ["D-196", "D-294"]}
    LEDGER_V2.write_text("\n".join([json.dumps(header, ensure_ascii=False), *old, *(json.dumps(r, ensure_ascii=False) for r in new)]) + "\n", encoding="utf-8")
    assert sha(V1) == v1_before and sha(LEDGER_V1).startswith(LEDGER_V1_SHA256_PREFIX), "the v1 files must not move"
    print(json.dumps({"v2_sha256": sha(V2)[:12], "v1_unchanged": True, "sibling_exception_pages": exceptions, "ledger_v2_lines": len(old) + 3, "limits": len(limits)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
