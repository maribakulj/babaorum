"""Convertisseur : sorties du producteur ngram8 (scripts/g0_lm_ngram8.py) -> entrées `checks` du registre d'usage (g0_suivi, AV-G0-SUIVI-LM).

Format imposé par le module (src/babaorum/g0_suivi.py) :
  {page: <clé CANONIQUE>, check: "ngram8", status: "complete", against: [sources LM de la table],
   against_manifests: {<source LM>: lm_fingerprint(files de sa ligne)}, findings: [{kind: "lm_overlap", source: <source LM>}]}

Garde-fous (revue pQ 33bf5875) :
- une sortie n'admet QUE SA PROPRE page : aucun membre de classe n'est admis sans son propre calcul ;
- une sortie sans verdict, sans par_source, ou calculée contre autre chose que les 322 textes, n'est pas convertie ;
- l'empreinte `comparés` émise par le producteur doit égaler les fichiers de la table (SHA), sinon la page n'est pas admise
  (un SHA LM qui change impose un nouveau calcul) ;
- against_manifests = lm_fingerprint du MODULE, sur la même liste files que la table.
Sans réseau. Sortie : JSON des entrées checks, dans un fichier NEUF.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]
from babaorum.g0_suivi import LM_CHECK, lm_fingerprint  # noqa: E402
from babaorum.g0_suivi_identity import _namespace_of, page_identity  # noqa: E402

N_TEXTES_ATTENDUS = 322
LM_VERDICT_PREFIX = "SUIVI_TEXT_OVERLAP:"


def lm_sources(table: dict) -> dict[str, list[dict]]:
    """Sources de la table qui portent lm_texts pleinement vérifiés (all_on_disk true)."""
    return {s["source"]: s["lm_texts"]["files"] for s in table["sources"] if s.get("lm_texts", {}).get("all_on_disk") is True}


def convert(producer_outputs: dict[str, dict], table: dict) -> list[dict]:
    """producer_outputs : {id brut de la page : sortie du producteur}. Une sortie n'admet que SA page."""
    lms = lm_sources(table)
    if not lms:
        raise ValueError("aucune source LM avec lm_texts vérifiés : rien à convertir")
    prints = {name: lm_fingerprint(files) for name, files in lms.items()}
    fresh = sorted({(f["path"]["path"], f["sha256"]) for files in lms.values() for f in files})   # fichiers distincts, pas de doublon par ligne
    entries = []
    for raw, out in producer_outputs.items():
        if out.get("verdict") == "incomplete":
            continue                                   # moins de 50 mots : pas de contrôle, donc refus (pas d'entrée)
        m = out.get("mots")
        if not (isinstance(m, int) and not isinstance(m, bool) and m >= 50):
            continue                                   # mots absent, non entier, booléen ou < 50 : aucune entrée (refus)
        if out.get("index_textes") != N_TEXTES_ATTENDUS or "verdict" not in out or "par_source" not in out:
            continue                                   # pas de calcul complet contre les 322 textes : pas d'entrée, donc refus
        comp = out.get("comparés")
        if not comp or sorted({(f["path"], f["sha256"]) for f in comp}) != fresh:
            continue                                   # empreinte des fichiers comparés absente ou périmée : refus
        ns = _namespace_of(raw)
        if ns is None:
            raise ValueError(f"page sans espace reconnu : {raw!r}")
        findings = []
        if any(v.startswith(LM_VERDICT_PREFIX) for v in out.get("verdicts", [])):
            findings = [{"kind": "lm_overlap", "source": name} for name in sorted(lms)]
        entries.append({"page": page_identity(raw, ns).page, "check": LM_CHECK, "status": "complete",
                        "against": sorted(lms), "against_manifests": dict(prints), "findings": findings})
    return entries


def merge_ngram8(entries: list[dict]) -> list[dict]:
    """Une seule entrée ngram8 par page. Deux empreintes différentes pour une même source : ERREUR, pas d'écrasement."""
    merged: dict[str, dict] = {}
    for e in entries:
        m = merged.setdefault(e["page"], {**e, "against": [], "against_manifests": {}, "findings": []})
        for name, fp in e["against_manifests"].items():
            if name in m["against_manifests"] and m["against_manifests"][name] != fp:
                raise ValueError(f"deux empreintes différentes pour la source {name} sur la page {e['page']} : refusé")
        m["against"] = sorted(set(m["against"]) | set(e["against"]))
        m["against_manifests"] = {**m["against_manifests"], **e["against_manifests"]}
        seen = {json.dumps(f, sort_keys=True) for f in m["findings"]}
        for f in e["findings"]:
            if json.dumps(f, sort_keys=True) not in seen:
                m["findings"].append(f); seen.add(json.dumps(f, sort_keys=True))
        m["status"] = "complete" if (m["status"] == "complete" and e["status"] == "complete") else "incomplete"
    return list(merged.values())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outputs", required=True, type=Path, help="JSON : {id brut: sortie du producteur}")
    ap.add_argument("--table", required=True, type=Path, help="sources_full.DRAFT.json (lm_texts vérifiés)")
    ap.add_argument("--out", required=True, type=Path, help="fichier NEUF")
    a = ap.parse_args(argv)
    if a.out.exists():
        print(f"refusé : {a.out} existe", file=sys.stderr)
        return 1
    try:
        entries = merge_ngram8(convert(json.loads(a.outputs.read_text(encoding="utf-8")),
                                       json.loads(a.table.read_text(encoding="utf-8"))))
    except ValueError as exc:
        print(f"refusé : {exc}", file=sys.stderr)
        return 2
    a.out.write_text(json.dumps(entries, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"entrees": len(entries)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
