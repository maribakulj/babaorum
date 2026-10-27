"""G0-03 B0 (classe A, lecture locale, aucune requête réseau) : listes nominatives des unités sans clé de numéro.

Pour ICDAR 2017/2019 : une unité = un fichier texte de l'arbre local (identifiant = chemin relatif à la racine de l'arbre).
Pour HIPE OCR-repair : une unité = un document_id d'un jsonl français (identifiant = document_id).
Chaque ligne donne l'identifiant d'origine et la source candidate du numéro. Rien n'est deviné : une unité sans nom de
fichier porteur d'un numéro reste sans clé.
Sortie : un fichier JSON dans le dossier donné (refusé s'il existe).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ICDAR = Path.home() / "corpus-vt/icdar2019"
HIPE = Path.home() / "corpus-vt/hipe-ocrepair/data"


def icdar_units() -> list[dict]:
    rows = []
    for txt in sorted(ICDAR.rglob("*.txt")):
        if "FR" not in txt.parts:
            continue
        rel = txt.relative_to(ICDAR).as_posix()
        rows.append({"unit": rel, "origin_id": rel, "number_key": None,
                     "candidate_source": "manifeste README icdar2019 (identifiants fr_periodical-NN, à lire) ; nom de fichier sans numéro"})
    return rows


def hipe_units() -> list[dict]:
    seen: dict[str, str] = {}
    for jl in sorted(HIPE.rglob("*_fr.jsonl")):
        with jl.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                doc = json.loads(line)["document_metadata"]
                if doc.get("language") != "fr":
                    continue
                seen.setdefault(doc["document_id"], jl.relative_to(HIPE).as_posix())
    return [{"unit": did, "origin_id": did, "number_key": None, "candidate_source": f"jsonl {src} (publication_title, date)"}
            for did, src in sorted(seen.items())]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        print(f"refusé : {out} existe", file=sys.stderr)
        return 1
    icdar, hipe = icdar_units(), hipe_units()
    data = {"lot": "G0-03 B0 (classe A, lecture locale)", "icdar2017_2019": {"units": len(icdar), "keyed": 0, "rows": icdar},
            "hipe_ocrepair_fr": {"units": len(hipe), "keyed": 0, "rows": hipe},
            "not_listed": {"gallicorpora": "manifestes (7) non localisés ; 484 unités sans clé annoncées dans l'inventaire",
                           "gallica_pseudo": "bloqué : Drive gallica_pages.json, Errno 60 à chaque tentative (21/10)"}}
    out.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"icdar": len(icdar), "hipe_fr": len(hipe)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
