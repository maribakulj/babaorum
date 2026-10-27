"""R2 — échantillon de texte de presse Gallica (PleIAs/French-PD-Newspapers) pour le LM.

Lecture par l'API datasets-server de Hugging Face (pas de fichier parquet,
rien à installer). Garde les numéros dont la note d'OCR (champ ``ocr``) est
≥ ``--min-ocr``, écarte tout numéro dont l'identifiant Gallica apparaît dans
un jeu de dev ou de test connu (noms de fichiers d'axel et de babaorum, sans
lire de vérité), jusqu'à ``--chars`` caractères. Le manifeste des numéros
gardés (identifiants) est écrit à côté : le pipeline doit pouvoir exclure
ces numéros d'un futur banc tenu à l'écart (G0), ou exclure ce LM.

Sortie hors dépôt : ~/corpus-vt/french-pd-sample/ (domaine public ;
réutilisation commerciale soumise aux conditions de la BnF).
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import re
import time
import urllib.request
from pathlib import Path

API = "https://datasets-server.huggingface.co/rows?dataset=PleIAs/French-PD-Newspapers&config=default&split=train&offset={o}&length={n}"
OUT = Path.home() / "corpus-vt" / "french-pd-sample"
SCAN = [Path.home() / "axel" / "data", Path.home() / "axel" / "runs", Path.home() / "babaorum" / "reports",
        Path.home() / "corpus-vt" / "newseye-read"]


def protected_ids() -> set[str]:
    """Identifiants Gallica (chiffres, 6-8) vus dans les noms de fichiers et
    manifestes de dev/test : on ne lit que des noms, jamais de vérité."""
    ids: set[str] = set()
    pat = re.compile(r"(?<!\d)(\d{6,8})(?!\d)")
    for root in SCAN:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            name = str(p.relative_to(root))
            if any(k in name.lower() for k in ("test", "dev", "bench", "indep", "e12", "e13", "e22", "g0", "g3")):
                ids |= {m.lstrip("0") for m in pat.findall(name)}
    for p in (Path.home() / "axel/data/prepared/presse_test_manifest.json",):
        if p.exists():
            ids |= {m.lstrip("0") for m in pat.findall(p.read_text())}
    return ids


def gid(file_id: str) -> str:
    """Identifiant Gallica d'un ark (contrôle NOID retiré seulement s'il est
    juste) : axel.data.doc_key.gallica_id_of_ark, en lecture seule."""
    import sys

    sys.path.insert(0, str(Path.home() / "axel" / "src"))
    from axel.data.doc_key import gallica_id_of_ark

    return (gallica_id_of_ark(file_id) or file_id).lstrip("0")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chars", type=int, default=50_000_000)
    ap.add_argument("--min-ocr", type=int, default=95)
    ap.add_argument("--seed", type=int, default=2)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    banned = protected_ids()
    print("identifiants protégés", len(banned), flush=True)
    total = json.load(urllib.request.urlopen(API.format(o=0, n=1)))["num_rows_total"]
    offsets = list(range(0, total, 100))
    random.Random(a.seed).shuffle(offsets)
    kept, skipped, n = [], {"ocr": 0, "protected": 0}, 0
    with gzip.open(OUT / "text.jsonl.gz", "wt") as g:
        for o in offsets:
            if n >= a.chars:
                break
            for _ in range(3):
                try:
                    d = json.load(urllib.request.urlopen(API.format(o=o, n=100), timeout=120))
                    break
                except Exception as e:  # noqa: BLE001
                    print("réessai", o, e, flush=True)
                    time.sleep(10)
            else:
                continue
            for r in d.get("rows", []):
                row = r["row"]
                try:
                    q = int(float(row.get("ocr") or 0))
                except ValueError:
                    q = 0
                if q < a.min_ocr:
                    skipped["ocr"] += 1
                    continue
                if gid(row["file_id"]) in banned or row["file_id"] in banned:
                    skipped["protected"] += 1
                    continue
                text = row.get("complete_text") or ""
                g.write(json.dumps({"file_id": row["file_id"], "date": row.get("date"), "title": row.get("title"),
                                    "ocr": q, "text": text}, ensure_ascii=False) + "\n")
                kept.append(row["file_id"])
                n += len(text)
            print(f"{n / 1e6:.1f} M caractères, {len(kept)} numéros", flush=True)
            time.sleep(1.0)
    (OUT / "manifest.json").write_text(json.dumps({"source": "PleIAs/French-PD-Newspapers (datasets-server)",
                                                    "min_ocr": a.min_ocr, "seed": a.seed, "chars": n,
                                                    "kept_file_ids": kept, "skipped": skipped,
                                                    "protected_ids": len(banned)}, indent=1))
    print("fini", n, len(kept), skipped)


if __name__ == "__main__":
    main()
