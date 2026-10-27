"""R7(b) — décodage K1 (CATMuS-Print Large) des 2 405 lignes eval de R6 (CLASSE B, figé, non lancé).

Pré-enregistrement : docs/READER_R7B_PREREG.txt (notices, portée D1/D2 seuls, options figées). Ce script
DÉCODE seulement : aucune mesure, aucun CER, aucun contraste (le rapport D1/D2 est un script à part).
Lignes = exactement les lignes eval des chaînes R6 figées (reports/reader_r6_B0_eval.jsonl.gz, mêmes
tiges que B4, contrôlé ; sha256 des deux extraits vérifiés avant la première ligne et écrits dans le JSON
de passage). Recadrages = ceux d'E8/R6 : axel/data/e8/kaggle/e8_items.jsonl.gz (sha256
contrôlé), image locale de axel/data/prepared/press/dev.jsonl, crop_line pad=4 sauf bnl/ (image
entière), niveaux de gris ; MD5 des pixels comparé à crop_md5, arrêt au premier écart.
Options figées : kraken 7.1.1 (arrêt si autre), modèle chargé par kraken.lib.models.load_any (sha256
contrôlé), chaque recadrage lu comme UNE ligne (Segmentation type bbox, boîte = image entière, aucune
segmentation), kraken.rpred.rpred avec ses valeurs par défaut (pad 16, bidi par défaut), sortie =
prediction brute. Python de kraken ; calcul lourd : par ~/outils-seg/lourd.sh.

    ~/outils-seg/lourd.sh ~/outils-seg/kraken-env/bin/python scripts/reader_r7b_k1_decode.py [--fumee N]

Sortie : reports/reader_r7b_k1_eval.jsonl.gz (stem, truth, hyp) + reports/reader_r7b_k1_eval_run.json
(provenance). --fumee N : N premières lignes, sortie dans le scratchpad donné par --out, rien dans reports/.
"""
import argparse
import gzip
import hashlib
import importlib.metadata
import json
import sys
import time
from pathlib import Path

AXEL = Path.home() / "axel"
sys.path.insert(0, str(AXEL / "src"))
from PIL import Image  # noqa: E402

from axel.image.crops import crop_line  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
R = HERE / "reports"
MODEL = Path.home() / "outils-seg/modeles/catmus-print-large/catmus-print-fondue-large.mlmodel"
MODEL_SHA = "1ed39e732b26ccdd0f14b9abd3edc66c4b04a02e3d827d7f9e96a7fe7b585b64"
ITEMS = AXEL / "data/e8/kaggle/e8_items.jsonl.gz"
ITEMS_SHA = "3def259037c596e28411fb7e634facc9ad11d9d7c4ca2053e08588f051a3163e"
DEV = AXEL / "data/prepared/press/dev.jsonl"
EXTRAITS = {"B0": (R / "reader_r6_B0_eval.jsonl.gz", "b3e33a9b0e12fca995e5daebc9d26be4678944fd82657060da36784bc3bd8451"),
            "B4": (R / "reader_r6_B4_eval.jsonl.gz", "dbb971e8e462bd118b09c726f0fbfb99ebf5c153b5d599d8c0f643da6ee56ce0")}
Image.MAX_IMAGE_PIXELS = None


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def stems_of(path: Path) -> list[str]:
    return [json.loads(x)["stem"] for x in gzip.open(path, "rt")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fumee", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    assert importlib.metadata.version("kraken") == "7.1.1", importlib.metadata.version("kraken")
    assert sha256(MODEL) == MODEL_SHA and sha256(ITEMS) == ITEMS_SHA
    for k, (f, h) in EXTRAITS.items():  # extraits R6 dont on prend les tiges : refus de toute autre empreinte (pM)
        assert sha256(f) == h, f"extrait {k} inattendu"
    if a.fumee:
        assert a.out and not Path(a.out).resolve().is_relative_to(R), "fumée : sortie hors de reports/"
    out = Path(a.out) if a.fumee else R / "reader_r7b_k1_eval.jsonl.gz"

    stems = stems_of(EXTRAITS["B0"][0])
    assert len(stems) == 2405 and stems == stems_of(EXTRAITS["B4"][0])
    items = {r["stem"]: r for r in map(json.loads, gzip.open(ITEMS, "rt")) if r["split"] == "dev"}
    local = {}
    for x in open(DEV):
        ln = json.loads(x)
        local[f"{ln['corpus']}__{ln['page']}__{ln['line_id']}".replace("/", "_").replace("#", "-")] = ln["image"]
    rows = [items[s] for s in stems]
    if a.fumee:
        rows = rows[: a.fumee]

    from kraken.containers import BBoxLine, Segmentation
    from kraken.lib.models import load_any
    from kraken.rpred import rpred

    net = load_any(str(MODEL))
    t0, path, page = time.time(), None, None
    with gzip.open(out, "wt") as g:
        for i, it in enumerate(sorted(rows, key=lambda r: (local[r["stem"]], r["stem"]))):
            if local[it["stem"]] != path:
                path = local[it["stem"]]
                with Image.open(path) as im:
                    page = im.convert("L")
            bnl = it["image"].startswith("bnl/")
            crop = page if bnl else crop_line(page, tuple(it["box"]), it["polygon"], pad=4).image
            assert hashlib.md5(crop.tobytes()).hexdigest() == it["crop_md5"], it["stem"]
            seg = Segmentation(type="bbox", imagename=it["stem"], text_direction="horizontal-lr",
                               script_detection=False,
                               lines=[BBoxLine(id="l", bbox=(0, 0, crop.width, crop.height))])
            recs = list(rpred(net, crop, seg))
            assert len(recs) == 1, (it["stem"], len(recs))
            g.write(json.dumps({"stem": it["stem"], "truth": it["truth"], "hyp": recs[0].prediction},
                               ensure_ascii=False) + "\n")
            if i % 200 == 0:
                print(i, f"{time.time() - t0:.0f} s", flush=True)
    run = {"lot": "R7(b) K1 — décodage eval R6 (classe B)", "lignes": len(rows), "secondes": round(time.time() - t0),
           "kraken": importlib.metadata.version("kraken"), "torch": importlib.metadata.version("torch"),
           "python": sys.version.split()[0], "modele_sha256": MODEL_SHA, "items_sha256": ITEMS_SHA,
           "extraits_sha256": {k: h for k, (_, h) in EXTRAITS.items()},
           "script_sha256": sha256(Path(__file__)), "sortie_sha256": sha256(out)}
    if not a.fumee:
        (R / "reader_r7b_k1_eval_run.json").write_text(json.dumps(run, indent=1, ensure_ascii=False))
    print(json.dumps(run, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
