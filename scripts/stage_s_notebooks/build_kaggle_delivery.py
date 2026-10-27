"""Carnet de LIVRAISON DATA-2 (docs/END_TO_END_DATA2_LOT.txt §18, accepté par pA le 24/10/2026). pC, GEL 9.

Une seule source : mabaku/babaorum-data2-fragments (kernel_sources), aucun dataset, internet coupé.
La cellule embarque data2_text_guard.py (bibliothèque standard seulement) et les constantes de build (gabarit LISEZMOI,
mention). Elle copie FRAG_OUT seulement, relance les contrôles AVANT toute copie, et n'imprime que des statuts,
des comptes, des sha et des identifiants.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DELIVERY_SLUG = "mabaku/babaorum-data2-delivery"
DELIVERY_SOURCE = "mabaku/babaorum-data2-fragments"
DELIVERY_CANDIDATES = [  # §22 (2) : deux formats de montage de sortie de carnet, avec et sans frag_out
    "/kaggle/input/babaorum-data2-fragments",
    "/kaggle/input/babaorum-data2-fragments/frag_out",
    "/kaggle/input/notebooks/mabaku/babaorum-data2-fragments",
    "/kaggle/input/notebooks/mabaku/babaorum-data2-fragments/frag_out",
]
DELIVERY_OUT = "/kaggle/working/delivery"

DELIVERY_BODY = r'''
import hashlib, shutil

INPUT_CANDIDATES = __CANDIDATES__
OUT_DIR = Path(__OUT__)
README_TEMPLATE = __TEMPLATE__
README_CFG = __README_CFG__
OVERLAP_MENTION = __MENTION__
ARMS = ("B3", "S2")

def _emit(status):
    # §19 (2) : stdout ne contient que le statut ; en cas de succès, des comptes et des sha (voir run)
    print(status, flush=True)

def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def run():
    hits = [Path(c) for c in INPUT_CANDIDATES if (Path(c) / "data1_manifest.json").is_file()]
    if len(hits) != 1:
        _emit("BLOCKED_NOT_FOUND")
        return
    src = hits[0]
    try:
        upstream = _read_json(src / "data2_notebook_manifest.json")
        manifest = _read_json(src / "data1_manifest.json")
    except Exception:
        _emit("BLOCKED_UPSTREAM")
        return
    # §20 (2) : rien n'est livré sauf si exit_status == DONE et status == "OK" ; SMOKE n'est jamais livrable
    if upstream.get("exit_status") != "DONE" or manifest.get("status") != "OK":
        _emit("BLOCKED_UPSTREAM")
        return
    pages = manifest["pages_fragments"]
    payload_rel = [f"fragments/{arm}/{pid}.json" for arm in ARMS for pid in pages]
    if not all((src / rel).is_file() for rel in payload_rel + ["LISEZMOI.txt"]):
        _emit("BLOCKED_UPSTREAM")
        return
    try:
        payloads = [_read_json(src / rel) for rel in payload_rel]
    except Exception:
        _emit("BLOCKED_UPSTREAM")
        return
    accepted = (src / "LISEZMOI_DATA2.txt").is_file()
    problems = check_all(payloads, manifest)
    if not readme_matches((src / "LISEZMOI.txt").read_text(encoding="utf-8"), manifest, README_TEMPLATE, README_CFG):
        problems.append("readme")
    if accepted and not exact_text_matches((src / "LISEZMOI_DATA2.txt").read_text(encoding="utf-8"),
                                           OVERLAP_MENTION + "\n"):
        problems.append("mention")
    extra = unexpected_output(src, expected_output(manifest, accepted))
    if problems or extra:
        _emit("BLOCKED_TEXT_FIELD")
        return
    listed = ["data1_manifest.json", "LISEZMOI.txt"] + (["LISEZMOI_DATA2.txt"] if accepted else []) + payload_rel
    try:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        for rel in listed:
            dst = OUT_DIR / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src / rel, dst)
        if any(_sha(src / rel) != _sha(OUT_DIR / rel) for rel in listed):
            _emit("BLOCKED_COPY")
            return
    except OSError:
        _emit("BLOCKED_COPY")
        return
    print("FICHIERS", len(listed), flush=True)
    print("PAGES", len(pages), flush=True)
    print("MANIFESTE_SHA", _sha(OUT_DIR / "data1_manifest.json"), flush=True)
    _emit("OK")

try:
    run()
except Exception:
    _emit("ECHEC")
'''


def build_delivery_cell(candidates=None, out_dir: str = DELIVERY_OUT) -> str:
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    import data2_text_guard as G
    cfg_e2e4 = json.loads((ROOT / "config" / "e2e4_extraction.json").read_text())
    readme_cfg = {"scale": {"target_median_height_px": cfg_e2e4["scale"]["target_median_height_px"]},
                  "extraction": {"stride": cfg_e2e4["extraction"]["stride"]},
                  "model": {"K": cfg_e2e4["model"]["K"]}}
    template = G.template_from_source((ROOT / "scripts" / "data1_fragments.py").read_text())
    mention = json.loads((ROOT / "config" / "data2_pages.json").read_text())["overlap_text_accepted_real"]["mention"]
    guard_src = (ROOT / "scripts" / "data2_text_guard.py").read_text(encoding="utf-8")
    body = (DELIVERY_BODY.replace("__CANDIDATES__", repr(list(candidates or DELIVERY_CANDIDATES)))
            .replace("__OUT__", repr(out_dir)).replace("__TEMPLATE__", repr(template))
            .replace("__README_CFG__", repr(readme_cfg)).replace("__MENTION__", repr(mention)))
    return ("# --- embarqué : data2_text_guard.py (bibliothèque standard seulement) ---\n" + guard_src +
            "\n# --- livraison (§18) ---\nfrom pathlib import Path\n" + body)


def delivery_metadata() -> dict:
    return {"id": DELIVERY_SLUG, "title": "babaorum-data2-delivery", "code_file": "kaggle_data2_delivery.ipynb",
            "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": False,
            "enable_internet": False, "dataset_sources": [], "competition_sources": [],
            "kernel_sources": [DELIVERY_SOURCE], "model_sources": []}


def delivery_notebook() -> dict:
    return {"nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}},
            "cells": [{"cell_type": "markdown", "metadata": {}, "source":
                       "# DATA-2 — LIVRAISON : copie de la sortie des fragments, contrôles relancés, sortie réduite"},
                      {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                       "source": build_delivery_cell()}]}


def write_delivery(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "kernel-metadata.json").write_text(json.dumps(delivery_metadata(), indent=2))
    (out_dir / "kaggle_data2_delivery.ipynb").write_text(json.dumps(delivery_notebook(), indent=1, ensure_ascii=False))


def main() -> None:
    write_delivery(Path(__file__).parent / "delivery")
    print(Path(__file__).parent / "delivery")


if __name__ == "__main__":
    main()
