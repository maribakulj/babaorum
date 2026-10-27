"""Résolveur de montages du carnet LD-1 (copie de build_kaggle_ld1.resolve_one, testée sans Kaggle)."""
from pathlib import Path


def resolve(candidates, marker):
    hits = [Path(c) for c in candidates if (Path(c) / marker).is_file()]
    if len(hits) != 1:
        raise RuntimeError(f"montage : {len(hits)} candidat(s) pour {marker}")
    return hits[0]
