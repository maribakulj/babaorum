"""Lot 2d — longueurs des lignes par rapport au plafond (CLASSE A, descriptif, sans revendication).

Demande w2:pA (17/10) : montrer si la moitié tune représente les lignes longues, où l'ancien
plafond max_len = T_A // 2 + 2 tronquait. Pour tune (les 400 lignes du panneau R1e), la moitié eval
(lignes A1 de dev_pred dont la page est eval) et photo_1983 (train), par ligne : T_A (trames A1),
len(vérité), rapport len(vérité) / T_A, et « vérité > ancien plafond ». Aucun décodage, aucun CER ;
seules les longueurs de vérité sont lues (eval déjà exposée).
Sortie : reports/reader_lengths.json.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
R = HERE / "reports"
PRED = Path.home() / "axel/runs/e17/kaggle/a1/dev_pred_a1.jsonl.gz"


def rows_of(path: Path, keep) -> list[tuple[int, int]]:
    out = []
    for x in gzip.open(path, "rt"):
        r = json.loads(x)
        if keep(r["stem"]):
            out.append((len(r["logp"]), len(r["truth"])))
    return out


def summary(rows: list[tuple[int, int]]) -> dict:
    t_a = np.array([a for a, _ in rows], float)
    n_t = np.array([b for _, b in rows], float)
    ratio = n_t / t_a
    old_cap = np.array([a // 2 + 2 for a, _ in rows], float)
    q = (0.5, 0.9, 0.99, 1.0)
    return {
        "lignes": len(rows),
        "len_verite_quantiles_50_90_99_max": [float(np.quantile(n_t, p)) for p in q],
        "T_A_quantiles_50_90_99_max": [float(np.quantile(t_a, p)) for p in q],
        "len_verite_sur_T_A_quantiles_50_90_99_max": [round(float(np.quantile(ratio, p)), 4) for p in q],
        "verite_plus_longue_que_ancien_plafond_T_A_div_2_plus_2": int((n_t > old_cap).sum()),
    }


def main() -> None:
    split = json.loads((R / "reader_r3_split.json").read_text())
    eval_pages = set(split["eval"])
    tune_stems = {r["stem"] for r in json.loads((R / "reader_r1e_tune.json").read_text())["rows"]}
    assert len(tune_stems) == 400
    page = lambda s: s.split("__")[1]  # noqa: E731
    tune_pages = set(split["tune"])
    assert all(page(s) in tune_pages and page(s) not in eval_pages for s in tune_stems), "tige hors moitié tune"
    tune = rows_of(PRED, lambda s: s in tune_stems)
    evl = rows_of(PRED, lambda s: page(s) in eval_pages)
    photo = rows_of(R / "reader_a1_photo1983.jsonl.gz", lambda s: True)
    assert (len(tune), len(evl), len(photo)) == (400, 2405, 2392), (len(tune), len(evl), len(photo))
    out = {
        "lot": "2d — longueurs (classe A, descriptif)",
        "ancien_plafond": "T_A // 2 + 2 (trames A1)",
        "tune_panneau_400": summary(tune),
        "eval": summary(evl),
        "photo_1983_train": summary(photo),
    }
    (R / "reader_lengths.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
