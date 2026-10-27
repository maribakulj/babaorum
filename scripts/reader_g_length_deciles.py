"""Lot G — coût de la porte selon len(vérité)/T_A, en déciles (CLASSE A, descriptif, sans revendication).

Demande w2:pA (17/10) : les signalements se concentrent-ils sur les lignes longues ? Si oui, une variable
de longueur dans S2 s'écrit AVANT l'ajustement. Pour ne rien voir des tranches C et M de G, le compte se
fait sur la tranche A SEULE (13 pages ; pages = stem sans le suffixe de ligne, triées puis
random.Random(7).shuffle, comme docs/READER_GATE_COST_PREREG.txt). Vraies lignes seulement.
Signalée = D > θ, θ = 0,1 de l'étape 3 (reports/reader_gate_calib_2d.json), D infini compris.
Déciles de len(vérité)/T_A (numpy.quantile, linéaire) calculés sur A ; par décile : lignes, signalées,
dont hypothèse vide, coût en %. Aucun décodage, aucun CER.
Sortie : reports/reader_g_length_deciles.json.
"""

from __future__ import annotations

import gzip
import json
import random
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
R = HERE / "reports"


def main() -> None:
    rows = json.loads((R / "reader_gate_calib_2d_rows.json").read_text())["true"]
    theta = json.loads((R / "reader_gate_calib_2d.json").read_text())["theta"]
    assert len(rows) == 2392 and theta == 0.1, (len(rows), theta)
    t_a = {}
    for x in gzip.open(R / "reader_a1_photo1983.jsonl.gz", "rt"):
        r = json.loads(x)
        if "stem" in r:
            t_a[r["stem"]] = len(r["logp"])
    page = lambda s: s.rsplit("__", 1)[0]  # noqa: E731
    pages = sorted({page(r["stem"]) for r in rows})
    assert len(pages) == 27
    random.Random(7).shuffle(pages)
    tranche_a = set(pages[:13])
    a = [r for r in rows if page(r["stem"]) in tranche_a]
    ratio = np.array([len(r["truth"]) / t_a[r["stem"]] for r in a])
    edges = np.quantile(ratio, np.linspace(0, 1, 11), method="linear")
    dec = np.clip(np.searchsorted(edges, ratio, side="right") - 1, 0, 9)
    out = []
    for k in range(10):
        sel = [r for r, d in zip(a, dec) if d == k]
        flagged = [r for r in sel if r["D"] > theta]
        out.append({"decile": k + 1, "bornes_len_sur_T_A": [round(float(edges[k]), 4), round(float(edges[k + 1]), 4)],
                    "lignes": len(sel), "signalees": len(flagged),
                    "signalees_hypothese_vide": sum(r["h"] == "" for r in flagged),
                    "cout_pc": round(100 * len(flagged) / len(sel), 2) if sel else None})
    res = {"lot": "G — coût selon len(vérité)/T_A, tranche A seule (classe A, descriptif)",
           "theta": theta, "pages_A": len(tranche_a), "lignes_A": len(a),
           "signalees_A": sum(r["D"] > theta for r in a), "deciles": out}
    (R / "reader_g_length_deciles.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
