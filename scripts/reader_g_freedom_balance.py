"""Lot G — liberté d'alignement s chez les vraies lignes et les négatifs, tranche A seule (CLASSE A, descriptif).

Demande w2:pA (17/10) : avant tout ajustement, vérifier que positifs et négatifs n'ont pas des
distributions de s très différentes (risque de raccourci « long = vrai »). s = (T_A − besoin(h)) / T_A,
besoin(h) = |y| + r(y) comme scripts/reader_feasibility.py, y = h encodé dans le codec A1 (encoder_for,
NFC avec repli) ; h hors codec : s non défini, compté à part. h = hypothèse du faisceau (vraie ligne,
perturbée) ou texte posé (voisine, autre). T_A = trames A1 de la ligne (perturbée : de son propre
treillis A1). Lignes à D infini (vides, sans kraken, perturbées sans mot) : comptées à part, hors
distribution. Tranche A seule (Random(7), 13 premières pages) : rien de C ni de M. Par groupe : effectif,
quantiles de s aux déciles (numpy, linéaire), et part de chaque groupe dans les déciles de s des vraies
lignes (valeurs hors des bornes des vraies rangées dans le décile extrême). Groupe ajouté (demande pA) : « fausses_lectures » = vraies lignes de A dont la
lecture h a un taux d'erreur D-150 > 0,35 (line_errors d'axel, sur photo_1983 TRAIN : seul usage d'un
taux d'erreur, pour définir ce groupe ; aucun CER rapporté). Aucun décodage. Sortie : reports/reader_g_freedom_balance.json.
"""

from __future__ import annotations

import gzip
import json
import math
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
sys.path.insert(0, str(HERE / "src"))

from axel.reader.metric import line_errors  # noqa: E402
from reader_r1d_joint import TIMING, encoder_for  # noqa: E402

R = HERE / "reports"
MODES = ("blank", "scramble", "blur_strong", "noise_strong")


def t_of(path: Path) -> dict[str, int]:
    out = {}
    for x in gzip.open(path, "rt"):
        r = json.loads(x)
        if "stem" in r and r.get("logp") is not None:
            out[r["stem"]] = len(r["logp"])
    return out


def main() -> None:
    rows = json.loads((R / "reader_gate_calib_2d_rows.json").read_text())
    assert len(rows["true"]) == 2392
    enc = encoder_for(("",) + tuple(json.load(open(TIMING))["chars"]), nfd=False)

    def s_of(h: str, t: int) -> float | None:
        ids: list[int] = []
        for ch in h:
            e = enc(ch)
            if e is None:
                return None
            ids += e
        return (t - (len(ids) + sum(1 for a, b in zip(ids, ids[1:]) if a == b))) / t

    page = lambda s: s.rsplit("__", 1)[0]  # noqa: E731
    pages = sorted({page(r["stem"]) for r in rows["true"]})
    assert len(pages) == 27
    random.Random(7).shuffle(pages)
    in_a = set(pages[:13])
    t_true = t_of(R / "reader_a1_photo1983.jsonl.gz")
    groups: dict[str, list] = {"vraies": [], "fausses_lectures": [], "voisine": [], "autre": [], "perturbee": []}
    for r in rows["true"]:
        if page(r["stem"]) in in_a:
            groups["vraies"].append((r["h"], t_true[r["stem"]], r["D"]))
            e, n = line_errors(r["h"], r["truth"], "d150")
            if n and e / n > 0.35:  # vraie lecture fausse (demande pA) : sous-ensemble des vraies lignes
                groups["fausses_lectures"].append((r["h"], t_true[r["stem"]], r["D"]))
    for k in ("voisine", "autre"):
        for r in rows[k]:
            if page(r["stem"]) in in_a:
                groups[k].append((r["text"], t_true[r["stem"]], r["D"]))
    for m in MODES:
        t_m = t_of(R / f"reader_a1_photo1983_{m}.jsonl.gz")
        for r in rows["perturbed"][m]:
            if page(r["stem"]) in in_a:
                groups["perturbee"].append((r["h"], t_m.get(r["stem"]), r["D"]))
    vals: dict[str, np.ndarray] = {}
    out: dict = {"lot": "G — liberté d'alignement s, tranche A seule (classe A, descriptif)", "groupes": {}}
    for k, g in groups.items():
        finite = [(h, t) for h, t, d in g if not math.isinf(d) and t]
        s = [s_of(h, t) for h, t in finite]
        vals[k] = np.array([x for x in s if x is not None])
        out["groupes"][k] = {"effectif": len(g), "D_infini_hors": len(g) - len(finite),
                             "hors_codec": sum(x is None for x in s), "n_s": int(vals[k].size),
                             "s_deciles_0_10_20_100": [round(float(np.quantile(vals[k], p)), 4)
                                                       for p in np.linspace(0, 1, 11)] if vals[k].size else None}
    edges = np.quantile(vals["vraies"], np.linspace(0, 1, 11))
    for k in groups:
        v = vals[k]
        dec = np.clip(np.searchsorted(edges, v, side="right") - 1, 0, 9)
        out["groupes"][k]["part_par_decile_s_des_vraies_pc"] = [
            round(100 * float((dec == i).sum()) / v.size, 1) if v.size else None for i in range(10)]
    (R / "reader_g_freedom_balance.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
