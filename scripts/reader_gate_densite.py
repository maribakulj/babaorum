"""Négatifs à densité appariée — contrôle d'appariement, puis S1-strat recalibré en validation croisée
(CLASSE B ; EXPLORATOIRE, AUCUNE revendication ; docs/READER_NEG_DENSITE_PREREG.txt).

Entrées (sha256 vérifiés) : reports/reader_gate_calib_2d_rows.json (vraies, voisines, autres, 4 modes
d'origine sur le sous-ensemble de 200 lignes) et reports/reader_gate_dens_rows.json (strips, occlude_donor,
noise_xstrong sur les 2 392 lignes). s comme dans G (scripts/reader_gate_cost_g.py) : (T_A − besoin(h))/T_A,
codec A1 (encoder_for recopié), hors codec → 0, borné à [−1 ; 1] ; T_A = trames A1 de la ligne (vraie,
voisine, autre) ou de son treillis perturbé. D infini : hors des distributions, gardé aux dénominateurs.

Partie 1 — contrôle d'appariement, tranche A du lot G (pages triées, random.Random(7).shuffle, 13
premières) : pour noise_strong et les trois nouveaux modes, contre les vraies de A : médiane de s, part
dans le dernier décile de s des vraies (bornes numpy.quantile linéaire), distance KS deux échantillons
(descriptif). ACCEPTÉ si |médiane(mode) − médiane(vraies)| ≤ 0,03 ET part dans [5 % ; 15 %] (seuils pA).

Partie 2 — S1-strat recalibré, validation croisée sur 3 plis de pages : les 27 pages triées,
random.Random(7).shuffle, plis = positions 0-8, 9-17, 18-26 ; rotation i : A = pli i, C = pli i+1,
M = pli i+2 (modulo 3). Négatifs de la garantie = voisines, autres, noise_strong et les modes ACCEPTÉS en
partie 1 ; blank, scramble, blur_strong et les modes refusés : rapportés par mode, hors de la garantie.
Bornes de s = quartiles de s des vraies du pli A. S1 = D, seuil = plus grand point de la grille tenant sur
C les bornes Clopper-Pearson 95 % ≤ 1 % (négatifs de la garantie) et ≤ 2 % (voisines) ; S1-strat = seuil
de la tranche k = quantile q des D finis des négatifs (garantie) de C de la tranche k, un seul q, le plus
grand tenant les mêmes bornes globales sur C. Grille de quantiles, pour les deux : 0 à 5 % par pas de
0,1 %, puis 6 à 100 % par pas de 1 % (plus fine près du bord, où G s'est arrêté). Mesure sur M une fois :
bornes globales et par type, coût = vraies non vides signalées / vraies non vides, IC de coût(S1) −
coût(S1-strat) par tirage des pages de M (random.Random(1), 10 000 tirages, indices 250 et 9 749).
Rapporté par rotation, puis dispersion (min, max) des trois. RÉSERVE DE PLATEFORME : treillis Mac MPS bf16.
Environnement : PYTHONPATH=~/axel/src ~/outils-seg/gate-env/bin/python scripts/reader_gate_densite.py
Sortie : reports/reader_gate_densite.json (aucun nom de page).
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import random
import sys
import unicodedata
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path.home() / "axel/src"))
from axel.image.lattice import CODEC_FOLD  # noqa: E402

R = HERE / "reports"
TIMING = Path.home() / "axel/runs/e17/kaggle/a1/timing.json"  # recopié de scripts/reader_r1d_joint.py
ROWS_2D = (R / "reader_gate_calib_2d_rows.json", "4dbfa1fd335beac5362b88460aa98e6e313af40998ef71dcc5e68d4c614d6668")
ROWS_DENS = (R / "reader_gate_dens_rows.json", None)  # empreinte écrite après son passage, avant relecture
MODES_2D = ("blank", "scramble", "blur_strong", "noise_strong")
MODES_DENS = ("strips", "occlude_donor", "noise_xstrong")
CONTROLE = ("noise_strong",) + MODES_DENS
INF = math.inf


def encoder_for(labels: tuple[str, ...], nfd: bool):
    """Recopié tel quel de scripts/reader_r1d_joint.py (l'importer charge torch via ancre.py)."""
    index = {c: i for i, c in enumerate(labels) if i}
    cache: dict[str, list[int] | None] = {}

    def enc(ch: str) -> list[int] | None:
        if ch in cache:
            return cache[ch]
        out: list[int] | None = None
        if not nfd and ch in index:
            out = [index[ch]]
        else:
            units = unicodedata.normalize("NFD", ch)
            if all(u in index for u in units):
                out = [index[u] for u in units]
            else:
                fold = CODEC_FOLD.get(ch)
                if fold and all(u in index for u in fold):
                    out = [index[u] for u in fold]
        cache[ch] = out
        return out

    return enc


def _log_binom_cdf(k: int, n: int, p: float) -> float:
    """Recopié de scripts/reader_gate_calib_r1e.py (40a15847)."""
    if p <= 0:
        return 0.0
    if p >= 1:
        return -math.inf if k < n else 0.0
    lp, lq = math.log(p), math.log1p(-p)
    lg = math.lgamma(n + 1)
    terms = [lg - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp + (n - i) * lq for i in range(k + 1)]
    m = max(terms)
    return m + math.log(sum(math.exp(t - m) for t in terms))


def clopper_pearson_upper(k: int, n: int, alpha: float = 0.05) -> float:
    """Recopié de scripts/reader_gate_calib_r1e.py (40a15847)."""
    if n == 0 or k >= n:
        return 1.0
    target = math.log(alpha / 2)
    lo, hi = k / n, 1.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if _log_binom_cdf(k, n, mid) > target:
            lo = mid
        else:
            hi = mid
    return hi


def t_of(path: Path) -> dict[str, int]:
    out = {}
    for x in gzip.open(path, "rt"):
        r = json.loads(x)
        if "stem" in r and r.get("logp") is not None:
            out[r["stem"]] = len(r["logp"])
    return out


def ks(a: np.ndarray, b: np.ndarray) -> float:
    """Distance de Kolmogorov-Smirnov à deux échantillons (sup des écarts des fonctions de répartition)."""
    grid = np.concatenate([a, b])
    fa = np.searchsorted(np.sort(a), grid, side="right") / a.size
    fb = np.searchsorted(np.sort(b), grid, side="right") / b.size
    return float(np.abs(fa - fb).max())


def main() -> None:
    for p, h in (ROWS_2D, ROWS_DENS):
        assert h is not None and hashlib.sha256(p.read_bytes()).hexdigest() == h, f"{p.name} inattendu"
    r2d = json.loads(ROWS_2D[0].read_text())
    rdn = json.loads(ROWS_DENS[0].read_text())
    enc = encoder_for(("",) + tuple(json.load(open(TIMING))["chars"]), nfd=False)

    def s_of(h: str, t: int) -> float:
        ids: list[int] = []
        for ch in h:
            e = enc(ch)
            if e is None:
                return 0.0
            ids += e
        need = len(ids) + sum(1 for a, b in zip(ids, ids[1:]) if a == b)
        return max(-1.0, min(1.0, (t - need) / t))

    page = lambda s: s.rsplit("__", 1)[0]  # noqa: E731
    t_true = t_of(R / "reader_a1_photo1983.jsonl.gz")
    lines = []
    for r in r2d["true"]:
        fin = not math.isinf(r["D"])
        lines.append({"page": page(r["stem"]), "g": "vraie", "D": r["D"], "vide": r["h"] == "",
                      "s": s_of(r["h"], t_true[r["stem"]]) if fin else None})
    for k in ("voisine", "autre"):
        for r in r2d[k]:
            fin = not math.isinf(r["D"])
            lines.append({"page": page(r["stem"]), "g": k, "D": r["D"], "vide": False,
                          "s": s_of(r["text"], t_true[r["stem"]]) if fin else None})
    for src, modes in ((r2d["perturbed"], MODES_2D), (rdn["perturbed"], MODES_DENS)):
        for m in modes:
            t_m = t_of(R / f"reader_a1_photo1983_{m}.jsonl.gz")
            for r in src[m]:
                fin = not math.isinf(r["D"])
                lines.append({"page": page(r["stem"]), "g": m, "D": r["D"], "vide": False,
                              "s": s_of(r["h"], t_m[r["stem"]]) if fin else None})

    pages = sorted({x["page"] for x in lines if x["g"] == "vraie"})
    assert len(pages) == 27
    random.Random(7).shuffle(pages)
    out: dict = {"lot": "Négatifs à densité appariée (classe B, EXPLORATOIRE, aucune revendication)",
                 "reserve_plateforme": "treillis Mac MPS bf16 ; θ et bornes sans garantie sur une autre plateforme"}

    # Partie 1 : contrôle d'appariement sur la tranche A du lot G
    tA = set(pages[:13])
    sv = np.array([x["s"] for x in lines if x["g"] == "vraie" and x["page"] in tA and x["s"] is not None])
    med_v, top = float(np.median(sv)), float(np.quantile(sv, 0.9, method="linear"))
    ctl, accepted = {}, []
    for m in CONTROLE:
        sm = np.array([x["s"] for x in lines if x["g"] == m and x["page"] in tA and x["s"] is not None])
        n_all = sum(1 for x in lines if x["g"] == m and x["page"] in tA)
        if not sm.size:
            ctl[m] = {"n": n_all, "n_s": 0, "accepte": False}
            continue
        part = 100 * float((sm > top).sum()) / sm.size
        ok = abs(float(np.median(sm)) - med_v) <= 0.03 and 5 <= part <= 15
        ctl[m] = {"n": n_all, "n_s": int(sm.size), "mediane_s": float(np.median(sm)), "ecart_mediane": float(np.median(sm)) - med_v,
                  "part_dernier_decile_pc": part, "ks_descriptif": ks(sm, sv), "accepte": ok}
        if ok:
            accepted.append(m)
    out["controle_A"] = {"vraies_n_s": int(sv.size), "mediane_s_vraies": med_v, "bord_dernier_decile": top, "modes": ctl,
                         "modes_acceptes": accepted}

    # Partie 2 : S1-strat recalibré, validation croisée sur 3 plis
    garantie = {"voisine", "autre", "noise_strong", *accepted}
    hors = [m for m in MODES_2D + MODES_DENS if m not in garantie]
    folds = [set(pages[0:9]), set(pages[9:18]), set(pages[18:27])]
    grid = sorted({round(i / 1000, 3) for i in range(0, 51)} | {i / 100 for i in range(6, 101)})

    def counts(ok, sel):
        neg = [x for x in sel if x["g"] in garantie]
        vois = [x for x in neg if x["g"] == "voisine"]
        a, av = sum(ok[id(x)] for x in neg), sum(ok[id(x)] for x in vois)
        par = {}
        for g in ["voisine", "autre", *MODES_2D, *MODES_DENS]:
            sg = [x for x in sel if x["g"] == g]
            k = sum(ok[id(x)] for x in sg)
            par[g] = {"acceptes": k, "n": len(sg), "ub": clopper_pearson_upper(k, len(sg)), "garantie": g in garantie}
        return {"neg_acceptes": a, "neg_n": len(neg), "ub_tous": clopper_pearson_upper(a, len(neg)),
                "voisines_acceptees": av, "voisines_n": len(vois), "ub_voisines": clopper_pearson_upper(av, len(vois)),
                "par_type": par}

    def holds(c):
        return c["ub_tous"] <= 0.01 and c["ub_voisines"] <= 0.02

    rotations = []
    for i in range(3):
        A, C, M = folds[i], folds[(i + 1) % 3], folds[(i + 2) % 3]
        sA = np.array([x["s"] for x in lines if x["g"] == "vraie" and x["page"] in A and x["s"] is not None])
        bornes = tuple(float(np.quantile(sA, q, method="linear")) for q in (0.25, 0.5, 0.75))
        ts = {id(x): (None if x["s"] is None else int(np.searchsorted(bornes, x["s"], side="right"))) for x in lines}
        CL = [x for x in lines if x["page"] in C]
        ML = [x for x in lines if x["page"] in M]
        negC = [x["D"] for x in CL if x["g"] in garantie and not math.isinf(x["D"])]

        thr1 = None  # S1 : seuil commun
        for q in grid:
            t = float(np.quantile(negC, q, method="linear"))
            if holds(counts({id(x): x["D"] <= t for x in lines}, CL)):
                thr1 = t  # grille croissante : le dernier tenu est le plus grand
        negCk = {k: np.array([x["D"] for x in CL if x["g"] in garantie and ts[id(x)] == k and not math.isinf(x["D"])])
                 for k in range(4)}

        def strat(q):
            thr = {k: float(np.quantile(v, q, method="linear")) if v.size else None for k, v in negCk.items()}
            return thr, {id(x): ts[id(x)] is not None and thr[ts[id(x)]] is not None and x["D"] <= thr[ts[id(x)]]
                         for x in lines}

        q_star = None
        for q in grid:
            if holds(counts(strat(q)[1], CL)):
                q_star = q
        res = {"rotation": i, "bornes_s": list(bornes), "S1": {"seuil_C": thr1}, "S1-strat": {"q_C": q_star}}
        fm = {}
        for name, ok in (("S1", {id(x): thr1 is not None and x["D"] <= thr1 for x in lines}),
                         ("S1-strat", strat(q_star)[1] if q_star is not None else {id(x): False for x in lines})):
            if name == "S1-strat" and q_star is not None:
                res[name]["seuils_par_tranche"] = {str(k): v for k, v in strat(q_star)[0].items()}
                res[name]["q_au_bord"] = q_star in (grid[0], grid[-1])
            c = counts(ok, ML)
            c["bornes_tiennent"] = holds(c)
            tv = [x for x in ML if x["g"] == "vraie" and not x["vide"]]
            f = sum(not ok[id(x)] for x in tv)
            c["vraies_non_vides"] = {"signalees": f, "n": len(tv), "cout_pc": 100 * f / len(tv) if tv else None}
            res[name]["M"] = c
            fm[name] = {p: [sum(not ok[id(x)] for x in tv if x["page"] == p), sum(x["page"] == p for x in tv)]
                        for p in sorted(M)}
        pm = sorted(M)
        rng = random.Random(1)
        ds = []
        for _ in range(10000):
            pick = [rng.choice(pm) for _ in pm]

            def cost(nm):
                n = sum(fm[nm][p][1] for p in pick)
                return 100 * sum(fm[nm][p][0] for p in pick) / n if n else 0.0
            ds.append(cost("S1") - cost("S1-strat"))
        ds.sort()
        res["ic_cout_S1_moins_S1strat_points"] = [ds[250], ds[9749]]
        rotations.append(res)
    out["validation_croisee"] = {"negatifs_garantie": sorted(garantie), "hors_garantie": hors, "rotations": rotations,
                                 "dispersion": {
                                     "cout_S1_pc": [min(r["S1"]["M"]["vraies_non_vides"]["cout_pc"] for r in rotations),
                                                    max(r["S1"]["M"]["vraies_non_vides"]["cout_pc"] for r in rotations)],
                                     "cout_S1strat_pc": [min(r["S1-strat"]["M"]["vraies_non_vides"]["cout_pc"] for r in rotations),
                                                         max(r["S1-strat"]["M"]["vraies_non_vides"]["cout_pc"] for r in rotations)],
                                     "ic_bas": [min(r["ic_cout_S1_moins_S1strat_points"][0] for r in rotations),
                                                max(r["ic_cout_S1_moins_S1strat_points"][0] for r in rotations)]},
                                 "revendication": "aucune (exploratoire ; confirmation sur le banc R7(a))"}
    out["provenance"] = {"rows_2d": ROWS_2D[1], "rows_dens": ROWS_DENS[1],
                         "script": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (R / "reader_gate_densite.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
