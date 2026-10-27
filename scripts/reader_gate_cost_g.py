"""Lot G — coût de la porte de désaccord (CLASSE B ; EXPLORATOIRE ; docs/READER_GATE_COST_PREREG.txt,
« Version finale de G », « Précisions techniques », entrées 2d et variable s). Feu vert pM requis.

Données : reports/reader_gate_calib_2d_rows.json seulement (photo_1983, TRAIN). Pages = stem sans le
suffixe de ligne, triées, random.Random(7).shuffle ; 13 premières = A (ajustement de S2), 7 suivantes =
C (un seuil par score), 7 dernières = M (mesure, une fois).
Scores (plus haut = plus suspect ; « acceptée » si score ≤ seuil ; score infini = jamais acceptée) :
  S1 = D ; S2 = P(négatif) d'une LogisticRegression(penalty="l2", solver="lbfgs", C=1.0, class_weight="balanced",
  max_iter=1000, fit_intercept=True) ajustée sur A, variables (min(D, 10), m_A/S_A, m_K/S_K, s) ;
  S3 = max(D/0,2 ; 1 − min(m_A/S_A, m_K/S_K)).
  S1-strat (ajout pA après le contrôle de s) = D avec un seuil PAR TRANCHE de s ; tranches par les
  quartiles de s des vraies de A, BORNES_S (prereg), tranche = numpy.searchsorted(BORNES_S, s, "right")
  (0 à 3) ; seuil de la tranche k = quantile q (numpy linéaire) des D finis des négatifs de C de la
  tranche k, UN q commun pris sur la grille 0, 1, …, 100 % : le plus grand q tenant les bornes globales
  (≤ 1 % tous négatifs, ≤ 2 % voisines) sur C. (Un seuil tenu tranche par tranche est impossible : sur C,
  200 à 543 négatifs par tranche [non recomptés, pas un résultat], et la borne CP à 1 acceptation ne passe 1 % qu'à partir d'environ 557.)
  Bornes par tranche et par type rapportées, sans condition ; perturbées PAR MODE (blank, scramble,
  blur_strong, noise_strong) ; à côté, en DESCRIPTIF, borne sur les négatifs réalistes (voisines,
  autres, noise_strong). Les rows sont vérifiées (sha256) avant l'ajustement ; les pages de M ne sont
  pas écrites dans la sortie. D infini : hors tranches, toujours rejeté.
Constat attendu (écrit avant C et M) : les négatifs perturbés ont une densité très différente des vraies
lignes (médiane de s 0,852 contre 0,548 sur A) quand voisines et autres sont appariées (0,545 ; 0,552) ;
un seuil commun hérite de ce biais. Bornes de fausse acceptation rapportées PAR TYPE de négatif
(voisine, autre, perturbée PAR MODE) et, sur M, PAR TRANCHE de s pour chaque score ; les tranches basses comptent
peu de perturbées : voisines et autres y portent la garantie, avec des bornes plus larges.
s = (T_A − besoin(h))/T_A borné à [−1 ; 1], besoin = |y| + répétitions consécutives, y = h dans le
codec A1 (encoder_for) ; hors codec : s = 0. h = hypothèse (vraie, perturbée) ou texte posé (voisine,
autre). Voisines/autres : m_A, m_K et T_A de la vraie ligne du même stem ; perturbées : les leurs.
D infini : hors quantiles et hors ajustement, gardé aux dénominateurs (vraie = signalée, négatif =
rejeté).
Seuil (C) : grille numpy.quantile(scores finis des négatifs de C, 0 %, 1 %, …, 100 %, method="linear") ;
plus grand seuil tenant les bornes Clopper-Pearson bilatérales 95 % ≤ 1 % (tous négatifs de C) et
≤ 2 % (voisines de C).
Mesure (M) : seuil appliqué une fois ; coût = vraies signalées, en deux parts (hypothèse vide / non
vide) ; fausse acceptation et bornes sur M.
Critère (M, part non vide) : IC de coût(S1) − coût(Sx) en points, pages de M triées, 10 000 tirages avec
remise, random.Random(1), statistique = 100 × somme des signalées / somme des effectifs sur les pages
tirées, percentiles aux indices 250 et 9 749. « Réduit le coût » seulement si l'IC est entièrement
positif ET si les bornes globales de fausse acceptation de Sx tiennent sur M ; S2 aussi : monotone (coef. de
min(D, 10) > 0, des deux marges < 0 ; s libre) et n_iter_ < 1 000. Sinon aucune revendication.
Environnement DÉDIÉ et figé (décision pA) : ~/outils-seg/gate-env (uv, Python 3.12.13), lancé ainsi :
    PYTHONPATH=~/axel/src ~/outils-seg/gate-env/bin/python scripts/reader_gate_cost_g.py
uv pip freeze : cloudpickle==3.1.2, joblib==1.6.0, numpy==2.5.3, rapidfuzz==3.13.0, scikit-learn==1.7.2,
scipy==1.18.1, threadpoolctl==3.7.0 ; Python 3.12, numpy et scikit-learn vérifiés au démarrage.
encoder_for est recopié de reader_r1d_joint.py (l'importer charge torch) ; CODEC_FOLD vient d'axel.
Aucun décodage, aucun CER. Sortie : reports/reader_gate_cost_g.json. Option --fumee : ajustement de S2
sur A puis arrêt, sans rien calculer sur C ni M.
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
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
sys.path.insert(0, str(HERE / "src"))

from axel.image.lattice import CODEC_FOLD  # noqa: E402

TIMING = Path.home() / "axel/runs/e17/kaggle/a1/timing.json"  # recopié de scripts/reader_r1d_joint.py


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

R = HERE / "reports"
ROWS = R / "reader_gate_calib_2d_rows.json"
S_A, S_K = 4.745910807473918, 10.500977670092407
MODES = ("blank", "scramble", "blur_strong", "noise_strong")
INF = math.inf
BORNES_S = (0.4896, 0.5481, 0.5862)  # quartiles de s des vraies de A (reports/reader_g_freedom_balance, prereg)
ROWS_SHA = "4dbfa1fd335beac5362b88460aa98e6e313af40998ef71dcc5e68d4c614d6668"  # vérifié avant l'ajustement (pM)
TYPES = ("voisine", "autre") + tuple(f"perturbee_{m}" for m in MODES)  # bornes par type, perturbées PAR MODE (pM)
REALISTES = ("voisine", "autre", "perturbee_noise_strong")  # garantie descriptive sur négatifs réalistes (pA)


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


def main() -> None:
    import importlib.metadata as md
    assert sys.version_info[:2] == (3, 12), sys.version
    assert (md.version("numpy"), md.version("scikit-learn")) == ("2.5.3", "1.7.2"), "environnement gate-env attendu"
    assert hashlib.sha256(ROWS.read_bytes()).hexdigest() == ROWS_SHA, "rows 2d inattendues"
    rows = json.loads(ROWS.read_text())
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
    true = {r["stem"]: r for r in rows["true"]}
    assert len(true) == 2392

    # lignes : (tranche page, groupe, D, m_A, m_K, s, vide)
    lines = []
    for r in rows["true"]:
        fin = not math.isinf(r["D"])
        lines.append({"page": page(r["stem"]), "g": "vraie", "D": r["D"], "mA": r.get("m_A"), "mK": r.get("m_K"),
                      "s": s_of(r["h"], t_true[r["stem"]]) if fin else None, "vide": r["h"] == ""})
    for k in ("voisine", "autre"):
        for r in rows[k]:
            t = true[r["stem"]]
            fin = not math.isinf(r["D"])
            lines.append({"page": page(r["stem"]), "g": k, "D": r["D"], "mA": t["m_A"], "mK": t["m_K"],
                          "s": s_of(r["text"], t_true[r["stem"]]) if fin else None, "vide": False})
    for m in MODES:
        t_m = t_of(R / f"reader_a1_photo1983_{m}.jsonl.gz")
        for r in rows["perturbed"][m]:
            fin = not math.isinf(r["D"])
            lines.append({"page": page(r["stem"]), "g": "perturbee", "mode": m, "D": r["D"], "mA": r.get("m_A"),
                          "mK": r.get("m_K"), "s": s_of(r["h"], t_m[r["stem"]]) if fin else None, "vide": False})
    for x in lines:
        if not math.isinf(x["D"]):
            assert x["mA"] is not None and x["mK"] is not None and x["s"] is not None, x

    pages = sorted({x["page"] for x in lines if x["g"] == "vraie"})
    assert len(pages) == 27
    random.Random(7).shuffle(pages)
    tranche = {p: "A" for p in pages[:13]} | {p: "C" for p in pages[13:20]} | {p: "M" for p in pages[20:]}
    for x in lines:
        x["tr"] = tranche[x["page"]]
        x["ts"] = None if x["s"] is None else int(np.searchsorted(BORNES_S, x["s"], side="right"))

    def feats(x):
        return [min(x["D"], 10.0), x["mA"] / S_A, x["mK"] / S_K, x["s"]]

    fit = [x for x in lines if x["tr"] == "A" and not math.isinf(x["D"])]
    X = np.array([feats(x) for x in fit])
    y = np.array([0 if x["g"] == "vraie" else 1 for x in fit])
    clf = LogisticRegression(penalty="l2", solver="lbfgs", C=1.0, class_weight="balanced", max_iter=1000, fit_intercept=True)
    clf.fit(X, y)
    coef = clf.coef_[0].tolist()
    n_iter = int(clf.n_iter_[0])
    monotone = coef[0] > 0 and coef[1] < 0 and coef[2] < 0
    if "--fumee" in sys.argv:  # essai de fumée : tranche A seule, rien n'est calculé sur C ni M
        print(json.dumps({"fumee": "tranche A seule", "lignes_ajustement": len(fit),
                          "coef": coef, "n_iter": n_iter, "monotone": monotone}, indent=1))
        return
    fin_lines = [x for x in lines if not math.isinf(x["D"])]
    p2 = clf.predict_proba(np.array([feats(x) for x in fin_lines]))[:, 1]
    s2 = {id(x): float(v) for x, v in zip(fin_lines, p2)}

    def score(name, x):
        if math.isinf(x["D"]):
            return INF
        if name == "S1":
            return x["D"]
        if name == "S2":
            return s2[id(x)]
        return max(x["D"] / 0.2, 1.0 - min(x["mA"] / S_A, x["mK"] / S_K))

    def kind(x):
        return f"perturbee_{x['mode']}" if x["g"] == "perturbee" else x["g"]

    def counts(ok, sel):
        """ok : id(ligne) -> acceptée. Négatifs de sel : global, voisines, et par type."""
        neg = [x for x in sel if x["g"] != "vraie"]
        vois = [x for x in neg if x["g"] == "voisine"]
        acc = sum(ok[id(x)] for x in neg)
        acc_v = sum(ok[id(x)] for x in vois)
        par_type = {}
        for g in TYPES:
            sg = [x for x in neg if kind(x) == g]
            a = sum(ok[id(x)] for x in sg)
            par_type[g] = {"acceptes": a, "n": len(sg), "ub": clopper_pearson_upper(a, len(sg))}
        sr = [x for x in neg if kind(x) in REALISTES]
        a = sum(ok[id(x)] for x in sr)
        par_type["realistes_descriptif"] = {"acceptes": a, "n": len(sr), "ub": clopper_pearson_upper(a, len(sr))}
        return {"neg_acceptes": acc, "neg_n": len(neg), "ub_tous": clopper_pearson_upper(acc, len(neg)),
                "voisines_acceptees": acc_v, "voisines_n": len(vois),
                "ub_voisines": clopper_pearson_upper(acc_v, len(vois)), "par_type": par_type}

    def accept(sc, thr):
        return {id(x): sc[id(x)] <= thr for x in lines}

    def pick_threshold(sc, sel):
        """Plus grand seuil de la grille des quantiles (négatifs finis de sel) tenant les deux bornes."""
        finite = [sc[id(x)] for x in sel if x["g"] != "vraie" and not math.isinf(sc[id(x)])]
        if not finite:
            return None
        grid = np.quantile(np.array(finite), np.linspace(0, 1, 101), method="linear")
        ok = [float(t) for t in grid if (c := counts(accept(sc, t), sel))["ub_tous"] <= 0.01 and c["ub_voisines"] <= 0.02]
        return max(ok) if ok else None

    C_lines = [x for x in lines if x["tr"] == "C"]
    M_lines = [x for x in lines if x["tr"] == "M"]
    out = {"lot": "G — coût de la porte (classe B, EXPLORATOIRE)", "tranches_pages": {"A": 13, "C": 7, "M": 7},
           "S2": {"coef_minD_mA_mK_s": coef, "intercept": float(clf.intercept_[0]),
           "n_iter": n_iter, "monotone": monotone, "lignes_ajustement": len(fit)}, "scores": {}}
    def measure(ok, res):
        """Mesure sur M avec l'acceptation ok ; remplit res["M"], rend le compte par page (non vides)."""
        res["M"] = counts(ok, M_lines)
        res["M"]["bornes_tiennent"] = res["M"]["ub_tous"] <= 0.01 and res["M"]["ub_voisines"] <= 0.02
        tv = [x for x in M_lines if x["g"] == "vraie"]
        for part, sel in (("vide", [x for x in tv if x["vide"]]), ("non_vide", [x for x in tv if not x["vide"]])):
            f = sum(not ok[id(x)] for x in sel)
            res["M"][f"vraies_{part}"] = {"signalees": f, "n": len(sel), "cout_pc": 100 * f / len(sel) if sel else None}
        par_ts = {}
        for k in (0, 1, 2, 3, None):
            sel = [x for x in M_lines if x["ts"] == k]
            v = [x for x in sel if x["g"] == "vraie"]
            f = sum(not ok[id(x)] for x in v)
            par_ts["D_infini" if k is None else str(k)] = dict(counts(ok, sel), vraies_signalees=f, vraies_n=len(v))
        res["M_par_tranche_s"] = par_ts
        fm = {p: [0, 0] for p in pages[20:]}
        for x in tv:
            if not x["vide"]:
                fm[x["page"]][0] += not ok[id(x)]
                fm[x["page"]][1] += 1
        return fm

    flag_m: dict[str, dict] = {}
    for name in ("S1", "S2", "S3"):
        sc = {id(x): score(name, x) for x in lines}
        thr = pick_threshold(sc, C_lines)
        res = {"seuil_C": thr}
        if thr is not None:
            ok = accept(sc, thr)
            res["C"] = counts(ok, C_lines)
            flag_m[name] = measure(ok, res)
        out["scores"][name] = res

    # S1-strat : D, seuil de la tranche k = quantile q des D finis des négatifs de C de la tranche k, un
    # seul q pour toutes (grille 0, 1, …, 100 %) : le plus grand q tenant les bornes GLOBALES sur C.
    sc = {id(x): score("S1", x) for x in lines}
    negC = {k: np.array([sc[id(x)] for x in C_lines if x["ts"] == k and x["g"] != "vraie"
                         and not math.isinf(sc[id(x)])]) for k in range(4)}

    def strat_ok(q):
        thr = {k: float(np.quantile(v, q, method="linear")) if v.size else None for k, v in negC.items()}
        return thr, {id(x): x["ts"] is not None and thr[x["ts"]] is not None and sc[id(x)] <= thr[x["ts"]]
                     for x in lines}

    q_ok = [q for q in np.linspace(0, 1, 101)
            if (c := counts(strat_ok(q)[1], C_lines))["ub_tous"] <= 0.01 and c["ub_voisines"] <= 0.02]
    q_star = float(max(q_ok)) if q_ok else None
    thr_k, ok = strat_ok(q_star) if q_star is not None else ({k: None for k in range(4)}, {id(x): False for x in lines})
    res = {"q_C": q_star, "seuils_C_par_tranche_s": {str(k): v for k, v in thr_k.items()}, "bornes_s": list(BORNES_S),
           "C": counts(ok, C_lines),
           "C_par_tranche_s": {str(k): counts(ok, [x for x in C_lines if x["ts"] == k]) for k in range(4)}}
    flag_m["S1-strat"] = measure(ok, res)
    out["scores"]["S1-strat"] = res

    pm = sorted(pages[20:])
    for name in ("S2", "S3", "S1-strat"):
        if "S1" not in flag_m or name not in flag_m:
            out["scores"][name]["critere"] = "pas de seuil tenu sur C : aucune revendication"
            continue
        rng = random.Random(1)
        ds = []
        for _ in range(10000):
            pick = [rng.choice(pm) for _ in pm]

            def cost(nm):
                f = sum(flag_m[nm][p][0] for p in pick)
                n = sum(flag_m[nm][p][1] for p in pick)
                return 100 * f / n if n else 0.0
            ds.append(cost("S1") - cost(name))
        ds.sort()
        ci = [ds[250], ds[9749]]
        bounds = out["scores"][name]["M"]["bornes_tiennent"]
        extra = monotone and n_iter < 1000 if name == "S2" else True
        out["scores"][name]["ic_cout_S1_moins_Sx_points"] = ci
        out["scores"][name]["reduit_le_cout"] = bool(ci[0] > 0 and bounds and extra)
    out["provenance"] = {"rows_sha256": hashlib.sha256(ROWS.read_bytes()).hexdigest(),
                         "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (R / "reader_gate_cost_g.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
