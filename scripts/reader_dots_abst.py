r"""Lot ABSTENTION (docs/READER_DOTS_ABST_PREREG.txt ; CLASSE B) — choix de τ sur la SÉLECTION, puis mesures sur P3.

RÉSERVE B4 (en tête) : la tête A1 a été choisie sur tout press/dev ; l'éval et les négatifs de P3 sont déjà exposés :
toutes les mesures sur P3 sont des SECONDS REGARDS, pour décider si l'abstention entre dans le dispositif gelé.

Règle : M = Σ_t (1 − exp(logp[t, blanc])) sur le treillis A1 de la ligne ; abstention (sortie vide) si M < τ.
Choix de τ : grille = quantiles 0, 1, …, 100 % (numpy linéaire) des M des négatifs de SÉLECTION (200 lignes
photo_1983 disjointes de pert200, blank et scramble, clé « #r-abst », treillis A1 Mac) ; τ = plus grande valeur
de la grille dont la fausse abstention sur les 400 vraies lignes TUNE est ≤ 0,5 % ET ≤ 2 % sur les TUNE COURTES ;
sinon τ = −∞ (pas d'abstention).
Lignes COURTES : vérité NORM de 1 à 3 caractères, OU seulement chiffres et ponctuation, OU un seul mot.
Compteurs d'invention sur clean : ANCIEN = ≥ 3 mots de 2 lettres ou plus ; NOUVEAU = ≥ 3 jetons, jeton = mot de 2
lettres ou plus OU nombre (\d+).
Mesures (une fois) : négatifs P3 (pert200 ; treillis Mac) blank / scramble / strips : parts inventées (ancien et
nouveau compteur) de libre, guidé λ 0,3 et A1, puis du guidé AVEC abstention, et part abstenue ; critères blank ≤ 5 %
et scramble ≤ 1/10 du libre (nouveau compteur). Éval P3 (2 405) : fausse abstention globale et sur les COURTES
(critères ≤ 1 % et ≤ 5 %) ; CER NORM du guidé avec abstention ; C1 et C2 relus avec abstention (descriptif, IC
apparié par ligne comme en P3).
Environnement : PYTHONPATH=~/axel/src:src ~/outils-seg/gate-env/bin/python scripts/reader_dots_abst.py \
    --p3-sortie <dossier rapatrié de P3> --p3-entrees <entrées P3>
Sortie : reports/reader_dots_abst.json.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import math
import re

from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
R = HERE / "reports"
A1_DEV = Path.home() / "axel/runs/e17/kaggle/a1/dev_pred_a1.jsonl.gz"
P3_MANIFEST = "9b875a049c7f45aab85d0e68685114b6eb96f37894ef8e5495c1410110326693"
_spec = importlib.util.spec_from_file_location("p3r", HERE / "scripts/reader_dots_p3_report.py")
p3r = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p3r)
norm, paired = p3r.norm, p3r.paired
OLD = re.compile(r"[^\W\d_]{2,}")
NEW = re.compile(r"[^\W\d_]{2,}|\d+")


def mass(logp) -> float:
    lp = np.asarray(logp, dtype=np.float64)
    return float(np.sum(1.0 - np.exp(lp[:, 0])))


def masses(path: Path, keys: set[str] | None = None, suffix: str = "") -> dict[str, float]:
    out = {}
    for x in gzip.open(path, "rt"):
        r = json.loads(x)
        s = r.get("stem")
        if s and r.get("logp") is not None and (keys is None or s + suffix in keys):
            out[s + suffix] = mass(r["logp"])
    return out


def short(truth: str) -> bool:
    t = norm(truth)
    return 1 <= len(t) <= 3 or (bool(t) and not re.search(r"[^\W\d_]", t)) or (bool(t) and " " not in t)


def _log_binom_cdf(k: int, n: int, p: float) -> float:
    """log P(X ≤ k), X ~ B(n, p) ; recopié de scripts/reader_gate_calib_r1e.py (sans scipy)."""
    if p <= 0:
        return 0.0
    if p >= 1:
        return -math.inf if k < n else 0.0
    lp, lq = math.log(p), math.log1p(-p)
    lg = math.lgamma(n + 1)
    terms = [lg - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp + (n - i) * lq for i in range(k + 1)]
    m = max(terms)
    return m + math.log(sum(math.exp(t - m) for t in terms))


def cp(k: int, n: int) -> list[float | None]:
    """IC 95 % de Clopper-Pearson d'une proportion k/n (bilatéral), par dichotomie sur la loi binomiale
    (borne haute : P(X ≤ k) = 0,025 ; borne basse : P(X ≥ k) = 0,025, soit P(X ≤ k − 1) = 0,975)."""
    if n == 0:
        return [None, None]
    target_hi, target_lo = math.log(0.025), math.log(0.975)
    hi = 1.0
    if k < n:
        a, b = k / n, 1.0
        for _ in range(80):
            mid = (a + b) / 2
            a, b = (mid, b) if _log_binom_cdf(k, n, mid) > target_hi else (a, mid)
        hi = b
    lo = 0.0
    if k > 0:
        a, b = 0.0, k / n
        for _ in range(80):
            mid = (a + b) / 2
            a, b = (mid, b) if _log_binom_cdf(k - 1, n, mid) > target_lo else (a, mid)
        lo = a
    return [lo, hi]


def rate(k: int, n: int, limit: float) -> dict:
    """Taux, effectif, IC et verdict (seulement si n ≥ 100 : précision pA (4))."""
    return {"k": k, "n": n, "taux": k / n if n else None, "ic95": cp(k, n),
            "verdict": (k / n <= limit) if n >= 100 else "effectif < 100 : pas de verdict"}


def invented(text: str, rx) -> bool:
    return len(rx.findall(text or "")) >= 3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p3-sortie", required=True)
    ap.add_argument("--p3-entrees", required=True)
    a = ap.parse_args()
    S, E = Path(a.p3_sortie), Path(a.p3_entrees)
    assert hashlib.sha256((E / "MANIFEST.json").read_bytes()).hexdigest() == P3_MANIFEST

    # sélection : TUNE (coût) + négatifs abst200 (détection)
    tune = {r["stem"]: r["truth"] for r in json.loads((R / "reader_r1e_tune.json").read_text())["rows"]}
    assert len(tune) == 400
    m_tune = masses(A1_DEV, set(tune))
    assert set(m_tune) == set(tune)
    sel = {}
    for mode in ("blank", "scramble"):
        sel |= {f"{k}#{mode}": v for k, v in masses(R / f"reader_a1_photo1983_abst_{mode}.jsonl.gz").items()}
    assert len(sel) == 400, len(sel)
    pert = {json.loads(x).get("stem") or "__".join(json.loads(x)[k] for k in ("corpus", "page", "line_id"))
            for x in open(R / "reader_items_photo1983_pert200.jsonl")}
    assert not {k.split("#")[0] for k in sel} & pert, "sélection non disjointe de P3"
    grid = np.quantile(np.array(list(sel.values())), np.linspace(0, 1, 101), method="linear")
    t_short = [s for s in tune if short(tune[s])]
    tau = -math.inf
    for t in grid:
        fa = sum(m_tune[s] < t for s in tune) / len(tune)
        fs = sum(m_tune[s] < t for s in t_short) / max(1, len(t_short))
        if fa <= 0.005 and fs <= 0.02:
            tau = float(t)
    out = {"reserve_B4_en_tete": "tête A1 choisie sur tout press/dev ; mesures P3 = SECOND REGARD",
           "tau": tau, "selection": {
               "tune_lignes": 400, "tune_courtes": len(t_short),
               "fausse_abstention_tune": rate(sum(m_tune[s] < tau for s in tune), 400, 0.005),
               "fausse_abstention_tune_courtes": rate(sum(m_tune[s] < tau for s in t_short), len(t_short), 0.02),
               "abstention_negatifs_selection": {m: sum(v < tau for k, v in sel.items() if k.endswith(m)) / 200
                                                 for m in ("#blank", "#scramble")},
               "M_quantiles_tune_5_50": [float(np.quantile(list(m_tune.values()), q)) for q in (0.05, 0.5)],
               "M_quantiles_negatifs_50_95": [float(np.quantile(list(sel.values()), q)) for q in (0.5, 0.95)]}}

    # P3 : sorties existantes, abstention a posteriori
    items = {it["stem"]: it for it in map(json.loads, open(E / "items.jsonl"))}
    rows: dict[str, dict] = {}
    for x in open(S / "dots_p1_eval.jsonl"):
        r = json.loads(x)
        rows.setdefault(r["config"], {})[r["stem"]] = r
    ev = [s for s, it in items.items() if it["part"] == "eval"]
    m_ev = masses(A1_DEV, set(ev))
    assert set(m_ev) == set(ev)
    neg = [s for s, it in items.items() if it["part"] == "neg"]
    m_neg = {}
    for mode in ("scramble", "blank", "strips"):
        m_neg |= masses(R / f"reader_a1_photo1983_{mode}.jsonl.gz", set(neg), f"#{mode}")
    assert set(m_neg) == set(neg)
    a1_neg = {}
    for mode in ("scramble", "blank", "strips"):
        for x in gzip.open(R / f"reader_a1_photo1983_{mode}.jsonl.gz", "rt"):
            r = json.loads(x)
            if f"{r.get('stem')}#{mode}" in items:
                a1_neg[f"{r['stem']}#{mode}"] = r.get("hyp") or ""
    res_neg = {}
    for mode in ("blank", "scramble", "strips"):
        ss = sorted(s for s in neg if items[s]["mode"] == mode)
        texts = {"libre": {s: rows["libre"][s]["clean"] for s in ss},
                 "guide_l0.3": {s: rows["guide_l0.3"][s]["clean"] for s in ss},
                 "A1": {s: a1_neg[s] for s in ss}}
        texts["guide_l0.3_abst"] = {s: ("" if m_neg[s] < tau else texts["guide_l0.3"][s]) for s in ss}
        res_neg[mode] = {c: {"ancien": sum(invented(t[s], OLD) for s in ss) / len(ss),
                             "nouveau": sum(invented(t[s], NEW) for s in ss) / len(ss)} for c, t in texts.items()}
        res_neg[mode]["part_abstenue"] = sum(m_neg[s] < tau for s in ss) / len(ss)
    out["negatifs_P3"] = res_neg
    libre_new = res_neg["scramble"]["libre"]["nouveau"]
    out["criteres"] = {
        "blank_le_5pc": res_neg["blank"]["guide_l0.3_abst"]["nouveau"] <= 0.05,
        "scramble_le_dixieme_du_libre": res_neg["scramble"]["guide_l0.3_abst"]["nouveau"] <= libre_new / 10 if libre_new else None}
    e_short = [s for s in ev if short(items[s]["truth"])]
    fa = rate(sum(m_ev[s] < tau for s in ev), len(ev), 0.01)
    fs = rate(sum(m_ev[s] < tau for s in e_short), len(e_short), 0.05)
    out["eval"] = {"lignes": len(ev), "courtes": len(e_short), "fausse_abstention": fa, "fausse_abstention_courtes": fs}
    out["criteres"] |= {"fausse_abstention_le_1pc": fa["verdict"], "fausse_abstention_courtes_le_5pc": fs["verdict"]}
    err = {}
    for name, txt in (("libre", lambda s: rows["libre"][s]["clean"]), ("guide_l0.0", lambda s: rows["guide_l0.0"][s]["clean"]),
                      ("guide_l0.3", lambda s: rows["guide_l0.3"][s]["clean"]),
                      ("guide_l0.3_abst", lambda s: "" if m_ev[s] < tau else rows["guide_l0.3"][s]["clean"])):
        err[name] = {}
        for s in ev:
            tn = norm(items[s]["truth"])
            err[name][s] = (p3r.Levenshtein.distance(norm(txt(s)), tn) if tn else 0, len(tn))
    cer = {k: 100 * sum(v[0] for v in d.values()) / sum(v[1] for v in d.values()) for k, d in err.items()}
    out["eval"]["cer_norm_pc"] = cer
    out["eval"]["C1_avec_abstention"] = {"point": cer["libre"] - cer["guide_l0.3_abst"],
                                         "ic95_par_ligne": paired(err, ev, "libre", "guide_l0.3_abst")}
    out["eval"]["C2_avec_abstention"] = {"point": cer["guide_l0.0"] - cer["guide_l0.3_abst"],
                                         "ic95_par_ligne": paired(err, ev, "guide_l0.0", "guide_l0.3_abst")}
    out["statut"] = "SECOND REGARD : décide si l'abstention entre dans le dispositif gelé pour R7(a)"
    (R / "reader_dots_abst.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print("RÉSERVE B4 :", out["reserve_B4_en_tete"])
    print(json.dumps({"tau": tau, "selection": out["selection"], "criteres": out["criteres"],
                      "eval": {k: v for k, v in out["eval"].items() if k != "cer_norm_pc"} | {"cer": cer}},
                     indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
