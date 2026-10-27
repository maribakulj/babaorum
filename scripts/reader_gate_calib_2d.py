"""Lot 2d ÉTAPE 3 (CLASSE B) : copie de scripts/reader_gate_calib_r1e.py (bras B, poids bornés,
échelles gelées) ; SEULS changements : max_len = min(T_A, T_K) au lieu de T_A // 2 + 2, et sorties
*_2d. θ recalculé par la même règle (θ = 0,2 ne se transfère pas). Texte d'origine ci-dessous.

Réglage de θ_D de la porte — VARIANTE R1e étape 2 (bras B) : copie de reader_gate_calib.py
(8267c7fb…) ; seul changement : le faisceau joint décode avec le poids par ligne w_A de la règle
« bounded » du prototype de pR (f800c469…), échelles gelées s_A, s_K ; sorties *_r1e.

Réglage de θ_D de la porte de désaccord croisé (docs/READER_GATE_PROTOCOL.txt v2).

Entrées (photo_1983, jeu de réglage) : treillis A1 et kraken des lignes
vraies, et des mêmes lignes perturbées (4 modes, sous-ensemble). Pour chaque
cas, la sortie h et D = max_i (best_i − log P_i(h)) / |h|, best_i = max de
log P_i sur {8 hypothèses finies du faisceau joint de la ligne, glouton A1,
glouton kraken, h} (définition figée le 15/10 après revue de w2:pM).
Clé de ligne = stem complet corpus__page__line_id (2 392 clés ; line_id seul
n'est pas unique). Ligne sans treillis kraken = abstention, gardée au
dénominateur ; mode de perturbation absent = arrêt. Couloir 8 nats.
- lignes vraies : h = faisceau joint (a 0,6, λ 0,35, β 2, LM press8x) ;
- négatifs « voisine immédiate » : h = vérité de la ligne au-dessus / au-dessous
  (même page, recouvrement horizontal), notée contre les treillis de la ligne ;
- négatifs « autre ligne » : h = vérité d'une ligne tirée au hasard ailleurs ;
- négatifs perturbés : h = faisceau joint sur les treillis perturbés ; compté
  « accepté » s'il contient au moins un mot de ≥ 2 lettres (plus strict que
  D-85, précision ajoutée avant mesure).
Règle : plus GRAND θ (pas 0,1) dont la borne haute Clopper-Pearson 95 % du
taux de négatifs acceptés est ≤ 1 % (tous) et ≤ 2 % (voisines). θ ne dépend
que des négatifs ; le coût en rappel n'est pas mesuré ici (photos).
"""

from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import math
import pickle
import random
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np

AXEL = Path.home() / "axel"
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AXEL / "src"))
sys.path.insert(0, str(HERE / "src"))
sys.path.insert(0, str(HERE / "scripts"))

from axel.image.fusion import EOS, LMCache, lm_score  # noqa: E402
from axel.image.lattice import ctc_logprob  # noqa: E402
from axel.systems.charlm import CharLM  # noqa: E402,F401
from babaorum.reader.ancre import multi_ctc_beam  # noqa: E402
from reader_r1d_joint import encoder_for  # noqa: E402

R = HERE / "reports"
TIMING = AXEL / "runs/e17/kaggle/a1/timing.json"
LM = AXEL / "runs/e2/lm.press8x.pkl"
CFG = {"a": 0.6, "lm": 0.35, "beta": 2.0, "corridor": 8.0}
PROTO = Path.home() / "mathematicien/proto/q3_weights.py"
PROTO_SHA = "f800c469bab71ca98bd33c3ddb7e62f062ec439f7d08decd89ccf8d34fbf6163"
S_A = 4.745910807473918
S_K = 10.500977670092407


def load_proto():
    """Fonction weight / frame_margin du prototype de pR, après contrôle de son sha256."""
    if hashlib.sha256(PROTO.read_bytes()).hexdigest() != PROTO_SHA:
        raise SystemExit("prototype q3_weights.py : sha256 inattendu")
    spec = importlib.util.spec_from_file_location("q3_weights_ro", PROTO)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load(path: Path) -> tuple[dict[str, dict], tuple[str, ...] | None]:
    rows, labels = {}, None
    for line in gzip.open(path, "rt"):
        r = json.loads(line)
        if "labels" in r:
            labels = tuple(r["labels"])
            continue
        rows[r["stem"]] = r
    return rows, labels


def _log_binom_cdf(k: int, n: int, p: float) -> float:
    """log P(X <= k), X ~ B(n, p), sommé en log (pas de débordement)."""
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
    """Borne haute bilatérale à 1 − alpha (comme avant : P(X <= k) = alpha/2),
    par bissection sur log P(X <= k)."""
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


def main() -> None:
    ancre_sha = hashlib.sha256((HERE / "src/babaorum/reader/ancre.py").read_bytes()).hexdigest()
    assert ancre_sha.startswith("9c3da908"), f"ancre.py : sha256 {ancre_sha[:8]}… ≠ 9c3da908"
    a1, _ = load(R / "reader_a1_photo1983.jsonl.gz")
    kr, klabels = load(R / "reader_kraken_photo1983.jsonl.gz")
    labels_a = ("",) + tuple(json.load(open(TIMING))["chars"])
    enc_a = encoder_for(labels_a, nfd=False)
    enc_k = encoder_for(klabels, nfd=True)
    lmc = LMCache(pickle.load(open(LM, "rb")))

    def lm(text: str, c: str | None) -> float:
        hist = list(unicodedata.normalize("NFD", text))
        if c is None:
            return lm_score(lmc, EOS, hist)
        tot = 0.0
        for u in unicodedata.normalize("NFD", c):
            tot += lm_score(lmc, u, hist)
            hist.append(u)
        return tot

    def lp_text(lp: np.ndarray, enc, text: str) -> float:
        ids = []
        for ch in unicodedata.normalize("NFC", text):
            e = enc(ch)
            if e is None:
                return -1e9
            ids += e
        return ctc_logprob_np(lp, ids)

    def ctc_logprob_np(lp: np.ndarray, ids: list[int]) -> float:
        from axel.image.lattice import Lattice

        lat = Lattice(lp.astype(np.float16), np.arange(len(lp), dtype=np.float32), 1.0, ("",) * lp.shape[1])
        return ctc_logprob(lat, ids)

    proto = load_proto()

    def joint(la: np.ndarray, lk: np.ndarray, klab) -> tuple[str, list[str]]:
        # bras B : poids w_A par ligne (règle bounded, échelles gelées ; marges frame_margin)
        m_a, m_k = proto.frame_margin(la)[0], proto.frame_margin(lk)[0]
        w_a = proto.weight("bounded", m_a, m_k, (S_A, S_K))
        cand = {labels_a[i] for i in np.nonzero(la.max(0) > np.log(1e-3))[0] if i}
        cand |= {klab[i] for i in np.nonzero(lk.max(0) > np.log(1e-3))[0] if i and not unicodedata.combining(klab[i][0])}
        cand = sorted(c for c in cand if enc_a(c))
        hs = multi_ctc_beam([(la, w_a, enc_a), (lk, 1 - w_a, enc_k)], cand, lm=lm,
                            lm_weight=CFG["lm"], beta=CFG["beta"], beam=8, corridor=8.0, max_len=min(la.shape[0], lk.shape[0]))
        texts = [h.text for h in hs[:8]]
        return (texts[0] if texts else ""), texts, {"w_A": w_a, "m_A": m_a, "m_K": m_k}

    def D(la, lk, h, cands) -> float:
        """Définition figée (15/10, après revue pM) : pour chaque treillis i,
        best_i = max de log P_i sur C = {8 hypothèses finies du faisceau joint de
        la ligne, glouton A1, glouton kraken, h} ; D = max_i (best_i − log P_i(h)) / |h|."""
        n = max(1, len(h))
        out = []
        for lp, enc in ((la, enc_a), (lk, enc_k)):
            sc = {t: lp_text(lp, enc, t) for t in set(cands) | {h} if t}
            out.append((max(sc.values()) - sc[h]) / n)
        return max(out)

    def has_word(t: str) -> bool:
        return bool(re.search(r"[^\W\d_]{2,}", t or ""))

    def stem_of(it: dict) -> str:
        return it.get("stem") or f"{it['corpus']}__{it['page']}__{it['line_id']}"

    items = {}
    for x in open(R / "reader_items_photo1983.jsonl"):
        it = json.loads(x)
        items[stem_of(it)] = it
    assert len(items) == 2392, f"{len(items)} clés de lignes, 2 392 attendues"
    assert set(a1) == set(items), "treillis A1 et lignes ne coïncident pas"
    by_page: dict[str, list] = {}
    for s_, it in items.items():
        by_page.setdefault(it["page"], []).append((s_, it))
    rng = random.Random(11)
    res = {"true": [], "voisine": [], "autre": [], "perturbed": {}, "excluded_negatives": 0}
    INF = float("inf")
    all_stems = sorted(items)
    for s_ in all_stems:
        it = items[s_]
        kr_ok = s_ in kr and kr[s_]["logp"] is not None
        if not kr_ok:
            # pas de treillis kraken : le système s'abstient (vraie ligne signalée ;
            # tout négatif posé sur cette ligne est rejeté) ; gardé au dénominateur
            res["true"].append({"stem": s_, "h": "", "truth": it["truth"], "D": INF, "no_kraken": True})
            negs_here = INF
        else:
            la, lk = np.asarray(a1[s_]["logp"], float), np.asarray(kr[s_]["logp"], float)
            ga, gk = a1[s_]["hyp"], kr[s_]["hyp"]
            h, beam, info = joint(la, lk, klabels)
            cands = beam + [ga or "", gk or ""]
            res["true"].append({"stem": s_, "h": h, "truth": it["truth"], "D": D(la, lk, h, cands) if h.strip() else INF,
                                **info})
            negs_here = None
        y = (it["box"][1] + it["box"][3]) / 2
        over = [(abs((o["box"][1] + o["box"][3]) / 2 - y), o) for t, o in by_page[it["page"]]
                if t != s_ and min(o["box"][2], it["box"][2]) - max(o["box"][0], it["box"][0]) > 0]
        above = sorted([q for q in over if (q[1]["box"][1] + q[1]["box"][3]) / 2 < y], key=lambda q: q[0])[:1]
        below = sorted([q for q in over if (q[1]["box"][1] + q[1]["box"][3]) / 2 > y], key=lambda q: q[0])[:1]
        other = items[rng.choice(all_stems)]
        for kind, txts in (("voisine", [o["truth"] for _, o in above + below]), ("autre", [other["truth"]])):
            for t in txts:
                if not has_word(t) or t.strip() == it["truth"].strip():
                    res["excluded_negatives"] += 1  # pas un négatif informatif
                    continue
                res[kind].append({"stem": s_, "text": t, "D": negs_here if negs_here is not None else D(la, lk, t, cands)})
    for mode in ("blank", "scramble", "blur_strong", "noise_strong"):
        pa, pk = R / f"reader_a1_photo1983_{mode}.jsonl.gz", R / f"reader_kraken_photo1983_{mode}.jsonl.gz"
        if not (pa.exists() and pk.exists()):
            raise SystemExit(f"mode de perturbation absent : {mode} (aucun saut silencieux)")
        A, _ = load(pa)
        K, kl = load(pk)
        kl = kl or klabels
        out = []
        for s_ in A:
            if s_ not in K or K[s_]["logp"] is None:
                out.append({"stem": s_, "h": "", "word": False, "D": INF, "no_kraken": True})  # rejeté, compté
                continue
            la, lk = np.asarray(A[s_]["logp"], float), np.asarray(K[s_]["logp"], float)
            h, beam, info = joint(la, lk, kl)
            word = has_word(h)
            out.append({"stem": s_, "h": h, "word": word,
                        "D": D(la, lk, h, beam + [A[s_]["hyp"] or "", K[s_]["hyp"] or ""]) if word else INF,
                        **info})
        res["perturbed"][mode] = out
    negs_all = [(r["D"], "voisine") for r in res["voisine"]] + [(r["D"], "autre") for r in res["autre"]]
    negs_all += [(r["D"], "pert") for rs in res["perturbed"].values() for r in rs]
    n_pert = sum(len(rs) for rs in res["perturbed"].values())
    n_all = len(res["voisine"]) + len(res["autre"]) + n_pert
    table = []
    theta = None
    for t10 in range(0, 101):
        th = t10 / 10
        acc_all = sum(d <= th for d, _ in negs_all)
        acc_v = sum(d <= th for d, k in negs_all if k == "voisine")
        ub_all, ub_v = clopper_pearson_upper(acc_all, n_all), clopper_pearson_upper(acc_v, len(res["voisine"]))
        flagged_true = sum(r["D"] > th for r in res["true"])
        table.append({"theta": th, "neg_accepted": acc_all, "neg_n": n_all, "ub_all": ub_all,
                      "voisine_accepted": acc_v, "voisine_n": len(res["voisine"]), "ub_voisine": ub_v,
                      "true_flagged": flagged_true, "true_n": len(res["true"])})
    ok = [r for r in table if r["ub_all"] <= 0.01 and r["ub_voisine"] <= 0.02]
    # θ grand = porte permissive ; on veut le plus grand θ qui tient (le moins de lignes vraies écartées)
    theta = max((r["theta"] for r in ok), default=None)
    assert n_all >= 600, f"{n_all} négatifs, au moins 600 exigés"
    rep = {"theta": theta, "rule": "plus grand θ (pas 0,1) tenant ub ≤ 1 % tous négatifs et ≤ 2 % voisines",
           "D_definition": "max_i (best_i − log P_i(h))/|h|, best_i sur {8 hypothèses finies du faisceau, glouton A1, glouton kraken, h}",
           "table": table, "n_true": len(res["true"]), "n_perturbed": n_pert,
           "n_voisine": len(res["voisine"]), "n_autre": len(res["autre"]), "excluded_negatives": res["excluded_negatives"],
           "true_no_kraken": sum(1 for r in res["true"] if r.get("no_kraken")),
           "regle_poids": "bounded", "echelles": {"S_A": S_A, "S_K": S_K},
           "provenance": {"ancre_py_sha256": ancre_sha,
                          "proto_q3_sha256": hashlib.sha256(PROTO.read_bytes()).hexdigest(),
                          "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
    (R / "reader_gate_calib_2d.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    (R / "reader_gate_calib_2d_rows.json").write_text(json.dumps(res, ensure_ascii=False))
    print(json.dumps({k: v for k, v in rep.items() if k != "table"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
