"""Lot 2d étape 2 — CER du bras B sur les 400 stems tune, plafond min(T_A, T_K) (classe B, descriptif).

Panneau : les 400 stems de reports/reader_r1e_tune.json (sha256 3e754b90…), un seul passage.
Décodeur : multi_ctc_beam d'ancre.py 9c3da908 (inchangé), bras B (poids par ligne « bounded »,
échelles gelées), réglages gelés (λ 0,35, β 2, couloir 8, faisceau 8, press8x), et
max_len = min(T_A, T_K) (et non T_A // 2 + 2 comme au passage d'étape 1).
Lecture et décodage repris de scripts/reader_r1e_tune.py (importé, non modifié).
Sortie : reports/reader_lot2d_tune.json. Descriptif : aucune revendication.

    ~/axel/.venv/bin/python scripts/reader_lot2d_tune.py
"""

from __future__ import annotations

import gzip
import json
import pickle
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
sys.path.insert(0, str(HERE / "src"))

from axel.image.fusion import EOS, LMCache, lm_score  # noqa: E402
from axel.image.lattice import load as load_lat  # noqa: E402
from axel.reader.metric import line_errors, score  # noqa: E402
from babaorum.reader.ancre import _multi_full, _multi_witness, multi_ctc_beam  # noqa: E402
from reader_dvides import witness_texts  # noqa: E402  (t* comme au diagnostic D-vides)
import reader_r1d_joint as r1d  # noqa: E402
from reader_r1e_tune import S_A, S_K, load_proto, sha256  # noqa: E402

PANEL = HERE / "reports/reader_r1e_tune.json"
SPLIT = HERE / "reports/reader_r3_split.json"
PANEL_SHA = "3e754b90ab1bab2ff7eb083c16874fc9e9d684d0f6d98da773eaca154b806fba"
ANCRE_SHA = "9c3da908"
R1E_B_CER = 1.0444398935080892
CFG = {"lm": 0.35, "beta": 2.0, "beam": 8, "corridor": 8.0}


def main() -> None:
    assert sha256(PANEL) == PANEL_SHA, "reader_r1e_tune.json : sha256 inattendu"
    assert sha256(HERE / "src/babaorum/reader/ancre.py")[:8] == ANCRE_SHA, "ancre.py modifié"
    proto = load_proto()
    panel = json.loads(PANEL.read_text())
    prow = {r["stem"]: r for r in panel["rows"]}
    assert len(prow) == 400, f"{len(prow)} stems, 400 attendus"
    split = json.loads(SPLIT.read_text())
    tune_pages, eval_pages = set(split["tune"]), set(split["eval"])
    for stem in prow:  # refus de toute tige d'une page eval
        page = stem.split("__")[1]
        assert page in tune_pages and page not in eval_pages, f"tige hors moitié tune : {stem}"

    labels = ("",) + tuple(json.load(open(r1d.TIMING))["chars"])
    enc_a = r1d.encoder_for(labels, nfd=False)
    lmc = LMCache(pickle.load(open(r1d.LM, "rb")))

    def lm(text: str, c: str | None) -> float:
        hist = list(unicodedata.normalize("NFD", text))
        if c is None:
            return lm_score(lmc, EOS, hist)
        tot = 0.0
        for u in unicodedata.normalize("NFD", c):
            tot += lm_score(lmc, u, hist)
            hist.append(u)
        return tot

    # lignes A1 du panneau (même fichier source que reader_r1d_joint.py)
    src = {}
    for x in gzip.open(r1d.PRED, "rt"):
        r = json.loads(x)
        if r["stem"] in prow:
            src[r["stem"]] = r
    assert len(src) == 400, "stems du panneau absents des prédictions A1"

    t0 = time.time()
    out_rows = []
    for stem, pr in prow.items():
        r = src[stem]
        assert r["truth"] == pr["truth"], f"vérité différente pour {stem}"
        la = np.asarray(r["logp"], dtype=np.float64)
        le_lat = load_lat(r1d.E8LAT / f"{stem}.npz")
        le = le_lat.logprobs.astype(np.float64)
        enc_e = r1d.encoder_for(tuple(le_lat.labels), nfd=True)
        m_a, m_k = proto.frame_margin(la)[0], proto.frame_margin(le)[0]
        w_b = proto.weight("bounded", m_a, m_k, (S_A, S_K))
        assert abs(w_b - pr["w_B"]) < 1e-12, f"poids différent pour {stem}"
        cand = {labels[i] for i in np.nonzero(la.max(0) > np.log(1e-3))[0] if i}
        cand |= {le_lat.labels[i] for i in np.nonzero(le.max(0) > np.log(1e-3))[0]
                 if i and not unicodedata.combining(le_lat.labels[i][0])}
        cand = sorted(c for c in cand if enc_a(c))
        max_len = min(la.shape[0], le.shape[0])
        lattices = [(la, w_b, enc_a), (le, 1 - w_b, enc_e)]
        L = _multi_witness(lattices, cand)  # témoin initial (glouton replié)
        # t* : chaîne qui atteint L (vide, ou glouton exprimable), comme reader_dvides.py
        vals = {t: v for t in witness_texts(lattices, cand) if (v := _multi_full(lattices, t)) is not None}
        attain = sorted(t for t, v in vals.items() if abs(v - L) <= 1e-9)
        t_star = "" if "" in attain else attain[0]
        assert len(t_star) <= max_len, f"len(t*) {len(t_star)} > max_len {max_len} pour {stem} : arrêt"
        hs = multi_ctc_beam(lattices, cand, lm=lm, lm_weight=CFG["lm"],
                            beta=CFG["beta"], beam=CFG["beam"], corridor=CFG["corridor"], max_len=max_len)
        hyp = hs[0].text if hs else ""
        # le témoin ne fait que monter : la meilleure fin admise n'est jamais retirée par le filtre
        best_admis = max(h.ctc for h in hs) if hs else None
        ref_finale = max(L, best_admis) if best_admis is not None else L
        out_rows.append({"stem": stem, "truth": pr["truth"], "hyp": hyp, "abstention": not hs,
                         "max_len": max_len, "len_truth": len(pr["truth"]),
                         "hyp_r1e_B": pr["B"]["hyp"],
                         "L_temoin": L, "t_star_len": len(t_star),
                         "meilleur_score_complet_fin_admise": best_admis,
                         "reference_finale": ref_finale})
    seconds = time.time() - t0

    pairs = [(o["hyp"], o["truth"]) for o in out_rows]
    sc = score(pairs)
    improved = degraded = same = 0
    for o in out_rows:
        eN, nN = line_errors(o["hyp"], o["truth"])
        eO, _ = line_errors(o["hyp_r1e_B"], o["truth"])
        if nN == 0:
            continue
        if eN < eO:
            improved += 1
        elif eN > eO:
            degraded += 1
        else:
            same += 1
    # le passage d'étape 1 doit être reproduit par ses propres sorties (contrôle de cohérence)
    cer_r1e = score([(o["hyp_r1e_B"], o["truth"]) for o in out_rows])["cer_d150"]
    assert abs(cer_r1e - R1E_B_CER) < 1e-9, f"CER r1e recalculé {cer_r1e} ≠ {R1E_B_CER}"
    res = {
        "lot": "2d etape 2 (tune, bras B, plafond min(T_A, T_K)) — descriptif, sans revendication",
        "n": len(out_rows),
        "cer_d150": sc["cer_d150"],
        "difference_cer_nouveau_moins_ancien": sc["cer_d150"] - R1E_B_CER,
        "cer_raw": sc["cer_raw"],
        "lignes_hypothese_vide": sum(o["hyp"] == "" for o in out_rows),
        "abstentions": sum(o["abstention"] for o in out_rows),
        "verites_vides_hors_cer": sc["empty_truth_lines_d150"],
        "cer_d150_etape1_reference": R1E_B_CER,
        "cer_d150_etape1_recalcule": cer_r1e,
        "ameliorees_vs_etape1": improved,
        "degradees_vs_etape1": degraded,
        "egales_vs_etape1": same,
        "lignes_truth_plus_longue_que_max_len": sum(o["len_truth"] > o["max_len"] for o in out_rows),
        "secondes": round(seconds, 1),
        "lignes": out_rows,
        "provenance": {"reader_lot2d_tune_py_sha256": sha256(Path(__file__)),
                       "panel_sha256": PANEL_SHA,
                       "ancre_py_sha256": sha256(HERE / "src/babaorum/reader/ancre.py"),
                       "proto_q3_sha256": sha256(Path.home() / "mathematicien/proto/q3_weights.py")},
    }
    (HERE / "reports/reader_lot2d_tune.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "lignes"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
