"""R1e étape 1 — tune seulement : poids fixe (F, w_A = 0,6) contre poids par ligne B (règle
« bounded » de pR, Q3) avec le décodeur corrigé (ancre.py 9c3da908…).

Lit les 400 stems tune de R1d (mêmes tirages que reader_r1d_joint.py : moitiés de pages
Random(1), échantillon Random(5)), les mêmes treillis, les mêmes réglages gelés
(λ 0,35, β 2, couloir 8, faisceau 8, press8x). Aucune ligne eval.

Poids B : weight('bounded', m_A, m_K, (s_A, s_K)) du prototype de pR, chargé depuis son
fichier après contrôle de son sha256 (f800c469…) ; m_i = frame_margin du treillis i.

    ~/outils-seg/lourd.sh ~/axel/.venv/bin/python scripts/reader_r1e_tune.py --tag _tune > reports/reader_r1e_tune.log 2>&1
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import pickle
import random
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
sys.path.insert(0, str(HERE / "src"))
import reader_r1d_joint as r1d  # noqa: E402  (lecture des treillis, lecture seule)

from axel.image.fusion import EOS, LMCache, lm_score  # noqa: E402
from axel.image.lattice import load as load_lat  # noqa: E402
from axel.reader.metric import line_errors, score  # noqa: E402
from babaorum.reader.ancre import multi_ctc_beam  # noqa: E402

PROTO = Path.home() / "mathematicien/proto/q3_weights.py"
PROTO_SHA = "f800c469bab71ca98bd33c3ddb7e62f062ec439f7d08decd89ccf8d34fbf6163"
ANCRE = HERE / "src/babaorum/reader/ancre.py"
ANCRE_SHA = "9c3da908"
S_A = 4.745910807473918
S_K = 10.500977670092407
PRIOR = 0.6
REF = HERE / "reports/reader_r1d_tune_corr8.json"
R1D_EVAL = HERE / "reports/reader_r1d_eval_eval_c8.json"
R1D_EVAL_CER = 1.002905197074767


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_proto():
    if sha256(PROTO) != PROTO_SHA:
        raise SystemExit("prototype q3_weights.py : sha256 inattendu")
    spec = importlib.util.spec_from_file_location("q3_weights_ro", PROTO)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # le module ne lance rien à l'import (garde __main__)
    return mod


def paired_ci(acc, x, y, pool, draws=10000, seed=1):
    """Copie de scripts/reader_r3_report.py : IC apparié par page, graine 1, indices 250 et 9749."""
    def cer_of(arm, ps):
        return 100 * sum(acc[arm][p][0] for p in ps) / max(1, sum(acc[arm][p][1] for p in ps))
    rng = random.Random(seed)
    ds = []
    for _ in range(draws):
        ps = [rng.choice(pool) for _ in pool]
        ds.append(cer_of(x, ps) - cer_of(y, ps))
    ds.sort()
    return [ds[int(0.025 * draws)], ds[int(0.975 * draws) - 1]]


def eval_contrasts(out_rows) -> dict:
    """Contrastes eval : bras − R1d (IC apparié par page contre R1d), CER(B) − CER(F) descriptif,
    BnL et NewsEye séparés ; abstentions et vérités vides rapportées à part."""
    r1d = {r["stem"]: r["joint"] for r in json.loads(R1D_EVAL.read_text())["rows"]}
    acc = {"F": {}, "B": {}, "R1D": {}}
    for o in out_rows:
        page = o["stem"].split("__")[1]
        for arm, hyp in (("F", o["F"]["hyp"]), ("B", o["B"]["hyp"]), ("R1D", r1d[o["stem"]])):
            e, n = line_errors(hyp, o["truth"])
            v = acc[arm].setdefault(page, [0, 0])
            v[0] += e
            v[1] += n
    pages = sorted(acc["R1D"])
    strata = {"bnl": [p for p in pages if p.isdigit()], "newseye": [p for p in pages if not p.isdigit()]}

    def cer_of(arm, ps):
        return 100 * sum(acc[arm][p][0] for p in ps) / max(1, sum(acc[arm][p][1] for p in ps))

    out = {"cer_R1D_recalcule": cer_of("R1D", pages), "cer_R1D_reference": R1D_EVAL_CER}
    for arm in ("F", "B"):
        out[arm] = {
            "cer_d150": cer_of(arm, pages),
            "contraste_point_vs_R1D": cer_of(arm, pages) - R1D_EVAL_CER,
            "ic95_apparie_vs_R1D": paired_ci(acc, arm, "R1D", pages),
            "par_strate": {k: {"cer": cer_of(arm, v), "ic95_vs_R1D": paired_ci(acc, arm, "R1D", v)}
                           for k, v in strata.items()},
            "abstentions": sum(o[arm]["abstain"] for o in out_rows),
            "lignes_verite_vide_hors_cer": sum(1 for o in out_rows if not line_errors(o[arm]["hyp"], o["truth"])[1]),
        }
    out["CER_B_moins_CER_F_descriptif"] = cer_of("B", pages) - cer_of("F", pages)
    out["CER_B_moins_CER_F_par_strate"] = {k: cer_of("B", v) - cer_of("F", v) for k, v in strata.items()}
    out["pages"] = len(pages)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=("tune", "eval"), default="tune")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--lm", type=float, default=0.35)
    ap.add_argument("--beta", type=float, default=2.0)
    ap.add_argument("--beam", type=int, default=8)
    ap.add_argument("--corridor", type=float, default=8.0)
    ap.add_argument("--a-fixed", type=float, default=0.6)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    assert sha256(ANCRE)[:8] == ANCRE_SHA, "ancre.py : empreinte inattendue"
    proto = load_proto()
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

    # sélection identique à reader_r1d_joint.py --part tune --n 400
    rows = [json.loads(x) for x in gzip.open(r1d.PRED, "rt")]
    pages = sorted({r1d.page_of(r["stem"]) for r in rows})
    random.Random(1).shuffle(pages)
    tp = set(pages[: len(pages) // 2])
    if a.part == "tune":
        rows = [r for r in rows if r1d.page_of(r["stem"]) in tp]
        rows = random.Random(5).sample(rows, min(a.n, len(rows)))
        ref = json.loads(REF.read_text())
        if a.n == 400:  # lot complet seulement : même échantillon que la référence R1d
            assert [r["stem"] for r in rows] == [r["stem"] for r in ref["rows"]], "stems tune ≠ référence R1d"
    else:  # eval : moitié complémentaire, ordre du fichier, comme reader_r1d_joint.py --part eval
        rows = [r for r in rows if r1d.page_of(r["stem"]) not in tp]
        ref = json.loads(R1D_EVAL.read_text())
        assert [r["stem"] for r in rows] == [r["stem"] for r in ref["rows"]], "stems eval ≠ référence R1d"
        assert len(rows) == 2405

    t0 = time.time()
    out_rows = []
    for r in rows:
        la = np.asarray(r["logp"], dtype=np.float64)
        le_lat = load_lat(r1d.E8LAT / f"{r['stem']}.npz")
        le = le_lat.logprobs.astype(np.float64)
        enc_e = r1d.encoder_for(tuple(le_lat.labels), nfd=True)
        cand = {labels[i] for i in np.nonzero(la.max(0) > np.log(1e-3))[0] if i}
        cand |= {le_lat.labels[i] for i in np.nonzero(le.max(0) > np.log(1e-3))[0]
                 if i and not unicodedata.combining(le_lat.labels[i][0])}
        cand = sorted(c for c in cand if enc_a(c))
        m_a = proto.frame_margin(la)[0]
        m_k = proto.frame_margin(le)[0]
        w_b = proto.weight("bounded", m_a, m_k, (S_A, S_K))
        arms = {}
        for name, w in (("F", a.a_fixed), ("B", w_b)):
            hs = multi_ctc_beam([(la, w, enc_a), (le, 1 - w, enc_e)], cand, lm=lm, lm_weight=a.lm,
                                beta=a.beta, beam=a.beam, corridor=a.corridor,
                                max_len=la.shape[0] // 2 + 2)
            arms[name] = {"hyp": hs[0].text if hs else "", "abstain": not hs}
        out_rows.append({"stem": r["stem"], "truth": r["truth"], "w_B": w_b, "m_A": m_a, "m_K": m_k,
                         "F": arms["F"], "B": arms["B"]})

    # mesures : CER D-150 (lignes à vérité vide hors CER, à part), améliorées / dégradées (B vs F)
    summary = {}
    for name in ("F", "B"):
        pairs = [(o[name]["hyp"], o["truth"]) for o in out_rows]
        summary[name] = score(pairs) | {"abstentions": sum(o[name]["abstain"] for o in out_rows)}
    improved = degraded = same = 0
    for o in out_rows:
        eF, nF = line_errors(o["F"]["hyp"], o["truth"])
        eB, _ = line_errors(o["B"]["hyp"], o["truth"])
        if nF == 0:
            continue
        if eB < eF:
            improved += 1
        elif eB > eF:
            degraded += 1
        else:
            same += 1
    passes = summary["B"]["cer_d150"] < summary["F"]["cer_d150"] and degraded <= improved
    wb = np.array([o["w_B"] for o in out_rows])
    res = {
        "lot": "R1e etape 1 (tune seulement)" if a.part == "tune" else "R1e etape 3 (eval, une fois)",
        "n": len(out_rows),
        "cfg": {"a_fixed": a.a_fixed, "lm": a.lm, "beta": a.beta, "beam": a.beam, "corridor": a.corridor,
                "s_A": S_A, "s_K": S_K, "regle_B": "bounded (q3_weights.weight)"},
        "provenance": {"ancre_py_sha256": sha256(ANCRE), "proto_sha256": sha256(PROTO),
                       "script_sha256": sha256(Path(__file__)),
                       "stems_identiques_r1d_tune_corr8": True},
        "F": summary["F"], "B": summary["B"],
        "lignes_ameliorees_B_vs_F": improved, "lignes_degradees_B_vs_F": degraded,
        "lignes_egales": same,
        "poids_B": {"min": float(wb.min()), "mediane": float(np.median(wb)), "max": float(wb.max()),
                    "egal_0_6": int(np.sum(np.isclose(wb, 0.6)))},
        "decision_passe_a_l_eval": {"B": bool(passes), "F": True,
                                    "regle": "B passe si CER(B) < CER(F) et degradees(B) <= ameliorees(B)"},
        "seconds": time.time() - t0,
    }
    print(json.dumps({k: res[k] for k in ("n", "lignes_ameliorees_B_vs_F", "lignes_degradees_B_vs_F",
                                          "decision_passe_a_l_eval")}, ensure_ascii=False))
    print(json.dumps({"F": summary["F"], "B": summary["B"]}, ensure_ascii=False))
    if a.part == "eval":
        res["eval"] = eval_contrasts(out_rows)
        res["decision_passe_a_l_eval"] = None
        out_name = f"reader_r1e_eval{a.tag.removeprefix('_eval')}.json"
    else:
        out_name = f"reader_r1e_tune{a.tag.removeprefix('_tune')}.json"
    res["rows"] = out_rows
    (HERE / "reports" / out_name).write_text(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
