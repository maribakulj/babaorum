"""DIAGNOSTIC D-vides (docs/READER_LOT2A.txt, p9 17/10) — pourquoi 392 vraies lignes photo_1983
sortent à hypothèse vide avec le bras B (script reader_gate_calib_r1e.py, 40a15847).

Population : les 392 lignes à h == "" de reports/reader_gate_calib_r1e_rows.json. Aucun CER,
aucune ligne tune/eval. ancre.py 9c3da908 inchangé : le faisceau forcé est une copie locale
de multi_ctc_beam (forced_beam), utilisée seulement ici.

Sept classes exclusives, somme 392, premier critère vérifié (version finale, docs/READER_LOT2A.txt) :
  NON REPRODUCTIBLE  le redécodage ne rend pas h == ""
  BOGUE DE SCORE     |Σ w·ctc_full(t*) − L| > 1e-9
  BOGUE D'ADMISSION  t* vide et liste de fins vide
  TÉMOIN VIDE        t* vide et liste de fins non vide
  HORS CANDIDATS     t* utilise un caractère absent des candidats
  ÉLAGAGE            t* admise quand le faisceau garde de force son préfixe
  AUTRE              à examiner à la main
Un t* non vide dont le meilleur texte rendu est vide continue vers HORS CANDIDATS / ÉLAGAGE / AUTRE.
t* = chaîne qui atteint L (vide, ou glouton exprimable) : le vide s'il atteint L, sinon la plus
petite chaîne dans l'ordre des points de code (tri de str Python).

    ~/axel/.venv/bin/python scripts/reader_dvides.py
"""

from __future__ import annotations

import hashlib
import json
import pickle
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
from babaorum.reader.ancre import (  # noqa: E402
    MultiHyp, _extend_ids, _multi_full, _multi_witness, ctc_full, ctc_prefix_init, end_ok,
    greedy_ids, multi_ctc_beam)
from reader_gate_calib_r1e import CFG, LM, PROTO_SHA, S_A, S_K, TIMING, load, load_proto  # noqa: E402
from reader_r1d_joint import encoder_for  # noqa: E402

R = HERE / "reports"
ROWS = R / "reader_gate_calib_r1e_rows.json"
GATE_SHA = "40a15847"
ANCRE_SHA = "9c3da908"
TOL = 1e-9
CLASSES = ("NON REPRODUCTIBLE", "BOGUE DE SCORE", "BOGUE D'ADMISSION", "TÉMOIN VIDE",
           "HORS CANDIDATS", "ÉLAGAGE", "AUTRE")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extend_one(h: MultiHyp, c: str, lattices, lm, lm_weight: float, beta: float):
    """Une extension par un caractère, même arithmétique que multi_ctc_beam."""
    sts = []
    for li, (lp, _, enc) in enumerate(lattices):
        seq = enc(c)
        if not seq:
            return None
        sts.append(_extend_ids(lp, [h.states[li]], [seq])[0])
    vis = sum(w * st.score for st, (_, w, _) in zip(sts, lattices))
    lmv = h.lm + (lm(h.text, c) if lm is not None else 0.0)
    return MultiHyp(h.text + c, sts, lmv, vis + lm_weight * lmv + beta * (len(h.text) + 1))


def forced_beam(lattices, candidates, lm, lm_weight, beta, beam, corridor, max_len, forced=None):
    """Copie de ancre.multi_ctc_beam (9c3da908) ; ``forced`` : préfixe gardé de force dans le
    faisceau (ajouté s'il sort du faisceau). Sans ``forced``, doit rendre la même liste."""
    root = MultiHyp("", [ctc_prefix_init(lp) for lp, _, _ in lattices], 0.0)
    witness = _multi_witness(lattices, candidates)
    hyps, finished = [root], []
    fchain = root
    for step in range(max_len):
        for h in hyps:
            full = sum(w * ctc_full(st) for st, (_, w, _) in zip(h.states, lattices))
            if end_ok(full, witness, corridor):
                lmv = h.lm + (lm(h.text, None) if lm is not None else 0.0)
                finished.append(MultiHyp(h.text, h.states, lmv, full + lm_weight * lmv + beta * len(h.text),
                                         True, full))
        if finished:
            witness = max(witness, max(f.ctc for f in finished))
            finished = [f for f in finished if end_ok(f.ctc, witness, corridor)]
        pairs = [(h, c) for h in hyps for c in candidates]
        new_states = [[None] * len(lattices) for _ in pairs]
        for li, (lp, _, enc) in enumerate(lattices):
            ok = [(k, enc(c)) for k, (_, c) in enumerate(pairs)]
            ok = [(k, s) for k, s in ok if s]
            ext = _extend_ids(lp, [pairs[k][0].states[li] for k, _ in ok], [s for _, s in ok])
            for (k, _), st in zip(ok, ext):
                new_states[k][li] = st
        live = []
        for (h, c), sts in zip(pairs, new_states):
            if any(s is None for s in sts):
                continue
            vis = sum(w * st.score for st, (_, w, _) in zip(sts, lattices))
            lmv = h.lm + (lm(h.text, c) if lm is not None else 0.0)
            live.append((vis, MultiHyp(h.text + c, sts, lmv, vis + lm_weight * lmv + beta * (len(h.text) + 1))))
        if not live:
            break
        best_vis = max(v for v, _ in live)
        live = [h for v, h in live if v >= best_vis - corridor]
        live.sort(key=lambda h: -h.score)
        hyps = live[:beam]
        if forced is not None and step < len(forced):
            # la chaîne forcée avance seule, indépendamment du faisceau
            fchain = extend_one(fchain, forced[step], lattices, lm, lm_weight, beta)
            if fchain is None:
                break
            if step == len(forced) - 1:
                # t* complet : sa fin est testée dans cette itération, avant l'arrêt anticipé
                full = sum(w * ctc_full(st) for st, (_, w, _) in zip(fchain.states, lattices))
                if end_ok(full, witness, corridor):
                    lmv = fchain.lm + (lm(fchain.text, None) if lm is not None else 0.0)
                    finished.append(MultiHyp(fchain.text, fchain.states, lmv,
                                             full + lm_weight * lmv + beta * len(fchain.text), True, full))
                    witness = max(witness, max(f.ctc for f in finished))
                    finished = [f for f in finished if end_ok(f.ctc, witness, corridor)]
            elif fchain.text not in {h.text for h in hyps}:
                hyps = hyps + [fchain]
        if finished and max(f.score for f in finished) >= hyps[0].score + 5.0 * max(1.0, beta):
            break
    return sorted(finished, key=lambda h: -h.score)


def witness_texts(lattices, cand) -> set[str]:
    """Textes qui peuvent attendre le témoin : vide, et gloutons exprimables (un caractère = un id)."""
    texts = {""}
    for lp, _, enc in lattices:
        inv = {}
        for c in cand:
            seq = enc(c)
            if seq and len(seq) == 1:
                inv[seq[0]] = c
        ids = greedy_ids(lp)
        if all(k in inv for k in ids):
            texts.add("".join(inv[k] for k in ids))
    return texts


def main() -> None:
    assert sha256(HERE / "scripts/reader_gate_calib_r1e.py")[:8] == GATE_SHA, "script de la porte modifié"
    assert sha256(HERE / "src/babaorum/reader/ancre.py")[:8] == ANCRE_SHA, "ancre.py modifié"
    proto = load_proto()
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

    population = [r for r in json.loads(ROWS.read_text())["true"] if r["h"] == ""]
    assert len(population) == 392, f"{len(population)} lignes vides, 392 attendues"

    rows_out, validation = [], []
    for idx, r in enumerate(population):
        s = r["stem"]
        la = np.asarray(a1[s]["logp"], float)
        lk = np.asarray(kr[s]["logp"], float)
        m_a, m_k = proto.frame_margin(la)[0], proto.frame_margin(lk)[0]
        w_a = proto.weight("bounded", m_a, m_k, (S_A, S_K))
        assert abs(w_a - r["w_A"]) < 1e-12 and abs(m_a - r["m_A"]) < 1e-12 and abs(m_k - r["m_K"]) < 1e-12, s
        # candidats : exactement ceux de reader_gate_calib_r1e.joint (union, tri)
        cand = {labels_a[i] for i in np.nonzero(la.max(0) > np.log(1e-3))[0] if i}
        cand |= {klabels[i] for i in np.nonzero(lk.max(0) > np.log(1e-3))[0]
                 if i and not unicodedata.combining(klabels[i][0])}
        cand = sorted(c for c in cand if enc_a(c))
        lattices = [(la, w_a, enc_a), (lk, 1 - w_a, enc_k)]
        max_len = la.shape[0] // 2 + 2
        orig = multi_ctc_beam(lattices, cand, lm=lm, lm_weight=CFG["lm"], beta=CFG["beta"], beam=8,
                              corridor=8.0, max_len=max_len)
        h = orig[0].text if orig else ""
        if idx < 30:  # validation de la copie : sans forçage, même liste que multi_ctc_beam
            rep = forced_beam(lattices, cand, lm, CFG["lm"], CFG["beta"], 8, 8.0, max_len)
            same = [(x.text, round(x.score, 9)) for x in rep] == [(x.text, round(x.score, 9)) for x in orig]
            validation.append({"stem": s, "identique": same})
            assert same, f"copie du faisceau ≠ multi_ctc_beam sur {s}"
        row = {"stem": s, "h_redecode": h, "fins_rendues": len(orig),
               "meilleur_score_fin": orig[0].score if orig else None,
               "meilleur_texte_fin": orig[0].text if orig else None}
        if h != "":
            row.update(classe="NON REPRODUCTIBLE")
            rows_out.append(row)
            continue
        L = _multi_witness(lattices, cand)
        texts = witness_texts(lattices, cand)
        vals = {t: v for t in texts if (v := _multi_full(lattices, t)) is not None}
        assert abs(max(vals.values()) - L) <= TOL, "témoin recalculé ≠ témoin de ancre.py"
        attain = sorted(t for t, v in vals.items() if abs(v - L) <= TOL)
        t_star = "" if "" in attain else attain[0]
        diff = _multi_full(lattices, t_star) - L
        row.update(t_star=t_star, L=L, ecart_T1=diff, longueur_t_star=len(t_star))
        if abs(diff) > TOL:
            cls = "BOGUE DE SCORE"
        elif t_star == "" and not orig:
            cls = "BOGUE D'ADMISSION"
        elif t_star == "":
            cls = "TÉMOIN VIDE"
        elif not all(ch in cand for ch in t_star):
            cls = "HORS CANDIDATS"
        else:
            forced = forced_beam(lattices, cand, lm, CFG["lm"], CFG["beta"], 8, 8.0, max_len, forced=t_star)
            row["fins_forcees"] = len(forced)
            cls = "ÉLAGAGE" if any(f.text == t_star for f in forced) else "AUTRE"
        row["classe"] = cls
        rows_out.append(row)

    counts = {c: sum(1 for r in rows_out if r["classe"] == c) for c in CLASSES}
    examples = {c: [r for r in rows_out if r["classe"] == c][:5] for c in CLASSES}
    assert sum(counts.values()) == 392, "sept classes ne couvrent pas les 392 lignes"
    out = {
        "population": 392,
        "effectifs": counts,
        "somme": sum(counts.values()),
        "exemples_5_par_classe": examples,
        "validation_copie_faisceau": {"lignes_testees": len(validation),
                                      "identiques": sum(v["identique"] for v in validation)},
        "lignes": rows_out,
        "provenance": {"reader_dvides_py_sha256": sha256(Path(__file__)),
                       "gate_script_sha256": sha256(HERE / "scripts/reader_gate_calib_r1e.py"),
                       "ancre_py_sha256": sha256(HERE / "src/babaorum/reader/ancre.py"),
                       "proto_q3_sha256": PROTO_SHA},
    }
    (R / "reader_dvides.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps({"effectifs": counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
