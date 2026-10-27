"""Lot 2d étape 1 — contrôle sans CER (docs/READER_LOT2A.txt, « Lot 2d — précisions avant lancement »).

Les 333 lignes AUTRE de reports/reader_dvides.json (sha256 663c5a91…) : t* figés, deux décodages
par ligne du bras B avec max_len = min(T_A, T_K) : forcé (t* gardé de force) et non forcé.
Comptes SÉPARÉS (non exclusifs) :
  forcé admis                     t* complet admis par le chemin forcé (end_ok, même itération) ;
  meilleur texte non forcé non vide   meilleure fin non forcée (premier rang du tri) non vide ;
  meilleur texte non forcé vide   meilleure fin non forcée vide, ou aucune fin ;
  forcé admis ∩ meilleur texte non forcé vide   parmi les lignes forcé admis, celles où le
                                  décodage non forcé reste à texte vide.
Aucun CER. ancre.py 9c3da908, scripts 40a15847, 70acbf9c inchangés : réutilisation seulement.

    ~/axel/.venv/bin/python scripts/reader_lot2d_check.py
"""

from __future__ import annotations

import hashlib
import json
import pickle
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np

AXEL = Path.home() / "axel"
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AXEL / "src"))
sys.path.insert(0, str(HERE / "src"))
sys.path.insert(0, str(HERE / "scripts"))

from axel.image.fusion import EOS, LMCache, lm_score  # noqa: E402
from reader_dvides import forced_beam  # noqa: E402
from reader_gate_calib_r1e import CFG, LM, S_A, S_K, TIMING, load, load_proto  # noqa: E402
from reader_r1d_joint import encoder_for  # noqa: E402

R = HERE / "reports"
GATE_SHA, DVIDES_SHA, ANCRE_SHA = "40a15847", "70acbf9c", "9c3da908"



def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    assert sha256(R / "reader_dvides.json").startswith("663c5a91"), "reader_dvides.json : sha256 inattendu"
    assert sha256(HERE / "scripts/reader_gate_calib_r1e.py")[:8] == GATE_SHA, "script de la porte modifié"
    assert sha256(HERE / "scripts/reader_dvides.py")[:8] == DVIDES_SHA, "reader_dvides.py modifié"
    assert sha256(HERE / "src/babaorum/reader/ancre.py")[:8] == ANCRE_SHA, "ancre.py modifié"
    proto = load_proto()

    dv = json.loads((R / "reader_dvides.json").read_text())
    autre = [r for r in dv["lignes"] if r.get("classe") == "AUTRE"]
    assert len(autre) == 333, f"{len(autre)} lignes AUTRE, 333 attendues"
    rows_gate = {r["stem"]: r for r in json.loads((R / "reader_gate_calib_r1e_rows.json").read_text())["true"]}

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

    out_rows = []
    t0 = time.time()
    for r in autre:
        s, t = r["stem"], r["t_star"]
        la = np.asarray(a1[s]["logp"], float)
        lk = np.asarray(kr[s]["logp"], float)
        m_a, m_k = proto.frame_margin(la)[0], proto.frame_margin(lk)[0]
        w_a = proto.weight("bounded", m_a, m_k, (S_A, S_K))
        assert abs(w_a - rows_gate[s]["w_A"]) < 1e-12, f"poids différent pour {s}"
        cand = {labels_a[i] for i in np.nonzero(la.max(0) > np.log(1e-3))[0] if i}
        cand |= {klabels[i] for i in np.nonzero(lk.max(0) > np.log(1e-3))[0]
                 if i and not unicodedata.combining(klabels[i][0])}
        cand = sorted(c for c in cand if enc_a(c))
        assert all(ch in cand for ch in t), f"t* hors candidats pour {s}"
        lattices = [(la, w_a, enc_a), (lk, 1 - w_a, enc_k)]
        max_len = min(la.shape[0], lk.shape[0])
        assert max_len >= len(t), f"max_len {max_len} < len(t*) {len(t)} pour {s}"
        forced = forced_beam(lattices, cand, lm, CFG["lm"], CFG["beta"], 8, 8.0, max_len, forced=t)
        admis = any(f.text == t for f in forced)
        free = forced_beam(lattices, cand, lm, CFG["lm"], CFG["beta"], 8, 8.0, max_len)
        best = free[0].text if free else ""  # meilleure fin non forcée (liste triée) ; vide si aucune
        best_forced = forced[0].text if forced else ""
        row = {"stem": s, "longueur_t_star": len(t), "T_A": la.shape[0], "T_K": lk.shape[0],
               "max_len": max_len, "force_admis": admis, "meilleur_texte_non_force": best,
               "meilleur_texte_force": best_forced, "fins_non_forcees": len(free),
               "fins_forcees": len(forced),
               "t_star_dans_fins_non_forcees": any(f.text == t for f in free)}
        if admis:
            row["score_fin_t_star"] = next(f.score for f in forced if f.text == t)
        out_rows.append(row)

    counts = {
        "forcé admis": sum(x["force_admis"] for x in out_rows),
        "meilleur texte non forcé non vide": sum(x["meilleur_texte_non_force"] != "" for x in out_rows),
        "meilleur texte non forcé vide": sum(x["meilleur_texte_non_force"] == "" for x in out_rows),
        "forcé admis ∩ meilleur texte non forcé vide":
            sum(x["force_admis"] and x["meilleur_texte_non_force"] == "" for x in out_rows),
        "t* contenu parmi les fins non forcées": sum(x["t_star_dans_fins_non_forcees"] for x in out_rows),
    }
    assert len(out_rows) == 333, f"dénominateur {len(out_rows)} ≠ 333"
    print(f"dénominateur : {len(out_rows)}")
    res = {
        "lot": "2d etape 1 (controle sans CER)",
        "n": len(out_rows),
        "denominateur": len(out_rows),
        "comptes_separes": counts,
        "max_len": "min(T_A, T_K)",
        "secondes": round(time.time() - t0, 1),
        "lignes": out_rows,
        "provenance": {"reader_lot2d_check_py_sha256": sha256(Path(__file__)),
                       "reader_dvides_py_sha256": sha256(HERE / "scripts/reader_dvides.py"),
                       "gate_script_sha256": sha256(HERE / "scripts/reader_gate_calib_r1e.py"),
                       "ancre_py_sha256": sha256(HERE / "src/babaorum/reader/ancre.py")},
    }
    (R / "reader_lot2d_check.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps({"comptes_separes": counts, "secondes": res["secondes"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
