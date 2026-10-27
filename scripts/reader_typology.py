"""Typologie des erreurs de six lecteurs sur les MÊMES 2 405 lignes eval (sorties EXISTANTES, aucun calcul
GPU, aucun décodage ; demande pA et Marcel du 17/10/2026). Descriptif, aucune revendication ; eval déjà
exposée (second regard).

Systèmes et sources (sha256 vérifiés) :
  A1     glouton A1 (champ a1 de reports/reader_r1d_eval_eval_c8.json) ;
  JOINT  faisceau joint R1d (champ joint du même fichier) — AFFECTÉ par la troncature max_len = T_A//2+2
         (inventaire pA du 17/10) : lignes longues tronquées ; rapporté avec cette réserve ;
  E8     glouton kraken affiné (champ e8 de reports/reader_r1b_cands.jsonl.gz ; point d'E8 choisi sur un dev
         contenant l'eval : réserve C4) ;
  B0, B4 bras R6 (reports/reader_r6_B0_eval.jsonl.gz, reports/reader_r6_B4_eval.jsonl.gz) ;
  K1     CATMuS-Print Large non affiné, re-décodage BaselineLine (reports/reader_r7b_k1bl_eval.jsonl.gz).
Mêmes tiges et mêmes vérités pour tous (contrôlé, arrêt sinon).
Métrique : babaorum.reader.metric_layers (CER par couche cumulée L0-L5, WER L0 et L4, sac de mots,
classement des différences entre mots) ; plus le CER D-150 d'axel (axel.reader.metric.line_errors) pour
la continuité avec les chiffres R6/R7. Par système : global, BnL, NewsEye ; 20 substitutions de
caractères les plus fréquentes à la couche L4 (editops de Levenshtein sur L4) ; puis, par paire de
systèmes, lignes justes (L4 égale, vérité non vide à L4) chez l'un et fausses chez l'autre.
Environnement : PYTHONPATH=~/axel/src:src ~/outils-seg/gate-env/bin/python scripts/reader_typology.py
Sortie : reports/reader_typology.json.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path.home() / "axel/src"))
sys.path.insert(0, str(HERE / "src"))
from axel.reader.metric import line_errors  # noqa: E402
from babaorum.reader.metric_layers import Tally, layers  # noqa: E402
from rapidfuzz.distance import Levenshtein  # noqa: E402

R = HERE / "reports"
SRC = {
    "R1D": (R / "reader_r1d_eval_eval_c8.json", "9975bc59cf9b7931f29b7eeb382aea674ab5e1a61872ac9e0ca5a1c55281fa12"),
    "R1B": (R / "reader_r1b_cands.jsonl.gz", "1875232b88dbacef319ad47696dcffee2dbcbdb15f91624c19d704e2d15e5850"),
    "B0": (R / "reader_r6_B0_eval.jsonl.gz", "b3e33a9b0e12fca995e5daebc9d26be4678944fd82657060da36784bc3bd8451"),
    "B4": (R / "reader_r6_B4_eval.jsonl.gz", "dbb971e8e462bd118b09c726f0fbfb99ebf5c153b5d599d8c0f643da6ee56ce0"),
    "K1": (R / "reader_r7b_k1bl_eval.jsonl.gz", "9e51532002e8ab079cf3b13ac4a8e1be2f467ab18a2e4fb4d9f0d024bdcc9030"),
}
SYSTEMS = ("A1", "JOINT", "E8", "B0", "B4", "K1")


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    for k, (p, h) in SRC.items():
        assert h is None or sha256(p) == h, f"{k} inattendu"
    text: dict[str, dict[str, str]] = {s: {} for s in SYSTEMS}
    truth: dict[str, str] = {}

    def put(sys_, stem, hyp, tru):
        text[sys_][stem] = hyp or ""
        assert truth.setdefault(stem, tru) == tru, (sys_, stem)

    r1d = json.loads(SRC["R1D"][0].read_text())["rows"]
    assert len(r1d) == len({r["stem"] for r in r1d}) == 2405, "R1D : tiges en double ou manquantes"
    for r in r1d:
        put("A1", r["stem"], r["a1"], r["truth"])
        put("JOINT", r["stem"], r["joint"], r["truth"])
    stems = sorted(truth)
    assert len(stems) == 2405
    seen: set[str] = set()
    for x in gzip.open(SRC["R1B"][0], "rt"):
        r = json.loads(x)
        assert r["stem"] not in seen, f"R1B : tige en double {r['stem']}"
        seen.add(r["stem"])
        if r["stem"] in truth:
            put("E8", r["stem"], r["e8"], r["truth"])
    for k in ("B0", "B4", "K1"):
        for x in gzip.open(SRC[k][0], "rt"):
            r = json.loads(x)
            put(k, r["stem"], r["hyp"], r["truth"])
    assert sorted(truth) == stems and all(set(t) == set(stems) for t in text.values()), "lignes différentes"

    strate = lambda s: "bnl" if s.split("__")[1].isdigit() else "newseye"  # noqa: E731
    out = {"lot": "Typologie des erreurs, 6 lecteurs, 2 405 lignes eval (descriptif, aucune revendication)",
           "reserves": {"JOINT": "troncature max_len = T_A//2+2 (lignes longues)",
                        "E8": "point choisi sur un dev contenant l'eval (réserve C4)",
                        "eval": "eval déjà exposée : second regard"},
           "lignes": len(stems), "par_strate_lignes": dict(Counter(strate(s) for s in stems)), "systemes": {}}
    ok_l4: dict[str, set[str]] = {}
    for sys_ in SYSTEMS:
        tallies = {"global": Tally(), "bnl": Tally(), "newseye": Tally()}
        d150 = {k: [0, 0] for k in tallies}
        subs: Counter = Counter()
        ok_l4[sys_] = set()
        for s in stems:
            h, t = text[sys_][s], truth[s]
            e, n = line_errors(h, t, "d150")
            for k in ("global", strate(s)):
                tallies[k].add(h, t)
                d150[k][0] += e
                d150[k][1] += n
            lh, lt = layers(h)["L4_ponctuation"], layers(t)["L4_ponctuation"]
            if lt and lh == lt:  # une vérité vide à L4 n'est pas une ligne « juste »
                ok_l4[sys_].add(s)
            for op in Levenshtein.editops(lh, lt):
                if op.tag == "replace":
                    subs[f"{lh[op.src_pos]}→{lt[op.dest_pos]}"] += 1  # lu → vrai
        out["systemes"][sys_] = {k: dict(v.report(), cer_d150_pc=100 * d150[k][0] / d150[k][1] if d150[k][1] else None)
                                 for k, v in tallies.items()}
        out["systemes"][sys_]["substitutions_L4_lu_vers_vrai_top20"] = subs.most_common(20)
        out["systemes"][sys_]["lignes_justes_L4"] = len(ok_l4[sys_])
    out["paires_lignes_justes_L4"] = {f"{a}|{b}": {f"{a}_seul": len(ok_l4[a] - ok_l4[b]), f"{b}_seul": len(ok_l4[b] - ok_l4[a])}
                                      for a, b in combinations(SYSTEMS, 2)}
    out["provenance"] = {k: sha256(p) for k, (p, _) in SRC.items()} | {
        "metric_layers": sha256(HERE / "src/babaorum/reader/metric_layers.py"), "script": sha256(Path(__file__))}
    (R / "reader_typology.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps({s: out["systemes"][s]["global"]["cer_pc"] for s in SYSTEMS}, indent=1))


if __name__ == "__main__":
    main()
