r"""P3 de docs/READER_DOTS_CTC_PLAN.txt — rapport de l'éval (SECOND REGARD) et des négatifs Q2, selon le
pré-enregistrement P3 (λ* = 0,3 et NORM gelés).

RÉSERVE B4 (EN TÊTE) : la tête A1, qui fournit les treillis du guide, a été choisie sur TOUT press/dev (axel E17,
commit ac1ec0f) ; l'éval de 2 405 lignes a donc déjà servi à sélectionner le guide, en plus de son exposition par
R6/R7. ÉTIQUETTE : « SECOND REGARD, éval déjà exposée » : ces chiffres servent à DÉCIDER si on continue, pas à
affirmer un gain ; la confirmation se fera sur un panneau hors de press/dev (banc R7(a)).

Entrées : la sortie Kaggle dots_p1_eval.jsonl et dots_p1_eval_run.json (rapatriées dans --sortie), items.jsonl
des entrées P3 (--entrees, MANIFEST épinglé), et pour A1 SEUL le champ hyp de axel
runs/e17/kaggle/a1/dev_pred_a1.jsonl.gz (lecture gloutonne sans LLM, déjà calculée).
Mesures (éval) : CER NORM sur clean (champ pré-enregistré), CER brut sur raw (NFC, blancs), D-150 sur clean,
métrique en couches ; MOTS INVENTÉS = « mot_invente » de metric_layers.classify, comme sur TUNE. A1 SEUL : même
chose sur son hyp (clean = hyp, A1 n'écrit pas de Markdown).
Contrastes (CER NORM, IC 95 % apparié par ligne : lignes tirées avec remise, 10 000 tirages, random.Random(1),
indices 250 et 9 749 ; statistique 100 × Σ erreurs / Σ longueurs) :
  C1 = libre − guide_l0.3 ; C2 = guide_l0.0 − guide_l0.3 ; C4 = A1 − guide_l0.3 (descriptif).
  Lecture : « le score CTC aide » si C2 entièrement positif ; « le dispositif lit mieux que dots libre » si C1 l'est ;
  SECOND REGARD, pour décider. Descriptifs : mêmes contrastes appariés PAR PAGE (pages tirées avec remise, même
  graine) ; par SOURCE (BnL, NewsEye) avec IC apparié par ligne.
Négatifs Q2 : sortie « inventée » si clean contient ≥ 3 mots de 2 lettres ou plus ([^\W\d_]{2,}) ; par
perturbation × configuration : part inventée, part non vide, n_plancher, causes d'arrêt ; critère scramble
(part guidée ≤ 1/10 de la part libre) rapporté sans revendication ; strips descriptif seulement.
Si le carnet a été interrompu (journal : interrompu), les négatifs sont rapportés sur les lignes faites, et l'éval
doit être complète (sinon arrêt).
Environnement : PYTHONPATH=~/axel/src:src ~/outils-seg/gate-env/bin/python scripts/reader_dots_p3_report.py \\
    --sortie <dossier rapatrié> --entrees <entrées P3>
Sortie : reports/reader_dots_p3_eval.json.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path.home() / "axel/src"))
sys.path.insert(0, str(HERE / "src"))
from axel.reader.metric import line_errors  # noqa: E402
from babaorum.reader.metric_layers import Tally, classify  # noqa: E402
from rapidfuzz.distance import Levenshtein  # noqa: E402

MANIFEST_SHA = "9b875a049c7f45aab85d0e68685114b6eb96f37894ef8e5495c1410110326693"
A1_PRED = Path.home() / "axel/runs/e17/kaggle/a1/dev_pred_a1.jsonl.gz"
EVAL_CONFIGS = ("libre", "guide_l0.0", "guide_l0.3")
NEG_CONFIGS = ("libre", "guide_l0.3")
MODES = ("scramble", "blank", "strips")
WORD = re.compile(r"[^\W\d_]{2,}")
_NORM = str.maketrans({**{c: "'" for c in "’‘ʼ´`′"}, **{c: '"' for c in "«»“”„‟‹›‚"},
                       **{c: "-" for c in "‐‑‒–—―−"}})


def norm(t: str) -> str:
    """NORM pré-enregistrée : NFC, apostrophes, guillemets, tirets, blancs compactés ; rien d'autre."""
    return " ".join(unicodedata.normalize("NFC", t or "").translate(_NORM).split())


def raw_form(t: str) -> str:
    return " ".join(unicodedata.normalize("NFC", t or "").split())


def paired(err: dict, keys: list, a: str, b: str, group=None) -> list[float]:
    """IC 95 % de CER(a) − CER(b) ; err[sys][clé ligne] = (erreurs, longueur) ; group : clé → unité tirée."""
    units = sorted({group(k) for k in keys}) if group else sorted(keys)
    by = {}
    for k in keys:
        by.setdefault(group(k) if group else k, []).append(k)
    sums = {u: [sum(err[s][k][i] for k in by[u]) for s in (a, b) for i in (0, 1)] for u in units}
    rng = random.Random(1)
    ds = []
    for _ in range(10000):
        pick = [rng.choice(units) for _ in units]
        ea, na, eb, nb = (sum(sums[u][i] for u in pick) for i in range(4))
        ds.append((100 * ea / na if na else 0.0) - (100 * eb / nb if nb else 0.0))
    ds.sort()
    return [ds[250], ds[9749]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sortie", required=True)
    ap.add_argument("--entrees", required=True)
    a = ap.parse_args()
    S, E = Path(a.sortie), Path(a.entrees)
    assert hashlib.sha256((E / "MANIFEST.json").read_bytes()).hexdigest() == MANIFEST_SHA, "entrées inattendues"
    man = json.loads((E / "MANIFEST.json").read_text())
    assert hashlib.sha256((E / "items.jsonl").read_bytes()).hexdigest() == man["items.jsonl"]
    run = json.loads((S / "dots_p1_eval_run.json").read_text())
    assert run["manifest_sha256"] == MANIFEST_SHA and run["phase"] == "EVAL", run
    items = {it["stem"]: it for it in map(json.loads, open(E / "items.jsonl"))}
    ev = [s for s, it in items.items() if it["part"] == "eval"]
    assert len(ev) == 2405
    rows: dict[str, dict[str, dict]] = {}
    for x in open(S / "dots_p1_eval.jsonl"):
        r = json.loads(x)
        assert r["stem"] in items
        assert r["stem"] not in rows.get(r["config"], {}), f"doublon {r['stem']} {r['config']}"
        rows.setdefault(r["config"], {})[r["stem"]] = r
    for c in EVAL_CONFIGS:
        missing = [s for s in ev if s not in rows.get(c, {})]
        assert not missing, f"éval incomplète pour {c} : {len(missing)} lignes manquantes"
    a1 = {}
    for x in gzip.open(A1_PRED, "rt"):
        r = json.loads(x)
        if r["stem"] in items and items[r["stem"]]["part"] == "eval":
            assert r["truth"] == items[r["stem"]]["truth"], r["stem"]
            a1[r["stem"]] = r["hyp"] or ""
    assert set(a1) == set(ev)
    a1_neg = {}  # pA (c) : A1 SEUL aussi sur les négatifs (lecture gloutonne déjà calculée, Mac)
    for m in MODES:
        for x in gzip.open(HERE / f"reports/reader_a1_photo1983_{m}.jsonl.gz", "rt"):
            r = json.loads(x)
            if f"{r.get('stem')}#{m}" in items:
                a1_neg[f"{r['stem']}#{m}"] = r.get("hyp") or ""
    assert len(a1_neg) == 600, len(a1_neg)
    if run.get("interrompu"):
        print("ATTENTION : carnet INTERROMPU au plafond ; négatifs rapportés en PARTIEL", flush=True)

    texts = {c: {s: rows[c][s]["clean"] for s in ev} for c in EVAL_CONFIGS} | {"A1": a1}
    raws = {c: {s: rows[c][s]["raw"] for s in ev} for c in EVAL_CONFIGS} | {"A1": a1}
    truth = {s: items[s]["truth"] for s in ev}
    source = lambda s: "bnl" if s.split("__")[1].isdigit() else "newseye"  # noqa: E731
    page = lambda s: s.split("__")[1]  # noqa: E731
    systems = (*EVAL_CONFIGS, "A1")
    err = {sy: {} for sy in systems}
    out: dict = {"reserve_B4_en_tete": "tête A1 choisie sur tout press/dev (axel E17, ac1ec0f) : l'éval a servi à "
                                       "sélectionner le guide ; confirmation seulement hors de press/dev",
                 "etiquette": "SECOND REGARD, éval déjà exposée : pour DÉCIDER si on continue, pas pour affirmer",
                 "lambda_star": 0.3, "run": run, "lignes_eval": len(ev),
                 "par_source_lignes": dict(Counter(source(s) for s in ev)), "systemes": {}}
    for sy in systems:
        acc = {k: {"norm": [0, 0], "raw": [0, 0], "d150": [0, 0]} for k in ("global", "bnl", "newseye")}
        tallies = {k: Tally() for k in ("global", "bnl", "newseye")}
        inv = Counter()  # gardé pour la ventilation ; le total vient des couches (même source que P2)
        for s in ev:
            t, h = truth[s], texts[sy][s]
            tn, hn = norm(t), norm(h)
            e = Levenshtein.distance(hn, tn) if tn else 0
            err[sy][s] = (e, len(tn))
            inv[source(s)] += classify(h, t).get("mot_invente", 0)
            for k in ("global", source(s)):
                acc[k]["norm"][0] += e
                acc[k]["norm"][1] += len(tn)
                tr = raw_form(t)
                if tr:
                    acc[k]["raw"][0] += Levenshtein.distance(raw_form(raws[sy][s]), tr)
                    acc[k]["raw"][1] += len(tr)
                ed, nd = line_errors(h, t, "d150")
                acc[k]["d150"][0] += ed
                acc[k]["d150"][1] += nd
                tallies[k].add(h, t)
        rate = lambda v: 100 * v[0] / v[1] if v[1] else None  # noqa: E731
        res = {f"cer_{m}_pc": {k: rate(acc[k][m]) for k in acc} for m in ("norm", "raw", "d150")}
        res["support_caracteres"] = {m: {k: acc[k][m][1] for k in acc} for m in ("norm", "raw", "d150")}
        res["mots_inventes"] = {k: tallies[k].report()["differences"].get("mot_invente", 0)
                                for k in ("global", "bnl", "newseye")}
        res["lignes_verite_vide_norm"] = sum(not norm(t) for t in truth.values())
        res["couches"] = {k: v.report() for k, v in tallies.items()}
        if sy in EVAL_CONFIGS:
            rs = [rows[sy][s] for s in ev]
            res["fins"] = dict(Counter(r["fin"] for r in rs))
            res["lignes_avec_plancher"] = sum(r["n_plancher"] > 0 for r in rs)
            res["s_par_ligne"] = round(sum(r["s"] for r in rs) / len(rs), 3)
            if res["fins"].get("autre"):
                print(f"ATTENTION {sy} : {res['fins']['autre']} lignes arrêtées ni par eos, ni au plafond, ni au temps (M4)")
        out["systemes"][sy] = res
    contrasts = {"C1_libre_moins_guide": ("libre", "guide_l0.3"), "C2_l0_moins_guide": ("guide_l0.0", "guide_l0.3"),
                 "C4_A1_moins_guide_descriptif": ("A1", "guide_l0.3")}
    out["contrastes"] = {}
    for name, (x, y) in contrasts.items():
        pt = out["systemes"][x]["cer_norm_pc"]["global"] - out["systemes"][y]["cer_norm_pc"]["global"]
        ci = paired(err, ev, x, y)
        out["contrastes"][name] = {
            "point": pt, "ic95_par_ligne": ci, "ic95_par_page_descriptif": paired(err, ev, x, y, group=page),
            "par_source": {src: {"point": out["systemes"][x]["cer_norm_pc"][src] - out["systemes"][y]["cer_norm_pc"][src],
                                 "ic95_par_ligne": paired(err, [s for s in ev if source(s) == src], x, y)}
                           for src in ("bnl", "newseye")},
            "lecture": ("entièrement positif" if ci[0] > 0 else ("entièrement négatif" if ci[1] < 0 else "contient 0"))}
    out["decision"] = {"le_score_CTC_aide": out["contrastes"]["C2_l0_moins_guide"]["ic95_par_ligne"][0] > 0,
                       "le_dispositif_lit_mieux_que_dots_libre": out["contrastes"]["C1_libre_moins_guide"]["ic95_par_ligne"][0] > 0,
                       "statut": "SECOND REGARD : décision de continuer, pas une affirmation"}
    neg = {}
    neg_done = {c: {s for s, r in rows.get(c, {}).items() if items[s]["part"] == "neg"} for c in NEG_CONFIGS}
    common = set.intersection(*neg_done.values())  # mêmes lignes pour libre et guidé (audit p3r, mineur 2)
    if not run.get("interrompu"):
        assert len(common) == 600, f"négatifs incomplets sans interruption : {len(common)}"
    for m in MODES:
        for c in (*NEG_CONFIGS, "A1"):
            if c == "A1":
                rs = [{"clean": a1_neg[s], "n_plancher": 0, "fin": "glouton"} for s in sorted(common) if items[s]["mode"] == m]
            else:
                rs = [rows[c][s] for s in sorted(common) if items[s]["mode"] == m]
            n = len(rs)
            neg[f"{m}|{c}"] = {"lignes": n,
                               "part_inventee": sum(len(WORD.findall(r["clean"])) >= 3 for r in rs) / n if n else None,
                               "part_non_vide": sum(bool(r["clean"].strip()) for r in rs) / n if n else None,
                               "lignes_avec_plancher": sum(r["n_plancher"] > 0 for r in rs),
                               "fins": dict(Counter(r["fin"] for r in rs))}
    out["negatifs_Q2"] = neg
    s_l, s_g = neg["scramble|libre"]["part_inventee"], neg["scramble|guide_l0.3"]["part_inventee"]
    out["negatifs_partiels"] = bool(run.get("interrompu")) or len(common) < 600
    out["critere_scramble"] = {"libre": s_l, "guide": s_g, "lignes": neg["scramble|libre"]["lignes"],
                               "guide_au_plus_un_dixieme_du_libre": (s_g <= s_l / 10) if s_l and s_g is not None else None,
                               "statut": "rapporté sans revendication"}
    (HERE / "reports/reader_dots_p3_eval.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print("RÉSERVE B4 :", out["reserve_B4_en_tete"])
    print("ÉTIQUETTE :", out["etiquette"], "| interrompu :", bool(run.get("interrompu")))
    print(json.dumps({"cer_norm": {sy: out["systemes"][sy]["cer_norm_pc"]["global"] for sy in systems},
                      "contrastes": {k: (round(v["point"], 3), [round(x, 3) for x in v["ic95_par_ligne"]])
                                     for k, v in out["contrastes"].items()},
                      "decision": out["decision"], "critere_scramble": out["critere_scramble"]}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
