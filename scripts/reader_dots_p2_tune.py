"""P2 de docs/READER_DOTS_CTC_PLAN.txt — lecture de la sortie TUNE et CHOIX de λ* (règle pré-enregistrée).

RÉSERVE B4 (audit pT), écrite EN TÊTE du rapport : la tête A1 qui fournit les treillis du guide a été choisie sur
TOUT press/dev (axel E17, commit ac1ec0f) ; l'éval de 2 405 lignes a donc déjà servi à sélectionner le guide. Le
système ne se présente pas comme « sélectionné sur TUNE seul » ; toute confirmation demandera un panneau hors de
press/dev.

Entrées : la sortie Kaggle dots_p1_tune.jsonl (400 lignes × 6 configurations) et son journal dots_p1_tune_run.json,
rapatriés dans un dossier donné par --sortie ; les vérités viennent d'items.jsonl des entrées (part = tune), dont
l'empreinte du MANIFEST est contrôlée (épinglée dans le carnet et ici).
Mesures par configuration, sur le champ pré-enregistré (M2) : clean = clean_text(join_paragraphs(raw)) déjà
calculé dans le carnet, puis NORM (NFC ; apostrophes → ' ; guillemets → " ; tirets → - ; blancs compactés) ; à
côté, le CER du raw (NFC, blancs compactés seulement), le D-150 (axel.reader.metric.line_errors) et la métrique
en couches (babaorum.reader.metric_layers.Tally). CER = 100 × Σ erreurs / Σ longueurs de vérité normalisée.
Choix de λ* : plus bas CER NORM parmi guide λ ∈ {0,3 ; 0,5 ; 0,7} ; égalité → λ le plus grand. guide_l0.0
n'entre pas dans le choix (contrôle B1).
Rapportés aussi (descriptif, aucune revendication) : causes d'arrêt (eos / plafond / autre), n_plancher (lignes
avec au moins un pas au plancher), hors codec, temps par ligne, par corpus (BnL, NewsEye).
Environnement : PYTHONPATH=~/axel/src:src ~/outils-seg/gate-env/bin/python scripts/reader_dots_p2_tune.py \\
    --sortie <dossier rapatrié> --entrees <dossier des entrées reconstruit par reader_dots_p1_build.py>
Sortie : reports/reader_dots_p2_tune.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path.home() / "axel/src"))
sys.path.insert(0, str(HERE / "src"))
from axel.reader.metric import line_errors  # noqa: E402
from babaorum.reader.metric_layers import Tally  # noqa: E402
from rapidfuzz.distance import Levenshtein  # noqa: E402

MANIFEST_SHA = "5029b3478bf5860b358492c3ab9af2e4b30d1ea5a46ccb1171984155a3b03d15"
CONFIGS = ("libre", "faisceau_libre", "guide_l0.0", "guide_l0.3", "guide_l0.5", "guide_l0.7")
LAMBDAS = {"guide_l0.3": 0.3, "guide_l0.5": 0.5, "guide_l0.7": 0.7}
_NORM = str.maketrans({**{c: "'" for c in "’‘ʼ´`′"}, **{c: '"' for c in "«»“”„‟‹›‚"},
                       **{c: "-" for c in "‐‑‒–—―−"}})


def norm(t: str) -> str:
    """NORM pré-enregistrée : NFC, apostrophes, guillemets, tirets, blancs compactés ; rien d'autre."""
    return " ".join(unicodedata.normalize("NFC", t or "").translate(_NORM).split())


def raw_form(t: str) -> str:
    return " ".join(unicodedata.normalize("NFC", t or "").split())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sortie", required=True)
    ap.add_argument("--entrees", required=True)
    a = ap.parse_args()
    S, E = Path(a.sortie), Path(a.entrees)
    assert hashlib.sha256((E / "MANIFEST.json").read_bytes()).hexdigest() == MANIFEST_SHA, "entrées inattendues"
    man = json.loads((E / "MANIFEST.json").read_text())
    assert hashlib.sha256((E / "items.jsonl").read_bytes()).hexdigest() == man["items.jsonl"], "items.jsonl"
    run = json.loads((S / "dots_p1_tune_run.json").read_text())
    assert run["manifest_sha256"] == MANIFEST_SHA and run["phase"] == "TUNE" and run["lignes"] == 400, run
    assert tuple(c[0] for c in run["configs"]) == CONFIGS, run["configs"]
    truth = {}
    for x in open(E / "items.jsonl"):
        it = json.loads(x)
        if it["part"] == "tune":
            truth[it["stem"]] = it["truth"]
    assert len(truth) == 400
    rows: dict[str, dict[str, dict]] = {c: {} for c in CONFIGS}
    for x in open(S / "dots_p1_tune.jsonl"):
        r = json.loads(x)
        assert r["config"] in rows and r["stem"] in truth, (r["config"], r["stem"])
        assert r["stem"] not in rows[r["config"]], f"doublon {r['stem']} {r['config']}"
        rows[r["config"]][r["stem"]] = r
    for c in CONFIGS:
        assert set(rows[c]) == set(truth), f"{c} : {len(rows[c])} lignes sur 400"

    strate = lambda s: "bnl" if s.split("__")[1].isdigit() else "newseye"  # noqa: E731
    out: dict = {"lot": "P2 dots + CTC — TUNE (choix de λ*)",
                 "reserve_B4_en_tete": "tête A1 choisie sur tout press/dev (axel E17, ac1ec0f) : l'éval a servi à "
                                       "sélectionner le guide ; confirmation seulement hors de press/dev",
                 "run": run, "lignes": 400, "par_strate_lignes": dict(Counter(strate(s) for s in truth)),
                 "configs": {}}
    for c in CONFIGS:
        acc = {k: {"norm": [0, 0], "raw": [0, 0], "d150": [0, 0]} for k in ("global", "bnl", "newseye")}
        tallies = {k: Tally() for k in ("global", "bnl", "newseye")}
        fins, n_pl, hc, secs = Counter(), 0, 0, 0.0
        for s, t in truth.items():
            r = rows[c][s]
            tn, tr = norm(t), raw_form(t)
            for k in ("global", strate(s)):
                if tn:
                    acc[k]["norm"][0] += Levenshtein.distance(norm(r["clean"]), tn)
                    acc[k]["norm"][1] += len(tn)
                if tr:
                    acc[k]["raw"][0] += Levenshtein.distance(raw_form(r["raw"]), tr)
                    acc[k]["raw"][1] += len(tr)
                e, n = line_errors(r["clean"], t, "d150")
                acc[k]["d150"][0] += e
                acc[k]["d150"][1] += n
                tallies[k].add(r["clean"], t)
            fins[r["fin"]] += 1
            n_pl += r["n_plancher"] > 0
            hc += r["hors_codec"] > 0
            secs += r["s"]
        rate = lambda v: 100 * v[0] / v[1] if v[1] else None  # noqa: E731
        if fins.get("autre"):
            print(f"ATTENTION {c} : {fins['autre']} lignes arrêtées ni par eos ni au plafond (M4)", flush=True)
        out["configs"][c] = {
            **{f"cer_{m}_pc": {k: rate(acc[k][m]) for k in acc} for m in ("norm", "raw", "d150")},
            "support_caracteres": {m: {k: acc[k][m][1] for k in acc} for m in ("norm", "raw", "d150")},
            "lignes_verite_vide": {"norm": sum(not norm(t) for t in truth.values()),
                                   "raw": sum(not raw_form(t) for t in truth.values()),
                                   "d150": sum(line_errors("", t, "d150")[1] == 0 for t in truth.values())},
            "couches": {k: v.report() for k, v in tallies.items()},
            "fins": dict(fins), "lignes_avec_plancher": n_pl, "lignes_hors_codec": hc,
            "s_par_ligne": round(secs / 400, 3)}
    cer = {c: out["configs"][c]["cer_norm_pc"]["global"] for c in LAMBDAS}
    best = min(cer.values())
    lam_star = max((LAMBDAS[c] for c in LAMBDAS if cer[c] == best))
    out["choix"] = {"regle": "plus bas CER NORM parmi λ ∈ {0,3 ; 0,5 ; 0,7} ; égalité → λ le plus grand",
                    "cer_norm_pc": cer, "lambda_star": lam_star}
    (HERE / "reports/reader_dots_p2_tune.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print("RÉSERVE B4 :", out["reserve_B4_en_tete"])
    print(json.dumps({"lambda_star": lam_star, "cer_norm_pc": {c: out["configs"][c]["cer_norm_pc"]["global"]
                                                               for c in CONFIGS}}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
