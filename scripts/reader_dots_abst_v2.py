r"""Lot ABSTENTION v2 (docs/READER_DOTS_ABST_V2_PREREG.txt ; CLASSE B) — test d'ENCRE sur les pixels, sans modèle.

RÉSERVE B4 (en tête) : la tête A1 a été choisie sur tout press/dev ; mesures sur P3 = SECOND REGARD.
Règle : σ = écart type des niveaux de gris du recadrage de la ligne ; abstention si σ < τ. τ choisi sur la SÉLECTION
(400 vraies lignes TUNE + 400 négatifs de sélection blank/scramble de reports/reader_items_photo1983_abst200.jsonl,
clé « stem#r-abst ») : grille = quantiles 0..100 % des σ réunis ; τ = plus grande valeur avec fausse abstention TUNE
≤ 0,5 % et aucune des lignes TUNE courtes abstenue. Descriptifs : fraction sombre sous Otsu ; M/T des treillis A1
(TUNE et sélection seulement). Mesures une fois sur P3 (sorties existantes, abstention a posteriori) : critères
blank ≤ 5 % (nouveau compteur), fausse abstention éval ≤ 1 % et ≤ 5 % sur les courtes (IC Clopper-Pearson, verdict
si n ≥ 100) ; scramble et strips rapportés ; CER NORM avec abstention, C1/C2 avec abstention (descriptif).
Recadrages : comme les lecteurs (axel crop_line polygone pad 4 ; BnL : image entière), MD5 contrôlés (crop_md5 des
entrées P3, pert_md5 des négatifs P3) ; images locales (axel data/prepared/press/dev.jsonl, photo_1983).
Environnement (PIL, numpy, rapidfuzz ; sans scipy) : PYTHONPATH=~/axel/src:src ~/outils-seg/dots-hf-env/bin/python \
    scripts/reader_dots_abst_v2.py \
    --p3-sortie <dossier rapatrié de P3> --p3-entrees <entrées P3>
Sortie : reports/reader_dots_abst_v2.json.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
AXEL = Path.home() / "axel"
R = HERE / "reports"
sys.path.insert(0, str(AXEL / "src"))
sys.path.insert(0, str(HERE / "src"))
_spec = importlib.util.spec_from_file_location("ab1", HERE / "scripts/reader_dots_abst.py")
ab1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ab1)
norm, paired, short, rate, invented, OLD, NEW, mass = (ab1.norm, ab1.paired, ab1.short, ab1.rate, ab1.invented,
                                                        ab1.OLD, ab1.NEW, ab1.mass)
P3_MANIFEST = ab1.P3_MANIFEST
SUFFIX = {"scramble": "#r-gate", "blank": "#r-gate", "strips": "#r-dens"}


def sigma(im) -> float:
    return float(np.asarray(im, dtype=np.float64).std())


def otsu_dark(im) -> float:
    a = np.asarray(im, dtype=np.uint8).ravel()
    if a.std() == 0:
        return 0.0
    hist = np.bincount(a, minlength=256).astype(np.float64)
    p = hist / hist.sum()
    w = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (mu[-1] * w - mu) ** 2 / (w * (1 - w))
    t = int(np.nanargmax(between))
    return float((a <= t).mean())


def main() -> None:
    from PIL import Image

    from axel.image.crops import crop_line
    from babaorum.reader.perturb import perturb

    Image.MAX_IMAGE_PIXELS = None
    ap = argparse.ArgumentParser()
    ap.add_argument("--p3-sortie", required=True)
    ap.add_argument("--p3-entrees", required=True)
    a = ap.parse_args()
    S, E = Path(a.p3_sortie), Path(a.p3_entrees)
    # audit pT (ABST v2) : liaison COMPLÈTE aux entrées et au passage P3 audités
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
    assert sha(E / "MANIFEST.json") == P3_MANIFEST, "MANIFEST P3 inattendu"
    man = json.loads((E / "MANIFEST.json").read_text())
    assert sha(E / "items.jsonl") == man["items.jsonl"], "items.jsonl ≠ MANIFEST"
    run = json.loads((S / "dots_p1_eval_run.json").read_text())
    assert (run["manifest_sha256"], run["phase"], run["revision"], run["dtype"]) == (
        P3_MANIFEST, "EVAL", "e539fbb52280393adc081b289ec597430a0f9031", "torch.float32"), run
    assert [c[0] for c in run["configs"]] == ["libre", "guide_l0.0", "guide_l0.3"], run["configs"]
    assert [c[0] for c in run["configs_negatifs"]] == ["libre", "guide_l0.3"], run["configs_negatifs"]
    assert not run["interrompu"] and run["lignes_traitees"] == run["lignes_prevues"] == 3005, run
    items = {it["stem"]: it for it in map(json.loads, open(E / "items.jsonl"))}
    assert len(items) == 3005
    local = {}
    for x in open(AXEL / "data/prepared/press/dev.jsonl"):
        ln = json.loads(x)
        local[f"{ln['corpus']}__{ln['page']}__{ln['line_id']}".replace("/", "_").replace("#", "-")] = ln["image"]
    by = {}
    for x in gzip.open(AXEL / "data/e8/kaggle/e8_items.jsonl.gz", "rt"):
        r = json.loads(x)
        by[r["stem"]] = r
    tune = {r["stem"]: r["truth"] for r in json.loads((R / "reader_r1e_tune.json").read_text())["rows"]}
    ev = [s for s, it in items.items() if it["part"] == "eval"]
    assert len(tune) == 400 and len(ev) == 2405 and not set(tune) & set(ev)

    cache: dict[str, object] = {}

    def page(path):
        if path not in cache:
            cache.clear()
            with Image.open(path) as im:
                cache[path] = im.convert("L")
        return cache[path]

    def dev_crop(s):
        r = by[s]
        pg = page(local[s])
        crop = pg if r["image"].startswith("bnl/") else crop_line(pg, tuple(r["box"]), r["polygon"], pad=4).image
        assert hashlib.md5(crop.tobytes()).hexdigest() == r["crop_md5"], s
        return crop

    feat = {}  # clé -> (σ, fraction sombre Otsu)
    for s in sorted(set(tune) | set(ev), key=lambda s: local[s]):
        c = dev_crop(s)
        feat[s] = (sigma(c), otsu_dark(c))
    key = lambda r: r.get("stem") or f"{r['corpus']}__{r['page']}__{r['line_id']}"  # noqa: E731
    sel = sorted((json.loads(x) for x in open(R / "reader_items_photo1983_abst200.jsonl")), key=lambda r: r["image"])
    pert_keys = {key(json.loads(x)) for x in open(R / "reader_items_photo1983_pert200.jsonl")}
    assert len(sel) == 200 and not {key(r) for r in sel} & pert_keys, "sélection non disjointe de pert200"
    for r in sel:
        crop = crop_line(page(r["image"]), tuple(r["box"]), r.get("polygon"), pad=4).image
        for mode in ("blank", "scramble"):
            p = perturb(crop, mode, key(r) + "#r-abst")
            feat[f"sel:{key(r)}#{mode}"] = (sigma(p), otsu_dark(p))
    neg = [s for s, it in items.items() if it["part"] == "neg"]
    local_pert = {key(r): r["image"] for r in map(json.loads, open(R / "reader_items_photo1983_pert200.jsonl"))}
    for s in sorted(neg, key=lambda s: local_pert[items[s]["src_stem"]]):
        it = items[s]
        crop = crop_line(page(local_pert[it["src_stem"]]), tuple(it["box"]), it.get("polygon"), pad=4).image
        assert hashlib.md5(crop.tobytes()).hexdigest() == it["crop_md5"], s
        p = perturb(crop, it["mode"], it["cle_perturbation"])
        assert hashlib.md5(p.tobytes()).hexdigest() == it["pert_md5"], s
        feat[s] = (sigma(p), otsu_dark(p))

    # choix de τ sur la sélection
    sel_keys = [k for k in feat if k.startswith("sel:")]
    assert len(sel_keys) == 400
    t_short = [s for s in tune if short(tune[s])]
    grid = np.quantile(np.array([feat[s][0] for s in tune] + [feat[k][0] for k in sel_keys]),
                       np.linspace(0, 1, 101), method="linear")
    tau = None
    for t in grid:
        if sum(feat[s][0] < t for s in tune) / 400 <= 0.005 and not any(feat[s][0] < t for s in t_short):
            tau = float(t)
    abst = (lambda k: tau is not None and feat[k][0] < tau)
    q = lambda xs: [float(np.quantile(xs, v)) for v in (0.05, 0.25, 0.5, 0.75, 0.95)]  # noqa: E731
    out = {"reserve_B4_en_tete": "tête A1 choisie sur tout press/dev ; mesures P3 = SECOND REGARD",
           "regle": "abstention si σ (écart type des gris du recadrage) < τ", "tau": tau,
           "selection": {"tune_courtes": len(t_short),
                         "fausse_abstention_tune": rate(sum(abst(s) for s in tune), 400, 0.005),
                         "fausse_abstention_tune_courtes": rate(sum(abst(s) for s in t_short), len(t_short), 0.02),
                         "abstention_selection": {m: sum(abst(k) for k in sel_keys if k.endswith(m)) / 200
                                                  for m in ("#blank", "#scramble")},
                         "sigma_q5_25_50_75_95": {"tune": q([feat[s][0] for s in tune]),
                                                  "sel_blank": q([feat[k][0] for k in sel_keys if k.endswith("#blank")]),
                                                  "sel_scramble": q([feat[k][0] for k in sel_keys if k.endswith("#scramble")])},
                         "otsu_sombre_q5_25_50_75_95": {"tune": q([feat[s][1] for s in tune]),
                                                        "sel_scramble": q([feat[k][1] for k in sel_keys if k.endswith("#scramble")])}}}
    (R / "reader_dots_abst_v2.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))  # τ écrit d'abord
    # descriptif M/T (TUNE et sélection seulement)
    mt = {"tune": [], "sel_blank": [], "sel_scramble": []}
    for x in gzip.open(AXEL / "runs/e17/kaggle/a1/dev_pred_a1.jsonl.gz", "rt"):
        r = json.loads(x)
        if r["stem"] in tune:
            mt["tune"].append(mass(r["logp"]) / len(r["logp"]))
    sel_keys_set = {key(x) for x in sel}
    for mode in ("blank", "scramble"):
        seen = []
        for x in gzip.open(R / f"reader_a1_photo1983_abst_{mode}.jsonl.gz", "rt"):
            r = json.loads(x)
            if r.get("logp") is not None:
                seen.append(r["stem"])
                mt[f"sel_{mode}"].append(mass(r["logp"]) / len(r["logp"]))
        assert len(seen) == len(set(seen)) == 200 and set(seen) == sel_keys_set, f"M/T {mode} : ensemble inexact"
    assert len(mt["tune"]) == 400 and len(mt["sel_blank"]) == len(mt["sel_scramble"]) == 200
    out["M_sur_T_descriptif_q5_25_50_75_95"] = {k: q(v) for k, v in mt.items()}

    # P3 (sorties existantes) : SECOND REGARD
    rows: dict[str, dict] = {}
    for x in open(S / "dots_p1_eval.jsonl"):
        r = json.loads(x)
        assert r["stem"] in items, f"tige étrangère {r['stem']}"
        assert r["stem"] not in rows.get(r["config"], {}), f"doublon {r['stem']} {r['config']}"
        rows.setdefault(r["config"], {})[r["stem"]] = r
    for c in ("libre", "guide_l0.0", "guide_l0.3"):
        assert set(ev) <= set(rows.get(c, {})), f"éval incomplète pour {c}"
    for c in ("libre", "guide_l0.3"):
        assert set(neg) <= set(rows.get(c, {})), f"négatifs incomplets pour {c}"
    assert "guide_l0.0" not in rows or not set(neg) & set(rows["guide_l0.0"]), "λ0 inattendu sur les négatifs"
    out["provenance"] = {"manifest_P3": P3_MANIFEST, "items": man["items.jsonl"],
                         "sortie_P3": sha(S / "dots_p1_eval.jsonl"), "journal_P3": sha(S / "dots_p1_eval_run.json"),
                         "script": sha(Path(__file__)),
                         "selection_abst200": sha(R / "reader_items_photo1983_abst200.jsonl"),
                         "negatifs_pert200": sha(R / "reader_items_photo1983_pert200.jsonl")}
    a1_neg = {}
    for mode in SUFFIX:
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
        texts["guide_l0.3_abst"] = {s: ("" if abst(s) else texts["guide_l0.3"][s]) for s in ss}
        res_neg[mode] = {c: {"ancien": sum(invented(t[s], OLD) for s in ss) / len(ss),
                             "nouveau": sum(invented(t[s], NEW) for s in ss) / len(ss)} for c, t in texts.items()}
        res_neg[mode]["part_abstenue"] = sum(abst(s) for s in ss) / len(ss)
    out["negatifs_P3"] = res_neg
    e_short = [s for s in ev if short(items[s]["truth"])]
    fa = rate(sum(abst(s) for s in ev), len(ev), 0.01)
    fs = rate(sum(abst(s) for s in e_short), len(e_short), 0.05)
    out["eval"] = {"fausse_abstention": fa, "fausse_abstention_courtes": fs}
    out["criteres"] = {"blank_le_5pc": res_neg["blank"]["guide_l0.3_abst"]["nouveau"] <= 0.05,
                       "fausse_abstention_le_1pc": fa["verdict"], "fausse_abstention_courtes_le_5pc": fs["verdict"],
                       "scramble": "échec attendu, rapporté (lot « second signal » à part)"}
    err = {}
    for name, txt in (("libre", lambda s: rows["libre"][s]["clean"]), ("guide_l0.0", lambda s: rows["guide_l0.0"][s]["clean"]),
                      ("guide_l0.3", lambda s: rows["guide_l0.3"][s]["clean"]),
                      ("guide_l0.3_abst", lambda s: "" if abst(s) else rows["guide_l0.3"][s]["clean"])):
        err[name] = {}
        for s in ev:
            tn = norm(items[s]["truth"])
            err[name][s] = (ab1.p3r.Levenshtein.distance(norm(txt(s)), tn) if tn else 0, len(tn))
    cer = {k: 100 * sum(v[0] for v in d.values()) / sum(v[1] for v in d.values()) for k, d in err.items()}
    out["eval"]["cer_norm_pc"] = cer
    out["eval"]["C1_avec_abstention"] = {"point": cer["libre"] - cer["guide_l0.3_abst"],
                                         "ic95_par_ligne": paired(err, ev, "libre", "guide_l0.3_abst")}
    out["eval"]["C2_avec_abstention"] = {"point": cer["guide_l0.0"] - cer["guide_l0.3_abst"],
                                         "ic95_par_ligne": paired(err, ev, "guide_l0.0", "guide_l0.3_abst")}
    out["statut"] = "SECOND REGARD : décide si l'abstention entre dans le dispositif gelé pour R7(a)"
    out["lecture_des_IC"] = ("IC de Clopper-Pearson binomiaux nominaux par ligne, pas une garantie sur de nouvelles pages "
                             "(sélection et négatifs P3 partagent les mêmes 27 pages) ; le verdict porte sur le taux "
                             "OBSERVÉ (contraintes ponctuelles écrites), pas sur la borne haute")
    (R / "reader_dots_abst_v2.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print("RÉSERVE B4 :", out["reserve_B4_en_tete"])
    print(json.dumps({"tau": tau, "selection": out["selection"], "criteres": out["criteres"],
                      "eval": {k: v for k, v in out["eval"].items() if k != "cer_norm_pc"}}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
