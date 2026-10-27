r"""Lot « second signal » (docs/READER_DOTS_SS_PREREG.txt ; CLASSE B) — choix de δ_a et δ_r sur la SÉLECTION, puis critères
lus une fois sur P3 (SECOND REGARD). RÉSERVE B4 en tête : la tête A1 a été choisie sur tout press/dev.

d = Levenshtein(NORM(A1 glouton), NORM(guidé clean)) / max(longueurs) (0 si les deux sont vides).
V-abst : sortie vide si d > δ_a ; V-repli : sortie = A1 glouton si d > δ_r ; les deux APRÈS l'abstention d'encre gelée
(σ < τ = reports/reader_dots_abst_v2.json : sortie vide).
Choix (sélection seulement) : grille = quantiles 0..100 % des d des 400 lignes TUNE et des 200 négatifs scramble de
sélection ; δ_a = plus petite valeur avec, sur TUNE, fausse abstention ≤ 0,5 %, aucune courte abstenue et CER NORM de V-abst
≤ CER NORM guidé + 0,05 point ; δ_r = plus petite valeur avec CER NORM de V-repli ≤ CER NORM guidé + 0,05 point (TUNE).
Critères P3 (une fois) : K-s scramble, part inventée (nouveau compteur) du dispositif ≤ 1/10 de dots libre ; K-cer CER NORM
éval à ± 0,05 point du dispositif gelé actuel (guidé + σ) ; K-fa (V-abst) ≤ 1 % et ≤ 5 % courtes (IC, verdict si n ≥ 100) ;
V-repli : part renvoyée à A1 ; lignes où le guidé corrige A1 et qui seraient coupées ou renvoyées.
Sur TUNE et la sélection, l'abstention σ n'a coupé aucune vraie ligne (ABST v2 : 0/400) et n'abstient pas le scramble ; elle
n'est donc appliquée que sur P3 (éval et négatifs), avec les σ recalculés comme dans ABST v2.
Environnement : PYTHONPATH=~/axel/src:src ~/outils-seg/dots-hf-env/bin/python scripts/reader_dots_ss.py \
    --tune-sortie <sortie TUNE> --ss-sortie <sortie du passage second signal> --ss-entrees <entrées du passage> \
    --p3-sortie <sortie P3> --p3-entrees <entrées P3>
Sortie : reports/reader_dots_ss.json.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
AXEL = Path.home() / "axel"
R = HERE / "reports"
_load = lambda n, f: (lambda s: (s.loader.exec_module(m := importlib.util.module_from_spec(s)), m)[1])(  # noqa: E731
    importlib.util.spec_from_file_location(n, HERE / f))
ab1 = _load("ab1", "scripts/reader_dots_abst.py")
abv2 = _load("abv2", "scripts/reader_dots_abst_v2.py")
norm, paired, short, rate, invented, NEW = ab1.norm, ab1.paired, ab1.short, ab1.rate, ab1.invented, ab1.NEW
Lev = ab1.p3r.Levenshtein
A1_DEV = AXEL / "runs/e17/kaggle/a1/dev_pred_a1.jsonl.gz"
TUNE_MANIFEST = "5029b3478bf5860b358492c3ab9af2e4b30d1ea5a46ccb1171984155a3b03d15"
SS_MANIFEST = "6ae365bf163a8b363d3abb32e32b48cb1501c40ff19fcbc491c12b19d32a57a8"  # audité par pT le 19/10
REVISION, DTYPE = "e539fbb52280393adc081b289ec597430a0f9031", "torch.float32"
TAU_GEL = 20.827896720535783  # gel R7(a), ABST v2


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def dist(a: str, b: str) -> float:
    x, y = norm(a), norm(b)
    m = max(len(x), len(y))
    return Lev.distance(x, y) / m if m else 0.0


def cer(pairs) -> float:
    e = n = 0
    for h, t in pairs:
        tn = norm(t)
        if tn:
            e += Lev.distance(norm(h), tn)
            n += len(tn)
    return 100 * e / n


def load_rows(path: Path, items: dict) -> dict:
    rows: dict = {}
    for x in open(path):
        r = json.loads(x)
        assert r["stem"] in items, f"tige étrangère {r['stem']}"
        assert r["stem"] not in rows.get(r["config"], {}), f"doublon {r['stem']}"
        rows.setdefault(r["config"], {})[r["stem"]] = r
    return rows


def main() -> None:
    from PIL import Image

    from axel.image.crops import crop_line
    from babaorum.reader.perturb import perturb

    Image.MAX_IMAGE_PIXELS = None
    ap = argparse.ArgumentParser()
    for k in ("tune-sortie", "ss-sortie", "ss-entrees", "p3-sortie", "p3-entrees"):
        ap.add_argument(f"--{k}", required=True)
    a = ap.parse_args()
    TS, SS, SE, PS, PE = map(Path, (a.tune_sortie, a.ss_sortie, a.ss_entrees, a.p3_sortie, a.p3_entrees))
    tau = json.loads((R / "reader_dots_abst_v2.json").read_text())["tau"]
    assert tau == TAU_GEL, f"tau {tau!r} ≠ gel {TAU_GEL!r}"
    sha_abst_v2 = sha(R / "reader_dots_abst_v2.json")

    # TUNE : sorties guidées λ 0,3 (journal TUNE vérifié) et A1
    trun = json.loads((TS / "dots_p1_tune_run.json").read_text())
    assert (trun["manifest_sha256"], trun["phase"], trun["revision"], trun["dtype"]) == (TUNE_MANIFEST, "TUNE", REVISION, DTYPE), trun
    assert ["guide_l0.3", 0.3, 4] in [list(c) for c in trun["configs"]], trun["configs"]
    tune = {r["stem"]: r["truth"] for r in json.loads((R / "reader_r1e_tune.json").read_text())["rows"]}
    trows = load_rows(TS / "dots_p1_tune.jsonl", tune)
    g_t = {s: trows["guide_l0.3"][s]["clean"] for s in tune}
    a1_dev = {}
    for x in gzip.open(A1_DEV, "rt"):
        r = json.loads(x)
        a1_dev[r["stem"]] = r["hyp"] or ""
    d_t = {s: dist(a1_dev[s], g_t[s]) for s in tune}
    # sélection : sorties guidées du passage second signal (journal vérifié) et A1 Mac
    sman = sha(SE / "MANIFEST.json")
    assert sman == SS_MANIFEST, f"MANIFEST SS {sman} ≠ audité"
    srun = json.loads((SS / "dots_p1_eval_run.json").read_text())
    assert srun["manifest_sha256"] == sman and not srun["interrompu"] and srun["lignes_traitees"] == 200, srun
    assert (srun["phase"], srun["revision"], srun["dtype"]) == ("EVAL", REVISION, DTYPE), srun
    assert srun["configs_negatifs"] == [["guide_l0.3", 0.3, 4]] and srun["lignes_prevues"] == 200, srun
    sm = json.loads((SE / "MANIFEST.json").read_text())
    for f in ("items.jsonl", "a1.jsonl"):
        assert sha(SE / f) == sm[f], f"entrée ss modifiée : {f}"
    sitems = {it["stem"]: it for it in map(json.loads, open(SE / "items.jsonl"))}
    srows = load_rows(SS / "dots_p1_eval.jsonl", sitems)
    assert set(srows) == {"guide_l0.3"} and set(srows["guide_l0.3"]) == set(sitems), "passage incomplet"
    k_ = lambda r: r.get("stem") or f"{r['corpus']}__{r['page']}__{r['line_id']}"  # noqa: E731
    abst200 = {k_(json.loads(x)) for x in open(R / "reader_items_photo1983_abst200.jsonl")}
    pert200 = {k_(json.loads(x)) for x in open(R / "reader_items_photo1983_pert200.jsonl")}
    assert len(abst200) == 200 and not abst200 & pert200, "abst200 / pert200"
    assert set(sitems) == {f"{s}#scramble" for s in abst200}, "panneau SS ≠ abst200#scramble"
    assert all(it["mode"] == "scramble" and it["cle_perturbation"] == it["src_stem"] + "#r-abst" for it in sitems.values())
    a1_sel = {}
    for x in gzip.open(R / "reader_a1_photo1983_abst_scramble.jsonl.gz", "rt"):
        r = json.loads(x)
        if f"{r.get('stem')}#scramble" in sitems:
            a1_sel[f"{r['stem']}#scramble"] = r.get("hyp") or ""
    assert set(a1_sel) == set(sitems)
    d_s = {s: dist(a1_sel[s], srows["guide_l0.3"][s]["clean"]) for s in sitems}

    # choix des seuils
    grid = np.quantile(np.array(list(d_t.values()) + list(d_s.values())), np.linspace(0, 1, 101), method="linear")
    t_short = [s for s in tune if short(tune[s])]
    base = cer((g_t[s], tune[s]) for s in tune)

    def v(text_g, text_a1, dd, delta, mode):
        if dd <= delta:
            return text_g
        return "" if mode == "abst" else text_a1

    delta_a = delta_r = None
    for t in sorted(grid, reverse=True):  # du plus large au plus sévère ; on garde le dernier tenable
        fa = sum(d_t[s] > t for s in tune) / 400
        if fa <= 0.005 and not any(d_t[s] > t for s in t_short) and \
                cer((v(g_t[s], a1_dev[s], d_t[s], t, "abst"), tune[s]) for s in tune) <= base + 0.05:
            delta_a = float(t)
        if cer((v(g_t[s], a1_dev[s], d_t[s], t, "repli"), tune[s]) for s in tune) <= base + 0.05:
            delta_r = float(t)
    out = {"reserve_B4_en_tete": "tête A1 choisie sur tout press/dev ; mesures P3 = SECOND REGARD",
           "delta_abst": delta_a, "delta_repli": delta_r, "tau_sigma": tau,
           "selection": {"cer_tune_guide": base,
                         "d_q5_50_95": {"tune": [float(np.quantile(list(d_t.values()), q)) for q in (0.05, 0.5, 0.95)],
                                        "scramble_sel": [float(np.quantile(list(d_s.values()), q)) for q in (0.05, 0.5, 0.95)]},
                         "scramble_sel_coupe": {"abst": None if delta_a is None else sum(x > delta_a for x in d_s.values()) / 200,
                                                "repli": None if delta_r is None else sum(x > delta_r for x in d_s.values()) / 200}}}
    (R / "reader_dots_ss.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))  # seuils écrits d'abord

    # P3 : SECOND REGARD (liaison complète, comme ABST v2)
    assert sha(PE / "MANIFEST.json") == ab1.P3_MANIFEST
    man = json.loads((PE / "MANIFEST.json").read_text())
    assert sha(PE / "items.jsonl") == man["items.jsonl"]
    prun = json.loads((PS / "dots_p1_eval_run.json").read_text())
    assert (prun["manifest_sha256"], prun["phase"], prun["revision"], prun["dtype"]) == (ab1.P3_MANIFEST, "EVAL", REVISION, DTYPE), prun
    assert [c[0] for c in prun["configs"]] == ["libre", "guide_l0.0", "guide_l0.3"], prun["configs"]
    assert [c[0] for c in prun["configs_negatifs"]] == ["libre", "guide_l0.3"], prun["configs_negatifs"]
    assert not prun["interrompu"] and prun["lignes_traitees"] == prun["lignes_prevues"] == 3005, prun
    items = {it["stem"]: it for it in map(json.loads, open(PE / "items.jsonl"))}
    rows = load_rows(PS / "dots_p1_eval.jsonl", items)
    ev = [s for s, it in items.items() if it["part"] == "eval"]
    neg = [s for s, it in items.items() if it["part"] == "neg"]
    for c in ("libre", "guide_l0.3"):
        assert set(ev) | set(neg) <= set(rows[c]), c
    # σ des lignes P3 (mêmes recadrages qu'ABST v2)
    local = {}
    for x in open(AXEL / "data/prepared/press/dev.jsonl"):
        ln = json.loads(x)
        local[f"{ln['corpus']}__{ln['page']}__{ln['line_id']}".replace("/", "_").replace("#", "-")] = ln["image"]
    by = {json.loads(x)["stem"]: json.loads(x) for x in gzip.open(AXEL / "data/e8/kaggle/e8_items.jsonl.gz", "rt")}
    key = lambda r: r.get("stem") or f"{r['corpus']}__{r['page']}__{r['line_id']}"  # noqa: E731
    local_pert = {key(r): r["image"] for r in map(json.loads, open(R / "reader_items_photo1983_pert200.jsonl"))}
    sig = {}
    cache: dict = {}

    def page(p):
        if p not in cache:
            cache.clear()
            with Image.open(p) as im:
                cache[p] = im.convert("L")
        return cache[p]
    for s in sorted(ev, key=lambda s: local[s]):
        r = by[s]
        c = page(local[s]) if r["image"].startswith("bnl/") else crop_line(page(local[s]), tuple(r["box"]), r["polygon"], pad=4).image
        assert hashlib.md5(c.tobytes()).hexdigest() == r["crop_md5"], s
        sig[s] = abv2.sigma(c)
    for s in sorted(neg, key=lambda s: local_pert[items[s]["src_stem"]]):
        it = items[s]
        c = crop_line(page(local_pert[it["src_stem"]]), tuple(it["box"]), it.get("polygon"), pad=4).image
        p = perturb(c, it["mode"], it["cle_perturbation"])
        assert hashlib.md5(p.tobytes()).hexdigest() == it["pert_md5"], s
        sig[s] = abv2.sigma(p)
    a1_neg = {}
    for mode in ("scramble", "blank", "strips"):
        for x in gzip.open(R / f"reader_a1_photo1983_{mode}.jsonl.gz", "rt"):
            r = json.loads(x)
            if f"{r.get('stem')}#{mode}" in items:
                a1_neg[f"{r['stem']}#{mode}"] = r.get("hyp") or ""
    a1 = {s: a1_dev[s] for s in ev} | a1_neg

    def final(s, mode, delta):
        g = rows["guide_l0.3"][s]["clean"]
        if sig[s] < tau:
            return ""
        if mode is None or delta is None:
            return g
        return v(g, a1[s], dist(a1[s], g), delta, mode)

    res = {}
    gel = cer((final(s, None, None), items[s]["truth"]) for s in ev)
    assert abs(gel - 1.848) < 0.001, f"dispositif gelé recalculé {gel} ≠ 1,848 % (liaison)"
    libre_scr = sum(invented(rows["libre"][s]["clean"], NEW) for s in neg if items[s]["mode"] == "scramble") / 200
    e_short = [s for s in ev if short(items[s]["truth"])]
    for name, mode, delta in (("V-abst", "abst", delta_a), ("V-repli", "repli", delta_r)):
        txt = {s: final(s, mode, delta) for s in ev + neg}
        c = cer((txt[s], items[s]["truth"]) for s in ev)
        err = {"gel": {}, name: {}}
        for s in ev:
            tn = norm(items[s]["truth"])
            err["gel"][s] = (Lev.distance(norm(final(s, None, None)), tn) if tn else 0, len(tn))
            err[name][s] = (Lev.distance(norm(txt[s]), tn) if tn else 0, len(tn))
        touched = [s for s in ev if delta is not None and sig[s] >= tau and dist(a1[s], rows["guide_l0.3"][s]["clean"]) > delta]
        corrige = [s for s in touched if err["gel"][s][0] < Lev.distance(norm(a1[s]), norm(items[s]["truth"]))]
        if delta is None:
            res[name] = {"delta": None, "variante_non_appliquee": True, "K_s_scramble": None, "K_cer_a_005": None}
            continue
        r_ = {"delta": delta, "cer_norm_eval": c, "cer_norm_gel": gel, "ecart_points": c - gel,
              "K_cer_a_005": abs(c - gel) <= 0.05, "ic95_gel_moins_variante": paired(err, ev, "gel", name),
              "lignes_eval_touchees": len(touched), "dont_ou_le_guide_corrige_A1": len(corrige)}
        for m in ("scramble", "blank", "strips"):
            ss = [s for s in neg if items[s]["mode"] == m]
            r_[f"{m}_inventee_nouveau"] = sum(invented(txt[s], NEW) for s in ss) / len(ss)
        r_["K_s_scramble"] = r_["scramble_inventee_nouveau"] <= libre_scr / 10
        if mode == "abst":
            r_["fausse_abstention"] = rate(len(touched), len(ev), 0.01)
            tot = [s for s in ev if sig[s] < tau or s in set(touched)]
            r_["fausse_abstention_avec_sigma"] = rate(len(tot), len(ev), 0.01)
            r_["fausse_abstention_courtes"] = rate(sum(s in set(touched) for s in e_short), len(e_short), 0.05)
            r_["fausse_abstention_courtes_avec_sigma"] = rate(sum(s in set(tot) for s in e_short), len(e_short), 0.05)
        else:
            r_["part_renvoyee_A1"] = len(touched) / len(ev)
        res[name] = r_
    out["libre_scramble_inventee_nouveau"] = libre_scr
    out["P3"] = res
    out["provenance"] = {"sortie_P3": sha(PS / "dots_p1_eval.jsonl"), "sortie_ss": sha(SS / "dots_p1_eval.jsonl"),
                         "sortie_tune": sha(TS / "dots_p1_tune.jsonl"), "manifest_ss": sman, "script": sha(Path(__file__)),
                         "journal_tune": sha(TS / "dots_p1_tune_run.json"), "journal_ss": sha(SS / "dots_p1_eval_run.json"),
                         "journal_P3": sha(PS / "dots_p1_eval_run.json"), "rapport_abst_v2": sha_abst_v2, "tau": tau}
    out["statut"] = "SECOND REGARD : décide si une variante entre dans le dispositif gelé"
    (R / "reader_dots_ss.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print("RÉSERVE B4 :", out["reserve_B4_en_tete"])
    print(json.dumps({"delta_abst": delta_a, "delta_repli": delta_r, "P3": res}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
