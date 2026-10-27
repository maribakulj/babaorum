"""R6 — contrastes pré-enregistrés (docs/READER_R6_PREREG.txt) sur la moitié eval.

Entrées : reports/reader_r6_<bras>_eval.jsonl.gz (lignes eval extraites des
<bras>_dev.jsonl.gz de Kaggle : stem, truth, hyp) et le texte glouton E8 des
mêmes lignes (reports/reader_r1b_cands.jsonl.gz, champ e8). IC : apparié par
page, 10 000 tirages, graine 1, percentiles (indices 250 et 9 749), BnL et
NewsEye séparés. Signe : diff = CER(base) − CER(variante) ; revendication
seulement si l'IC est entièrement positif ; entièrement négatif = variante pire.
C4 calculé comme CER(E8) − CER(B0) (même signe que les autres : positif = B0 meilleur) ;
porte du témoin sur la borne haute de CER(B0) − CER(E8) = − borne basse (porte du témoin : borne haute > 0,5 point -> les bras
ne se lisent que contre B0). Les bras absents sont sautés.
Sortie : reports/reader_r6_report.json.
"""
import gzip
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path.home() / "axel/src"))
from axel.reader.metric import line_errors  # noqa: E402

R = HERE / "reports"
EVAL_PAGES = set(json.load(open(R / "reader_r3_split.json"))["eval"])  # pages eval (996ab0cd…)
TEXT: dict = {}
for line in gzip.open(R / "reader_r1b_cands.jsonl.gz", "rt"):
    r = json.loads(line)
    TEXT.setdefault("E8", {})[r["stem"]] = r["e8"]
TRUTH: dict = {}
for arm in ("B0", "B1", "B2", "B3", "B4", "B5"):
    p = R / f"reader_r6_{arm}_eval.jsonl.gz"
    if p.exists():
        for line in gzip.open(p, "rt"):
            r = json.loads(line)
            TEXT.setdefault(arm, {})[r["stem"]] = r["hyp"]
            TRUTH[r["stem"]] = r["truth"]
stems = sorted(TRUTH)
assert len(stems) == 2405 and all(s.split("__")[1] in EVAL_PAGES for s in stems), len(stems)
for arm, t in TEXT.items():
    assert set(stems) <= set(t), f"{arm} : lignes manquantes"
page = lambda s: s.split("__")[1]  # noqa: E731
E: dict = {}
for arm in TEXT:
    for s in stems:
        e, n = line_errors(TEXT[arm][s], TRUTH[s], "d150")
        acc = E.setdefault(arm, {}).setdefault(page(s), [0, 0])
        acc[0] += e
        acc[1] += n


def cer(arm, ps):
    return 100 * sum(E[arm][p][0] for p in ps) / max(1, sum(E[arm][p][1] for p in ps))


def paired_ci(x, y, pool, draws=10000, seed=1):
    rng = random.Random(seed)
    ds = sorted(cer(x, ps) - cer(y, ps) for ps in ([rng.choice(pool) for _ in pool] for _ in range(draws)))
    return [ds[int(0.025 * draws)], ds[int(0.975 * draws) - 1]]


pages = sorted({page(s) for s in stems})
strata: dict = {}
for p in pages:
    strata.setdefault("bnl" if p.isdigit() else "newseye", []).append(p)
CONTRASTS = {"C1": ("B1", "B2"), "C2": ("B2", "B4"), "C3": ("B2", "B5"), "C4": ("E8", "B0"),
             "desc_B1_vs_B0": ("B0", "B1"), "desc_B3_vs_B0": ("B0", "B3")}
out = {"lines": len(stems), "pages": len(pages), "cer_d150": {a: cer(a, pages) for a in E},
       "by_stratum": {k: {a: cer(a, v) for a in E} for k, v in strata.items()}, "contrasts": {}}
for name, (base, var) in CONTRASTS.items():
    if base in E and var in E:
        ci = paired_ci(base, var, pages)
        out["contrasts"][name] = {
            "diff": f"CER({base}) - CER({var})", "point": cer(base, pages) - cer(var, pages), "ci95": ci,
            "by_stratum": {k: paired_ci(base, var, v) for k, v in strata.items()},
            "reading": "variante meilleure (revendication)" if ci[0] > 0 else
                       ("variante pire" if ci[1] < 0 else "IC contient 0 : aucune revendication"),
            "descriptive": name.startswith("desc")}
if "C4" in out["contrasts"]:
    ub = -out["contrasts"]["C4"]["ci95"][0]  # borne haute de CER(B0) − CER(E8) en points
    out["witness_gate"] = {"upper_CER_B0_minus_E8": ub, "read_against_B0_only": ub > 0.5}
out["ci_spec"] = "apparié par page, 10 000 tirages, graine 1, percentiles ; BnL et NewsEye séparés"
(R / "reader_r6_report.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
print(json.dumps(out, indent=1, ensure_ascii=False))
