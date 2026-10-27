"""R1c — IC apparié par page (bootstrap) et dégradations du consensus R1b (eval).

Bras comparés sur la moitié eval, réglages pris sur tune (reports/reader_r1b.json) :
consensus (a, λ, β choisis), E8 seul + LM (son meilleur réglage tune), A1 seul + LM.
Dégradation = ligne où le bras fait plus d'erreurs que le meilleur des deux gloutons.
"""
import gzip, json, random, sys
from pathlib import Path
sys.path.insert(0, str(Path.home() / "axel/src"))
from axel.reader.metric import line_errors

HERE = Path(__file__).resolve().parents[1]
rep = json.load(open(HERE / "reports/reader_r1b.json"))
rows = [json.loads(x) for x in gzip.open(HERE / "reports/reader_r1b_cands.jsonl.gz", "rt")]
page = lambda s: s.split("__")[1]
pages = sorted({page(r["stem"]) for r in rows}); random.Random(1).shuffle(pages)
tp = set(pages[: len(pages) // 2])
ev = [r for r in rows if page(r["stem"]) not in tp]
cfg = {"consensus": tuple(rep["chosen"].values()), "e8_lm": tuple(rep["eval_e8_only"]["cfg"]),
       "a1_lm": tuple(rep["eval_a1_only"]["cfg"])}
def pick(r, a, lam, beta):
    return max(r["cands"], key=lambda c: a * c["a"] + (1 - a) * c["e"] + lam * c["lm"] + beta * len(c["t"]))["t"]
E = {}  # arm -> page -> [err, n]
deg = {}
for arm, k in list(cfg.items()) + [("e8_greedy", None), ("a1_greedy", None)]:
    E[arm] = {}; deg[arm] = 0
    for r in ev:
        h = pick(r, *k) if k else r["e8" if arm == "e8_greedy" else "a1"]
        e, n = line_errors(h, r["truth"], "d150")
        ref = min(line_errors(r["a1"], r["truth"], "d150")[0], line_errors(r["e8"], r["truth"], "d150")[0])
        deg[arm] += e > ref
        acc = E[arm].setdefault(page(r["stem"]), [0, 0]); acc[0] += e; acc[1] += n
pk = sorted(E["consensus"])
def cer(arm, ps): return 100 * sum(E[arm][p][0] for p in ps) / sum(E[arm][p][1] for p in ps)
out = {"eval_lines": len(ev), "pages": len(pk), "cer": {a: cer(a, pk) for a in E}, "degraded_lines": deg, "ci95_delta": {}}
rng = random.Random(7)
# BnL : une « page » par ligne ; NewsEye : quelques pages -> bootstrap stratifié par corpus
strata = {}
for p in pk: strata.setdefault("bnl" if p.isdigit() else "newseye", []).append(p)
for a, b in (("consensus", "e8_lm"), ("consensus", "a1_lm"), ("consensus", "e8_greedy")):
    ds = []
    for _ in range(5000):
        ps = [rng.choice(v) for v in strata.values() for _ in v]
        ds.append(cer(a, ps) - cer(b, ps))
    ds.sort(); out["ci95_delta"][f"{a}-{b}"] = [ds[125], ds[-126]]
out["newseye_pages"] = len(strata.get("newseye", []))
print(json.dumps(out, indent=1))
(HERE / "reports/reader_r1c.json").write_text(json.dumps(out, indent=1))
