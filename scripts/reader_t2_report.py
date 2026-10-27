"""T2 — rapport (docs/READER_T2_PROTOCOL.txt) depuis reports/reader_t2_rows.jsonl.gz."""
import gzip
import json
import unicodedata
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
rows = [json.loads(x) for x in gzip.open(HERE / "reports/reader_t2_rows.jsonl.gz", "rt")]
DELTA = 20.0


def frac_guaranteed(vals, lens, delta=None, per_char=None):
    ok = n = 0
    for s, L in zip(vals, lens):
        n += 1
        if s is None:
            continue  # moins de 3 mots : pas d'invention de 3 mots possible par substitution
        thr = delta if delta is not None else per_char * L
        ok += s > thr
    return ok / max(1, n)


def per_char_equiv(vals, lens, target):
    lo, hi = 0.0, 20.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if frac_guaranteed(vals, lens, per_char=mid) > target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


rep = {"lines": len(rows)}
for name in ("a1", "kr"):
    R = [r[name] for r in rows if name in r]
    multi = [x for x in R if x.get("S_w") is not None]
    clean = [x for x in R if x["err"] == 0 and x["n"] > 0]
    m_clean = np.array([m for x in clean for m in x["m"]])
    out = {"lines": len(R), "lines_ge3_words": len(multi), "clean_lines": len(clean),
           "m1_clean_quantiles": {str(q): float(np.quantile(m_clean, q)) for q in (0.01, 0.05, 0.1, 0.5)}
           if len(m_clean) else None}
    for key in ("S_w", "S_Nk"):
        vals = [x.get(key) for x in multi]
        lens = [len(x["hyp"]) for x in multi]
        f = frac_guaranteed(vals, lens, delta=DELTA)
        out[key] = {"frac_guaranteed_delta20_on_ge3words": f,
                    "per_char_delta_same_frac": per_char_equiv(vals, lens, f),
                    "median": float(np.median([v for v in vals if v is not None]))}
        # erreurs : les lignes avec erreurs sont-elles moins garanties ?
        werr = [x for x in multi if x["err"] > 0]
        out[key]["frac_guaranteed_lines_with_errors"] = frac_guaranteed([x.get(key) for x in werr], [len(x["hyp"]) for x in werr], delta=DELTA)
    adds = [x["add"] for x in R if "add" in x]
    if adds:
        ratio = np.array([a["joint"] / a["sum"] for a in adds if a["sum"] > 0])
        out["additivity"] = {"n": len(ratio), "ratio_joint_over_sum_quantiles":
                             {str(q): float(np.quantile(ratio, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
                             "frac_joint_ge_sum": float((ratio >= 1).mean())}
    rep[name] = out

# système combiné (R1d, a = 0,6) : lignes où les deux gloutons coïncident (NFC), marge par mot
agree = comb = 0
vals, lens = [], []
for r in rows:
    if "a1" not in r or "kr" not in r:
        continue
    ha = unicodedata.normalize("NFC", r["a1"]["hyp"])
    hk = unicodedata.normalize("NFC", r["kr"]["hyp"])
    if ha != hk:
        continue
    agree += 1

    def word_min(x):
        d = {}
        for w, m in zip(x["words"], x["m"]):
            if w >= 0:
                d[w] = min(d.get(w, 1e9), m)
        return d
    wa, wk = word_min(r["a1"]), word_min(r["kr"])
    if len(wa) != len(wk) or len(wa) < 3:
        vals.append(None)
        lens.append(len(ha))
        continue
    combo = sorted(0.6 * wa[i] + 0.4 * wk[i] for i in wa)
    vals.append(float(sum(combo[:3])))
    lens.append(len(ha))
    comb += 1
f = frac_guaranteed(vals, lens, delta=DELTA)
rep["combined_a0.6"] = {"lines_both": sum(1 for r in rows if "a1" in r and "kr" in r), "agreeing_greedy": agree,
                        "agreeing_ge3_words": comb, "frac_guaranteed_delta20_among_agreeing": f,
                        "per_char_delta_same_frac": per_char_equiv(vals, lens, f),
                        "note": "lignes en désaccord : aucune garantie (comptées dans lines_both − agreeing)"}
(HERE / "reports/reader_t2_report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
print(json.dumps(rep, indent=1, ensure_ascii=False))
