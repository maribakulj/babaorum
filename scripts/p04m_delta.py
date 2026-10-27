"""AV-P04 (texte 939a1430, avenants v1.1 à v1.4, GO pQ) : Δ de la mesure unique du blla affiné, sur le Mac. Writes ONE small JSON (--out,
refused if it exists), ALWAYS with a structured outcome (closed / incomplete / one of the 4 issues); its SHA is pinned in the note B before
the pass-2 bench is read. Exit 0 only for one of the 4 issues.

Order (pQ M1): the gate FIRST (nothing of pass 2 is opened or hashed before it is open), then the chain of SHA links (pQ B4):
  porte_pairage.json --(sha in report 1 gate.sha256; sha in report 2 consumed.porte_sha256)--> report 1 --(sha in report 2
  consumed.passe1_report_sha256)--> report 2; each report --(report_sha256, archive_sha256 = runs_archive_sha256)--> its bench LINK
  (p04m_bench.py) --(bench_sha256)--> its bench. Complete schemas (no None == None), same axel commit, manifest, freeze, CPU, truth manifest,
  config and bench axel head in both passes; bench rows: no duplicate (page, chain), finite CER, positive integer chars.
  Per pass (pQ B4, 2nd audit): link.axel_head == bench.provenance.axel_head, both starting with the report's axel_commit; the bench axel
  worktree was clean; the producers actually read by the bench (link.producers, its sha recomputed), p04m_bench.py itself and the bench
  interpreter with its installed distributions (link.bench_env) are the SAME in both passes.
  Any unexpected error also ends in a structured "MESURE INCOMPLÈTE" (pQ minor).
Then: the 24 main pages present in both benches with the same true chars; δ_p = cer(pass 1) − cer(pass 2), c1+ma; Δ = Σ chars δ / Σ chars;
95 % CI by bootstrap_by_groups (p08_oracles 5dc107c9, key "delta", 10 000 draws, seed 20261017, 21 groups); outcome (TEM 0.005, inclusive;
lower bound strictly > 0): (a) Δ >= TEM and lower > 0; (b) lower > 0 and Δ < TEM; (c′) lower <= 0 and Δ >= TEM; (c) lower <= 0 and Δ < TEM.
Outside the criterion: the 2 disagreement pages; the CER re-read from the delivered ALTO (p01c.alto.round_trip.cer_reordered_from_export,
pages common to both passes); pass 1 against the historic reference bench.
    PYTHONDONTWRITEBYTECODE=1 ~/axel/.venv/bin/python scripts/p04m_delta.py --porte porte_pairage.json --report1 p04m_passe1_report.json \
        --report2 p04m_passe2_report.json --link1 reports/bench_p04m_passe1.link.json --link2 reports/bench_p04m_passe2.link.json \
        --bench1 reports/bench_p04m_passe1.json --bench2 reports/bench_p04m_passe2.json --reference reports/bench_pipeline_p07b_26.json \
        --out runs/p04m/delta.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from babaorum.p08_oracles import bootstrap_by_groups  # noqa: E402

TEM = 0.005
CONFIG_SHA = "c8216a7ea475b5abc3ce1e4917e06e44288ec906d226a437bb65a2019f034389"
TRUTH_MANIFEST_SHA = "7f2b90e23611750e02777b8dc84877c664bb6062687e65b32ac468ede7f0b16d"
ORACLES_SHA = "5dc107c9"
CHAIN = "c1+ma"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
REQUIRED_PRODUCERS = ("babaorum:scripts/bench_pipeline.py", "babaorum:config/bench_pipeline_v1.json", "babaorum:src/babaorum/schemas/alto-4-4.xsd",
                      "babaorum:src/babaorum/schemas/page-2019-07-15.xsd", "babaorum:src/babaorum/schemas/xlink.xsd")   # = p04m_bench.py
# the 24 main pages and their 21 groups (number, title), frozen in AV-P04 § 3
GROUPS = {
    "Marie-Claire_4613173_T0000004": "4613173|marie claire", "Marie-Claire_4621234_T0000002": "4621234|marie claire",
    "Marie-Claire_4621234_T0000008": "4621234|marie claire", "L_OEuvre_4613537_T0000001": "4613537|l oeuvre",
    "L_OEuvre_4613814_T0000002": "4613814|l oeuvre", "L_OEuvre_4615451_T0000001": "4615451|l oeuvre", "L_OEuvre_4616168_T0000002": "4616168|l oeuvre",
    "L_OEuvre_4617562_T0000007": "4617562|l oeuvre", "L_OEuvre_4620081_T0000004": "4620081|l oeuvre", "L_OEuvre_4621234_T0000001": "4621234|l oeuvre",
    "L_OEuvre_4621234_T0000010": "4621234|l oeuvre", "La_Presse_0476474_T0000001": "476474|la presse", "La_Presse_0477885_T0000002": "477885|la presse",
    "La_Presse_0514199_T0000004": "514199|la presse", "La_Presse_0542128_T0000001": "542128|la presse", "La_Presse_0543209_T0000001": "543209|la presse",
    "La_Presse_0545076_T0000002": "545076|la presse", "Le_Matin_0551841_T0000003": "551841|le matin", "Le_Matin_0569308_T0000002": "569308|le matin",
    "Le_Matin_0574578_T0000002": "574578|le matin", "Le_Matin_0576768_T0000002": "576768|le matin", "Le_Matin_0576768_T0000006": "576768|le matin",
    "Le_Matin_0586574_T0000003": "586574|le matin", "La_Fronde_6704254_T0000002": "6704254|la fronde",
}
APART = ("Le_Matin_0554721_T0000004", "Le_Matin_0558051_T0000005")
# note B (pA, 25/10): the pass-1 gate closed on 3 pages produced on the Mac; ONLY this exact gate file is accepted as closed. Δ stays on the
# 24 pages / 21 groups; a sensitivity on the 21 pages / 19 groups where the gate held is reported beside it; different outcomes = FRAGILE
NOTE_B_GATE_SHA = "05c1bb5267a31c1396f48ddd69a71eba38a3c6804d039927a8e2b6cf9d9b6e07"
NOTE_B_CLOSING = ["L_OEuvre_4613537_T0000001", "Le_Matin_0551841_T0000003", "Marie-Claire_4621234_T0000008"]


class Incomplete(Exception):
    pass


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def producers_sha(producers: dict) -> str:                                     # same canonical form as p04m_bench.py
    return hashlib.sha256(json.dumps(producers, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def load(p: Path, what: str):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError) as e:
        raise Incomplete(f"{what}: unreadable ({type(e).__name__})") from None


def check_report(r, k: int, name: str) -> None:
    if not isinstance(r, dict):
        raise Incomplete(f"{name}: not a JSON object")
    need = {"passe": int, "incomplete": list, "elapsed_hours": (int, float), "axel_commit": str, "manifest_p04m": str, "freeze": dict,
            "machine": dict, "runs_archive_sha256": str}
    bad = [key for key, t in need.items() if not isinstance(r.get(key), t) or isinstance(r.get(key), bool)]
    if bad:
        raise Incomplete(f"{name}: missing or mistyped fields {bad}")
    if r["passe"] != k or r["incomplete"]:
        raise Incomplete(f"{name}: pass {r['passe']}, incomplete {r['incomplete']}")
    if not finite(r["elapsed_hours"]) or not 0 < r["elapsed_hours"] <= 12:
        raise Incomplete(f"{name}: elapsed hours {r['elapsed_hours']} not in (0, 12]")
    if not HEX64.match(r["manifest_p04m"]) or not HEX64.match(r["runs_archive_sha256"]) or not r["freeze"] \
            or not isinstance(r["machine"].get("cpu_model"), str) or not r["machine"]["cpu_model"]:
        raise Incomplete(f"{name}: manifest, archive sha, freeze or CPU model invalid")


def rows_of(bench, name: str) -> dict:
    rows, seen = {}, set()
    if not isinstance(bench, dict) or not isinstance(bench.get("chains"), list) or not all(isinstance(r, dict) for r in bench["chains"]):
        raise Incomplete(f"{name}: not a bench (object with a list of row objects)")
    for r in bench["chains"]:
        key = (r.get("chain"), r.get("page"))
        if key in seen:
            raise Incomplete(f"{name}: duplicate row {key}")
        seen.add(key)
        if r.get("chain") == CHAIN:
            if not finite(r.get("cer_reordered")) or not isinstance(r.get("chars"), int) or isinstance(r.get("chars"), bool) or r["chars"] <= 0:
                raise Incomplete(f"{name}: invalid cer or chars on {r.get('page')}")
            rows[r["page"]] = r
    return rows


def alto_cer(r: dict | None):
    try:
        v = r["p01c"]["alto"]["round_trip"]["cer_reordered_from_export"]
    except (KeyError, TypeError):
        return None
    return v if finite(v) else None


def outcome(delta: float, lo: float) -> str:
    if lo > 0:
        return "(a) SUCCÈS" if delta >= TEM else "(b) gain réel, sous le seuil utile : pas d'adoption, modèle archivé"
    return ("(c′) NON DÉTECTÉ, estimation ≥ 0,5 point : modèle NON archivé, confirmation sur un jeu élargi (lot neuf)" if delta >= TEM
            else "(c) NON DÉTECTÉ (puissance d'environ 50 % à 0,5 point), pas absent : modèle archivé")


def measure(a, rep: dict) -> int:
    P = {k: Path(getattr(a, k)) for k in ("porte", "report1", "report2", "link1", "link2", "bench1", "bench2", "reference")}
    # 1. the gate, before anything of pass 2
    if not P["porte"].is_file():
        rep["outcome"] = "PORTE ABSENTE : traitée comme FERMÉE ; rien n'est mesuré"
        return 1
    porte = load(P["porte"], "porte")
    rep["inputs"] = {"porte": sha(P["porte"])}
    pages = porte.get("pages") if isinstance(porte, dict) else None
    note_b = rep["inputs"]["porte"] == NOTE_B_GATE_SHA
    closing = NOTE_B_CLOSING if note_b else []
    if not (isinstance(porte, dict) and porte.get("porte_ouverte") is (not note_b) and porte.get("pages_closing") == closing
            and isinstance(pages, dict) and len(pages) == 26
            and all(isinstance(v, dict) and v.get("page_ok") is (p not in closing) for p, v in pages.items())):
        rep["outcome"] = "PORTE FERMÉE (ou incohérente) : ARRÊT, rien n'est mesuré"
        return 1
    rep["gate_note_b"] = note_b
    # 2. pass 1 report bound to the gate
    r1 = load(P["report1"], "report1")
    check_report(r1, 1, "report1")
    if not isinstance(r1.get("gate"), dict) or r1["gate"].get("porte_ouverte") is not (not note_b) or r1["gate"].get("pages_closing") != closing \
            or r1["gate"].get("sha256") != rep["inputs"]["porte"]:
        raise Incomplete("report1 does not record THIS open gate")
    rep["inputs"]["report1"] = sha(P["report1"])
    # 3. pass 2 (only now)
    r2 = load(P["report2"], "report2")
    check_report(r2, 2, "report2")
    cons = r2.get("consumed")
    if not isinstance(cons, dict) or cons.get("porte_sha256") != rep["inputs"]["porte"] or cons.get("passe1_report_sha256") != rep["inputs"]["report1"]:
        raise Incomplete("report2 is not bound to this gate and this pass-1 report")
    if cons.get("porte_note_b", False) is not note_b:                         # the notebook records whether it took the note-B gate
        raise Incomplete("report2: porte_note_b does not match the gate")
    rep["inputs"]["report2"] = sha(P["report2"])
    for key in ("axel_commit", "manifest_p04m", "freeze"):
        if r1[key] != r2[key]:
            raise Incomplete(f"{key} differs between the passes")
    if r1["machine"]["cpu_model"] != r2["machine"]["cpu_model"]:
        raise Incomplete("CPU model differs between the passes")
    # 4. bench links and benches
    benches = {}
    for k, r in ((1, r1), (2, r2)):
        lk = load(P[f"link{k}"], f"link{k}")
        rep["inputs"][f"link{k}"] = sha(P[f"link{k}"])
        if not isinstance(lk, dict) or lk.get("passe") != k or lk.get("report_sha256") != rep["inputs"][f"report{k}"] \
                or lk.get("archive_sha256") != r["runs_archive_sha256"] or lk.get("truth_manifest_sha256") != TRUTH_MANIFEST_SHA \
                or lk.get("config_sha256") != CONFIG_SHA:
            raise Incomplete(f"link{k}: not bound to report {k}, its archive, the frozen truth or the config")
        if not P[f"bench{k}"].is_file() or sha(P[f"bench{k}"]) != lk.get("bench_sha256"):
            raise Incomplete(f"bench{k}: absent or not the linked bench")
        rep["inputs"][f"bench{k}"] = lk["bench_sha256"]
        b = load(P[f"bench{k}"], f"bench{k}")
        prov = b.get("provenance") if isinstance(b, dict) else None
        if not isinstance(prov, dict) or prov.get("config_sha256") != CONFIG_SHA:
            raise Incomplete(f"bench{k}: config differs")
        head, commit = lk.get("axel_head"), r["axel_commit"]
        if not (isinstance(head, str) and HEX40.match(head) and len(commit) >= 7 and head.startswith(commit) and prov.get("axel_head") == head):
            raise Incomplete(f"link{k}/bench{k}: axel head {head!r} / {prov.get('axel_head')!r} is not the report's commit {commit!r}")
        prods = lk.get("producers")
        if lk.get("axel_clean") is not True or not isinstance(prods, dict) or not prods or not isinstance(lk.get("script_sha256"), str) \
                or any(x not in prods for x in REQUIRED_PRODUCERS) or not any(x.startswith("axel:src/") for x in prods) \
                or not all(isinstance(x, str) and isinstance(y, str) and HEX64.match(y) for x, y in prods.items()) \
                or lk.get("producers_sha256") != producers_sha(prods):
            raise Incomplete(f"link{k}: bench producers absent, malformed or axel worktree not clean")
        benches[k] = (b, lk)
    if benches[1][1]["axel_head"] != benches[2][1]["axel_head"]:
        raise Incomplete("bench axel head differs between the passes")
    if benches[1][1]["script_sha256"] != benches[2][1]["script_sha256"]:
        raise Incomplete("p04m_bench.py differs between the passes")
    if not isinstance(benches[1][1].get("bench_env"), list) or not benches[1][1]["bench_env"] \
            or benches[1][1]["bench_env"] != benches[2][1].get("bench_env"):
        raise Incomplete("bench interpreter or installed distributions absent or different between the passes")
    if benches[1][1]["producers_sha256"] != benches[2][1]["producers_sha256"]:
        changed = sorted(set(benches[1][1]["producers"].items()) ^ set(benches[2][1]["producers"].items()))
        raise Incomplete(f"bench producers differ between the passes: {[x for x, _ in changed][:5]}")
    rep["producers_sha256"] = benches[1][1]["producers_sha256"]
    ref = load(P["reference"], "reference")
    rep["inputs"]["reference"] = sha(P["reference"])
    rw1, rw2, rr = rows_of(benches[1][0], "bench1"), rows_of(benches[2][0], "bench2"), rows_of(ref, "reference")
    missing = [p for p in GROUPS if p not in rw1 or p not in rw2]
    if missing:
        raise Incomplete(f"main pages missing: {missing}")
    chars_diff = [p for p in GROUPS if rw1[p]["chars"] != rw2[p]["chars"]]
    if chars_diff:
        raise Incomplete(f"true chars differ between passes: {chars_diff}")
    per_page = {p: {"cer_pass1": rw1[p]["cer_reordered"], "cer_pass2": rw2[p]["cer_reordered"], "chars": rw1[p]["chars"],
                    "delta": rw1[p]["cer_reordered"] - rw2[p]["cer_reordered"]} for p in GROUPS}
    rep["per_page"] = per_page
    rep["apart"] = {p: ({"cer_pass1": rw1[p]["cer_reordered"], "cer_pass2": rw2[p]["cer_reordered"],
                         "delta": rw1[p]["cer_reordered"] - rw2[p]["cer_reordered"]} if p in rw1 and p in rw2 else "non mesurée") for p in APART}
    common = [p for p in GROUPS if alto_cer(rw1[p]) is not None and alto_cer(rw2[p]) is not None]
    rep["alto_outside_criterion"] = {"pages_common": len(common), "not_measured": sorted(set(GROUPS) - set(common)),
                                     "per_page": {p: {"alto_cer_pass1": alto_cer(rw1[p]), "alto_cer_pass2": alto_cer(rw2[p])} for p in common}}
    rep["pass1_vs_reference_outside_gate"] = {p: {"cer_pass1": rw1[p]["cer_reordered"], "cer_reference": rr[p]["cer_reordered"]}
                                              for p in list(GROUPS) + list(APART) if p in rw1 and p in rr}
    rows = [{"page": p, "group": GROUPS[p], "chars": v["chars"], "delta": v["delta"]} for p, v in per_page.items()]
    w = sum(r["chars"] for r in rows)
    delta = sum(r["chars"] * r["delta"] for r in rows) / w
    boot = bootstrap_by_groups(rows, ["delta"])
    lo, hi = boot["ci95"]["delta"]
    if not all(finite(x) for x in (delta, lo, hi)):
        raise Incomplete("non-finite Δ or CI")
    rep.update({"delta": delta, "ci95": [lo, hi], "bootstrap": {k: boot[k] for k in ("groups", "draws", "seed")}, "outcome": outcome(delta, lo)})
    if note_b:                                                                # same Δ, same bootstrap, on the pages where the gate held
        sub = [r for r in rows if r["page"] not in closing]
        if len(sub) != 21 or len({r["group"] for r in sub}) != 19:
            raise Incomplete("note B sensitivity: not 21 pages in 19 groups")
        d_s = sum(r["chars"] * r["delta"] for r in sub) / sum(r["chars"] for r in sub)
        b_s = bootstrap_by_groups(sub, ["delta"])
        lo_s, hi_s = b_s["ci95"]["delta"]
        if not all(finite(x) for x in (d_s, lo_s, hi_s)):
            raise Incomplete("non-finite sensitivity Δ or CI")
        rep["sensitivity_note_b"] = {"pages": len(sub), "groups": b_s["groups"], "excluded": closing, "delta": d_s, "ci95": [lo_s, hi_s],
                                     "outcome": outcome(d_s, lo_s)}
        rep["fragile"] = rep["sensitivity_note_b"]["outcome"] != rep["outcome"]
        rep["robustness"] = "FRAGILE : la sensibilité (21 pages) ne donne pas la même issue" if rep["fragile"] else "concordant"
        if rep["fragile"] and rep["outcome"].startswith("(a)"):               # pA, 25/10: a FRAGILE success does not allow adoption
            rep["decision_note_b"] = ("(a) FRAGILE traité comme (c′) : PAS d'adoption, modèle NON archivé, confirmation sur un jeu élargi "
                                      "d'environ 38 pages")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    for k in ("porte", "report1", "report2", "link1", "link2", "bench1", "bench2", "reference", "out"):
        ap.add_argument(f"--{k}", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)                             # the structured outcome must always be writable (Sonnet)
    if out.exists():
        raise SystemExit(f"--out exists, refused: {out}")
    rep: dict = {"note": "AV-P04 : mesure unique du blla affiné ; Δ = passe 1 − passe 2", "tem": TEM, "script": sha(Path(__file__))}
    if not sha(ROOT / "src/babaorum/p08_oracles.py").startswith(ORACLES_SHA) or len(GROUPS) != 24 or len(set(GROUPS.values())) != 21:
        rep["outcome"] = "MESURE INCOMPLÈTE : pièces épinglées non conformes (p08_oracles ou groupes)"
        code = 1
    else:
        try:
            code = measure(a, rep)
        except Incomplete as e:
            rep["problems"] = [str(e)]
            rep["outcome"] = "MESURE INCOMPLÈTE : aucun verdict ; valeurs disponibles à titre descriptif seulement"
            code = 1
        except Exception as e:                                                # pQ minor: the structured outcome is promised for ANY input
            rep["problems"] = [f"unexpected {type(e).__name__}: {str(e)[:300]}"]
            rep["outcome"] = "MESURE INCOMPLÈTE : entrée inattendue, aucun verdict"
            code = 1
    with open(out, "x", encoding="utf-8") as f:
        f.write(json.dumps(rep, ensure_ascii=False, indent=1))
    print(json.dumps({k: rep.get(k) for k in ("delta", "ci95", "outcome", "problems")}, ensure_ascii=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
