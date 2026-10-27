"""LOT-P07b v11, point (1) of the pQ audit: compare two executions of the same session (v3 vs relaunch), protocol written BEFORE any gap is examined.

Usage: p07b_compare_runs.py <dirA> <dirB> <out.json>
Each dir holds the extracted step archives (p07b_s1_<k>_<step>/runs/...). Nothing is written except <out.json>.

Normalisation N: `seconds` dropped; a line UUID is replaced by a canonical key (integer baseline, ties broken by integer boundary then file order),
taken in the file that PRODUCES the line in the same execution: e12/lines/ppdl3_z (read by e12/read and e22/read) and e28/lines/rest_z (selected by
e28/add/a_z and e32/dev/add/a_m, read by e28/read and e32/dev/read). Lines of B whose key is not in A are paired ONE TO ONE with lines of A by
symmetric baseline coverage >= 0.95 (deterministic greedy: best coverage first, ties by key). Layers, page by page: geometry (blocks, lines),
text (per reader, CER between runs), selections (kept sets, candidate counts), links (every UUID cited downstream exists upstream in the same run,
no duplicate id). Verdict: identique / équivalent / anomalie (thresholds of the lot; the CER ceiling holds for EVERY page and reader, not pooled).
v2 (Sonnet review): reader inventory and empty input feed the verdict; e12/blocks and e28/lines/rest_z compared; one-to-one deterministic pairing.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path.home() / "axel/src"))
from axel.layout.geom import baseline_cover, line_height  # noqa: E402

BLOCKS = "e12/blocks/ppdl3"
LINE_PRODUCERS = {
    "e12/lines/ppdl3_z": {"readers": ["e12/read/ppdl3_z", "e22/read/ppdl3_z"], "selections": []},
    "e28/lines/rest_z": {"readers": ["e28/read/german_print_s/a_z", "e28/read/mccatmus_press/a_z",
                                     "e32/dev/read/german_print_s/extra", "e32/dev/read/mccatmus_press/extra"],
                         "selections": ["e28/add/a_z", "e32/dev/add/a_m"]},
}
COVER = 0.95


def files(root: Path, folder: str) -> dict[str, Path]:
    return {p.stem: p for p in root.glob(f"*/runs/{folder}/*.json")}


def load(p: Path) -> dict:
    return json.loads(p.read_text())


def tup(pts):
    return tuple(tuple(float(v) for v in p) for p in pts)


def ints(pts) -> tuple:
    return tuple(int(round(v)) for p in pts for v in p)


def keyed(lines: list[dict]) -> tuple[dict, dict, list]:
    """Canonical keys; equal integer baselines told apart by integer boundary, then file order. Returns (key->line, id->key, duplicate ids)."""
    groups: dict[tuple, list] = {}
    for i, x in enumerate(lines):
        groups.setdefault(ints(x["baseline"]), []).append((ints(x["boundary"]), i, x))
    by_key, uid, dup = {}, {}, []
    for k, members in groups.items():
        for r, (_, _, x) in enumerate(sorted(members, key=lambda t: (t[0], t[1]))):
            by_key[k + (r,)] = x
            if x["id"] in uid:
                dup.append(x["id"])
            uid[x["id"]] = k + (r,)
    return by_key, uid, dup


def sym_cover(la: dict, lb: dict) -> float:
    ba, bb = tup(la["baseline"]), tup(lb["baseline"])
    return min(baseline_cover(ba, bb, 0.5 * line_height(tup(la["boundary"]), ba)),
               baseline_cover(bb, ba, 0.5 * line_height(tup(lb["boundary"]), bb)))


def pair(ka: dict, kb: dict) -> tuple[dict, int]:
    """B key -> A key, one to one: exact keys first, then the best coverage >= COVER (greedy, deterministic). Returns (map, pairs)."""
    m = {k: k for k in kb if k in ka}
    ra, rb = sorted(set(ka) - set(m.values())), sorted(set(kb) - set(m))
    cand = sorted((-sym_cover(kb[j], ka[i]), j, i) for j in rb for i in ra)
    used_a, used_b = set(), set()
    for c, j, i in cand:
        if -c < COVER:
            break
        if i in used_a or j in used_b:
            continue
        m[j] = i
        used_a.add(i)
        used_b.add(j)
    return m, len(m)


def edit(a: str, b: str) -> int:
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def lines_of(data: dict) -> list[dict]:
    return [ln for b in data["blocks"] for ln in b["lines"]]


def compare(a: Path, b: Path) -> dict:
    rep: dict = {"inventory": {}, "pages": {}, "anomalies": []}
    tot = {"lines": 0, "exact": 0, "found": 0, "chars": 0, "edits": 0, "sel_changed": 0, "sel_lines": 0, "broken": 0, "duplicates": 0,
           "blocks": 0, "blocks_exact": 0, "candidates_diff": 0, "ctc_diff": 0, "fallback_diff": 0, "unread_diff": 0}
    max_cer = 0.0
    folders = [BLOCKS] + [f for p, s in LINE_PRODUCERS.items() for f in [p, *s["readers"], *s["selections"]]]
    inv_ok = True
    for f in folders:
        fa, fb = files(a, f), files(b, f)
        rep["inventory"][f] = {"A": len(fa), "B": len(fb), "only_A": sorted(set(fa) - set(fb)), "only_B": sorted(set(fb) - set(fa))}
        if not fa or not fb or set(fa) != set(fb):
            inv_ok = False
            rep["anomalies"].append(f"inventory {f}: A={len(fa)} B={len(fb)}")
    fa, fb = files(a, BLOCKS), files(b, BLOCKS)   # blocks: integer polygons and labels, as multisets per page
    for page in sorted(set(fa) & set(fb)):
        pa = sorted(ints(x["polygon"]) + (x["label"],) for x in load(fa[page])["blocks"])
        pb = sorted(ints(x["polygon"]) + (x["label"],) for x in load(fb[page])["blocks"])
        tot["blocks"] += max(len(pa), len(pb))
        tot["blocks_exact"] += sum((Counter(pa) & Counter(pb)).values())   # multiset: identical polygons are not merged
    for producer, spec in LINE_PRODUCERS.items():
        fa, fb = files(a, producer), files(b, producer)
        for page in sorted(set(fa) & set(fb)):
            ka, uid_a, dup_a = keyed(lines_of(load(fa[page])))
            kb, uid_b, dup_b = keyed(lines_of(load(fb[page])))
            tot["duplicates"] += len(dup_a) + len(dup_b)
            m, paired = pair(ka, kb)
            exact = [k for k, v in m.items() if k == v and ints(ka[v]["boundary"]) == ints(kb[k]["boundary"])]
            tot["lines"] += max(len(ka), len(kb))
            tot["exact"] += len(exact)
            tot["found"] += paired
            to_a = {u: m.get(k, ("B-only",) + k) for u, k in uid_b.items()}   # B id -> A key, or a B-only marker
            pg = {"lines_A": len(ka), "lines_B": len(kb), "exact": len(exact), "paired": paired}
            for sel in spec["selections"]:
                sa, sb = files(a, sel).get(page), files(b, sel).get(page)
                if sa is None or sb is None:
                    continue   # counted by the inventory
                da, db = load(sa), load(sb)
                tot["broken"] += sum(x["id"] not in uid_a for x in da["lines"]) + sum(x["id"] not in uid_b for x in db["lines"])
                A = {uid_a[x["id"]] for x in da["lines"] if x["id"] in uid_a}
                B = {to_a[x["id"]] for x in db["lines"] if x["id"] in uid_b}
                tot["sel_changed"] += len(A ^ B)
                tot["sel_lines"] += len(A) + len(B)
                tot["candidates_diff"] += da.get("candidates") != db.get("candidates")
                pg[f"sel:{sel}"] = {"A": len(A), "B": len(B), "changed": len(A ^ B), "candidates": [da.get("candidates"), db.get("candidates")]}
            for reader in spec["readers"]:
                ra, rb = files(a, reader).get(page), files(b, reader).get(page)
                if ra is None or rb is None:
                    continue   # counted by the inventory
                da, db = load(ra), load(rb)
                ta, tb = da["texts"], db["texts"]
                bad = sum(u not in uid_a for u in ta) + sum(u not in uid_b for u in tb)
                tot["broken"] += bad
                ca = {uid_a[u]: t for u, t in ta.items() if u in uid_a}
                cb = {to_a[u]: t for u, t in tb.items() if u in uid_b}
                chars = edits = changed = 0
                for k in set(ca) | set(cb):
                    x, y = ca.get(k, ""), cb.get(k, "")
                    chars += max(len(x), len(y))
                    e = edit(x, y)
                    edits += e
                    changed += e > 0
                cer = edits / max(chars, 1)
                max_cer = max(max_cer, cer)
                tot["chars"] += chars
                tot["edits"] += edits
                ctc_a = {uid_a[u]: v for u, v in da.get("ctc", {}).items() if u in uid_a}
                ctc_b = {to_a[u]: v for u, v in db.get("ctc", {}).items() if u in uid_b}
                ctc_diff = sum(ctc_a.get(k) != ctc_b.get(k) for k in set(ctc_a) | set(ctc_b))
                fb_a = sorted(map(str, (uid_a.get(u, u) for u in da.get("fallback", []))))
                fb_b = sorted(map(str, (to_a.get(u, u) for u in db.get("fallback", []))))
                tot["ctc_diff"] += ctc_diff
                tot["fallback_diff"] += fb_a != fb_b
                if producer == "e12/lines/ppdl3_z":   # the c1 chain reads every line: unread counts must agree
                    tot["unread_diff"] += abs((len(ka) - len(ca)) - (len(kb) - len(cb)))
                pg[f"text:{reader}"] = {"lines": len(set(ca) | set(cb)), "changed": changed, "cer_between_runs": cer, "ctc_diff": ctc_diff,
                                        "broken": bad}
            rep["pages"][f"{producer}|{page}"] = pg
    found_share = tot["found"] / max(tot["lines"], 1)
    sel = tot["sel_changed"] / max(tot["sel_lines"], 1)
    if tot["lines"] == 0 or tot["chars"] == 0:
        inv_ok = False
        rep["anomalies"].append("empty input")
    clean = inv_ok and tot["broken"] == 0 and tot["duplicates"] == 0
    if (clean and tot["exact"] == tot["lines"] and tot["edits"] == 0 and tot["sel_changed"] == 0 and tot["blocks_exact"] == tot["blocks"]
            and tot["candidates_diff"] == 0 and tot["ctc_diff"] == 0 and tot["fallback_diff"] == 0 and tot["unread_diff"] == 0):
        verdict = "identique"
    elif clean and found_share >= 0.99 and max_cer <= 0.001 and sel <= 0.01:
        verdict = "équivalent"
    else:
        verdict = "anomalie"
    rep["totals"] = {**tot, "found_share": found_share, "cer_pooled": tot["edits"] / max(tot["chars"], 1), "cer_max_page_reader": max_cer,
                     "selection_changed_share": sel, "inventory_complete": inv_ok}
    rep["verdict"] = verdict
    return rep


def main() -> int:
    a, b, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    rep = compare(a, b)
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": rep["verdict"], **rep["totals"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
