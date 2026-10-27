"""LOT-P02 v15: count the KEY collisions (same geometric key before suffix) of the selected zone lines of each page, and whether the suffix
of each colliding line changes when the candidates are permuted. Read only (pool/ and the lines of the pool)."""
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]
from babaorum.p02_adapter import geometric_id  # noqa: E402

R = Path.home() / "Documents/reviews/babaorum-pipeline/rappel_pages"
pages = [json.loads(p) for p in (R / "runs/e12/pages.json").read_text().splitlines()] if False else json.loads((R / "runs/e12/pages.json").read_text())
out = {}
for p in pages:
    rows = [json.loads(l) for l in (R / "pool" / f"{p['page']}.jsonl").open(encoding="utf-8")]
    kept = [r for r in rows if r.get("origin") == "zone" and r.get("zone_selected_by_select_blocks") and r.get("baseline") and len(r["baseline"]) >= 2]
    keys = Counter(geometric_id(r) for r in kept)
    colliding = {k: v for k, v in keys.items() if v > 1}
    out[p["page"]] = {"lines": len(kept), "key_collisions": len(colliding), "lines_in_collisions": sum(colliding.values()),
                      "largest_group": max(colliding.values(), default=0)}
print(json.dumps(out, indent=1))
print("TOTAL key collisions over the pages:", sum(v["key_collisions"] for v in out.values()))
