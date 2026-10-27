"""Lot abstention (docs/READER_DOTS_ABST_PREREG.txt) — tirage des 200 lignes de SÉLECTION de photo_1983 (TRAIN),
disjointes des 200 de pert200 (négatifs de P3) : random.Random(19).sample sur les 2 192 restantes, triées par
tige. Aucune image lue, aucun décodage. Sortie : reports/reader_items_photo1983_abst200.jsonl (même format que
reader_items_photo1983.jsonl), à décoder ensuite par A1 sur le Mac :
    lourd.sh ~/axel/.venv/bin/python scripts/reader_a1_read.py --items reports/reader_items_photo1983_abst200.jsonl \\
        --perturb <blank|scramble> --suffix '#r-abst' --out reports/reader_a1_photo1983_abst_<mode>.jsonl.gz
"""

import json
import random
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "reports"


def main() -> None:
    key = lambda r: r.get("stem") or f"{r['corpus']}__{r['page']}__{r['line_id']}"  # noqa: E731
    rows = [json.loads(x) for x in open(R / "reader_items_photo1983.jsonl")]
    used = {key(json.loads(x)) for x in open(R / "reader_items_photo1983_pert200.jsonl")}
    assert len(rows) == 2392 and len(used) == 200
    rest = sorted((r for r in rows if key(r) not in used), key=key)
    assert len(rest) == 2192
    pick = sorted(random.Random(19).sample(rest, 200), key=key)
    (R / "reader_items_photo1983_abst200.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in pick))
    print(len(pick), "lignes ;", len({key(r) for r in pick} & used), "en commun avec pert200")


if __name__ == "__main__":
    main()
