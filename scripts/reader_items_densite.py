"""Négatifs à densité appariée — items de photo_1983 avec leur ligne DONNEUSE (mode occlude_donor).

docs/READER_NEG_DENSITE_PREREG.txt. Pour chaque ligne de reports/reader_items_photo1983.jsonl : donneuse =
une autre ligne de la MÊME image, de vérité différente, dont la hauteur de boîte est la plus proche ;
égalités départagées par la clé (corpus, page, line_id). Aucune lecture d'image, aucun décodage.
Sortie : reports/reader_items_photo1983_densite.jsonl (items d'origine + donor_key, donor_box, donor_polygon).
"""
import json
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "reports"


def main() -> None:
    rows = [json.loads(x) for x in open(R / "reader_items_photo1983.jsonl")]
    assert len(rows) == 2392
    key = lambda r: (r["corpus"], r["page"], r["line_id"])  # noqa: E731
    h = lambda r: r["box"][3] - r["box"][1]  # noqa: E731
    by_image: dict[str, list[dict]] = {}
    for r in rows:
        by_image.setdefault(r["image"], []).append(r)
    out = []
    for r in rows:
        cands = [c for c in by_image[r["image"]] if key(c) != key(r) and c["truth"] != r["truth"]]
        d = min(cands, key=lambda c: (abs(h(c) - h(r)), key(c)))
        out.append(dict(r, donor_key="__".join(key(d)), donor_box=d["box"], donor_polygon=d.get("polygon")))
    (R / "reader_items_photo1983_densite.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out))
    print(len(out), "items ;", len({x["donor_key"] for x in out}), "donneuses distinctes")


if __name__ == "__main__":
    main()
