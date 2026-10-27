"""Lot 2d — lignes infaisables pour CTC (CLASSE A, descriptif, sans décodage ni CER).

Demande w2:pA (17/10, question de cadence T1). Un treillis CTC de T trames ne peut émettre une suite
de labels y que si |y| + r(y) ≤ T, r(y) = nombre de paires de labels consécutifs identiques (un blanc
obligatoire entre eux). Par ligne et par treillis (A = A1, K = kraken) : y = vérité encodée dans le
codec du treillis (encoder_for de reader_r1d_joint : A en NFC avec repli, K en NFD), besoin = |y| + r(y).
Classes : « hors codec » (un caractère non encodable : besoin non défini, compté à part),
« infaisable » (besoin > T), « marge < 10 % » (0,9 T < besoin ≤ T), sinon faisable.
Jeux : tune (400 lignes du panneau R1e), eval (2 405 lignes, eval déjà exposée : seules les vérités sont
lues), photo_1983 (2 392, train). T_A = len(logp) A1 ; T_K = trames du treillis kraken (treillis E8 de
press pour tune/eval, reader_kraken_photo1983 pour photo_1983). Vérités vides comptées à part ; ligne sans treillis K = « sans_treillis » pour K.
Sortie : reports/reader_feasibility.json.
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
sys.path.insert(0, str(HERE / "src"))

from axel.image.lattice import load as load_lat  # noqa: E402
from reader_r1d_joint import E8LAT, PRED, TIMING, encoder_for  # noqa: E402

R = HERE / "reports"


def need(enc, truth: str) -> int | None:
    ids: list[int] = []
    for ch in truth:
        e = enc(ch)
        if e is None:
            return None
        ids += e
    return len(ids) + sum(1 for a, b in zip(ids, ids[1:]) if a == b)


def classify(n: int | None, t: int | None) -> str:
    if t is None:
        return "sans_treillis"
    if n is None:
        return "hors_codec"
    if n > t:
        return "infaisable"
    if n > 0.9 * t:
        return "marge_moins_10pc"
    return "faisable"


def tally(rows: list[dict]) -> dict:
    out = {"lignes": len(rows), "verites_vides": sum(r["vide"] for r in rows)}
    for k in ("A", "K"):
        c: dict[str, int] = {}
        for r in rows:
            if not r["vide"]:
                c[r[k]] = c.get(r[k], 0) + 1
        out[k] = dict(sorted(c.items()))
    out["infaisable_A_ou_K"] = sum(1 for r in rows if not r["vide"] and "infaisable" in (r["A"], r["K"]))
    out["infaisable_A_et_K"] = sum(1 for r in rows if not r["vide"] and r["A"] == r["K"] == "infaisable")
    return out


def main() -> None:
    labels_a = ("",) + tuple(json.load(open(TIMING))["chars"])
    enc_a = encoder_for(labels_a, nfd=False)
    split = json.loads((R / "reader_r3_split.json").read_text())
    eval_pages, tune_pages = set(split["eval"]), set(split["tune"])
    tune_stems = {r["stem"] for r in json.loads((R / "reader_r1e_tune.json").read_text())["rows"]}
    page = lambda s: s.split("__")[1]  # noqa: E731
    assert len(tune_stems) == 400
    assert all(page(s) in tune_pages and page(s) not in eval_pages for s in tune_stems), "tige hors moitié tune"

    def row(stem, truth, t_a, t_k, enc_k):
        return {"stem": stem, "vide": truth == "", "T_A": t_a, "T_K": t_k,
                "A": classify(need(enc_a, truth), t_a), "K": classify(need(enc_k, truth), t_k)}

    sets: dict[str, list[dict]] = {"tune": [], "eval": []}
    for x in gzip.open(PRED, "rt"):
        r = json.loads(x)
        s = r["stem"]
        key = "tune" if s in tune_stems else ("eval" if page(s) in eval_pages else None)
        if key is None:
            continue
        f = E8LAT / f"{s}.npz"
        if f.exists():
            lat = load_lat(f)
            sets[key].append(row(s, r["truth"], len(r["logp"]), lat.logprobs.shape[0],
                                 encoder_for(tuple(lat.labels), nfd=True)))
        else:
            sets[key].append(row(s, r["truth"], len(r["logp"]), None, enc_a))

    kr, klabels = {}, None
    for x in gzip.open(R / "reader_kraken_photo1983.jsonl.gz", "rt"):
        r = json.loads(x)
        if "labels" in r:
            klabels = tuple(r["labels"])
        else:
            kr[r["stem"]] = len(r["logp"])
    enc_kp = encoder_for(klabels, nfd=True)
    sets["photo_1983"] = []
    for x in gzip.open(R / "reader_a1_photo1983.jsonl.gz", "rt"):
        r = json.loads(x)
        sets["photo_1983"].append(row(r["stem"], r["truth"], len(r["logp"]), kr.get(r["stem"]), enc_kp))

    assert tuple(len(sets[k]) for k in ("tune", "eval", "photo_1983")) == (400, 2405, 2392)
    out = {"lot": "2d — faisabilité CTC (classe A, descriptif ; besoin = |y| + répétitions consécutives)",
           **{k: tally(v) for k, v in sets.items()},
           "lignes_infaisables": {k: [r for r in v if "infaisable" in (r["A"], r["K"])] for k, v in sets.items()}}
    (R / "reader_feasibility.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in out.items() if k != "lignes_infaisables"}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
