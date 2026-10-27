"""R7(b) — contrastes D1 et D2 (K1 contre B0 et B4) sur le panneau 1 (CLASSE B, figé, non lancé).

docs/READER_R7B_PREREG.txt (corrections pM, notices, re-décodage K1 en BaselineLine). D3 et D4 ABSENTS
(P1 non téléchargé). Panneau 1 = les 2 405 lignes eval de R6, SECOND REGARD sur une eval déjà exposée :
D1 et D2 RAPPORTÉS, AUCUNE revendication, même si l'IC exclut 0. Interprétation après l'inventaire de
recouvrement de pN.
Entrées (sha256 vérifiés avant tout calcul) : reports/reader_r6_B0_eval.jsonl.gz,
reports/reader_r6_B4_eval.jsonl.gz, reports/reader_r7b_k1bl_eval.jsonl.gz (re-décodage BaselineLine ;
l'archive bbox reader_r7b_k1_eval n'est PAS lue). Mêmes 2 405 tiges, mêmes vérités (contrôlé).
Contraste (signe pré-écrit) : diff = CER(K1) − CER(bras), positif = notre bras meilleur ; D1 bras = B0,
D2 bras = B4. Erreurs = axel.reader.metric.line_errors(hyp, vérité, "d150") sur les chaînes entières
(caractères hors codec compris). CER d'un ensemble de pages = 100 × somme des erreurs / somme des
longueurs (fonction recopiée de scripts/reader_r6_report.py, non importée : ce script réécrit le JSON R6).
IC apparié par page : pages triées, 10 000 tirages avec remise, random.Random(1), percentiles aux indices
250 et 9 749 ; global, puis BnL (page en chiffres) et NewsEye séparés. MDE indicative = 2,8 × écart-type
des diff tirées (puissance 80 %, α 5 % bilatéral), sans valeur de décision. Pas d'équivalence (pas de
marge fixée, pas de TOST).
Hors codec (descriptif, sans revendication) : caractère de la vérité, en NFKD (normalisation de K1),
absent des « graphemes » de ~/outils-seg/modeles/catmus-print-large/metadata.json (sha256 vérifié) ;
compté en lignes et en caractères, global et par strate.
Environnement : PYTHONPATH=~/axel/src ~/outils-seg/gate-env/bin/python scripts/reader_r7b_report.py
Sortie : reports/reader_r7b_report.json.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import random
import statistics
import sys
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path.home() / "axel/src"))
from axel.reader.metric import line_errors  # noqa: E402

R = HERE / "reports"
INPUTS = {
    "B0": (R / "reader_r6_B0_eval.jsonl.gz", "b3e33a9b0e12fca995e5daebc9d26be4678944fd82657060da36784bc3bd8451"),
    "B4": (R / "reader_r6_B4_eval.jsonl.gz", "dbb971e8e462bd118b09c726f0fbfb99ebf5c153b5d599d8c0f643da6ee56ce0"),
    "K1": (R / "reader_r7b_k1bl_eval.jsonl.gz", "9e51532002e8ab079cf3b13ac4a8e1be2f467ab18a2e4fb4d9f0d024bdcc9030"),
}
META = Path.home() / "outils-seg/modeles/catmus-print-large/metadata.json"
META_SHA = "143db2643936e60a5efdb64731129ba0d3278f66b6d7690ca133e5012b34db60"
CONTRASTS = {"D1": ("K1", "B0"), "D2": ("K1", "B4")}


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    for k, (p, h) in INPUTS.items():
        assert h is not None and sha256(p) == h, f"entrée {k} inattendue"
    assert sha256(META) == META_SHA
    text: dict[str, dict[str, str]] = {}
    truth: dict[str, str] = {}
    for arm, (p, _) in INPUTS.items():
        for x in gzip.open(p, "rt"):
            r = json.loads(x)
            text.setdefault(arm, {})[r["stem"]] = r["hyp"]
            assert truth.setdefault(r["stem"], r["truth"]) == r["truth"], r["stem"]
    stems = sorted(truth)
    assert len(stems) == 2405 and all(set(t) == set(stems) for t in text.values())

    page = lambda s: s.split("__")[1]  # noqa: E731
    E: dict[str, dict[str, list[int]]] = {}
    for arm in text:
        for s in stems:
            e, n = line_errors(text[arm][s], truth[s], "d150")
            acc = E.setdefault(arm, {}).setdefault(page(s), [0, 0])
            acc[0] += e
            acc[1] += n

    def cer(arm, ps):
        return 100 * sum(E[arm][p][0] for p in ps) / max(1, sum(E[arm][p][1] for p in ps))

    def paired(x, y, pool, draws=10000, seed=1):
        rng = random.Random(seed)
        ds = sorted(cer(x, ps) - cer(y, ps) for ps in ([rng.choice(pool) for _ in pool] for _ in range(draws)))
        return [ds[250], ds[9749]], 2.8 * statistics.pstdev(ds)

    pages = sorted({page(s) for s in stems})
    strata: dict[str, list[str]] = {}
    for p in pages:
        strata.setdefault("bnl" if p.isdigit() else "newseye", []).append(p)

    graphemes = set(json.loads(META.read_text())["graphemes"])
    hors: dict[str, list[int]] = {"global": [0, 0]}
    for s in stems:
        miss = sum(ch not in graphemes for ch in unicodedata.normalize("NFKD", truth[s]) if not ch.isspace())
        for k in ("global", "bnl" if page(s).isdigit() else "newseye"):
            h = hors.setdefault(k, [0, 0])
            h[0] += miss > 0
            h[1] += miss

    out = {"lot": "R7(b) — D1, D2 sur le panneau 1 (second regard sur une eval exposée ; aucune revendication)",
           "lignes": len(stems), "pages": len(pages), "pages_par_strate": {k: len(v) for k, v in strata.items()},
           "cer_d150": {a: cer(a, pages) for a in E},
           "cer_d150_par_strate": {k: {a: cer(a, v) for a in E} for k, v in strata.items()},
           "hors_codec_K1_descriptif": {k: {"lignes": v[0], "caracteres": v[1]} for k, v in hors.items()},
           "contrastes": {}}
    for name, (sys_, arm) in CONTRASTS.items():
        ci, mde = paired(sys_, arm, pages)
        out["contrastes"][name] = {
            "diff": f"CER({sys_}) - CER({arm}), positif = {arm} meilleur", "point": cer(sys_, pages) - cer(arm, pages),
            "ic95": ci, "mde_indicative": mde,
            "par_strate": {k: dict(zip(("ic95", "mde_indicative"), paired(sys_, arm, v))) for k, v in strata.items()},
            "revendication": "aucune (panneau 1, second regard sur une eval exposée)"}
    out["provenance"] = {k: sha256(p) for k, (p, _) in INPUTS.items()} | {"metadata": META_SHA,
                                                                            "script": sha256(Path(__file__))}
    (R / "reader_r7b_report.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
