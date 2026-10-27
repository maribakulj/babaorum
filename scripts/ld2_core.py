"""LD-2 : logique pure, SANS torch (docs/END_TO_END_LD2_LOT.txt §0, §4-§7 ; conception §10 (5), §11).

Formules de pas, critère (a)+(b), bootstrap apparié, lecture de la référence B3 + λ, réserves à écrire en tête du rapport,
délais. Importé par scripts/train_ld2.py et par les tests légers. Aucune page n'est lue ici.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "ld2.json"

# Réserves du lot, recopiées MOT POUR MOT (§0, §5 (4), §6, §7 (5)) ; un test les retrouve dans docs/END_TO_END_LD2_LOT.txt.
RESERVES = [
    "mise à l'échelle par la GT, partagée avec la référence ;",
    "pages d'évaluation déjà vues trois fois (E2E-1, E2E-4, E2E-5) ;",
    "bootstrap peu puissant sur 10 pages ;",
    "zones PP réglées sur Gallica et NewsEye ;",
    "LD-0R : résultat sur pages réelles, à citer.",
    "le seuil de score vient de LD-1, choisi sur des pages vues par l'encodeur, en Z-GT, sans CTC (voir LD-1 §10 (9)).",
    "Sur 26 pages réelles de presse (LD-0R, pages de test du pipeline, rapport 6acc06f2…), les zones PP-DocLayoutV3 par "
    "blocs couvrent 0,934 des lignes, avec la marge de 16 px (0,901 sans marge), et 0,756 sur la pire page "
    "(Le_Matin_0554721_T0000004). Le rappel en zones PP sur du réel est donc plafonné vers 0,93. Les résultats de LD-2, "
    "obtenus sur des pages synthétiques (couverture 0,998), ne se transposent pas tels quels au réel.",
    "LD-1B (rapport e7750d10…) : en Z-GT, rappel 0,843 pour les lignes à moins de 8 px du bord de zone contre 1,000 au-delà "
    "de 32 px ; piste de bord citée, sans effet sur ce lot.",
]
DECODER_PAIRING_LIMIT = ("down, l'encodeur du transformer et le décodeur viennent de LD-1, entraînés avec un enc_det affiné ; "
                         "ils sont appariés ici à l'enc_det d'E2E-5 non affiné (lettre du lot).")
TILE_SELECTION_LIMIT = ("témoin : sélection des lignes de tuile héritée d'E2E-5 (bande traitée comme verticale), non corrigée pour "
                        "garder la définition de référence")
ALTO_MENTION = "pas d'ALTO livré dans LD-2, règle du 21/10 (texte brut ET ALTO) non applicable"
CER_SCOPE = "CER du texte brut des lignes appariées, " + ALTO_MENTION + " ; conditionnel aux lectures obtenues"


def load_config(path=CONFIG) -> dict:
    return json.loads(Path(path).read_text())


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_sha(path, expected: str, what: str) -> str:
    """Refus si le sha256 complet diffère (checkpoints d'initialisation)."""
    got = sha256_file(path)
    if got != expected:
        raise SystemExit(f"refusé : {what} de sha différent ({got[:12]}… != {expected[:12]}…)")
    return got


def eval_page_ids() -> list:
    """Les 10 pages d'évaluation (mf_0050-0059). Appelée par l'étape EVAL seule, jamais en fumée."""
    return [f"mf_{i:04d}" for i in range(50, 60)]


# ---------- pas (§5 (1), (2)) ----------

def steps_main(s_per_step: float, cfg: dict) -> int:
    """S2 = min(20 000, floor(8 400 s / s_par_pas_principal))."""
    return int(min(cfg["steps"]["max_steps"], math.floor(cfg["steps"]["main_budget_s"] / s_per_step)))


def steps_witness(s2: int, s_per_step: float, cfg: dict) -> int:
    """S_t = min(S2, floor(4 200 s / s_par_pas_témoin))."""
    return int(min(s2, math.floor(cfg["steps"]["witness_budget_s"] / s_per_step)))


def allowed(cap: float, remaining: float, reserve: float = 0.0) -> float:
    """max(0, min(plafond, restant - réserve)) (LD-1 gel 7)."""
    return max(0.0, min(cap, remaining - reserve))


def check_reader_keys(have: dict, want: dict) -> None:
    """Gel 2, Q1 : le Reader vient d'E2E-5 ; refus si une clé manque, est en trop ou a une autre forme. have, want : {clé: forme}."""
    if set(have) != set(want):
        raise SystemExit(f"refusé : clés reader d'E2E-5 différentes (manquantes {sorted(set(want) - set(have))[:3]}, "
                         f"en trop {sorted(set(have) - set(want))[:3]})")
    bad = [k for k in want if tuple(have[k]) != tuple(want[k])]
    if bad:
        raise SystemExit(f"refusé : formes reader d'E2E-5 différentes : {bad[:3]}")


def witness_report(out) -> dict:
    """État du témoin tel qu'il est : manifeste (S_t), fichier d'état de EW, et codes de sortie de TW et EW écrits par le carnet
    dans <RUN>/../exits. Un arrêt dur (code non nul sans fichier d'état) donne HARD_STOP ; S_t = 0 donne NOT_RUN."""
    out = Path(out)
    man = read_json(out / "ld2_manifest.json")
    st_path = out / "ld2_witness_status.json"
    st = read_json(st_path) if st_path.is_file() else {}
    rcs = {}
    for ph in ("TW", "EW"):
        f = out.parent / "exits" / f"ld2_{ph}_rc.txt"
        if f.is_file():
            rcs[ph] = f.read_text().strip()
    status = st.get("status")
    if not man.get("S_t"):
        status = "NOT_RUN"
    elif status is None:
        status = "HARD_STOP" if rcs.get("EW", "0") != "0" else "NOT_RUN"
    return dict(st, status=status, S2=man.get("S2"), S_t=man.get("S_t"), timing=man["witness"].get("status"),
                tw_rc=rcs.get("TW"), ew_rc=rcs.get("EW"))


# ---------- manifeste ----------

def read_json(path) -> dict:
    return json.loads(Path(path).read_text())


def write_json(path, obj) -> None:
    Path(path).write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str))


def freeze_record(cfg: dict, ckpt_name: str, ckpt_sha: str, step: int, frozen_sha: dict) -> dict:
    """§4 (1) et §7 (2) : seuil, NMS et checkpoint, écrits AVANT l'évaluation. Rien ne dépend de mf_0050-0059."""
    fz = cfg["freeze"]
    return {"threshold": fz["threshold"], "nms": dict(fz["nms"]), "checkpoint": ckpt_name, "checkpoint_sha256": ckpt_sha,
            "checkpoint_step": step, "checkpoint_rule": fz["checkpoint_rule"], "frozen_files_sha256": frozen_sha,
            "written_before_EVAL": True}


# ---------- critère (§0 (a)+(b), conception §11) ----------

def page_pr(row: dict) -> tuple:
    """(P, R) d'une page : n_pred = 0 ou précision None donne P = 0 ; le rappel manquant vaut 0 (conception §10 (5))."""
    p = row["precision"] if (row.get("n_pred", 1) > 0 and row.get("precision") is not None) else 0.0
    return float(p), float(row.get("recall") or 0.0)


def mean_pr(rows: list, n_pages: int) -> dict:
    """Moyenne non pondérée sur exactement n_pages pages (dénominateur fixe)."""
    if len(rows) != n_pages:
        raise ValueError(f"{len(rows)} pages au lieu de {n_pages}")
    pr = [page_pr(r) for r in rows]
    return {"P": float(np.mean([p for p, _ in pr])), "R": float(np.mean([r for _, r in pr])), "pages": n_pages,
            "pages_without_prediction": int(sum(1 for r in rows if r.get("n_pred", 1) == 0)),
            "precision_none_pages": int(sum(1 for r in rows if r.get("n_pred", 1) > 0 and r.get("precision") is None))}


def load_reference(path, page_ids: list, cfg: dict, smoke: bool = False) -> dict:
    """Valeurs B3 + λ lues dans e2e5_lambda_run_report.json (metrics.after), sans nouveau calcul. Refus si le sha ne
    commence pas par 3abfe0d7 (sauf en fumée, où le fichier est un leurre)."""
    ref = cfg["reference"]
    if not smoke:
        sha = sha256_file(path)
        if not sha.startswith(ref["sha256_prefix"]):
            raise SystemExit(f"refusé : rapport de référence de sha {sha[:8]}…, attendu {ref['sha256_prefix']}…")
    by_id = {e["page_id"]: e["metrics"]["after"] for e in read_json(path)["eval"]}
    missing = [p for p in page_ids if p not in by_id]
    if missing:
        raise SystemExit(f"refusé : pages absentes du rapport de référence : {missing}")
    return {p: page_pr(by_id[p]) for p in page_ids}


def paired_bootstrap(d_p, d_r, cfg: dict) -> dict:
    """Bootstrap apparié : 10 000 tirages avec remise des pages (mêmes indices pour ΔP et ΔR), graine 20261018,
    percentiles 2,5 et 97,5 (interpolation linéaire de numpy) de la moyenne des Δ."""
    bs = cfg["bootstrap"]
    d_p, d_r = np.asarray(d_p, dtype=np.float64), np.asarray(d_r, dtype=np.float64)
    rng = np.random.default_rng(bs["seed"])
    idx = rng.integers(0, len(d_p), size=(bs["draws"], len(d_p)))
    out = {"draws": bs["draws"], "seed": bs["seed"], "percentiles": list(bs["percentiles"]), "n_pages": int(len(d_p)),
           "method": "np.random.default_rng(seed).integers(0, n, (draws, n)) ; np.percentile (linear)"}
    for name, d in (("dP", d_p), ("dR", d_r)):
        lo, hi = np.percentile(d[idx].mean(axis=1), bs["percentiles"])
        out[name] = {"mean": float(d.mean()), "low": float(lo), "high": float(hi)}
    return out


def criterion(main_rows: list, ref: dict, page_ids: list, cfg: dict) -> dict:
    """(a) P moyen >= 0,80 ET R moyen >= 0,90 ; (b) borne basse de ΔP > 0 ET borne basse de ΔR >= -0,02."""
    cr = cfg["criterion"]
    n = len(page_ids)
    by_id = {r["page"]: r for r in main_rows}
    agg = mean_pr([by_id[p] for p in page_ids], n)
    a = bool(agg["P"] >= cr["a"]["precision_min"] and agg["R"] >= cr["a"]["recall_min"])
    d_p = [page_pr(by_id[p])[0] - ref[p][0] for p in page_ids]
    d_r = [page_pr(by_id[p])[1] - ref[p][1] for p in page_ids]
    bs = paired_bootstrap(d_p, d_r, cfg)
    b = bool(bs["dP"]["low"] > cr["b"]["dp_low_gt"] and bs["dR"]["low"] >= cr["b"]["dr_low_min"])
    ref_mean = {"P": float(np.mean([ref[p][0] for p in page_ids])), "R": float(np.mean([ref[p][1] for p in page_ids]))}
    return {"a": {"met": a, "P": agg["P"], "R": agg["R"], "precision_min": cr["a"]["precision_min"],
                  "recall_min": cr["a"]["recall_min"], "pages_without_prediction": agg["pages_without_prediction"],
                  "precision_none_pages": agg["precision_none_pages"]},
            "b": {"met": b, "bootstrap": bs, "dp_low_gt": cr["b"]["dp_low_gt"], "dr_low_min": cr["b"]["dr_low_min"],
                  "reference_mean": ref_mean, "reference_per_page": {p: list(ref[p]) for p in page_ids}},
            "verdict": "CRITERE_REMPLI" if (a and b) else "CRITERE_NON_REMPLI"}
