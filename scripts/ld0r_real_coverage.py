"""LD-0R (docs/END_TO_END_LD0R_LOT.txt §0-§8, amendements 1, 2 et 7). Indicateur sur 26 pages de TEST (P-07/P-07b),
pas une évaluation de détecteur.

Ce script réutilise les fonctions gelées de LD-0 (scripts/ld0_zone_coverage.py, gel 6, sha 4fae48f8…) sans les copier,
et n'ajoute qu'un adaptateur de lecture (VT réelle -> même structure de lignes que LD-0).
Le champ text de la VT n'est JAMAIS chargé (object_pairs_hook l'écarte à la lecture). Rien ne sort du Mac.

Modes :
  (défaut)          pilote : téléchargement (étape séparée, sha contrôlés) puis mesure en HF_HUB_OFFLINE=1, chacune par
                    L0.run_step (groupe de processus, délai 1 800 s) ; poids effacés dans un finally (§8 (1)).
  --download-only   étape de téléchargement seule.
  --measure-only    étape de mesure seule (HF_HUB_OFFLINE=1 posé par le pilote).
Fumée : --pages L_OEuvre_4613537_T0000001 --smoke. Complet : --confirmed-go, après la relecture Sonnet.
"""
from __future__ import annotations

import argparse
import faulthandler
import hashlib
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import ld0_zone_coverage as L0  # noqa: E402  (gel 6, importé sans modification)

os.environ.update(L0.HF_ENV)     # §8 (1): avant tout import de huggingface_hub (imports lazy, dans les fonctions)

CONFIG = ROOT / "config" / "ld0r_pages.json"
DIVS = ("Z-PP-blocs", "Z-PP-colonnes", "Z-GT")
MARGINS = (("marge16", 16.0), ("marge0", 0.0))
EXPECTED_PAGES = 26
STEP_TIMEOUT_S = 1800
DEFAULT_CACHE = "/private/tmp/ld0r_hf_cache"


class ImageRefused(RuntimeError):
    """Un sha diffère (juge, run ou image) : la page est refusée (lot §4)."""


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _drop_text(pairs):
    """Le champ text n'entre jamais en mémoire : écarté au moment de la lecture du JSON."""
    return {k: v for k, v in pairs if k != "text"}


def read_json_no_text(path) -> dict:
    return json.loads(Path(path).read_text(), object_pairs_hook=_drop_text)


def band_fn():
    """stage_s_data._band, comme dans LD-0 (amendement 1). Import tardif, après L0._load_deps()."""
    L0._load_deps()
    from babaorum.experimental.stage_s_data import _band
    return _band


def vt_lines(true_lines, band):
    """Hauteur par _band (neg + pos). Une ligne dont _band échoue est comptée à part (amendement 1)."""
    ok, n_band_fail = [], 0
    for tl in true_lines:
        try:
            neg, pos = band(tl["baseline"], tl["polygon"])
            h = float(neg + pos)
            if not h > 0:
                raise ValueError("hauteur nulle")
        except (ValueError, TypeError, KeyError, IndexError):
            n_band_fail += 1
            continue
        ok.append({"id": tl["gt_line_id"], "region": tl["region"], "baseline": tl["baseline"],
                   "polygon": tl["polygon"], "h": h})
    return ok, n_band_fail


def z_gt_boxes(ok_lines):
    """Z-GT (contrôle de plomberie, lot §7 (4)) : une boîte englobante par region, à partir des polygones VT."""
    by_region: dict = {}
    for l in ok_lines:
        by_region.setdefault(l["region"], []).extend((float(x), float(y)) for x, y in l["polygon"])
    return [L0.bbox(pts) for pts in by_region.values()]


def load_page_inputs(entry: dict, juge_dir: Path):
    """Contrôle des sha (run, juge, image) puis lecture de la géométrie. Refus si un sha diffère (lot §4)."""
    run_path = juge_dir.parent / f"{entry['page']}.json"
    juge_path = juge_dir / f"{entry['page']}.json"
    if sha256(run_path) != entry["run_sha256"]:
        raise ImageRefused(f"sha du fichier run diffère pour {entry['page']}")
    if sha256(juge_path) != entry["juge_sha256"]:
        raise ImageRefused(f"sha du fichier juge diffère pour {entry['page']}")
    run = read_json_no_text(run_path)
    img_path = Path(run["image"])
    got = sha256(img_path)
    if got != entry["image_sha256"] or got != run["image_sha256"]:
        raise ImageRefused(f"sha de l'image diffère pour {entry['page']}")
    vt = read_json_no_text(juge_path)
    return img_path, vt["true_lines"]


def measure_page(entry: dict, juge_dir: Path, det_fn, cfg: dict, band) -> dict:
    t0 = time.monotonic()
    pid = entry["page"]
    img_path, true_lines = load_page_inputs(entry, juge_dir)
    im = Image.open(img_path).convert("RGB")
    W, H = im.size
    ok, n_band_fail = vt_lines(true_lines, band)
    if not ok:
        raise ValueError(f"{pid}: aucune ligne à hauteur définie")
    factor = float(cfg["measure"]["target_median_height_px"]) / float(np.median([l["h"] for l in ok]))
    W_s, H_s = round(W * factor), round(H * factor)
    sx, sy = W_s / W, H_s / H

    gt_lines, n_valid_fail = [], 0
    for l in ok:
        try:
            gt_lines.append(L0.GroundTruthLine(id=str(l["id"]),
                                               baseline=[[x * factor, y * factor] for x, y in l["baseline"]],
                                               height_px=l["h"] * factor, baseline_source="ground_truth",
                                               geometry_status="available"))
        except (TypeError, ValueError):
            n_valid_fail += 1
    evaluable = []
    for g in gt_lines:
        try:
            ev, _inc = L0._validated_gt([g])
            evaluable.extend(ev)
        except (ValueError, TypeError, IndexError):
            n_valid_fail += 1
    if not evaluable:
        raise ValueError(f"{pid}: aucune ligne validée")
    samples, heights = [], []
    for line, points, height in evaluable:
        spacing = min(cfg["measure"]["sample_spacing_max_px"], height / cfg["measure"]["sample_spacing_height_divisor"])
        samples.append(np.asarray(L0._sample(points, spacing), dtype=float))
        heights.append(height)
    hb = np.asarray(heights) > cfg["measure"]["height_bucket_px"]

    det = L0.normalize_blocks(det_fn(im), image_size=(W, H))
    kept = L0.select_blocks(det["blocks"], L0.Params(conf=cfg["zones"]["conf"], dedup=cfg["zones"]["dedup"]))
    kept = [b for b in kept if L0.is_read(b.get("label"))]
    kept_boxes = [L0.bbox([(float(x), float(y)) for x, y in b["polygon"]]) for b in kept]
    divisions, _groups = L0.zones_from_blocks(kept_boxes, H, cfg["zones"]["gap_fraction_of_native_height"],
                                              cfg["zones"]["max_h_native_px"])
    divisions["Z-GT"] = z_gt_boxes(ok)
    scaled = {d: [L0.scale_box(b, sx, sy) for b in divisions[d]] for d in DIVS}

    measures, zone_heights = {}, {}
    for d in DIVS:
        measures[d], zone_heights[d] = {}, [round(b[3] - b[1], 2) for b in scaled[d]]
        nz = len(scaled[d])
        for mname, margin in MARGINS:
            m = L0.measure_lines(samples, scaled[d], margin, cfg["measure"]["threshold_a"],
                                 cfg["measure"]["threshold_a_prime"], cfg["measure"]["threshold_b"])
            lpz = L0.lines_per_zone(m, nz)
            cov = m["covered_one"]
            measures[d][mname] = {
                "n_lines": m["n_lines"], "a": m["a"], "a_prime": m["a_prime"], "b": m["b"], "c": m["c"],
                "n_zones": nz, "lines_per_zone": lpz,
                "lines_height_le_32": int((~hb).sum()), "covered_height_le_32": int(cov[~hb].sum()),
                "lines_height_gt_32": int(hb.sum()), "covered_height_gt_32": int(cov[hb].sum()),
                "frac_lines_in_zone_le_32": L0.fraction_in_small_zones(m, lpz, 32),
                "frac_lines_in_zone_le_64": L0.fraction_in_small_zones(m, lpz, 64),
            }
    return {
        "page": pid, "W": W, "H": H, "factor": factor, "W_s": W_s, "H_s": H_s,
        "n_lignes_vt": len(true_lines), "n_lignes_band_echec": n_band_fail,
        "n_lignes_validation_echec": n_valid_fail, "n_lignes_validees": len(evaluable),
        "n_detections": len(det["blocks"]), "n_kept": len(kept),
        "heights_scaled_median": float(np.median(heights)), "measures": measures,
        "zone_heights_scaled": zone_heights, "timings_s": round(time.monotonic() - t0, 2),
    }


def decide_status(n_ok: int, n_errors: int, expected: int) -> str:
    """§8 (2): INCOMPLETE si une erreur est survenue ou si moins de `expected` pages sont mesurées."""
    return "INCOMPLETE" if (n_errors > 0 or n_ok < expected) else "COMPLETE"


def reading(per_page: list, n_band_fail: int, n_lines: int, cfg: dict, smoke: bool) -> dict:
    """Règles fixées d'avance (lot §5, amendements 3 et 7). Le reste est descriptif."""
    rd = cfg["reading"]
    fail_share = n_band_fail / max(1, n_lines + n_band_fail)
    out = {"band_failure_share": fail_share, "smoke": smoke}
    if fail_share > rd["band_failure_max_share"]:
        out.update(status="ADAPTER_SUSPECT", reason="part des lignes à _band en échec > 5 % : aucune lecture R1/R2")
        return out
    z_gt = float(np.mean([p["measures"]["Z-GT"]["marge16"]["a"] for p in per_page]))
    if z_gt < rd["z_gt_control_min_mean"]:
        out.update(status="ADAPTER_SUSPECT", z_gt_mean_a=z_gt,
                   reason=f"contrôle Z-GT moyen {z_gt:.4f} < 0,99 (plomberie)")
        return out
    a_mean = float(np.mean([p["measures"]["Z-PP-blocs"]["marge16"]["a"] for p in per_page]))
    counts = [c for p in per_page for c in p["measures"]["Z-PP-blocs"]["marge16"]["lines_per_zone"]]
    lines_all = sum(p["measures"]["Z-PP-blocs"]["marge16"]["n_lines"] for p in per_page)
    p95 = float(np.percentile(counts, 95)) if counts else None
    share32 = sum(c for c in counts if c > 32) / lines_all
    share64 = sum(c for c in counts if c > 64) / lines_all
    out.update({"z_gt_mean_a": z_gt, "pp_blocs_mean_a": a_mean, "n_zones": len(counts),
                "p95_lines_per_zone": p95,
                "p99_lines_per_zone": float(np.percentile(counts, 99)) if counts else None,
                "max_lines_per_zone": max(counts) if counts else None,
                "share_lines_in_zones_gt32": share32, "share_lines_in_zones_gt64": share64,
                "min_a_per_page": min(p["measures"]["Z-PP-blocs"]["marge16"]["a"] for p in per_page)})
    if smoke:
        out.update(status="SMOKE_NON_LU", reason="fumée 1 page : R1/R2 se lisent sur les 26 pages seulement")
        return out
    if a_mean < rd["usable_mean_a"]:
        out.update(status="ZONES_INSUFFISANTES_REEL",
                   reason=f"moyenne (a) Z-PP-blocs {a_mean:.4f} < 0,92 : LD-1 non lancé sur les zones PP")
    elif p95 <= 32 and share32 <= rd["share_max_in_big_zones"]:
        out.update(status="OK", N=32)
    elif p95 <= 64 and share64 <= rd["share_max_in_big_zones"]:
        out.update(status="OK_N64", N=64,
                   reason="N=64 : amendement LD-1 requis (taille d'entrée, p99 de hauteur à relire)")
    else:
        out.update(status="ZONES_TROP_CHARGEES",
                   reason="p95 > 64 ou part > 2 % dans zones > 64 : revoir le découpage avant LD-1")
    return out


def _load_cfg() -> dict:
    cfg = json.loads(CONFIG.read_text())
    if sha256(ROOT / cfg["ld0_script"]["path"]) != cfg["ld0_script"]["sha256"]:
        raise SystemExit("refusé : sha du script LD-0 différent du gel 6")
    return cfg


def _select_pages(cfg: dict, args) -> list:
    if args.smoke:
        if not args.pages or len(args.pages) != 1:
            raise SystemExit("fumée : exactement une page")
        pages = [p for p in cfg["pages"] if p["page"] in args.pages]
        if len(pages) != 1:
            raise SystemExit("page absente de la config")
        return pages
    if not args.confirmed_go:
        raise SystemExit("refusé : lancement complet sans --confirmed-go (relecture Sonnet d'abord)")
    if len(cfg["pages"]) != EXPECTED_PAGES or args.pages:
        raise SystemExit("refusé : le lancement complet porte exactement les 26 pages de la config")
    return list(cfg["pages"])


def download_mode(args) -> int:
    cfg = _load_cfg()
    local = L0.download_detector(cfg["detector"], Path(args.cache_dir))     # sha contrôlés, RuntimeError sinon
    out = Path(args.out) if args.out else Path(args.cache_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "download.json").write_text(json.dumps({n: str(p.name) for n, p in local.items()}, indent=1))
    print(json.dumps({"download": "ok", "files": sorted(local)}), flush=True)
    return 0


def measure_mode(args) -> int:
    os.environ["HF_HUB_OFFLINE"] = "1"                 # §8 (1): mesure hors ligne, poids déjà téléchargés
    cfg = _load_cfg()
    pages = _select_pages(cfg, args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    juge_dir = Path(cfg["juge_dir"])
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    t_start = time.monotonic()
    per_page, errors = [], []
    band = band_fn()
    proc, model, _size = L0.load_detector(cfg["detector"], Path(args.cache_dir))
    floor = cfg["detector"]["floor"]
    det_fn = lambda im: L0.detect(proc, model, im, floor)  # noqa: E731
    try:
        for entry in pages:
            try:
                res = measure_page(entry, juge_dir, det_fn, cfg, band)
                per_page.append(res)
                (out / f"{entry['page']}.json").write_text(json.dumps(res, indent=1))
            except ImageRefused as exc:
                errors.append({"page": entry["page"], "error": f"ImageRefused: {exc}"})
            except Exception as exc:  # noqa: BLE001  (une page en erreur est notée ; les autres continuent)
                errors.append({"page": entry["page"], "error": f"{type(exc).__name__}: {exc}",
                               "trace": traceback.format_exc()[-1500:]})
        expected = 1 if args.smoke else EXPECTED_PAGES
        status = decide_status(len(per_page), len(errors), expected)
        n_lines = sum(p["n_lignes_validees"] for p in per_page)
        n_fail = sum(p["n_lignes_band_echec"] for p in per_page)
        if status == "INCOMPLETE":
            rd = {"status": "INCOMPLETE", "reason": f"{len(errors)} erreur(s), {len(per_page)}/{expected} pages : aucun verdict"}
        else:
            rd = reading(per_page, n_fail, n_lines, cfg, args.smoke)
        report = {"schema": "ld0r_report_v2", "status": status, "reading": rd,
                  "pages_mesurees": [p["page"] for p in per_page], "errors": errors,
                  "per_page": [{k: v for k, v in p.items() if k != "measures"} for p in per_page],
                  "identity": {"ld0_sha256": cfg["ld0_script"]["sha256"], "config_sha256": sha256(CONFIG)},
                  "text_field_read": False, "started_utc": started,
                  "duration_s": round(time.monotonic() - t_start, 1)}
        (out / "ld0r_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
        print(json.dumps({"status": status, "reading": rd.get("status"), "pages": len(per_page),
                          "errors": len(errors)}, ensure_ascii=False), flush=True)
        return 0 if status == "COMPLETE" else 1
    finally:
        (out / "ld0r_measure_exit.json").write_text(json.dumps(
            {"pages_ok": len(per_page), "pages_error": len(errors), "duration_s": round(time.monotonic() - t_start, 1)},
            indent=1))


def driver(args) -> int:
    """Pilote : téléchargement puis mesure, chacun par L0.run_step (délai 1 800 s) ; poids effacés dans le finally."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    logs = out / "logs"
    cache = Path(args.cache_dir)
    py, me = sys.executable, str(Path(__file__).resolve())
    recs, status, reason = [], "INCOMPLETE", "pilote interrompu"
    try:
        recs.append(L0.run_step("download", [py, me, "--download-only", "--cache-dir", str(cache), "--out", str(out)],
                                STEP_TIMEOUT_S, logs, env=dict(os.environ)))
        if recs[-1]["rc"] != 0:
            reason = f"téléchargement en échec (rc={recs[-1]['rc']})"
            return 1
        argv = [py, me, "--measure-only", "--out", str(out), "--cache-dir", str(cache)]
        argv += ["--smoke"] if args.smoke else []
        argv += ["--pages", *args.pages] if args.pages else []
        argv += ["--confirmed-go"] if args.confirmed_go else []
        recs.append(L0.run_step("measure", argv, STEP_TIMEOUT_S, logs, env=dict(os.environ, HF_HUB_OFFLINE="1")))
        rep = out / "ld0r_report.json"
        if recs[-1]["rc"] == 0 and rep.exists():
            status = json.loads(rep.read_text())["status"]
            reason = "mesure terminée"
        else:
            reason = f"mesure en échec (rc={recs[-1]['rc']})"
        return 0 if status == "COMPLETE" else 1
    finally:
        shutil.rmtree(cache, ignore_errors=True)                                  # §8 (1) : poids effacés
        (out / "ld0r_run_exit.json").write_text(json.dumps(
            {"status": status, "reason": reason, "steps": recs, "weights_removed": not cache.exists()},
            indent=1, default=str))


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--pages", nargs="*", default=None)
    ap.add_argument("--cache-dir", default=DEFAULT_CACHE)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--confirmed-go", action="store_true")
    ap.add_argument("--download-only", action="store_true")
    ap.add_argument("--measure-only", action="store_true")
    return ap


def main(argv=None) -> int:
    faulthandler.enable()
    args = build_parser().parse_args(argv)
    if args.download_only:
        return download_mode(args)
    if args.measure_only:
        return measure_mode(args)
    return driver(args)


if __name__ == "__main__":
    raise SystemExit(main())
