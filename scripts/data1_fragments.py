"""DATA-1 — extraction de fragments pour pR (données seulement : aucune métrique, aucune classe, aucune vérité).

Lot : docs/END_TO_END_DATA1_LOT.txt (p6, 17/10/2026, amendements 1 et 2). Pour B3 et S2 (checkpoints E2E-1
inchangés) : extraction d'origine (stage_s_targets.extract_polylines, config e2e4 : stride 4, seuil 0,5,
min_cells 4) puis affinage (refine_lines). Pas de fusion (amendement 2, §6). Les pages mf_0000..mf_0049 sont
réparties en 3 morceaux (--shard k/3, amendement 1). Les pages mf_0050..mf_0059 ne sont jamais listées, ouvertes
ni chargées (test_data1_fragments.py). Entrées lues sous --tmp (lecture seule) ; sorties sous --out-dir seulement
(/kaggle/working sur Kaggle).

Sortie : fragments/<bras>/<page>.json (liste before seulement) et data1_manifest.json (sha256 de chaque
fichier de sortie, des entrées, du kit ; versions ; temps ; indice du morceau).
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import traceback
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

import e2e3_fp_classes as E3  # noqa: E402  (reuse by import: sha256, alphabet, checkpoint loading, guard)
import e2e4_extraction as E4  # noqa: E402  (reuse by import: refine)
from babaorum.experimental.stage_s_targets import extract_polylines  # noqa: E402

CONFIG = ROOT / "config/e2e4_extraction.json"
DATA_CONFIG = ROOT / "config/data1_fragments.json"
ARMS = ("B3", "S2")
PAGES_ALL = [f"mf_{i:04d}" for i in range(50)]
ALPHABET_SIZE_REFERENCE = 74      # alphabet size of the E2E-4 v3 report (alphabet_size); used in DATA-2 mode
FORBIDDEN = {f"mf_{i:04d}" for i in range(50, 60)}
KAGGLE_OUT = Path("/kaggle/working")
KIT_FILES = ["scripts/e2e3_fp_classes.py", "scripts/g_diag_mf.py", "scripts/e2e4_extraction.py",
             "config/e2e4_extraction.json", "scripts/data1_fragments.py", "config/data1_fragments.json"]


# ---------- I/O wrappers (the only place where files are opened; tests spy on them) ----------

def _sha(path) -> str:
    return E3.sha256(Path(path))


def _read_json(root: Path, pid: str) -> dict:
    return json.loads((root / "pages" / f"{pid}.json").read_text())


def _load_page(root: Path, pid: str, target_h: float) -> dict:
    """Scaled page (train_stage_s.load_scaled_page) and the page sizes: native (from the PNG header), scaled."""
    from PIL import Image
    from train_stage_s import load_scaled_page
    pg = load_scaled_page(root, pid, target_h)
    with Image.open(root / "images" / f"{pid}.png") as im:
        native_wh = [int(im.size[0]), int(im.size[1])]
    h_s, w_s = pg["image"].shape
    return {"image": pg["image"], "factor": pg["factor"], "native_wh": native_wh, "scaled_wh": [int(w_s), int(h_s)]}


def _prefix_ok(path: Path, prefix: str) -> bool:
    return E3._check_prefix(Path(path), prefix)


def _load_model(arm: str, ckpt: Path, alphabet_size: int):
    return E3.load_arm(arm, ckpt, alphabet_size)


# ---------- cells of each detected component (same components as extract_polylines) ----------

def components_with_cells(prob, *, stride: int, threshold: float, min_cells: int) -> list:
    """(polyline, cells) for each component, in the order of extract_polylines. The polyline is built and
    sorted exactly as in stage_s_targets.extract_polylines; infer_fragments checks that they are equal."""
    hc, wc = prob.shape
    fg = prob >= threshold
    seen = np.zeros((hc, wc), dtype=bool)
    out = []
    for r0 in range(hc):
        for c0 in range(wc):
            if not fg[r0, c0] or seen[r0, c0]:
                continue
            queue = deque([(r0, c0)])
            seen[r0, c0] = True
            cells = []
            while queue:
                r, c = queue.popleft()
                cells.append((r, c))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        rr, cc = r + dr, c + dc
                        if (dr or dc) and 0 <= rr < hc and 0 <= cc < wc and fg[rr, cc] and not seen[rr, cc]:
                            seen[rr, cc] = True
                            queue.append((rr, cc))
            if len(cells) < min_cells:
                continue
            by_col: dict[int, list[int]] = {}
            for r, c in cells:
                by_col.setdefault(c, []).append(r)
            poly = []
            for c in sorted(by_col):
                rows = by_col[c]
                r_mean = sum(rows) / len(rows)
                poly.append([(c + 0.5) * stride, (r_mean + 0.5) * stride])
            out.append((poly, cells))
    out.sort(key=lambda t: (min(q[1] for q in t[0]), min(q[0] for q in t[0])))
    return out


def infer_fragments(model, image, extraction: dict):
    """Original extraction (extract_polylines, len >= 2) with, per fragment, its score (mean of the detection
    probability over its cells) and n_cells; also the encoder features for the refinement."""
    import torch
    with torch.no_grad():
        features = model.enc_det(torch.as_tensor(image)[None, None])
        prob = torch.sigmoid(model.head_d(features)[0])[0, 0].numpy()
    kw = dict(stride=extraction["stride"], threshold=extraction["threshold"], min_cells=extraction["min_cells"])
    official = [p for p in extract_polylines(prob, **kw) if len(p) >= 2]
    comps = [(poly, cells) for poly, cells in components_with_cells(prob, **kw) if len(poly) >= 2]
    same = [[tuple(p) for p in poly] for poly, _ in comps] == [[tuple(p) for p in poly] for poly in official]
    if not same:
        raise RuntimeError("cell components differ from extract_polylines: refused")
    scores = [(float(np.mean([prob[r, c] for r, c in cells])), len(cells)) for _, cells in comps]
    return official, scores, features


# ---------- pure assembly of the output (tested without a model) ----------

def assemble_payload(arm, pid, page_sizes, extraction, extracted, scores, refined_b) -> dict:
    """Output of §2 as amended by §6: the before list only (no fusion, no after, no G)."""
    before = []
    for k, poly in enumerate(extracted):
        score, n_cells = scores[k]
        before.append({"id": f"{arm}:{pid}:b{k}", "polyline_px": [[float(x), float(y)] for x, y in poly],
                       "refined_baseline_px": refined_b[k], "score": score, "n_cells": n_cells})
    factor = page_sizes["factor"]
    return {"page_id": pid, "bras": arm, "échelle": float(factor), "factor": float(factor),
            "native_wh": page_sizes["native_wh"], "scaled_wh": page_sizes["scaled_wh"],
            "stride": extraction["stride"],
            "seuil": extraction["threshold"], "before": before}


# ---------- guards ----------

def check_output_dir(out, root, smoke: bool) -> None:
    """Never under --tmp (read-only input on Kaggle); outside /kaggle/working only for a smoke test."""
    out = Path(out).resolve()
    tmp = Path(root).resolve().parent
    if out == tmp or tmp in out.parents:
        raise SystemExit(f"refused: output {out} under the input folder {tmp}")
    if not smoke and not (out == KAGGLE_OUT or KAGGLE_OUT in out.parents):
        raise SystemExit(f"refused: full run writes only under {KAGGLE_OUT}, not {out}")


def _kit_sha() -> dict:
    return {rel: _sha(ROOT / rel) for rel in KIT_FILES}


def _versions() -> dict:
    import torch
    return {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__}


# ---------- run ----------

def run(args) -> dict:
    dcfg = json.loads(DATA_CONFIG.read_text())
    cfg = json.loads(CONFIG.read_text())
    gm_path = getattr(args, "generation_manifest", None)
    if gm_path:
        # DATA-2 (docs/END_TO_END_DATA2_LOT.txt, §3): the list of pages comes from the generation manifest
        used = [p["page_id"] for p in json.loads(Path(gm_path).read_text())["pages"]]
        assert len(used) in (1, 20) and used == [f"mfn_{i:04d}" for i in range(len(used))], "DATA-2 pages: mfn_0000.."
    else:
        used = list(dcfg["pages"]["used"])
        assert used == PAGES_ALL, "the page list of DATA-1 is mf_0000..mf_0049"
    assert not set(used) & FORBIDDEN
    smoke = args.pages is not None
    shard = parse_shard(args.shard, dcfg) if args.shard else None
    scope = shard["pages"] if shard else used           # pages this run may list at all
    frag_pages = list(args.pages) if smoke else scope
    assert set(frag_pages) <= set(scope) and not set(frag_pages) & FORBIDDEN
    root = Path(args.tmp) / ("mfn" if gm_path else "mf")     # DATA-2: <generation root>/mfn
    out = Path(args.out_dir)
    check_output_dir(out, root, smoke)
    out.mkdir(parents=True, exist_ok=True)
    times: dict[str, float] = {}

    def add(name, t0):
        times[name] = round(times.get(name, 0.0) + time.monotonic() - t0, 3)

    t_all = time.monotonic()
    manifest = {"schema_version": "data1_manifest_v1", "status": None, "smoke": smoke,
                "shard": shard, "pages_fragments": frag_pages,
                "alphabet": "placeholder of the reference size (74), see docs/END_TO_END_DATA1_LOT.txt §5 note",
                "arms": list(ARMS),
                "inputs": {}, "versions": _versions(), "kit_sha256": _kit_sha(),
                "config_sha256": {"data1": _sha(DATA_CONFIG), "e2e4": _sha(CONFIG)}}

    # 1. inputs: E2E-4 report (page reference), checkpoints, pages (sha against the reference)
    t0 = time.monotonic()
    if gm_path:
        # DATA-2: the sha reference is the generation manifest (per file), not the E2E-4 report
        ref_path = None
        gm = json.loads(Path(gm_path).read_text())
        ref = {"steps": {"page_cache": {"sha256": dict(gm["files_sha256"])}},
               "alphabet_size": ALPHABET_SIZE_REFERENCE}
    else:
        if not args.e2e4_report:
            raise SystemExit("refused: --e2e4-report is required without --generation-manifest")
        ref_path = Path(args.e2e4_report)
        if not _sha(ref_path).startswith(dcfg["page_reference"]["sha256_prefix"]):
            return _finish(manifest, out, times, t_all, "MISMATCH", "E2E-4 report sha prefix differs")
        ref = json.loads(ref_path.read_text())
    ref_sha = ref["steps"]["page_cache"]["sha256"]
    ckpts = {}
    for arm in ARMS:
        info = dcfg["checkpoints"][arm]
        ckpts[arm] = Path(args.calibration) / info["checkpoint"]
        if not _prefix_ok(ckpts[arm], info["sha256_prefix"]):
            return _finish(manifest, out, times, t_all, "MISMATCH", f"checkpoint {arm} prefix differs")
    page_sha, mismatch = {}, []
    for pid in frag_pages:                              # pages read (JSON for the geometry, PNG for the image)
        key = f"pages/{pid}.json"
        page_sha[key] = _sha(root / key)
        if ref_sha.get(key) != page_sha[key]:
            mismatch.append(key)
        key = f"images/{pid}.png"
        page_sha[key] = _sha(root / key)
        if ref_sha.get(key) != page_sha[key]:
            mismatch.append(key)
    manifest["inputs"].update({"e2e4_report_sha256": _sha(ref_path) if ref_path else None,
                               "generation_manifest_sha256": _sha(gm_path) if gm_path else None,
                               "checkpoints_sha256": {a: _sha(p) for a, p in ckpts.items()},
                               "pages_sha256": page_sha, "pages_checked_against_e2e4_report": True,
                               "pages_sha_mismatch": mismatch})
    add("inputs_check", t0)
    if mismatch:
        return _finish(manifest, out, times, t_all, "MISMATCH", f"{len(mismatch)} page files differ from the E2E-4 report")

    # 2. alphabet (as at E2E-1: pages 0..49, texts only), models
    t0 = time.monotonic()
    # placeholder alphabet of the reference size: the fragments depend on the size of the CTC head (checked by
    # the strict checkpoint load), not on the characters; no page other than the shard's is read
    alphabet = ["<blank>"] + [f"c{i}" for i in range(ref["alphabet_size"] - 1)]
    if len(alphabet) != ref["alphabet_size"]:
        return _finish(manifest, out, times, t_all, "MISMATCH", "alphabet size differs from the E2E-4 report")
    models = {arm: _load_model(arm, ckpts[arm], len(alphabet)) for arm in ARMS}
    add("alphabet_models", t0)

    # 3. fragments per page and arm; the cap starts here
    ext, mcfg = cfg["extraction"], cfg["model"]
    target = cfg["scale"]["target_median_height_px"]
    cap_min = dcfg["budget"]["cap_minutes"]
    started = time.monotonic()
    written = []
    status = "SMOKE" if smoke else "OK"
    for pid in frag_pages:
        if E3.guard_exceeded(time.monotonic() - started, cap_min):
            status = "INCOMPLETE"
            break
        t0 = time.monotonic()
        pg = _load_page(root, pid, target)
        add("page_load", t0)
        for arm in ARMS:
            t0 = time.monotonic()
            extracted, scores, features = infer_fragments(models[arm], pg["image"], ext)
            add("infer_and_score", t0)
            t0 = time.monotonic()
            refined_b = E4.refine(models[arm], features, extracted, mcfg)
            add("refine", t0)
            payload = assemble_payload(arm, pid, pg, ext, extracted, scores, refined_b)
            t0 = time.monotonic()
            path = out / "fragments" / arm / f"{pid}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, ensure_ascii=True, indent=1))
            written.append(path)
            add("write", t0)
    return _finish(manifest, out, times, t_all, status, None, written=written, pages_done=len(written))


README = """LISEZMOI — DATA-1, fragments d'extraction pour pR (données seulement : aucune métrique, aucune vérité)

Morceau : <shard> sur 3 — pages <pages> — bras <arms>. Statut : <status>. Généré le <date> par scripts/data1_fragments.py.

1. Repère. Toutes les coordonnées (polyline_px, refined_baseline_px) sont en pixels de la PAGE MISE À L'ÉCHELLE,
   et non de la page native. La mise à l'échelle (train_stage_s.load_scaled_page) vise une hauteur médiane de ligne
   de <target> px : factor = <target> / médiane(normal_negative_px + normal_positive_px) des lignes GT de la page.
   L'image est redimensionnée en (round(W·factor), round(H·factor)), en bilinéaire.
2. Origine et convention. Coordonnées continues « bords de pixels ». (0, 0) est le coin haut gauche de l'image,
   x va vers la droite, y vers le bas ; le pixel (i, j) couvre [i, i+1] × [j, j+1].
3. Effet du stride <stride>. La carte de détection a une cellule pour <stride> × <stride> pixels. Un point de
   polyline_px est (<stride>·c + <half>, <stride>·r̄ + <half>), avec c la colonne de la cellule et r̄ la moyenne des
   rangées de la composante dans cette colonne. x est donc sur une grille de pas <stride> ; y est une moyenne, donc
   pas sur la grille. Un point par colonne de cellules, de gauche à droite.
4. refined_baseline_px : <K> points, rééchantillonnés le long de la polyligne puis déplacés le long de la normale par
   la tête de géométrie (refine_lines). Ce sont des coordonnées continues, dans le même repère.
5. Retour à la page native. x_natif = x · W / W_s et y_natif = y · H / H_s. Les valeurs native_wh [W, H], scaled_wh
   [W_s, H_s] et factor sont écrites dans chaque fichier de page. L'écart entre W_s / W et factor est inférieur à 1/W.
6. Identifiants : "<bras>:<page>:b<k>", où k est l'indice d'extraction (tri par y minimal, puis par x minimal). Ils
   n'ont AUCUNE correspondance avec les lignes GT : aucun appariement n'est fait dans ce lot. Les lignes GT sont dans
   pages/mf_XXXX.json (lines[].id, lines[].baseline en pixels NATIFS, normal_negative_px et normal_positive_px). Pour
   comparer, il faut multiplier la GT par factor, ou ramener les fragments au natif comme au point 5.
7. score = moyenne de la probabilité de détection sur les n_cells cellules du fragment. Pas de fusion (amendement 2).
   Un seul bloc « before » par page et par bras ; pas de liste after, pas de G.

Fichiers : fragments/<bras>/<page>.json (une par page et par bras), data1_manifest.json (sha256 de chaque fichier,
des entrées et du kit ; versions ; temps), LISEZMOI.txt (ce fichier).
"""


def readme_text(manifest: dict) -> str:
    cfg = json.loads(CONFIG.read_text())
    shard = manifest.get("shard")
    pages = manifest.get("pages_fragments", [])
    values = {
        "<shard>": f"{shard['index']} (--shard {shard['index']}/{shard['of']})" if shard else "aucun (run complet)",
        "<pages>": f"{pages[0]} à {pages[-1]} ({len(pages)} pages)" if pages else "aucune",
        "<arms>": ", ".join(ARMS),
        "<status>": str(manifest.get("status")),
        "<date>": time.strftime("%Y-%m-%d %H:%M"),
        "<target>": f"{cfg['scale']['target_median_height_px']:g}",
        "<stride>": str(cfg["extraction"]["stride"]),
        "<half>": f"{cfg['extraction']['stride'] / 2:g}",
        "<K>": str(cfg["model"]["K"]),
    }
    text = README
    for key, val in values.items():
        text = text.replace(key, val)
    return text


def _finish(manifest, out, times, t_all, status, why, written=(), pages_done=0) -> dict:
    manifest["status"] = status
    if why:
        manifest["reason"] = why
    manifest["pages_written"] = pages_done
    manifest["outputs_sha256"] = {str(p.relative_to(out)): _sha(p) for p in sorted(written)}
    manifest["times_s"] = dict(times, total=round(time.monotonic() - t_all, 3))
    manifest["counts"] = {}
    for arm in ARMS:
        b = 0
        for p in written:
            if p.parent.name == arm:
                b += len(json.loads(p.read_text())["before"])
        manifest["counts"][arm] = {"before_total": b}
    (out / "data1_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    (out / "LISEZMOI.txt").write_text(readme_text(manifest), encoding="utf-8")
    return manifest


def parse_shard(text: str, dcfg: dict) -> dict:
    """'k/3' -> the consecutive block k of the config's used pages (blocks in dcfg['shards'])."""
    k_text, of_text = text.split("/")
    k, of = int(k_text), int(of_text)
    blocks = dcfg["shards"]
    assert of == len(blocks) and 0 <= k < of, ("shard", text)
    first, last = blocks[k]
    used = dcfg["pages"]["used"]
    pages = used[used.index(f"mf_{first:04d}"):used.index(f"mf_{last:04d}") + 1]
    assert pages == [f"mf_{i:04d}" for i in range(first, last + 1)] and not set(pages) & FORBIDDEN
    return {"index": k, "of": of, "range": [first, last], "pages": pages}


def build_parser():
    p = argparse.ArgumentParser(description="DATA-1 : fragments B3 et S2 pour pR (données seulement)")
    p.add_argument("--confirmed-go", action="store_true", required=True)
    p.add_argument("--tmp", required=True, help="folder that contains mf/ (read only)")
    p.add_argument("--calibration", required=True, help="folder that contains B3/ and S2/ checkpoint_last.pt")
    p.add_argument("--generation-manifest", default=None,
                   help="DATA-2: data2_generation_manifest.json (pages and sha reference), replaces the E2E-4 report")
    p.add_argument("--e2e4-report", default=None, help="E2E-4 v3 report (page sha reference; DATA-1)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--pages", default=None, help="smoke test only: explicit comma list inside the scope")
    p.add_argument("--shard", default=None, help="k/3 : block k of the config's shards (k = 0, 1, 2)")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.pages is not None:
        args.pages = [x.strip() for x in args.pages.split(",") if x.strip()]
    try:
        manifest = run(args)
    except Exception:
        print("FAILED", flush=True)
        traceback.print_exc()
        return 1
    print(json.dumps({"status": manifest["status"], "times_s": manifest["times_s"],
                      "pages_written": manifest["pages_written"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
