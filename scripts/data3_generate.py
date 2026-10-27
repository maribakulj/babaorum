"""DATA-3 (classe A) : 10 pages neuves mf_0060..mf_0069 pour LD-3, étape de génération (docs/END_TO_END_LD3_PROPOSITION.txt §8 (2), (3)).

Réutilise la mécanique de DATA-2 (scripts/data2_generate.py : commande du générateur, rename_page, normalise_text) sans la modifier :
mille-feuilles au commit 23875ab, profil pilot-v0.2, mêmes options de rendu ; seuls la graine, le nombre de pages et les noms changent.
Deux modes SEULEMENT :
  real  : graine 20261026, exactement 10 pages, noms mf_0060..mf_0069 ; autorisé UNIQUEMENT quand l'EVAL de LD-3 l'appelle
          (drapeau explicite called_from_eval_ld3 de produce_pages ; AUCUNE option de ligne de commande) ; refusé sinon ;
  decoy : fumée, graine différente, noms decoy_0000.. (jamais mf_*).
Gardes (testées) : tout nom de sortie correspondant à mf_00[0-5]\\d ou mfn_ est refusé ; graine réelle avec noms leurres (et l'inverse)
refusée ; dossier de sortie non vide refusé.
Disposition de sortie = racine de pages attendue par train_line_detr.load_scaled, train_ld2.load_page_text et TL.pp_zones :
<racine>/pages/{pid}.json (page_id, image{sha256,height,width,path}, lines, blocks, ...) et <racine>/images/{pid}.png.
RÉSERVE : OVERLAP_TEXT_FLAGGED est attendu (même réservoir de textes que mf_0000-0049). Il est rapporté et ne bloque pas LD-3, qui
mesure la géométrie ; OVERLAP_PAGE (image identique à une page d'entraînement) bloque.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import data2_generate as D2  # noqa: E402

CONFIG = ROOT / "config" / "ld3.json"
MODES = ("real", "decoy")
TAIL_CHARS = 2000


class GenerationFailed(RuntimeError):
    """Code de sortie non nul ou délai du générateur ; porte la fin de stderr et de stdout (statut INCOMPLETE à l'EVAL)."""


def load_config(path=CONFIG) -> dict:
    return json.loads(Path(path).read_text())


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------- noms, graines, drapeau, sortie vide (gardes) ----------

def real_names(cfg: dict) -> list:
    p = cfg["pages"]
    return [f"mf_{p['real_first'] + i:04d}" for i in range(p["n"])]


def decoy_names(cfg: dict, n: int = None) -> list:
    p = cfg["pages"]
    return [f"{p['decoy_prefix']}{i:04d}" for i in range(n or p["decoy_n"])]


def check_names(names: list, mode: str, cfg: dict) -> list:
    """Refuse tout nom interdit (mf_0000-0059, mfn_*) ; réel : exactement mf_0060..0069 ; leurre : decoy_NNNN seulement."""
    p = cfg["pages"]
    bad = [n for n in names if re.match(p["forbidden_name_regex"], n)]
    if bad:
        raise SystemExit(f"refusé : noms de sortie interdits {bad[:3]}")
    if mode == "real":
        if list(names) != real_names(cfg):
            raise SystemExit("refusé : le mode réel produit exactement les 10 pages mf_0060..mf_0069")
    elif mode == "decoy":
        if not names or any(not re.fullmatch(re.escape(p["decoy_prefix"]) + r"\d{4}", n) for n in names):
            raise SystemExit("refusé : le mode leurre ne produit que des noms decoy_NNNN")
    else:
        raise SystemExit(f"refusé : mode inconnu {mode!r}")
    return list(names)


def check_seed(seed: int, mode: str, cfg: dict) -> int:
    real = cfg["pages"]["seed"]
    if mode == "real" and seed != real:
        raise SystemExit(f"refusé : le mode réel exige la graine {real}")
    if mode == "decoy" and seed == real:
        raise SystemExit("refusé : la graine réelle ne peut pas produire de pages leurres")
    return seed


def check_call(mode: str, called_from_eval_ld3: bool) -> None:
    if mode == "real" and called_from_eval_ld3 is not True:
        raise SystemExit("refusé : le mode réel n'est appelé que par la phase EVAL de LD-3 (drapeau explicite)")


def check_out_empty(out: Path) -> None:
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusé : dossier de sortie non vide ({out})")


# ---------- génération ----------

def d2_style_cfg(cfg: dict, seed: int, pages: int) -> dict:
    """Configuration au format attendu par data2_generate.generate_command."""
    g = cfg["generator"]
    return {"generator": {"render": g["render"], "seed": seed, "pages": pages}}


def generate(src_root: Path, raw: Path, cfg: dict, pages: int, seed: int, extra_libs=None, timeout_s=None) -> int:
    """Lance le générateur du commit extrait (PYTHONPATH : bibliothèques pip isolées puis src). Délai géré par l'appelant."""
    paths = [str(Path(src_root) / "src")]
    if extra_libs:
        paths.insert(0, str(extra_libs))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(paths))
    raw = Path(raw)
    raw.mkdir(parents=True, exist_ok=True)
    with open(raw.parent / "generate.stdout", "w") as out_f, open(raw.parent / "generate.stderr", "w") as err_f:
        proc = subprocess.run(D2.generate_command(src_root, raw, d2_style_cfg(cfg, seed, pages), pages, seed),
                              cwd=str(src_root), env=env, stdout=out_f, stderr=err_f, timeout=timeout_s)
    return proc.returncode


def _tail(path: Path, n: int = TAIL_CHARS) -> str:
    try:
        return Path(path).read_text(errors="replace")[-n:]
    except OSError:
        return ""


def record_failure(work: Path, raw: Path, rc, timeout_s=None) -> dict:
    """Écrit data3_generate_failure.json (fins de stderr et stdout, 2000 caractères) dans `work` ; retourne l'enregistrement."""
    rec = {"schema_version": "data3_generate_failure_v1", "rc": rc, "timeout_s": timeout_s,
           "stderr_tail": _tail(Path(raw).parent / "generate.stderr"), "stdout_tail": _tail(Path(raw).parent / "generate.stdout")}
    Path(work).mkdir(parents=True, exist_ok=True)
    (Path(work) / "data3_generate_failure.json").write_text(json.dumps(rec, indent=2, ensure_ascii=False))
    return rec


def rename_all(raw: Path, pages_root: Path, names: list) -> list:
    """mf_0000.. (intermédiaires du générateur) -> noms de sortie ; seuls page_id et image.path changent (DATA-2 rename_page)."""
    raw_names = [f"mf_{i:04d}" for i in range(len(names))]
    return [D2.rename_page(raw, pages_root, r, n) for r, n in zip(raw_names, names)]


def overlap_check(pages_root: Path, names: list, mf_dir: Path, reference: dict, cfg: dict) -> dict:
    """Comme DATA-2 : textes normalisés des lignes GT des pages neuves comparés à ceux de mf_0000-0049 (JSON lus un par un,
    chemins explicites, aucune image de mf ouverte) ; images comparées aux sha du rapport de référence si fourni."""
    ov = cfg["overlap"]
    mf_list = list(ov["mf_pages"])
    mf_texts = set()
    for p in mf_list:
        page = json.loads((Path(mf_dir) / "pages" / f"{p}.json").read_text())
        mf_texts.update(D2.normalise_text(ln["text"]) for ln in page["lines"] if ln.get("text"))
    image_hits = []
    if reference is not None:
        ref_sha = reference["steps"]["page_cache"]["sha256"]
        mf_image_sha = {ref_sha[f"images/{p}.png"] for p in mf_list}
        image_hits = [n for n in names if sha256(Path(pages_root) / "images" / f"{n}.png") in mf_image_sha]
    per_page, total, hit = [], 0, 0
    for n in names:
        page = json.loads((Path(pages_root) / "pages" / f"{n}.json").read_text())
        lines = [D2.normalise_text(ln["text"]) for ln in page["lines"] if ln.get("text")]
        n_hit = sum(1 for t in lines if t in mf_texts)
        per_page.append({"page_id": n, "lines": len(lines), "lines_in_mf_text": n_hit, "share": (n_hit / len(lines)) if lines else None})
        total += len(lines)
        hit += n_hit
    share = hit / total if total else 0.0
    status = "OVERLAP_PAGE" if image_hits else ("OVERLAP_TEXT_FLAGGED" if share > ov["text_share_threshold"] else "OK")
    return {"schema_version": "data3_overlap_report_v1", "status": status, "image_sha_matches": image_hits,
            "image_check": reference is not None, "text_share_global": share, "text_share_threshold": ov["text_share_threshold"],
            "lines_total": total, "lines_in_mf_text": hit, "per_page": per_page,
            "blocking": bool(ov["blocking"].get(status, False)), "mf_pages_read": mf_list,
            "note": "OVERLAP_TEXT_FLAGGED : rapporté, non bloquant pour LD-3 (géométrie) ; OVERLAP_PAGE : bloquant"}


def produce_pages(mode: str, cfg: dict, src_root: Path, work: Path, pages_root: Path, *, called_from_eval_ld3: bool = False,
                  seed: int = None, n_pages: int = None, mf_dir: Path = None, reference: dict = None, extra_libs=None,
                  timeout_s=None) -> dict:
    """Génère, renomme, contrôle le recouvrement, écrit data3_generation_manifest.json dans `work` et retourne le manifeste.
    Les gardes passent AVANT toute action. Le dossier de pages (pages_root) doit être vide."""
    check_call(mode, called_from_eval_ld3)
    names = real_names(cfg) if mode == "real" else decoy_names(cfg, n_pages)
    check_names(names, mode, cfg)
    seed = cfg["pages"]["seed"] if (mode == "real" and seed is None) else (cfg["pages"]["decoy_seed"] if seed is None else seed)
    check_seed(seed, mode, cfg)
    check_out_empty(pages_root)
    work, pages_root = Path(work), Path(pages_root)
    raw = work / "gen_raw"
    t0 = time.monotonic()
    try:
        rc = generate(src_root, raw, cfg, len(names), seed, extra_libs, timeout_s)
    except subprocess.TimeoutExpired:
        rc = "TIMEOUT"
    if rc != 0:
        rec = record_failure(work, raw, rc, timeout_s)
        raise GenerationFailed(f"générateur : code de sortie {rc}\n--- stderr (fin) ---\n{rec['stderr_tail']}\n"
                               f"--- stdout (fin) ---\n{rec['stdout_tail']}")
    rows = rename_all(raw, pages_root, names)
    files = {}
    for r in rows:
        files[f"pages/{r['page_id']}.json"] = r["json_sha256"]
        files[f"images/{r['page_id']}.png"] = r["png_sha256"]
    overlap = overlap_check(pages_root, names, mf_dir, reference, cfg) if mf_dir is not None else None
    g = cfg["generator"]
    manifest = {"schema_version": "data3_generation_manifest_v1", "mode": mode, "seed": seed, "names": names, "pages": rows,
                "files_sha256": files,
                "generator": {"commit": g["commit"], "profile": g["profile"], "render": g["render"],
                              "pilot_config_sha256": g["pilot_config_sha256"], "bundle_sha256": g["bundle_sha256"],
                              "ld3_config_sha256": sha256(CONFIG) if Path(CONFIG).is_file() else None},
                "overlap": overlap, "versions": D2.versions(), "seconds_total": round(time.monotonic() - t0, 1),
                "note": "noms mf_0000.. du dossier brut = intermédiaires, jamais des sorties"}
    work.mkdir(parents=True, exist_ok=True)
    (work / "data3_generation_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    if overlap is not None:
        (work / "data3_overlap_report.json").write_text(json.dumps(overlap, indent=2, ensure_ascii=False))
    return manifest


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="DATA-3 : pages neuves pour LD-3 (modes real / decoy)")
    p.add_argument("--mode", choices=MODES, required=True)
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--pages-root", type=Path, required=True)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--n-pages", type=int, default=None, help="leurre seulement")
    p.add_argument("--mf-dir", type=Path, default=None)
    p.add_argument("--reference-report", type=Path, default=None)
    p.add_argument("--extra-libs", type=Path, default=None)
    args = p.parse_args(argv)
    if args.mode == "real":                       # le mode réel n'existe que par produce_pages, appelé par eval_ld3.phase_EVAL
        raise SystemExit("refusé : le mode réel n'est jamais lancé en ligne de commande ; il passe uniquement par produce_pages "
                         "appelé par la phase EVAL de LD-3")
    cfg = load_config()
    ref = json.loads(args.reference_report.read_text()) if args.reference_report else None
    try:
        m = produce_pages(args.mode, cfg, args.src, args.work, args.pages_root, called_from_eval_ld3=False,
                          seed=args.seed, n_pages=args.n_pages, mf_dir=args.mf_dir, reference=ref, extra_libs=args.extra_libs)
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return 1
    print(json.dumps({"mode": m["mode"], "names": m["names"], "overlap": (m["overlap"] or {}).get("status")}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
