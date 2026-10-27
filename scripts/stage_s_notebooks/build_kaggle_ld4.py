"""Carnets Kaggle LD-4 (docs/END_TO_END_LD4_PROPOSITION.txt §8 ; fusion DETR + B3+λ). Non lancé.

Même mécanique que build_kaggle_ld3 (gel 4) : pip isolé des bibliothèques du générateur, bundle v5 (unzip, clone, checkout, contrôle des
fichiers), D, controls, CAL (une page d'ENTRAÎNEMENT : DETR + B3), porte d'estimation, EVAL (génération des pages mf_0070..mf_0079 PUIS
lecture, une fois). Chaque étape argv est bornée par max(0, min(plafond, restant - marge d'arrêt)) ; les étapes fn sont gardées ; délai propre
de la génération ; fins de journaux en cas d'échec. Échéance globale 5 400 s. La fumée (CPU) génère UNE page leurre et utilise une référence B3
factice pour le critère (jamais pour les lignes B3 de la fusion, calculées sur la page leurre).
Le texte des cellules est celui de build_kaggle_ld3 (importé, non modifié) avec ld3 -> ld4 ; un test vérifie qu'il ne reste aucune trace de LD-3.
"""
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_kaggle_ld1 as B  # noqa: E402
import build_kaggle_ld3 as L3  # noqa: E402

ROOT = B.ROOT
CONFIG = ROOT / "config" / "ld4.json"
KIT_SLUG = "mabaku/babaorum-ld4-kit"
RUN_SLUG = "mabaku/babaorum-ld4-run"
SMOKE_SLUG = "mabaku/babaorum-ld4-smoke"
KIT_CANDIDATES = [c.replace("ld3", "ld4") for c in L3.KIT_CANDIDATES]
KIT_ROOTS = ["scripts/eval_ld4.py", "config/ld4.json", "tests/test_ld4.py", "scripts/eval_ld3.py", "scripts/data3_generate.py",
             "scripts/data2_generate.py", "scripts/data2_common.py", "scripts/ld1b_border_diag.py", "scripts/train_ld2.py",
             "scripts/ld2_core.py", "scripts/ld0_zone_coverage.py", "scripts/train_line_detr.py", "scripts/train_stage_s.py",
             "scripts/train_stage_s_v2.py", "src/babaorum/experimental/line_detr.py", "src/babaorum/experimental/stage_s_arms.py",
             "tests/resolve_probe.py"]
KIT_READS = ["config/ld4.json", "config/ld3.json", "config/ld2.json", "config/ld1_line_detr.json", "config/stage_s_smoke_v2.json",
             "config/data2_pages.json", "scripts/data2_common.py", "scripts/data2_generate.py", "scripts/data3_generate.py",
             "scripts/eval_ld3.py", "scripts/eval_ld4.py", "scripts/train_ld2.py", "scripts/ld2_core.py", "scripts/ld1b_border_diag.py",
             "scripts/ld0_zone_coverage.py", "scripts/train_line_detr.py", "scripts/train_stage_s.py", "scripts/train_stage_s_v2.py",
             "scripts/diag_ld_o1.py", "config/ld_o1.json"]               # épinglés par config/ld4.json (frozen_files_sha256)
SMOKE_CAL_S, SMOKE_EVAL_S = L3.SMOKE_CAL_S, L3.SMOKE_EVAL_S
resolve_one, resolve_file = B.resolve_one, B.resolve_file


def _t(text: str) -> str:
    """ld3 -> ld4 dans le texte des cellules de LD-3 (noms de fichiers, de dossiers, de sorties)."""
    return text.replace("ld3", "ld4").replace("LD-3", "LD-4").replace("LD3", "LD4")


def kit_files() -> list:
    vendor = sorted(str(Path(p).relative_to(ROOT)) for p in glob.glob(str(ROOT / "scripts" / "ld0_vendor" / "**" / "*.py"), recursive=True))
    return sorted(set(B.import_closure(KIT_ROOTS)) | set(vendor) | set(KIT_READS) | {"scripts/stage_s_notebooks/KAGGLE_INSTALL.txt"})


def _cfg() -> dict:
    return json.loads(CONFIG.read_text())


def _limits() -> dict:
    r = _cfg()["robustness"]
    return {k: r[k] for k in ("deadline_s", "stop_margin_s", "eval_margin_s", "estimate_factor", "generation_estimate_s", "heartbeat_s")}


def cell(kind: str, out: str = "/kaggle/working", kit=None, pages=None, ckpt=None, ld2=None, e2e4=None, bundle=None,
         e2e5_sha=None, ld2_sha=None, e2e4_sha=None, bundle_sha=None) -> str:
    cfg = _cfg()
    head = (L3.HEAD.replace("__OUT__", repr(out)).replace("__KIT__", repr(kit or KIT_CANDIDATES))
            .replace("__PAGES__", repr(pages or B.PAGES_CANDIDATES)).replace("__CKPT__", repr(ckpt or B.CKPT_CANDIDATES))
            .replace("__LD2__", repr(ld2 or L3.LD2_CANDIDATES)).replace("__E2E4__", repr(e2e4 or L3.E2E4_CANDIDATES))
            .replace("__BUNDLE__", repr(bundle or L3.BUNDLE_CANDIDATES))
            .replace("__E2E5_SHA__", repr(e2e5_sha or cfg["e2e5"]["checkpoint_sha256"]))
            .replace("__LD2_SHA__", repr(ld2_sha or cfg["ld2"]["checkpoint_sha256"]))
            .replace("__LD2_FILE__", repr(cfg["ld2"]["checkpoint_file"]))
            .replace("__E2E4_SHA__", repr(e2e4_sha or cfg["overlap"]["reference_report_sha256"]))
            .replace("__BUNDLE_SHA__", repr(bundle_sha or cfg["generator"]["bundle_sha256"]))
            .replace("__COMMIT__", repr(cfg["generator"]["commit"])).replace("__PIP__", repr(cfg["generator"]["required_pip"]))
            .replace("__RUNFILES__", repr(cfg["run_identity_files"])).replace("__TIMEOUTS__", repr(cfg["robustness"]["timeouts_s"]))
            .replace("__LIMITS__", repr(_limits())))
    if kind == "full":
        return _t(head + L3.RESOLVERS + L3._argv_builder("cuda") + L3.FULL_BODY)
    return _t(head + L3.RESOLVERS + L3._argv_builder("cpu")
              + L3.SMOKE_BODY.replace("__CALS__", repr(SMOKE_CAL_S)).replace("__EVALS__", repr(SMOKE_EVAL_S)))


def build_notebook() -> dict:
    return {"nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}},
            "cells": [{"cell_type": "markdown", "metadata": {}, "source": "# LD-4 — fusion DETR + B3+λ, inférence seule (GPU)"},
                      {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": cell("full")}]}


def build_smoke_notebook() -> dict:
    return {"nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}},
            "cells": [{"cell_type": "markdown", "metadata": {}, "source": "# LD-4 — FUMÉE Kaggle CPU : une page leurre, référence B3 factice"},
                      {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": cell("smoke")}]}


def kernel_metadata(slug: str, title: str, gpu: bool = False) -> dict:
    """Internet réglé par carnet : nécessaire à pip et à D ; hors de ces étapes, HF_HUB_OFFLINE=1."""
    m = L3.kernel_metadata(slug, title, gpu)
    m["dataset_sources"] = [KIT_SLUG, L3.BUNDLE_SLUG]
    return m


def write(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "kaggle_ld4.ipynb").write_text(json.dumps(build_notebook(), indent=1, ensure_ascii=False))
    (out_dir / "kaggle_ld4_smoke.ipynb").write_text(json.dumps(build_smoke_notebook(), indent=1, ensure_ascii=False))
    (out_dir / "kit_files.json").write_text(json.dumps(kit_files(), indent=1))
    for sub, slug, title, gpu, nb in (("smoke_kernel", SMOKE_SLUG, "babaorum-ld4-smoke", False, build_smoke_notebook()),
                                      ("full_kernel", RUN_SLUG, "babaorum-ld4-run", True, build_notebook())):
        d = out_dir / sub
        d.mkdir(parents=True, exist_ok=True)
        (d / "kernel-metadata.json").write_text(json.dumps(kernel_metadata(slug, title, gpu=gpu), indent=2))
        (d / "notebook.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    write(Path(__file__).parent / "ld4")
    print(Path(__file__).parent / "ld4")
