"""LD-1B, test LOURD (torch) : tourne sur Kaggle seulement (LD1B_HEAVY=1 et LD1B_PAGES_ROOT). Non exécuté en local.
Un dossier LD-1 factice (checkpoint aléatoire de même forme, pas S = 2) et la page mf_0000 (ENTRAÎNEMENT, usage de test
seulement) : vérifie que la chaîne complète tourne et produit un rapport JSON, sans rien lire de mf_0050-0059."""
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HEAVY = pytest.mark.skipif(os.environ.get("LD1B_HEAVY") != "1" or not os.environ.get("LD1B_PAGES_ROOT"),
                           reason="heavy: LD1B_HEAVY=1 et LD1B_PAGES_ROOT (Kaggle)")


@HEAVY
def test_the_full_chain_runs_on_a_factice_ld1_directory(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    import torch
    import ld1b_border_diag as D
    import train_line_detr as TR
    from babaorum.experimental.line_detr import LineDetr
    ld1 = tmp_path / "ld1"
    ld1.mkdir()
    torch.manual_seed(0)
    torch.save({"model": LineDetr().state_dict(), "step": 2}, ld1 / "ld1_step000002.pt")
    (ld1 / "ld1_timing_manifest.json").write_text(json.dumps({"steps": 2}))
    (ld1 / "ld1_report.json").write_text("{}")
    cfg = json.loads((ROOT / "config" / "ld1b_border.json").read_text())
    cfg["pages"] = ["mf_0000"]

    class Args:
        ld1_dir = str(ld1)
        root = os.environ["LD1B_PAGES_ROOT"]
        out = str(tmp_path / "out")
        cache_dir = str(tmp_path / "cache")

    report = D.measure(Args(), cfg)
    assert report["descriptive_only"] is True
    assert set(report["summary"]) == {"Z-GT", "Z-PP-blocs"}
    assert report["summary"]["Z-GT"]["0-8"]["gt_lines_total"] + report["summary"]["Z-GT"][">32"]["gt_lines_total"] >= 0
    json.dumps(report, default=str)
    assert "mf_0050" not in json.dumps(report)
