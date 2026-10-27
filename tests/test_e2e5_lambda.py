"""E2E-5 checks (docs/END_TO_END_E2E5_LOT.txt §4 and amendment §6). CPU, synthetic, no GPU, no data.

(1) s = 1 gives outputs and gradients bit for bit identical to the code before GradScale (B3, B4, S2);
(2) s = 1/6 divides the L_ctc gradient on enc_det by 6 and leaves the Reader gradient unchanged;
(3) the L_det gradient on enc_det is unchanged;
(4) S2 is not affected by the scale;
(5) reading rules: summary printed = status, sha, step-0 ratio only; R7 boundaries; pages disjoint;
    the output never goes under the pages folder; the step-0 control refuses a wrong implementation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import e2e5_lambda as E  # noqa: E402
sys.path.insert(0, str(ROOT / "scripts/stage_s_notebooks"))
from babaorum.experimental import stage_s_arms as arms  # noqa: E402

ALPHABET = 12
SMALL = dict(channels=16, K=8, hidden=32)


def _inputs(seed=0):
    g = torch.Generator().manual_seed(seed)
    image = torch.rand((1, 1, 96, 96), generator=g)
    polylines = [np.array([[10.0 + 2 * k, 40.0] for k in range(10)]) + np.array([0.0, 0.0]),
                 np.array([[12.0 + 2 * k, 62.0 + 0.1 * k] for k in range(10)])]
    labels = [[1, 2, 3], [4, 5]]
    return image, polylines, labels


def _model(arm, scale, seed=3):
    torch.manual_seed(seed)
    return arms.StageSModel(arm, ALPHABET, ctc_det_grad_scale=scale, **SMALL)


def _grads(model, loss):
    model.zero_grad(set_to_none=True)
    loss.backward()
    return {n: (p.grad.clone() if p.grad is not None else None) for n, p in model.named_parameters()}


def _ctc(model, image, polylines, labels):
    out = model.forward_lines(image, polylines, labels, sigma_px=2.0)
    inc = [x for x in out["ctc_losses"] if x is not None]
    return sum(inc) / len(inc)


# ---- (1) s = 1 bit for bit identical to the old code (old path: features_read = features_det)

@pytest.mark.parametrize("arm", ["B3", "B4", "S2"])
def test_scale_one_is_bit_identical_to_the_old_code(arm, monkeypatch):
    image, polylines, labels = _inputs()
    new = _model(arm, 1.0)
    old = _model(arm, 1.0)
    monkeypatch.setattr(arms.GradScale, "apply", staticmethod(lambda x, s: x))  # old code path
    loss_old = _ctc(old, image, polylines, labels)
    g_old = _grads(old, loss_old)
    monkeypatch.undo()
    loss_new = _ctc(new, image, polylines, labels)
    g_new = _grads(new, loss_new)
    assert torch.equal(loss_new.detach(), loss_old.detach())
    for name in g_old:
        if g_old[name] is None:
            assert g_new[name] is None
        else:
            assert torch.equal(g_new[name], g_old[name]), name


# ---- (2) s = 1/6 on B3: enc_det gradient of L_ctc divided by 6, Reader unchanged

def test_one_sixth_divides_the_ctc_gradient_on_enc_det_and_not_the_reader():
    image, polylines, labels = _inputs()
    base = _model("B3", 1.0)
    g1 = _grads(base, _ctc(base, image, polylines, labels))
    scaled = _model("B3", 1.0 / 6.0)
    g6 = _grads(scaled, _ctc(scaled, image, polylines, labels))
    num = den = 0.0
    for name, _ in base.enc_det.named_parameters():
        ref = g1["enc_det." + name]
        if ref is not None:
            num += float(((6.0 * g6["enc_det." + name] - ref) ** 2).sum())
            den += float((ref ** 2).sum())
    assert den > 0
    assert (num / den) ** 0.5 < 1e-5   # measured 6.4e-07 on the synthetic batch (float32 rounding)
    for name in g1:
        if name.startswith("reader."):
            assert torch.equal(g6[name], g1[name]), name


# ---- (3) L_det gradient on enc_det unchanged

def test_detection_gradient_on_enc_det_is_unchanged():
    image, polylines, labels = _inputs()
    def l_det(model):
        out = model.forward_lines(image, [], [], sigma_px=2.0)
        return (out["det"]["logit"] ** 2).mean()
    base, scaled = _model("B3", 1.0), _model("B3", 1.0 / 6.0)
    gb = _grads(base, l_det(base))
    gs = _grads(scaled, l_det(scaled))
    for name in gb:
        if name.startswith("enc_det.") and gb[name] is not None:
            assert torch.equal(gs[name], gb[name]), name


# ---- (4) S2 not affected by the scale

def test_s2_is_not_affected_by_the_scale():
    image, polylines, labels = _inputs()
    a, b = _model("S2", 1.0), _model("S2", 1.0 / 6.0)
    la, lb = _ctc(a, image, polylines, labels), _ctc(b, image, polylines, labels)
    assert torch.equal(la.detach(), lb.detach())
    ga, gb = _grads(a, la), _grads(b, lb)
    for name in ga:
        if ga[name] is not None:
            assert torch.equal(ga[name], gb[name]), name


def test_scale_is_validated():
    with pytest.raises(ValueError):
        _model("B3", 0.0)
    with pytest.raises(ValueError):
        _model("B3", 1.5)


# ---- (5) reading rules and guards

def test_pages_are_0000_to_0049_for_training_and_0050_to_0059_for_evaluation():
    cfg = json.loads(E.CONFIG.read_text())
    assert cfg["pages"]["train"] == [f"mf_{i:04d}" for i in range(50)]
    assert cfg["pages"]["eval"] == [f"mf_{i:04d}" for i in range(50, 60)]
    assert not set(cfg["pages"]["train"]) & set(cfg["pages"]["eval"])


def test_config_values_of_the_lot():
    cfg = json.loads(E.CONFIG.read_text())
    assert cfg["steps"] == 2000 and cfg["eval_every"] == 1000
    assert cfg["time_cap_min"] == 35 and cfg["hard_stop_min"] == 38
    assert cfg["lambda"]["float"] == pytest.approx(1 / 6)


def test_r7_passes_at_the_thresholds_and_counts_empty_predictions_as_zero_precision():
    def page(rec, prec, n_pred):
        return {"metrics": {"after": {"recall": rec, "precision": prec, "n_pred": n_pred}}}
    passing = [page(0.5, 0.5, 3)] * 10
    assert E.r7_reading(passing)["pass"] is True
    assert E.r7_reading([page(0.5, 0.5, 3)] * 9 + [page(0.0, None, 0)])["pass"] is False
    empty = [page(0.6, None, 0)] * 10          # no prediction: precision counts 0
    assert E.r7_reading(empty)["precision_mean"] == 0.0


def test_r7_is_not_evaluable_with_fewer_than_ten_pages():
    assert E.r7_reading([{"metrics": {"after": {"recall": 1.0, "precision": 1.0, "n_pred": 1}}}] * 9)["evaluable"] is False


def test_printed_summary_has_only_status_sha_and_step0_ratio(tmp_path):
    path = tmp_path / "r.json"
    path.write_text("{}")
    report = {"status": "COMPLETE", "step0": {"ratio": 0.16666, "norm_s1": 2.0}, "r7": {"pass": True}}
    summary = E.public_summary(path, report)
    assert set(summary) == set(E.SUMMARY_KEYS)
    assert "r7" not in json.dumps(summary) and "recall" not in json.dumps(summary)


def test_output_never_under_the_pages_folder(tmp_path):
    pages = tmp_path / "e2e3" / "mf"
    with pytest.raises(SystemExit):
        E.check_output_dir(pages / "out", pages, smoke=True)


def test_output_refused_outside_kaggle_working_without_smoke(tmp_path):
    with pytest.raises(SystemExit):
        E.check_output_dir(tmp_path / "out", tmp_path / "mf", smoke=False)


def test_run_refuses_without_confirmed_go():
    with pytest.raises(SystemExit):
        E.main(["--tmp", "x", "--run3-report", "x", "--out-dir", "x"])


def test_step0_control_fails_when_the_scale_is_not_applied(monkeypatch):
    image, polylines, labels = _inputs()
    model = _model("B3", 1.0)
    monkeypatch.setattr(arms.GradScale, "apply", staticmethod(lambda x, s: x))  # scale ignored
    result = E.step0_control(model, (image[0, 0].numpy(), polylines, labels), torch.device("cpu"),
                             sigma_px=2.0, tolerance=1e-4)
    assert result["ok"] is False


# ---- amendment 2 §7: common install cell, read from KAGGLE_INSTALL.txt

def test_install_cell_of_the_builder_is_the_block_of_kaggle_install_txt():
    import build_kaggle_e2e5 as builder  # noqa: E402  (scripts/stage_s_notebooks is on the path below)
    txt = (ROOT / "scripts/stage_s_notebooks/KAGGLE_INSTALL.txt").read_text().splitlines()
    start = txt.index("Cellule (première cellule de code) :") + 1
    end = txt.index("Lancement des scripts (PYTHONPATH vers les sources du bundle, jamais d'import depuis le dossier courant) :")
    block = "\n".join(ln[4:] for ln in txt[start:end] if ln.startswith("    ")) + "\n"
    assert builder.install_cell() == block
    assert "REQUIRED_OUTSIDE_IMAGE = []" in block


def test_notebook_first_code_cell_is_the_install_cell(tmp_path):
    import build_kaggle_e2e5 as builder  # noqa: E402
    nb = builder.build({k: "0" * 64 for k in builder.KIT_FILES}, "0" * 64)
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert "".join(code_cells[0]["source"]) == builder.install_cell()


# ---- amendment 3 §8: explicit page lists; no page mf_0050-0059 is opened by a smoke test

def test_explicit_ids_never_open_pages_50_to_59(tmp_path, monkeypatch):
    import argparse
    opened = []
    class Fake(dict):
        pass

    def spy_load(root, pid, target_h):
        opened.append(f"load:{pid}")
        return Fake(image=None, lines=[], factor=1.0, page_id=pid, alphabet=[])

    def spy_run(cfg, arm, output, *, train_pages, eval_pages, **kw):
        opened.append(f"run:{list(train_pages)}|{list(eval_pages)}")
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text("{}")
        return {"steps_completed": 0, "elapsed_seconds": 0.0, "criterion_eval_step": None,
                "final_eval": "FINAL_EVAL_SKIPPED_BUDGET", "eval": [], "seconds_per_step_mean": None}

    monkeypatch.setattr(E, "load_scaled_page", spy_load)
    monkeypatch.setattr(E, "build_alphabet", lambda pages: ["<blank>", "a"])
    monkeypatch.setattr(E, "first_batch", lambda *a, **k: (None, [], []))
    monkeypatch.setattr(E, "step0_control", lambda *a, **k: {"ok": True, "ratio": 1 / 6})
    monkeypatch.setattr(E.T, "run_v2", spy_run)
    args = argparse.Namespace(smoke=True, device="cpu", steps=2, train_ids=["mf_0000"], eval_ids=["mf_0020"],
                              tmp=tmp_path, run3_report=tmp_path / "r.json", out_dir=tmp_path / "out")
    (tmp_path / "mf" / "pages").mkdir(parents=True)
    (tmp_path / "r.json").write_text(json.dumps({"steps": {"page_cache": {"sha256": {}}}}))
    E.run_pilot(args, json.loads(E.CONFIG.read_text()))
    joined = " ".join(opened)
    for n in range(50, 60):
        assert f"mf_{n:04d}" not in joined, n
    assert "run:['mf_0000']|['mf_0020']" in joined
