"""LD-1 checks (docs/END_TO_END_LD1_LOT.txt §6, §10 (7)). Les tests marqués HEAVY calculent un réseau : lancés seulement
avec LD1_HEAVY=1, via ~/outils-seg/lourd.sh, quand le disque et pA le permettent (§7, §10 (8))."""
import builtins
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from babaorum.experimental import line_detr as LD  # noqa: E402
import train_line_detr as TR  # noqa: E402

PAGES_ROOT = Path(os.environ.get("LD1_PAGES_ROOT", str(ROOT.parent / "mathematicien" / "inputs" / "rg" / "acquisition" / "e2e3" / "mf")))
CARNET_FILES = ROOT / "scripts" / "stage_s_notebooks"
NEEDS_CARNET = pytest.mark.skipif(
    not (CARNET_FILES / "build_kaggle_ld1.py").is_file() or not (CARNET_FILES / "ld1" / "kaggle_ld1.ipynb").is_file(),
    reason="build_kaggle_ld1.py / ld1/*.ipynb absents du kit (Kaggle) : contrôlés en local par tests/test_ld1_carnet.py")
HEAVY = pytest.mark.skipif(os.environ.get("LD1_HEAVY") != "1", reason="heavy: LD1_HEAVY=1 via lourd.sh")


# ---- §6 : appariement hongrois, repères, NMS ----

def test_hungarian_recovers_the_permutation_on_a_toy_case():
    cost = np.array([[9.0, 1.0, 9.0, 9.0], [9.0, 9.0, 1.0, 9.0], [9.0, 9.0, 9.0, 1.0]])
    assert LD.hungarian(cost) == [(0, 1), (1, 2), (2, 3)]


def test_hungarian_equals_brute_force_on_random_cases():
    from itertools import permutations
    rng = np.random.default_rng(3)
    for _ in range(10):
        c = rng.random((4, 6))
        best = min(sum(c[i, p[i]] for i in range(4)) for p in permutations(range(6), 4))
        assert abs(sum(c[i, j] for i, j in LD.hungarian(c)) - best) < 1e-9


def test_zone_coordinates_round_trip_to_1e4_px_including_a_downscaled_zone():
    page = np.full((900, 2000), 0.5, dtype=np.float32)
    for box in ((100, 100, 600, 140), (50, 50, 1900, 800)):          # la seconde est réduite
        canvas, meta = LD.zone_canvas(page, box)
        pts = np.array([[120.0, 120.0], [300.3, 130.2], [500.0, 118.0]])
        back = LD.zone_to_page(LD.page_to_zone(pts, meta), meta)
        assert np.abs(back - pts).max() < 1e-4
    assert meta.fx < 1.0 and meta.fy < 1.0 and canvas.shape == (LD.CANVAS_H, LD.CANVAS_W)


def _line(x0, x1, y=50.0, n=8):
    xs = np.linspace(x0, x1, n)
    return np.stack([xs, np.full(n, y)], axis=-1)


def test_nms_suppresses_an_exact_duplicate_and_keeps_distant_lines():
    a = _line(0, 200)
    preds = [{"score": 0.9, "zone": 0, "query": 0, "pts": a, "h": 20.0},
             {"score": 0.8, "zone": 0, "query": 1, "pts": a.copy(), "h": 20.0},
             {"score": 0.7, "zone": 0, "query": 2, "pts": _line(0, 200, y=300.0), "h": 20.0}]
    kept = LD.nms_lines(preds)
    assert [p["query"] for p in kept] == [0, 2]


def test_nms_keeps_two_halves_of_a_line_cut_by_a_zone_border():
    left, right = _line(0, 100), _line(100, 200)
    preds = [{"score": 0.9, "zone": 0, "query": 0, "pts": left, "h": 20.0},
             {"score": 0.8, "zone": 1, "query": 0, "pts": right, "h": 20.0}]
    assert len(LD.nms_lines(preds)) == 2                              # coupure : deux morceaux gardés


def test_nms_suppresses_a_short_fragment_contained_in_a_long_kept_line():
    """§12 (10) : fragment court contenu dans une longue ligne gardée -> supprimé (t = distance des points candidats)."""
    long_line = _line(0, 400)
    fragment = _line(100, 160)
    preds = [{"score": 0.9, "zone": 0, "query": 0, "pts": long_line, "h": 20.0},
             {"score": 0.8, "zone": 0, "query": 1, "pts": fragment, "h": 20.0}]
    assert [p["query"] for p in LD.nms_lines(preds)] == [0]


def test_nms_order_is_deterministic_on_equal_scores():
    a = _line(0, 200)
    preds = [{"score": 0.5, "zone": 1, "query": 4, "pts": a, "h": 20.0},
             {"score": 0.5, "zone": 0, "query": 7, "pts": a.copy(), "h": 20.0}]
    assert [(p["zone"], p["query"]) for p in LD.nms_lines(preds)] == [(0, 7)]


# ---- §10 (9) : métriques et lecture ----

def test_mean_pr_counts_a_page_without_prediction_as_zero_and_a_none_precision_apart():
    r = TR.mean_pr([{"precision": 1.0, "recall": 1.0}, {"precision": 0.0, "recall": 0.0}])
    assert (r["P"], r["R"]) == (0.5, 0.5) and r["precision_none_pages"] == 0
    r2 = TR.mean_pr([{"precision": None, "recall": 0.5}, {"precision": 0.5, "recall": 0.5}])
    assert r2["P"] == 0.5 and r2["precision_none_pages"] == 1          # None n'est pas remplacée par 0


def test_the_train_and_held_lists_are_exact():
    assert TR.TRAIN_PAGES == [f"mf_{i:04d}" for i in range(0, 45)]
    assert TR.HELD_PAGES == [f"mf_{i:04d}" for i in range(45, 50)]


def test_threshold_choice_maximises_f1_and_takes_the_lowest_on_ties():
    grid = {0.3: (0.6, 0.9), 0.4: (0.7, 0.8), 0.5: (0.75, 0.75), 0.6: (0.75, 0.75), 0.7: (0.5, 0.9)}
    assert TR.choose_threshold(grid) == 0.5


def test_reading_thresholds_are_applied_to_the_minimum_of_precision_and_recall():
    assert TR.reading(0.80, 0.90) == "CONVERGE"
    assert TR.reading(0.79, 0.95) == "PARTIEL"
    assert TR.reading(0.50, 0.50) == "PARTIEL"
    assert TR.reading(0.49, 0.99) == "ECHEC"


# ---- §10 (1)-(2) : Z-GT par bloc, jamais mf_0050-0059 ----

def test_the_forbidden_pages_are_refused_and_the_lists_are_disjoint():
    assert not (set(TR.TRAIN_PAGES) & TR.FORBIDDEN) and not (set(TR.HELD_PAGES) & TR.FORBIDDEN)
    with pytest.raises(SystemExit):
        TR.guard_pages(["mf_0050"])
    with pytest.raises(SystemExit):
        TR.guard_pages(["mf_0059"])


def test_the_script_never_names_the_forbidden_pages_except_in_the_guard():
    """Seules les lignes de réserve (définition, garde, docstring) mentionnent mf_005x."""
    src = (ROOT / "scripts" / "train_line_detr.py").read_text()
    allowed = ("FORBIDDEN", "startswith", "mf_0050-0059")
    for line in src.splitlines():
        if "mf_005" in line:
            assert any(tag in line for tag in allowed), line


def test_only_mf_0000_is_read_in_the_smoke_path(tmp_path, monkeypatch):
    opened = []
    real_open = builtins.open

    def spy(file, *a, **k):
        opened.append(os.fspath(file))
        return real_open(file, *a, **k)

    monkeypatch.setattr(builtins, "open", spy)
    root = PAGES_ROOT
    if not (root / "pages" / "mf_0000.json").exists():
        pytest.skip("mf_0000 absent (Mac only)")
    TR.load_scaled(root, "mf_0000")
    monkeypatch.undo()
    assert not any("mf_005" in p for p in opened)


# ---- §6 : checkpoint E2E-5 : clés, formes, pas ----

def test_enc_det_loads_only_with_the_expected_keys_and_shapes_and_step():
    model = LD.LineDetr()
    state = {f"enc_det.{k}": v.clone() for k, v in model.enc_det.state_dict().items()}
    state["enc_det.extra"] = torch.zeros(1)
    with pytest.raises(ValueError):
        LD.load_enc_det_from_e2e5(model, state, expected_step=2000, step=2000)
    del state["enc_det.extra"]
    with pytest.raises(ValueError):
        LD.load_enc_det_from_e2e5(model, state, expected_step=2000, step=1000)
    LD.load_enc_det_from_e2e5(model, state, expected_step=2000, step=2000)


# ---- §10 (7) : remplissage et sur-apprentissage (lourd) ----

def test_masked_group_norm_equals_group_norm_without_padding():
    torch.manual_seed(0)
    gn = torch.nn.GroupNorm(8, 32)
    x = torch.randn(2, 32, 16, 24)
    out = LD.masked_group_norm(x, torch.ones(2, 1, 16, 24), gn)
    assert float((out - gn(x)).abs().max()) <= 1e-6


def test_masked_encoder_equals_the_original_encoder_without_padding():
    from babaorum.experimental.stage_s_model import Encoder
    torch.manual_seed(0)
    enc = Encoder(64).eval()
    x = torch.rand(1, 1, 128, 192)
    with torch.no_grad():
        ref = enc(x)
        out, _ = LD.masked_encoder(enc, x, torch.ones_like(x))
    scale = max(1.0, float(ref.abs().max()))
    assert float((out - ref).abs().max()) <= 1e-6 * scale


@pytest.mark.parametrize("off", [(0, 0), (5, 3)], ids=["aligne_sur_0", "non_aligne_5_3"])
def test_the_same_crop_in_two_canvas_sizes_gives_the_same_outputs(off):
    """§12 (6) et §13 (5) : test NON trivial. Le MÊME recadrage (aligné sur 0, ou décalé de (5, 3), non multiple de 4 ni
    de 16) est placé au même coin dans deux canevas de tailles différentes (écart de 64 px et plus). Les sorties du
    décodeur sont identiques à 1e-5 près : le hors-recadrage n'est jamais lu, quelle que soit la taille du canevas."""
    torch.manual_seed(0)
    model = LD.LineDetr().eval()
    oy, ox = off

    def run(hc, wc):
        canvas = torch.full((1, 1, hc, wc), 1.0)
        canvas[:, :, oy + 8:oy + 72, ox + 8:ox + 200] = 0.2
        mask = torch.zeros(1, 1, hc, wc)
        mask[:, :, oy:oy + 80, ox:ox + 208] = 1.0
        with torch.no_grad():
            return model(canvas, mask)[-1]
    a = run(128, 256)
    b = run(192, 320)                                              # écart de 64 px sur chaque dimension
    for key in ("logit", "pts", "neg", "pos"):
        assert float((a[key] - b[key]).abs().max()) <= 1e-5, key


def test_load_l0_sets_the_hf_environment_and_the_dependencies_before_any_download():
    TR_mod = TR
    L0 = TR_mod.load_l0({"detector": {"repo": "x", "revision": "y", "files": {}, "floor": 0.1}})
    assert os.environ["HF_HUB_DISABLE_XET"] == "1"
    assert L0.bbox is not None and L0.Params is not None


def test_pp_zones_runs_on_a_factice_page_with_an_injected_detector(tmp_path):
    """§13 (N2) : pp_zones s'exécute sur une page factice, avec un détecteur factice injecté (sans réseau)."""
    L0 = TR.load_l0({"detector": {"repo": "x", "revision": "y", "files": {}, "floor": 0.1}})
    root = tmp_path
    (root / "images").mkdir()
    Image.fromarray(np.zeros((200, 300, 3), dtype=np.uint8)).save(root / "images" / "mf_9999.png")
    page = {"scaled_wh": (600, 400)}

    def fake_det(arr):
        return {"width": 300, "height": 200, "blocks": [
            {"id": "b0", "label": "text", "score": 0.9, "polygon": [[10, 20], [110, 20], [110, 60], [10, 60]]}]}

    zones = TR.pp_zones(page, root, "mf_9999", tmp_path / "cache", L0, det_fn=fake_det)
    assert len(zones) == 1
    zid, box = zones[0]
    assert zid == 0 and box == (20.0, 40.0, 220.0, 120.0)           # scale 2 (600/300, 400/200)


def test_pages_without_prediction_are_counted_and_have_precision_zero():
    rows = [{"precision": 0.0, "recall": 0.0, "n_pred": 0}, {"precision": 0.8, "recall": 0.9, "n_pred": 3}]
    r = TR.mean_pr(rows)
    assert r["pages_without_prediction"] == 1 and r["P"] == 0.4
def test_params_by_block_adds_up_to_the_total():
    model = LD.LineDetr()
    parts = LD.params_by_block(model)
    assert sum(v for k, v in parts.items() if k != "total") == parts["total"]


def test_the_mount_resolver_takes_exactly_one_candidate_and_never_lists(tmp_path, monkeypatch):
    """§11 (b) : deux formats de montage et un leurre mf_0050 ; aucun listage, aucune lecture hors marqueur."""
    import resolve_probe as RP
    fmt_a = tmp_path / "input" / "babaorum-e2e3-run3" / "e2e3" / "mf"
    fmt_b = tmp_path / "input" / "notebooks" / "mabaku" / "babaorum-e2e3-run3" / "e2e3" / "mf"
    decoy = fmt_a / "pages" / "mf_0050.json"
    for base in (fmt_a, fmt_b):
        (base / "pages").mkdir(parents=True)
    decoy.write_text("{}")
    (fmt_b / "pages" / "mf_0000.json").write_text("{}")
    forbidden = []
    for name in ("listdir", "scandir"):
        monkeypatch.setattr(os, name, lambda *a, _n=name, **k: forbidden.append(_n))
    monkeypatch.setattr(Path, "glob", lambda *a, **k: forbidden.append("glob") or iter(()))
    monkeypatch.setattr(Path, "iterdir", lambda *a, **k: forbidden.append("iterdir") or iter(()))
    monkeypatch.setattr(Path, "rglob", lambda *a, **k: forbidden.append("rglob") or iter(()))
    hit = RP.resolve(
        [str(fmt_a), str(fmt_b)], "pages/mf_0000.json")
    assert hit == fmt_b
    assert forbidden == []
    assert str(decoy) not in str(hit)
    with pytest.raises(RuntimeError):
        RP.resolve([str(tmp_path / "absent")], "pages/mf_0000.json")


@HEAVY
def test_overfit_one_zone_of_mf_0000_divides_the_loss_by_ten_in_200_steps():
    root = PAGES_ROOT
    page = TR.load_scaled(root, "mf_0000")
    blk = next(b for b in page["blocks"] if len(b["lines"]) >= 2)
    s = TR.zone_sample(page, blk, blk["box"])
    torch.manual_seed(0)
    model = LD.LineDetr()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    first = None
    for step in range(200):
        loss, _, _ = TR.batch_loss(model, [s], torch.device("cpu"))
        if first is None:
            first = float(loss)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert float(loss) <= first / 10.0


# ---- carnet LD-1 : égalité avec build(), taille, entrées explicites, aucun mf_005x, KAGGLE_INSTALL inchangé ----

def _nb_cells():
    sys.path.insert(0, str(ROOT / "scripts" / "stage_s_notebooks"))
    import build_kaggle_ld1 as B
    return B, json.loads((ROOT / "scripts" / "stage_s_notebooks" / "ld1" / "kaggle_ld1.ipynb").read_text())


@NEEDS_CARNET
def test_the_carnet_on_disk_equals_build_and_is_small():
    B, nb = _nb_cells()
    assert nb == B.build_notebook()
    assert (ROOT / "scripts" / "stage_s_notebooks" / "ld1" / "kaggle_ld1.ipynb").stat().st_size < 300 * 1024


@NEEDS_CARNET
def test_the_carnet_reads_only_explicit_candidates_and_never_names_forbidden_pages():
    _B, nb = _nb_cells()
    code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    for bad in ("glob(", "rglob", "listdir", "scandir", "os.walk", "iterdir", "mf_005"):
        assert bad not in code, bad


@NEEDS_CARNET
def test_the_kit_closure_contains_every_babaorum_module_the_model_imports():
    B, _nb = _nb_cells()
    files = set(json.loads((ROOT / "scripts" / "stage_s_notebooks" / "ld1" / "kit_files.json").read_text()))
    for rel in ("src/babaorum/experimental/line_detr.py", "src/babaorum/experimental/stage_s_model.py",
                "src/babaorum/experimental/joint_geometry.py", "src/babaorum/experimental/joint_reader_bridge.py",
                "src/babaorum/line_geometry.py", "scripts/train_line_detr.py", "config/ld1_line_detr.json"):
        assert rel in files, rel


@NEEDS_CARNET
def test_kaggle_install_is_unchanged_and_the_carnet_carries_the_full_checkpoint_sha():
    install = (ROOT / "scripts" / "stage_s_notebooks" / "KAGGLE_INSTALL.txt").read_bytes()
    import hashlib
    assert hashlib.sha256(install).hexdigest().startswith("a0cda2416808a1ab")
    _B, nb = _nb_cells()
    code = "".join("".join(c["source"]) for c in nb["cells"][1:])
    assert "ef5e826803c347f73e49c83c0f21f61399f9ea06ee985905bd17644dff3844fa" in code
    assert "INIT_SHA = 'ef5e826803c347f73e49c83c0f21f61399f9ea06ee985905bd17644dff3844fa'" in code


# ---- gel 7 : §15 (1)-(3), §17 (2) (B1, B2, reprise interdite, sorties d'étape hors de RUN) ----
# Tests torch : ils ne tournent qu'en fumée Kaggle (pas de torch dans le venv local).

def test_the_targets_are_created_on_the_device_of_the_predictions(monkeypatch):
    """§15 (1) [B1] : points, neg, pos et indices de classes portent le device des prédictions (aucun appel sans device)."""
    seen = []
    real_as_tensor, real_tensor = torch.as_tensor, torch.tensor

    def spy_as_tensor(*a, **k):
        seen.append(("as_tensor", k.get("device")))
        return real_as_tensor(*a, **k)

    def spy_tensor(*a, **k):
        seen.append(("tensor", k.get("device")))
        return real_tensor(*a, **k)

    out_l = {"logit": torch.randn(1, TR.N_QUERIES), "pts": torch.rand(1, TR.N_QUERIES, TR.K, 2),
             "neg": torch.rand(1, TR.N_QUERIES), "pos": torch.rand(1, TR.N_QUERIES)}
    sample = {"pts": np.random.rand(2, TR.K, 2) * 50.0, "neg": np.array([5.0, 6.0]), "pos": np.array([7.0, 8.0])}
    monkeypatch.setattr(torch, "as_tensor", spy_as_tensor)
    monkeypatch.setattr(torch, "tensor", spy_tensor)
    TR.layer_loss(out_l, 0, sample, [(0, 3), (1, 5)])
    monkeypatch.undo()
    assert len(seen) >= 4                                          # indices, points, neg, pos
    assert all(dev is not None and torch.device(dev) == out_l["pts"].device for _, dev in seen), seen


def test_only_an_out_of_memory_triggers_the_batch_fallback(monkeypatch):
    """§15 (1) [B1] : un OOM fait passer à la taille suivante ; toute autre erreur remonte au premier essai."""
    opt = types.SimpleNamespace(zero_grad=lambda **k: None)
    cfg = {"batch_candidates": [8, 4, 2]}
    sizes = []
    monkeypatch.setattr(TR, "sample_batch", lambda pages, zi, rng, bs: sizes.append(bs) or [])

    def oom_until(limit):
        def fake(model, s, device):
            if sizes[-1] > limit:
                raise RuntimeError("CUDA out of memory. Tried to allocate 20.00 MiB")
            return types.SimpleNamespace(backward=lambda: None), None, None
        return fake

    monkeypatch.setattr(TR, "batch_loss", oom_until(2))
    assert TR.choose_batch(None, opt, None, None, None, torch.device("cpu"), cfg)[0] == 2
    assert sizes == [8, 4, 2]
    sizes.clear()

    def device_bug(model, s, device):
        raise RuntimeError("Expected all tensors to be on the same device")

    monkeypatch.setattr(TR, "batch_loss", device_bug)
    with pytest.raises(RuntimeError, match="same device"):
        TR.choose_batch(None, opt, None, None, None, torch.device("cpu"), cfg)
    assert sizes == [8]                                            # pas de repli sur 4 puis 2
    sizes.clear()
    monkeypatch.setattr(TR, "batch_loss", oom_until(0))
    with pytest.raises(TR.RepeatedOOM):
        TR.choose_batch(None, opt, None, None, None, torch.device("cpu"), cfg)
    assert sizes == [8, 4, 2]
    assert TR.is_oom(torch.cuda.OutOfMemoryError("x")) and TR.is_oom(RuntimeError("CUDA out of memory"))
    assert not TR.is_oom(RuntimeError("Expected all tensors to be on the same device"))
    assert not TR.is_oom(ValueError("out of memory"))


def test_the_device_control_makes_two_steps_and_turns_a_device_error_into_device_error(monkeypatch):
    steps = []
    net = torch.nn.Linear(1, 1)
    opt = types.SimpleNamespace(zero_grad=lambda **k: None, step=lambda: steps.append(1), state_dict=lambda: {},
                                load_state_dict=lambda sd: None)
    cfg = {"batch_candidates": [8, 4, 2]}
    monkeypatch.setattr(TR, "sample_batch", lambda pages, zi, rng, bs: [])
    monkeypatch.setattr(TR, "batch_loss", lambda m, s, d: (types.SimpleNamespace(backward=lambda: None), None, None))
    TR.device_control(net, opt, None, None, None, torch.device("cpu"), cfg)
    assert len(steps) == 2

    def device_bug(model, s, device):
        raise RuntimeError("Expected all tensors to be on the same device, cuda:0 and cpu")

    monkeypatch.setattr(TR, "batch_loss", device_bug)
    with pytest.raises(TR.DeviceError, match="same device"):
        TR.device_control(net, opt, None, None, None, torch.device("cpu"), cfg)

    def oom(model, s, device):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(TR, "batch_loss", oom)
    with pytest.raises(RuntimeError, match="out of memory") as ei:
        TR.device_control(net, opt, None, None, None, torch.device("cpu"), cfg)
    assert not isinstance(ei.value, TR.DeviceError)


def test_the_device_control_leaves_parameters_and_optimizer_state_unchanged(monkeypatch):
    """Les 2 pas de contrôle ne modifient ni les poids ni l'état de l'optimiseur (restauration dans un finally)."""
    import copy
    torch.manual_seed(0)
    net = torch.nn.Linear(3, 2)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-2)
    x = torch.randn(4, 3)
    net(x).sum().backward()
    opt.step()                                                     # l'état d'Adam n'est pas vide
    before_m, before_o = copy.deepcopy(net.state_dict()), copy.deepcopy(opt.state_dict())
    monkeypatch.setattr(TR, "sample_batch", lambda pages, zi, rng, bs: [])
    monkeypatch.setattr(TR, "batch_loss", lambda m, s, d: ((m(x) ** 2).sum(), None, None))
    TR.device_control(net, opt, None, None, None, torch.device("cpu"), {"batch_candidates": [8, 4, 2]})
    for k, v in before_m.items():
        assert torch.equal(net.state_dict()[k], v), k
    after_o = opt.state_dict()
    assert after_o["param_groups"] == before_o["param_groups"]
    for k, st in before_o["state"].items():
        for name, v in st.items():
            assert torch.equal(after_o["state"][k][name], v), (k, name)

    def bug(m, s, d):
        (m(x) ** 2).sum().backward()
        opt.step()
        raise RuntimeError("Expected all tensors to be on the same device")

    monkeypatch.setattr(TR, "batch_loss", bug)
    with pytest.raises(TR.DeviceError):
        TR.device_control(net, opt, None, None, None, torch.device("cpu"), {"batch_candidates": [8, 4, 2]})
    for k, v in before_m.items():                                  # restauré aussi après une erreur
        assert torch.equal(net.state_dict()[k], v), k


def _main_args(tmp_path, phase):
    run = tmp_path / "out" / "ld1"
    return run, ["--phase", phase, "--out", str(run), "--root", str(tmp_path), "--init-checkpoint", str(tmp_path / "x.pt")]


def _exit_record(tmp_path, phase):
    return json.loads((tmp_path / "out" / "exits" / f"ld1_run_exit_{phase}.json").read_text())


def test_the_seeds_are_set_before_build_model(tmp_path, monkeypatch):
    """§15 (2) [B2] : à l'entrée de build_model, torch, numpy et random sont déjà à l'état de la graine 20261023."""
    import random
    seed = json.loads(TR.CONFIG.read_text())["seed"]
    assert seed == 20261023
    seen = {}

    def spy(args, cfg, device):
        seen.update(torch=torch.initial_seed(), py=random.random(), np=float(np.random.rand()))
        raise RuntimeError("sentinelle")

    monkeypatch.setattr(TR, "build_model", spy)
    _run, argv = _main_args(tmp_path, "T")
    with pytest.raises(RuntimeError, match="sentinelle"):
        TR.main(argv)
    assert seen["torch"] == seed
    assert seen["py"] == random.Random(seed).random()
    assert seen["np"] == float(np.random.RandomState(seed).rand())


def test_two_constructions_with_the_same_seed_give_identical_parameters_except_enc_det():
    TR.set_seeds(20261023)
    a = LD.LineDetr()
    TR.set_seeds(20261023)
    b = LD.LineDetr()
    differs_from_other_seed = False
    TR.set_seeds(1)
    c = LD.LineDetr()
    for (n, pa), (_, pb), (_, pc) in zip(a.named_parameters(), b.named_parameters(), c.named_parameters()):
        if n.startswith("enc_det."):
            continue
        assert torch.equal(pa, pb), n
        differs_from_other_seed = differs_from_other_seed or not torch.equal(pa, pc)
    assert differs_from_other_seed


def test_phase_D_writes_nothing_in_RUN_and_the_real_guard_then_passes(tmp_path, monkeypatch):
    """§17 (2) [N1] : main D avec téléchargement factice, puis la VRAIE require_empty_output(RUN)."""
    class FakeL0:
        @staticmethod
        def download_detector(det_cfg, cache_dir):
            return {"model.safetensors": Path("/x/model.safetensors")}

    monkeypatch.setattr(TR, "load_l0", lambda cfg: FakeL0)
    run, argv = _main_args(tmp_path, "D")
    assert TR.main(argv) == 0
    assert run.is_dir() and list(run.rglob("*")) == []             # aucun fichier créé dans RUN
    assert _exit_record(tmp_path, "D")["status"] == "COMPLETE"
    assert (tmp_path / "out" / "exits" / "ld1_download.json").is_file()
    TR.require_empty_output(run)                                   # la vraie garde passe
    # T sur le même RUN : la garde passe (build_model est atteint), puis une erreur de device donne FAILED à T
    monkeypatch.setattr(TR, "build_model", lambda a, c, d: (None, None))
    monkeypatch.setattr(TR, "load_scaled", lambda root, pid: {"blocks": [object()]})

    def boom(*a, **k):
        raise TR.DeviceError("RuntimeError: Expected all tensors to be on the same device")

    monkeypatch.setattr(TR, "phase_T", boom)
    _run, argv_t = _main_args(tmp_path, "T")
    with pytest.raises(TR.DeviceError):
        TR.main(argv_t)
    rec = _exit_record(tmp_path, "T")
    assert rec["status"] == "FAILED" and "same device" in rec["error"]
    monkeypatch.setattr(TR, "phase_T", lambda *a, **k: (_ for _ in ()).throw(TR.RepeatedOOM("aucune taille de lot ne tient")))
    with pytest.raises(TR.RepeatedOOM):
        TR.main(argv_t)
    assert _exit_record(tmp_path, "T")["status"] == "INCOMPLETE"


def test_a_non_empty_RUN_makes_T_fail_and_leaves_RUN_untouched(tmp_path, monkeypatch):
    """§15 (3) : reprise interdite, la sortie doit être vide au départ (FAILED)."""
    run, argv = _main_args(tmp_path, "T")
    run.mkdir(parents=True)
    (run / "ld1_step000100.pt").write_bytes(b"x")
    monkeypatch.setattr(TR, "build_model", lambda a, c, d: pytest.fail("build_model ne doit pas être atteint"))
    with pytest.raises(SystemExit):
        TR.main(argv)
    assert _exit_record(tmp_path, "T")["status"] == "FAILED"
    assert sorted(p.name for p in run.iterdir()) == ["ld1_step000100.pt"]


def test_V_reads_S_from_the_manifest_checks_the_unique_checkpoint_and_writes_its_sha(tmp_path):
    """§15 (3) : ld1_step{S:06d}.pt unique, S lu dans le manifeste de T, sha256 dans ld1_report.json (checkpoint_sha256)."""
    import hashlib
    cfg = json.loads(TR.CONFIG.read_text())
    net = torch.nn.Linear(2, 2)
    out = tmp_path / "ld1"
    out.mkdir()
    (out / "ld1_timing_manifest.json").write_text(json.dumps({"steps": 5}))
    torch.save({"model": net.state_dict()}, out / "ld1_step000004.pt")
    args = types.SimpleNamespace(cache_dir=str(tmp_path / "hf"), diagnostic=False)
    with pytest.raises(SystemExit):                                # le pas S=5 manque : le plus récent n'est PAS pris
        TR.phase_V(args, cfg, torch.device("cpu"), out, tmp_path, net, [], None)
    torch.save({"model": net.state_dict()}, out / "ld1_step000005.pt")
    rep = TR.phase_V(args, cfg, torch.device("cpu"), out, tmp_path, net, [], None)
    want = hashlib.sha256((out / "ld1_step000005.pt").read_bytes()).hexdigest()
    assert rep["checkpoint"] == "ld1_step000005.pt" and rep["checkpoint_sha256"] == want
    assert json.loads((out / "ld1_report.json").read_text())["checkpoint_sha256"] == want
