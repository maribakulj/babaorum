"""LD-2 : tests avec torch (docs/END_TO_END_LD2_LOT.txt §4-§7). Pages synthétiques et leurres dans tmp_path ; AUCUNE page réelle.

La plupart des tests utilisent un LineDetr factice (même interface que LD-1, calcul minimal) : ils vérifient la mécanique des
phases, pas la qualité du modèle. Les tests marqués HEAVY construisent le vrai LineDetr (LD2_HEAVY=1)."""
import builtins
import copy
import hashlib
import io
import json
import os
import random
import sys
import types
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import ld2_core as C2  # noqa: E402
import train_ld2 as TD  # noqa: E402
import train_line_detr as TL  # noqa: E402
from babaorum.experimental import line_detr as LD  # noqa: E402
from babaorum.experimental.stage_s_model import Encoder  # noqa: E402
from babaorum.experimental.stage_s_reader import Reader  # noqa: E402

HEAVY = pytest.mark.skipif(os.environ.get("LD2_HEAVY") != "1", reason="heavy: LD2_HEAVY=1 (vrai LineDetr)")
DEVICE = torch.device("cpu")


# ---------- LineDetr factice : même interface, calcul minimal ----------

class FakeDetr(nn.Module):
    """enc_det réel pour les clés d'E2E-5 mais NON exécuté (features = convolution 1x1 sur le canevas moyenné par 4 : stride 4,
    64 canaux), puis un décodeur minuscule. La baseline prédite est une droite horizontale, ce qui rend la CTC calculable.
    down[0] reçoit les features stride 4, comme dans le vrai LineDetr. FakeDetrEnc exécute le vrai encodeur."""

    def __init__(self, n_queries=32, **kw):
        super().__init__()
        self.n_queries, self.k = n_queries, 8
        self.enc_det = Encoder(64)
        self.stem = nn.Conv2d(1, 64, 1)
        self.down = nn.Sequential(nn.Conv2d(64, 16, 1), nn.ReLU())
        self.encoder = nn.Identity()
        self.queries = nn.Embedding(n_queries, 16)
        self.decoder_layers = nn.ModuleList([])
        self.cls = nn.Linear(16, 1)
        self.pts_head = nn.Linear(16, 16)
        self.size_head = nn.Linear(16, 2)
        nn.init.zeros_(self.pts_head.weight)
        with torch.no_grad():
            xs = torch.logit(torch.tensor(np.linspace(0.1, 0.9, 8), dtype=torch.float32))
            ys = torch.zeros(8)
            self.pts_head.bias.copy_(torch.stack([xs, ys], dim=-1).reshape(-1))
            self.cls.bias.fill_(-0.2)                       # quelques requêtes seulement dépassent le seuil 0,6

    def features(self, canvas):
        return self.stem(F.avg_pool2d(canvas, 4))

    def forward(self, canvas, mask=None):
        g = self.down(self.features(canvas)).mean(dim=(2, 3))
        q = g[:, None, :] + self.queries.weight[None]
        pts = torch.sigmoid(self.pts_head(q)).view(q.shape[0], self.n_queries, 8, 2)
        sz = F.softplus(self.size_head(q)) * LD.H_T
        out = {"logit": self.cls(q)[..., 0], "pts": pts, "neg": sz[..., 0], "pos": sz[..., 1]}
        return [dict(out), dict(out)]


class FakeDetrEnc(FakeDetr):
    """Même décodeur factice, mais les features viennent du VRAI enc_det (le gradient CTC lui parvient)."""

    def features(self, canvas):
        return self.enc_det(canvas)


# ---------- pages synthétiques ----------

def make_page(root, pid, seed=0, w=900, h=1100):
    rng = np.random.default_rng(seed)
    (root / "pages").mkdir(parents=True, exist_ok=True)
    (root / "images").mkdir(parents=True, exist_ok=True)
    img = np.full((h, w), 255, dtype=np.uint8)
    lines, blocks, k = [], [], 0
    for bi, (y_start, n) in enumerate(((100, 5), (400, 5))):
        ids = []
        for j in range(n):
            y = y_start + 40 * j
            lid = f"{pid}_l{k:05d}"
            k += 1
            img[y - 18:y + 4, 60:760:3] = 0
            txt = "".join(rng.choice(list("abcdefgh "), 12)).strip() or "abc"
            lines.append({"id": lid, "block_id": f"{pid}_b{bi}", "baseline": [[60.0, float(y)], [760.0, float(y)]],
                          "polygon": [[60.0, float(y - 24)], [760.0, float(y - 24)], [760.0, float(y + 8)], [60.0, float(y + 8)]],
                          "text": txt, "legibility": "readable"})
            ids.append(lid)
        blocks.append({"id": f"{pid}_b{bi}", "category": "texte", "line_ids": ids, "article_id": None,
                       "polygon": [[50.0, y_start - 30.0], [770.0, y_start - 30.0], [770.0, y_start + 40.0 * n],
                                   [50.0, y_start + 40.0 * n]]})
    Image.fromarray(img).save(root / "images" / f"{pid}.png")
    sha = hashlib.sha256((root / "images" / f"{pid}.png").read_bytes()).hexdigest()
    (root / "pages" / f"{pid}.json").write_text(json.dumps({
        "page_id": pid, "image": {"sha256": sha, "height": h, "width": w, "path": "x", "dpi": 150, "color_mode": "L"},
        "lines": lines, "blocks": blocks}))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fake_det(arr):
    h, w = arr.shape[:2]
    return {"width": w, "height": h, "blocks": [
        {"id": "b0", "label": "text", "score": 0.9, "polygon": [[40, 80], [780, 80], [780, 330], [40, 330]]}]}


class Env:
    """Un lancement factice complet : pages (entraînement, repli et leurres mf_005x), checkpoints factices, config corrigée."""

    def __init__(self, tmp_path, monkeypatch, reference_pages=None):
        self.tmp = tmp_path
        self.root = tmp_path / "mf"
        for i, pid in enumerate(("mf_0000", "mf_0001", *C2.eval_page_ids())):
            make_page(self.root, pid, seed=i)
        monkeypatch.setattr(TD, "TRAIN_ALL", ["mf_0000", "mf_0001"])              # alphabet et pages d'entraînement de la fumée
        self.alphabet = TD.training_alphabet(self.root)
        torch.manual_seed(0)
        proto = FakeDetr()
        self.enc = tmp_path / "enc.pt"
        reader = Reader(64 * TD.ROWS, len(self.alphabet), hidden=128)             # Reader d'E2E-5 factice (gel 2, Q1)
        torch.save({"model": {**{f"enc_det.{k}": v for k, v in proto.enc_det.state_dict().items()},
                               **{f"reader.{k}": v for k, v in reader.state_dict().items()}}, "step": 2000}, self.enc)
        self.dec = tmp_path / "ld1_step001871.pt"
        torch.save({"model": proto.state_dict(), "step": 1871}, self.dec)
        ids = reference_pages or [*C2.eval_page_ids(), "mf_0000", "mf_0001"]       # + pages de remplacement de la fumée
        self.ref = tmp_path / "e2e5_lambda_run_report.json"
        self.ref.write_text(json.dumps({"eval": [{"page_id": p, "metrics": {"after": {"recall": 0.5, "precision": 0.4, "n_pred": 5}}}
                                                for p in ids]}))
        cfg = copy.deepcopy(TD.load_all())
        cfg["init"]["encoder"]["sha256_full"] = sha(self.enc)
        cfg["init"]["decoder"]["sha256_full"] = sha(self.dec)
        cfg["reference"]["sha256_prefix"] = sha(self.ref)[:8]
        cfg["ld1"]["batch_candidates"] = [2]
        cfg["ld1"]["checkpoint_every"] = 2
        cfg["robustness"]["log_every_steps"] = 2
        cfg["stage"]["ctc_lines_per_step_max"] = 3
        cfg["model"]["n_queries_witness"] = 64
        self.cfg = cfg
        monkeypatch.setattr(TD, "load_all", lambda path=None: cfg)
        monkeypatch.setattr(TD, "LineDetr", FakeDetr)
        monkeypatch.setattr(TD, "make_det_fn", lambda L0, cache: fake_det)
        self.run = tmp_path / "out" / "ld2"
        self.exits = tmp_path / "out" / "exits"
        self.cal = tmp_path / "out" / "ld2_cal"

    def argv(self, phase, *extra, train_pages=True):
        a = ["--phase", phase, "--root", str(self.root), "--out", str(self.run), "--init-checkpoint", str(self.enc),
             "--decoder-checkpoint", str(self.dec), "--reference-report", str(self.ref), "--cal-out", str(self.cal),
             "--cache-dir", str(self.tmp / "hf")]
        if train_pages and phase in ("T", "TW", "E", "EW"):
            a += ["--pages", "mf_0000", "mf_0001"]
        return a + list(extra)

    def main(self, phase, *extra, **kw):
        return TD.main(self.argv(phase, *extra, **kw))

    def exit_record(self, phase):
        return json.loads((self.exits / f"ld2_run_exit_{phase}.json").read_text())

    def t(self):
        """T (principal) puis TW (témoin), séparés (lot §8 (1))."""
        self.main("T", "--timing-steps", "2", "--max-steps", "4")
        return self.main("TW", "--timing-steps", "2")

    def train(self):
        self.t()
        self.main("E")
        self.main("EW", "--deadline-s", "600")
        self.main("FREEZE")

    def calibrate_and_eval(self, smoke=False):
        if smoke:
            self.main("CAL", "--pages", "mf_0000", "--smoke", "--eval-pages", "mf_0000", "mf_0001")
            return self.main("EVAL", "--smoke", "--eval-pages", "mf_0000", "mf_0001")
        self.main("CAL", "--pages", "mf_0001")
        return self.main("EVAL")

    def manifest(self):
        return json.loads((self.run / "ld2_manifest.json").read_text())


@pytest.fixture
def env(tmp_path, monkeypatch):
    return Env(tmp_path, monkeypatch)


# ---------- manifeste, ordre des phases, gel avant l'évaluation ----------

def test_the_manifest_has_S2_and_S_t_before_any_training_and_freeze_is_written_before_eval(env):
    env.main("T", "--timing-steps", "2", "--max-steps", "4")
    man = env.manifest()
    assert man["S2"] == min(4, int(8400 // man["main"]["s_per_step"])) and man["S_t"] == 0   # T écrit S2 seul
    env.main("TW", "--timing-steps", "2")
    man = env.manifest()
    assert man["S_t"] == min(man["S2"], int(4200 // man["witness"]["s_per_step"]))
    assert man["written_before_E"] is True and "freeze" not in man
    assert not list(env.run.glob("ld2_*_step*.pt"))                  # aucun entraînement n'a eu lieu
    assert man["main"]["batch_size"] == 2 and man["witness"]["status"] == "TIMED" and man["witness"]["n_queries"] == 64
    with pytest.raises(SystemExit, match="FREEZE"):
        env.main("EVAL", train_pages=False)                           # EVAL refuse sans gel (pages jamais ouvertes)
    env.main("E")
    env.main("EW", "--deadline-s", "600")
    assert not (env.run / "ld2_report.json").exists()
    env.main("FREEZE")
    man = env.manifest()
    fz = man["freeze"]
    assert fz["threshold"] == 0.6 and fz["checkpoint"] == f"ld2_main_step{man['S2']:06d}.pt"
    assert fz["checkpoint_sha256"] == sha(env.run / fz["checkpoint"]) and fz["written_before_EVAL"] is True
    assert fz["nms"]["overlap_fraction_of_min_length"] == 0.5 and fz["nms"]["max_median_distance_in_h"] == 0.5
    assert not (env.run / "ld2_report.json").exists()                # le rapport n'existe qu'après EVAL


def test_the_full_run_writes_the_report_with_the_sha_and_the_reserves_first(env):
    env.train()
    env.calibrate_and_eval()
    rep = json.loads((env.run / "ld2_report.json").read_text())
    assert list(rep)[:3] == ["schema", "checkpoint_sha256", "reserves"]
    assert rep["checkpoint_sha256"] == env.manifest()["freeze"]["checkpoint_sha256"]
    assert rep["reserves"] == C2.RESERVES and rep["cer_scope"] == C2.CER_SCOPE and rep["status"] == "COMPLETE"
    assert rep["pages"] == C2.eval_page_ids() and rep["Z-PP-blocs"]["pages"] == 10
    assert rep["criterion"]["verdict"] in ("CRITERE_REMPLI", "CRITERE_NON_REMPLI")
    assert rep["steps"]["S2"] == env.manifest()["S2"] and rep["steps"]["S_t"] == env.manifest()["S_t"]
    cal = json.loads((env.cal / "ld2_cal.json").read_text())
    assert cal["page"] == "mf_0001" and cal["estimate_s"] >= cal["load_seconds"]
    assert not (env.run / "ld2_cal.json").exists()                   # l'étalonnage reste hors de RUN
    with pytest.raises(SystemExit, match="existe déjà"):
        env.main("EVAL", train_pages=False)                           # pas de seconde soumission
    assert (env.run / "ld2_eval_started.json").is_file()              # gel 8 (B3) : marqueur créé avant la première page


def test_steps_are_written_in_the_manifest_with_the_lot_formulas_for_a_given_timing(env, monkeypatch):
    seq = iter([{"batch_size": 8, "batch_tries": [], "s_per_step": 2.5, "timing_steps": 200, "params_by_block": {}, "n_queries": 32, "key_sources": {"k": "v"}},
                {"batch_size": 4, "batch_tries": [], "s_per_step": 1.0, "timing_steps": 200, "params_by_block": {}, "n_queries": 512, "key_sources": {"k": "v"}}])
    monkeypatch.setattr(TD, "time_kind", lambda *a, **k: dict(next(seq)))
    env.main("T")                                                     # 200 pas, sans --max-steps
    env.main("TW")
    man = env.manifest()
    assert man["S2"] == 3360 and man["S_t"] == 3360                   # floor(8400/2.5) ; min(3360, floor(4200/1.0)=4200)


# ---------- l'échec du témoin ne touche pas le critère principal ----------

def test_a_witness_failure_is_reported_as_is_with_exit_code_zero_and_does_not_affect_the_main_criterion(env, monkeypatch):
    env.t()
    env.main("E")
    real = TD.train_kind

    def boom(kind, *a, **k):
        if kind == "witness":
            raise RuntimeError("CUDA error: fumée du témoin")
        return real(kind, *a, **k)

    monkeypatch.setattr(TD, "train_kind", boom)
    assert env.main("EW", "--deadline-s", "600") == 0                 # code 0 : le pipeline continue
    st = json.loads((env.run / "ld2_witness_status.json").read_text())
    assert st["status"] == "FAILED" and "fumée du témoin" in st["error"]
    monkeypatch.setattr(TD, "train_kind", real)
    env.main("FREEZE")
    env.calibrate_and_eval()
    rep = json.loads((env.run / "ld2_report.json").read_text())
    assert rep["witness"]["status"] == "FAILED" and "result" not in rep["witness"]
    rows = [dict(r["Z-PP-blocs"], page=r["page"]) for r in rep["per_page"]]
    again = C2.criterion(rows, C2.load_reference(env.ref, C2.eval_page_ids(), env.cfg), C2.eval_page_ids(), env.cfg)
    assert again == rep["criterion"]                                  # le critère ne dépend que des lignes du principal


def test_a_witness_timeout_is_not_reached_and_the_watchdog_writes_the_status(env):
    env.t()
    assert env.main("EW", "--deadline-s", "0") == 0
    st = json.loads((env.run / "ld2_witness_status.json").read_text())
    assert st["status"] == "NOT_REACHED" and st["steps_done"] == 0 and not list(env.run.glob("ld2_witness_step*.pt"))
    fired = []
    timer = TD.arm_watchdog(0.01, lambda: fired.append(1))
    timer.join(2.0)
    assert fired == [1]


def test_the_witness_timing_failure_leaves_the_main_run_possible(env, monkeypatch):
    real = TD.time_kind

    def maybe(kind, *a, **k):
        if kind == "witness":
            raise TL.RepeatedOOM("aucune taille de lot ne tient")
        return real(kind, *a, **k)

    monkeypatch.setattr(TD, "time_kind", maybe)
    env.main("T", "--timing-steps", "2", "--max-steps", "4")
    assert env.main("TW") == 0
    man = env.manifest()
    assert man["S_t"] == 0 and man["witness"]["status"] == "TIMING_FAILED" and man["S2"] == 4
    env.main("E")
    assert env.main("EW") == 0
    assert json.loads((env.run / "ld2_witness_status.json").read_text())["status"] == "NOT_RUN"
    env.main("FREEZE")
    env.calibrate_and_eval()
    assert json.loads((env.run / "ld2_report.json").read_text())["witness"]["status"] == "NOT_RUN"


# ---------- refus : shas, sortie vide, reprise, S2 non atteint ----------

def test_wrong_shas_are_refused_for_the_encoder_and_for_the_ld1_decoder(env):
    other = env.tmp / "other.pt"
    torch.save({"model": {}, "step": 2000}, other)
    args = types.SimpleNamespace(init_checkpoint=str(other), decoder_checkpoint=str(env.dec))
    model = TD.new_model("main", env.cfg, env.alphabet, DEVICE)
    with pytest.raises(SystemExit, match="encodeur"):
        TD.load_inits(model, "main", env.cfg, args)
    args = types.SimpleNamespace(init_checkpoint=str(env.enc), decoder_checkpoint=str(other))
    with pytest.raises(SystemExit, match="décodeur"):
        TD.load_inits(model, "main", env.cfg, args)
    cfg2 = copy.deepcopy(env.cfg)
    cfg2["init"]["decoder"]["step"] = 1000
    torch.save({"model": {}, "step": 2000}, other)
    with pytest.raises(SystemExit):                                   # le pas du checkpoint de LD-1 doit être 1871
        TD.load_inits(model, "main", cfg2, types.SimpleNamespace(init_checkpoint=str(env.enc), decoder_checkpoint=str(env.dec)))


def test_the_real_ld1_sha_and_step_are_the_pre_registered_ones():
    cfg = C2.load_config()
    assert cfg["init"]["decoder"]["sha256_full"] == "a1c5fbab288c99e4911789f926d2d568815fd2a165310da5020ac7c140f8197b"
    assert cfg["init"]["decoder"]["file"] == "ld1_step001871.pt" and cfg["init"]["decoder"]["step"] == 1871
    assert cfg["init"]["encoder"]["sha256_full"].startswith("ef5e8268") and cfg["init"]["encoder"]["step"] == 2000


def test_the_decoder_comes_from_ld1_and_the_encoder_from_e2e5_and_the_witness_decoder_is_new(env):
    proto = FakeDetr()
    ck1 = {k: torch.randn_like(v) if v.dtype.is_floating_point else v for k, v in proto.state_dict().items()}
    torch.save({"model": ck1, "step": 1871}, env.dec)
    env.cfg["init"]["decoder"]["sha256_full"] = sha(env.dec)
    enc_ck = torch.load(env.enc)
    enc_ck["model"] = {k: v + 1.0 if v.dtype.is_floating_point else v for k, v in enc_ck["model"].items()}
    torch.save(enc_ck, env.enc)
    env.cfg["init"]["encoder"]["sha256_full"] = sha(env.enc)
    args = types.SimpleNamespace(init_checkpoint=str(env.enc), decoder_checkpoint=str(env.dec))
    m = TD.new_model("main", env.cfg, env.alphabet, DEVICE)
    TD.load_inits(m, "main", env.cfg, args)
    for k, v in m.reader.state_dict().items():
        assert torch.equal(v, enc_ck["model"]["reader." + k]), k                 # Reader d'E2E-5 (principal)
    for k, v in m.detr.state_dict().items():
        want = enc_ck["model"][k] if k.startswith("enc_det.") else ck1[k]
        assert torch.equal(v, want), k
    w = TD.new_model("witness", env.cfg, env.alphabet, DEVICE)
    TD.load_inits(w, "witness", env.cfg, args)
    assert torch.equal(w.detr.enc_det.state_dict()["net.0.weight"], enc_ck["model"]["enc_det.net.0.weight"])
    assert not torch.equal(w.detr.cls.weight, ck1["cls.weight"]) and w.detr.n_queries == 64
    assert torch.equal(w.reader.out.weight, enc_ck["model"]["reader.out.weight"])        # même Reader pour le témoin


def test_a_missing_or_misshaped_reader_key_in_the_e2e5_checkpoint_is_refused(env):
    ck = torch.load(env.enc)
    miss = {k: v for k, v in ck["model"].items() if k != "reader.out.bias"}
    torch.save({"model": miss, "step": 2000}, env.enc)
    env.cfg["init"]["encoder"]["sha256_full"] = sha(env.enc)
    args = types.SimpleNamespace(init_checkpoint=str(env.enc), decoder_checkpoint=str(env.dec))
    model = TD.new_model("main", env.cfg, env.alphabet, DEVICE)
    with pytest.raises(SystemExit, match="clés reader"):
        TD.load_inits(model, "main", env.cfg, args)
    bad = dict(ck["model"], **{"reader.out.bias": torch.zeros(len(env.alphabet) + 1)})
    torch.save({"model": bad, "step": 2000}, env.enc)
    env.cfg["init"]["encoder"]["sha256_full"] = sha(env.enc)
    with pytest.raises(SystemExit, match="formes reader"):
        TD.load_inits(model, "main", env.cfg, args)


def test_the_manifest_records_which_keys_came_from_which_checkpoint(env):
    env.main("T", "--timing-steps", "2", "--max-steps", "4")
    assert env.manifest()["init"]["witness_key_sources"] is None            # rempli par TW seulement
    env.main("TW", "--timing-steps", "2")
    ks = env.manifest()["init"]["key_sources"]
    assert ks["reader.*"].startswith("E2E-5 ") and ks["enc_det.*"] == ks["reader.*"] and ks["detr.* sauf enc_det"].startswith("LD-1 ")
    assert env.manifest()["init"]["witness_key_sources"]["detr.* sauf enc_det"] == "neuf (graine)"


def test_T_requires_an_empty_output_and_E_refuses_a_resume(env):
    env.run.mkdir(parents=True)
    (env.run / "x.pt").write_bytes(b"x")
    with pytest.raises(SystemExit, match="non vide"):
        env.main("T", "--timing-steps", "2", "--max-steps", "4")
    assert env.exit_record("T")["status"] == "FAILED" and sorted(p.name for p in env.run.iterdir()) == ["x.pt"]
    (env.run / "x.pt").unlink()
    env.main("T", "--timing-steps", "2", "--max-steps", "4")
    (env.run / "ld2_main_step000002.pt").write_bytes(b"old")
    with pytest.raises(SystemExit, match="reprise interdite"):
        env.main("E")


def test_phase_D_writes_nothing_in_RUN_and_T_then_passes_the_guard(env, monkeypatch):
    class FakeL0:
        @staticmethod
        def download_detector(det, cache):
            return {"model.safetensors": Path("/x/model.safetensors")}

    monkeypatch.setattr(TL, "load_l0", lambda cfg: FakeL0)
    assert env.main("D") == 0
    assert env.run.is_dir() and list(env.run.rglob("*")) == []
    assert (env.exits / "ld2_download.json").is_file()
    env.main("T", "--timing-steps", "2", "--max-steps", "4")         # la garde « sortie vide » passe


def test_FREEZE_refuses_when_the_S2_checkpoint_is_missing_and_E_failure_stops_before_evaluation(env):
    env.main("T", "--timing-steps", "2", "--max-steps", "4")
    with pytest.raises(SystemExit, match="S2 non atteint"):
        env.main("FREEZE")
    assert "freeze" not in env.manifest() and env.exit_record("FREEZE")["status"] == "FAILED"
    with pytest.raises(SystemExit, match="FREEZE"):
        env.main("CAL", "--pages", "mf_0001")


def test_a_gated_file_that_differs_from_the_frozen_sha_is_refused(env, monkeypatch):
    env.main("T", "--timing-steps", "2", "--max-steps", "4")
    env.main("E")
    env.cfg["freeze"]["frozen_files_sha256_prefix"]["src/babaorum/experimental/line_detr.py"] = "00000000"
    with pytest.raises(SystemExit, match="gelée"):
        env.main("FREEZE")


def test_a_tampered_main_checkpoint_after_freeze_is_refused_by_eval(env):
    env.train()
    ck = env.run / env.manifest()["freeze"]["checkpoint"]
    ck.write_bytes(ck.read_bytes() + b"x")
    with pytest.raises(SystemExit, match="checkpoint gelé"):
        env.main("CAL", "--pages", "mf_0001")


def test_the_reference_report_with_another_sha_is_refused_before_any_page_is_opened(env, monkeypatch):
    env.train()
    env.ref.write_text(json.dumps({"eval": []}))
    spy = []
    real = TD.load_page_text
    monkeypatch.setattr(TD, "load_page_text", lambda root, pid: spy.append(pid) or real(root, pid))
    with pytest.raises(SystemExit, match="rapport de référence"):
        env.main("EVAL", train_pages=False)
    assert spy == []


# ---------- seeds, appariement, perte du témoin ----------

def test_seeds_are_set_before_the_modules_are_built(env, monkeypatch):
    seen = {}
    seed = env.cfg["seed"]

    class Spy(FakeDetr):
        def __init__(self, *a, **k):
            seen.setdefault("torch", torch.initial_seed())
            seen.setdefault("py", random.random())
            seen.setdefault("np", float(np.random.rand()))
            super().__init__(*a, **k)

    monkeypatch.setattr(TD, "LineDetr", Spy)
    torch.manual_seed(1)
    random.seed(2)
    np.random.seed(3)
    TD.new_model("main", env.cfg, env.alphabet, DEVICE)
    assert seen["torch"] == seed and seen["py"] == random.Random(seed).random()
    assert seen["np"] == float(np.random.RandomState(seed).rand())
    assert seed == 20261023


def test_two_builds_with_the_same_seed_give_identical_parameters(env):
    a = TD.new_model("main", env.cfg, ["<blank>", "a", "b"], DEVICE)
    b = TD.new_model("main", env.cfg, ["<blank>", "a", "b"], DEVICE)
    for (n, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert torch.equal(pa, pb), n


def test_vectorised_hungarian_and_witness_matching_equal_the_ld1_functions():
    rng = np.random.default_rng(0)
    for _ in range(30):
        n = int(rng.integers(1, 9))
        c = rng.random((n, n + int(rng.integers(0, 25))))
        assert TD.hungarian_vec(c) == LD.hungarian(c)
    for _ in range(10):
        N, Tn = int(rng.integers(8, 40)), int(rng.integers(1, 8))
        logit, h = rng.normal(size=N), rng.random(Tn) * 30 + 10
        pred, gt = rng.random((N, 8, 2)) * [1536, 512], rng.random((Tn, 8, 2)) * [1536, 512]
        assert TD.match_wh(logit, pred, gt, h, (LD.CANVAS_W, LD.CANVAS_H)) == LD.match_zone(logit, pred, gt, h)


def test_layer_loss_wh_equals_the_ld1_layer_loss_at_the_ld1_canvas():
    rng = np.random.default_rng(1)
    out_l = {"logit": torch.randn(1, 32), "pts": torch.rand(1, 32, 8, 2), "neg": torch.rand(1, 32) * 10,
             "pos": torch.rand(1, 32) * 10}
    s = {"pts": rng.random((3, 8, 2)) * [1536, 512], "neg": np.array([5.0, 6.0, 7.0]), "pos": np.array([7.0, 8.0, 9.0])}
    m = [(0, 3), (1, 5), (2, 9)]
    for a, b in zip(TL.layer_loss(out_l, 0, s, m), TD.layer_loss_wh(out_l, 0, s, m, (LD.CANVAS_W, LD.CANVAS_H))):
        assert float(a) == float(b)


def test_layer_loss_wh_normalises_by_the_tile_not_by_the_ld1_canvas():
    out_l = {"logit": torch.zeros(1, 4), "pts": torch.full((1, 4, 8, 2), 0.5), "neg": torch.zeros(1, 4), "pos": torch.zeros(1, 4)}
    s = {"pts": np.full((1, 8, 2), 384.0), "neg": np.zeros(1), "pos": np.zeros(1)}
    _l, _c, l_pts, _h = TD.layer_loss_wh(out_l, 0, s, [(0, 0)], (768, 768))
    assert l_pts == 0.0                                                # 384 / 768 = 0,5 : points exacts


# ---------- GradScale λ et géométrie détachée pour la CTC ----------

def _ctc_grads(model, page_pid, env, scale, monkeypatch):
    monkeypatch.setattr(TD, "LAMBDA", scale)
    page = TD.load_page_text(env.root, page_pid)
    task = TD.Task("main", env.cfg, [page], ["<blank>", *"abcdefgh "])
    blk = page["blocks"][0]
    s = TD.zone_sample_ctc(page, blk, blk["box"])
    canv = torch.as_tensor(s["canvas"])[None, None]
    msk = torch.as_tensor(s["mask"])[None, None]
    model.zero_grad(set_to_none=True)
    outs, f4 = model(canv, msk)
    matches = [[(g, g) for g in range(len(s["pts"]))]]
    rng = np.random.default_rng(0)
    l_ctc, info = task.ctc_term(model, [s], matches, outs[-1], f4, 1.0, rng, DEVICE)
    assert info["ctc_included"] >= 1
    l_ctc.backward()
    enc = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in model.detr.enc_det.parameters()])
    return float(enc.norm()), model


def test_the_ctc_gradient_to_enc_det_is_multiplied_by_one_sixth_and_the_geometry_is_detached(env, monkeypatch):
    monkeypatch.setattr(TD, "LineDetr", FakeDetrEnc)
    model = TD.new_model("main", env.cfg, ["<blank>", *"abcdefgh "], DEVICE)
    n1, _ = _ctc_grads(model, "mf_0000", env, 1.0, monkeypatch)
    n6, model = _ctc_grads(model, "mf_0000", env, 1.0 / 6.0, monkeypatch)
    assert n1 > 0 and abs(n6 / n1 - 1.0 / 6.0) <= 1e-4
    for name in ("cls", "pts_head", "size_head", "queries", "down"):
        for p in getattr(model.detr, name).parameters():
            assert p.grad is None or float(p.grad.abs().max()) == 0.0, name   # aucune CTC vers la géométrie
    assert any(p.grad is not None and float(p.grad.abs().max()) > 0 for p in model.reader.parameters())


def test_lambda_constant_is_one_sixth():
    assert TD.LAMBDA == 1.0 / 6.0 and C2.load_config()["lambda"]["value"] == "1/6"


def test_a_full_loss_step_updates_the_reader_and_the_geometry(env):
    page = TD.load_page_text(env.root, "mf_0000")
    task = TD.Task("main", env.cfg, [page], ["<blank>", *"abcdefgh "])
    model = TD.new_model("main", env.cfg, ["<blank>", *"abcdefgh "], DEVICE)
    for seed in range(20):                       # gel 9 : depuis B2, un recadrage perturbé peut couper les lignes ; premier tirage utile
        rng = np.random.default_rng(seed)
        loss, info = task.loss(model, task.sample(rng, 2), 1.0, rng, DEVICE)
        if info["ctc_included"] >= 1:
            break
    else:
        pytest.fail("aucun tirage (graines 0..19) n'a de ligne CTC entière dans son recadrage")
    loss.backward()
    assert np.isfinite(float(loss)) and info["ctc_included"] >= 1 and "ctc_cut_by_crop" in info
    assert any(float(p.grad.abs().max()) > 0 for p in model.detr.cls.parameters())
    assert any(float(p.grad.abs().max()) > 0 for p in model.reader.parameters())


def test_the_witness_samples_tiles_with_N_512_queries_and_matches_with_the_tile_canvas(env):
    page = TD.load_page_text(env.root, "mf_0000")
    task = TD.Task("witness", env.cfg, [page], ["<blank>", *"abcdefgh "])
    rng = np.random.default_rng(0)
    s = task.sample(rng, 2)
    assert s[0]["canvas"].shape == (768, 768) and s[0]["meta"].fx == 1.0
    model = TD.new_model("witness", env.cfg, ["<blank>", *"abcdefgh "], DEVICE)
    loss, info = task.loss(model, s, 1.0, rng, DEVICE)
    assert np.isfinite(float(loss))


# ---------- évaluation : n_pred = 0, tuiles, exposition ----------

def test_tile_origins_cover_the_page_and_end_on_the_border():
    assert TD.tile_origins(900, 768, 384) == [0, 132]
    assert TD.tile_origins(768, 768, 384) == [0]
    assert TD.tile_origins(2680, 768, 384)[-1] == 2680 - 768 and TD.tile_origins(500, 768, 384) == [0]


def test_score_lines_gives_zero_precision_and_recall_without_prediction_and_counts_cer_on_matches(env):
    page = TD.load_page_text(env.root, "mf_0000")
    assert TD.score_lines(page, [], [])["precision"] == 0.0 and TD.score_lines(page, [], [])["n_pred"] == 0
    kept = [{"pts": np.array(g["baseline"]), "zone": 0, "query": i, "score": 0.9, "h": float(g["neg"] + g["pos"])}
            for i, g in enumerate(page["gt_lines"])]
    raws = [(0, None, {"hyp": {i: g["text"] for i, g in enumerate(page["gt_lines"])}})]
    res = TD.score_lines(page, kept, raws)
    assert res["recall"] == 1.0 and res["precision"] == 1.0 and res["cer"]["edits"] == 0 and res["cer"]["lines"] == 10
    raws[0][2]["hyp"][0] = None
    assert TD.score_lines(page, kept, raws)["cer"]["skipped"] == 1


def _spy_everything(monkeypatch, touched):
    """open / io.open / Path.read_text / read_bytes / Image.open / listdir / scandir / iterdir / glob / rglob / walk / stat."""
    real_open, real_io_open = builtins.open, io.open
    real_text, real_bytes, real_image = Path.read_text, Path.read_bytes, Image.open
    real_stat, real_exists, real_isfile = Path.stat, Path.exists, Path.is_file

    def rec(p):
        touched.append(os.fspath(p) if isinstance(p, (str, os.PathLike)) else "stream")

    monkeypatch.setattr(builtins, "open", lambda f, *a, **k: (rec(f), real_open(f, *a, **k))[1])
    monkeypatch.setattr(io, "open", lambda f, *a, **k: (rec(f), real_io_open(f, *a, **k))[1])
    monkeypatch.setattr(Path, "read_text", lambda self, *a, **k: (rec(self), real_text(self, *a, **k))[1])
    monkeypatch.setattr(Path, "read_bytes", lambda self, *a, **k: (rec(self), real_bytes(self, *a, **k))[1])
    monkeypatch.setattr(Image, "open", lambda fp, *a, **k: (rec(fp), real_image(fp, *a, **k))[1])
    monkeypatch.setattr(Path, "stat", lambda self, *a, **k: (rec(self), real_stat(self, *a, **k))[1])
    monkeypatch.setattr(Path, "exists", lambda self, *a, **k: (rec(self), real_exists(self, *a, **k))[1])
    monkeypatch.setattr(Path, "is_file", lambda self, *a, **k: (rec(self), real_isfile(self, *a, **k))[1])
    for name in ("listdir", "scandir"):
        real = getattr(os, name)
        monkeypatch.setattr(os, name, lambda *a, _r=real, **k: (rec("LIST:" + os.fspath(a[0] if a else ".")), _r(*a, **k))[1])
    for name in ("iterdir", "glob", "rglob"):
        real = getattr(Path, name)
        monkeypatch.setattr(Path, name, lambda self, *a, _r=real, **k: (rec("LIST:" + os.fspath(self)), _r(self, *a, **k))[1])
    real_walk = os.walk
    monkeypatch.setattr(os, "walk", lambda top, *a, **k: (rec("LIST:" + os.fspath(top)), real_walk(top, *a, **k))[1])


def test_no_evaluation_page_is_touched_by_any_phase_before_EVAL_and_never_in_the_smoke_variant(env, monkeypatch):
    touched = []
    _spy_everything(monkeypatch, touched)
    env.train()
    env.main("CAL", "--pages", "mf_0001")
    assert touched and not any("mf_005" in t for t in touched), [t for t in touched if "mf_005" in t][:3]
    assert not [t for t in touched if t.startswith("LIST:" + str(env.root))]       # aucun listage du dossier des pages
    touched.clear()
    env.calibrate_and_eval(smoke=True)                                # fumée : pages de remplacement, jamais mf_005x
    assert not any("mf_005" in t for t in touched)
    rep = json.loads((env.run / "ld2_report.json").read_text())
    assert rep["smoke"] is True and rep["reference"]["decoy"] is True and rep["pages"] == ["mf_0000", "mf_0001"]


def test_EVAL_opens_exactly_the_ten_evaluation_pages_after_the_freeze_and_nothing_else_named_mf_005x(env, monkeypatch):
    env.train()
    env.main("CAL", "--pages", "mf_0001")
    touched = []
    _spy_everything(monkeypatch, touched)
    env.main("EVAL", train_pages=False)
    names = {Path(t).stem for t in touched if "mf_005" in t}
    assert names == set(C2.eval_page_ids())
    assert all(Path(t).parent.name in ("pages", "images") for t in touched if "mf_005" in t)
    assert not [t for t in touched if t.startswith("LIST:" + str(env.root))]       # aucun listage du dossier des pages


def test_smoke_eval_refuses_the_reserved_pages_and_the_real_eval_refuses_eval_pages(env):
    env.train()
    with pytest.raises(SystemExit):
        env.main("EVAL", "--smoke", "--eval-pages", "mf_0050", train_pages=False)
    with pytest.raises(SystemExit, match="fumée"):
        env.main("EVAL", "--eval-pages", "mf_0001", train_pages=False)
    with pytest.raises(SystemExit, match="remplacement"):
        env.main("EVAL", "--smoke", train_pages=False)
    with pytest.raises(SystemExit, match="réservées"):                  # l'étalonnage ne peut pas viser une page d'évaluation
        env.main("CAL", "--pages", "mf_0050")
    assert not (env.run / "ld2_report.json").exists()


def test_training_phases_refuse_the_reserved_pages(env):
    env.main("T", "--timing-steps", "2", "--max-steps", "4")
    with pytest.raises(SystemExit, match="réservées"):
        TD.main(env.argv("E", "--pages", "mf_0050", train_pages=False))


def test_the_script_names_the_reserved_pages_only_in_comments_and_docstrings():
    for rel in ("scripts/train_ld2.py", "scripts/ld2_core.py"):
        for line in (ROOT / rel).read_text().splitlines():
            if "mf_005" in line:
                assert "mf_0050-0059" in line, line


# ---------- vrai LineDetr (HEAVY) ----------

@HEAVY
def test_the_tap_captures_the_stride_4_features_of_the_real_encoder_and_lambda_holds_on_the_real_model(env, monkeypatch):
    monkeypatch.setattr(TD, "LineDetr", LD.LineDetr)
    model = TD.new_model("main", env.cfg, ["<blank>", *"abcdefgh "], DEVICE)
    c = torch.rand(1, 1, LD.CANVAS_H, LD.CANVAS_W)
    m = torch.ones_like(c)
    with torch.no_grad():
        _outs, f4 = model(c, m)
        ref, _ = LD.masked_encoder(model.detr.enc_det, c, m)
    assert f4.shape == (1, 64, 128, 384) and torch.equal(f4, ref)
    n1, _ = _ctc_grads(model, "mf_0000", env, 1.0, monkeypatch)
    n6, model = _ctc_grads(model, "mf_0000", env, 1.0 / 6.0, monkeypatch)
    assert abs(n6 / n1 - 1.0 / 6.0) <= 1e-4
    for name in ("down", "encoder", "decoder_layers", "cls", "pts_head", "size_head", "queries"):
        for p in getattr(model.detr, name).parameters():
            assert p.grad is None or float(p.grad.abs().max()) == 0.0, name
