"""LD-1 carnets et montages, SANS torch (lançable en local). docs/END_TO_END_LD1_LOT.txt §4, §12 (1)-(4), (11), (15)."""
import ast
import builtins
import hashlib
import io
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "stage_s_notebooks"))
import build_kaggle_ld1 as B  # noqa: E402

LD1 = ROOT / "scripts" / "stage_s_notebooks" / "ld1"


def _code(name):
    nb = json.loads((LD1 / name).read_text())
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")


# ---- égalité des carnets avec build() (§12 (11)) ----

def test_the_complete_carnet_on_disk_equals_build():
    assert json.loads((LD1 / "kaggle_ld1.ipynb").read_text()) == B.build_notebook()


def test_the_smoke_carnet_on_disk_equals_build():
    assert json.loads((LD1 / "kaggle_ld1_smoke.ipynb").read_text()) == B.build_smoke_notebook()


def test_the_smoke_kernel_folder_equals_build():
    nb = json.loads((LD1 / "smoke_kernel" / "notebook.ipynb").read_text())
    assert nb == B.build_smoke_notebook()
    meta = json.loads((LD1 / "smoke_kernel" / "kernel-metadata.json").read_text())
    assert meta == B.kernel_metadata("mabaku/babaorum-ld1-smoke", "babaorum-ld1-smoke")


# ---- montages : exactement un candidat, aucun listage, leurre au niveau 3 (§12 (11)) ----

def _tree(tmp_path):
    fmt_a = tmp_path / "input" / "babaorum-e2e3-run3-fp-classes" / "e2e3" / "mf"
    fmt_b = tmp_path / "input" / "notebooks" / "mabaku" / "babaorum-e2e3-run3-fp-classes" / "e2e3" / "mf"
    decoy_l2 = fmt_a / "pages" / "mf_0050.json"
    decoy_l3 = tmp_path / "input" / "x" / "y" / "mf_0050.json"          # leurre au niveau 3
    for base in (fmt_a, fmt_b):
        (base / "pages").mkdir(parents=True)
    decoy_l2.write_text("{}")
    decoy_l3.parent.mkdir(parents=True)
    decoy_l3.write_text("{}")
    (fmt_b / "pages" / "mf_0000.json").write_text("{}")
    return fmt_a, fmt_b, decoy_l2, decoy_l3


def test_resolve_one_takes_the_real_page_root_in_both_formats_and_never_lists(tmp_path, monkeypatch):
    fmt_a, fmt_b, decoy_l2, decoy_l3 = _tree(tmp_path)
    touched = []
    real_open = io.open
    monkeypatch.setattr(os, "listdir", lambda *a, **k: touched.append("listdir") or [])
    monkeypatch.setattr(os, "scandir", lambda *a, **k: touched.append("scandir") or iter(()))
    for name in ("glob", "iterdir", "rglob"):
        monkeypatch.setattr(Path, name, lambda *a, _n=name, **k: touched.append(_n) or iter(()))
    monkeypatch.setattr(os, "walk", lambda *a, **k: touched.append("walk") or iter(()))
    reads = []

    def spy(file, *a, **k):
        reads.append(os.fspath(file))
        return real_open(file, *a, **k)

    monkeypatch.setattr(io, "open", spy)
    monkeypatch.setattr(builtins, "open", spy)
    from PIL import Image
    real_read_text, real_read_bytes, real_image_open = Path.read_text, Path.read_bytes, Image.open

    def spy_text(self, *a, **k):
        reads.append(os.fspath(self))
        return real_read_text(self, *a, **k)

    def spy_bytes(self, *a, **k):
        reads.append(os.fspath(self))
        return real_read_bytes(self, *a, **k)

    def spy_image(fp, *a, **k):
        reads.append(os.fspath(fp) if isinstance(fp, (str, os.PathLike)) else "stream")
        return real_image_open(fp, *a, **k)

    monkeypatch.setattr(Path, "read_text", spy_text)
    monkeypatch.setattr(Path, "read_bytes", spy_bytes)
    monkeypatch.setattr(Image, "open", spy_image)
    hit = B.resolve_one([str(fmt_a), str(fmt_b)], "pages/mf_0000.json")
    monkeypatch.undo()
    assert hit == fmt_b
    assert touched == []
    assert not any("mf_0050" in r for r in reads)
    assert str(decoy_l2) not in str(hit) and str(decoy_l3) not in str(hit)


def test_resolve_one_refuses_zero_or_two_candidates(tmp_path):
    fmt_a, fmt_b, _, _ = _tree(tmp_path)
    with pytest.raises(RuntimeError):
        B.resolve_one([str(tmp_path / "absent")], "pages/mf_0000.json")
    (fmt_a / "pages" / "mf_0000.json").write_text("{}")
    with pytest.raises(RuntimeError):
        B.resolve_one([str(fmt_a), str(fmt_b)], "pages/mf_0000.json")


def test_resolve_file_takes_exactly_one_checkpoint_file(tmp_path):
    a = tmp_path / "a" / "logs" / "checkpoint_last.pt"
    b = tmp_path / "b" / "logs" / "checkpoint_last.pt"
    a.parent.mkdir(parents=True)
    a.write_bytes(b"x")
    assert B.resolve_file([str(a), str(b)]) == a
    b.parent.mkdir(parents=True)
    b.write_bytes(b"y")
    with pytest.raises(RuntimeError):
        B.resolve_file([str(a), str(b)])


# ---- étapes argv, sortie propre, contrôle CUDA, plafonds lus dans la config (§12 (2), (3), (4), (12)) ----

def test_the_complete_carnet_runs_D_T_E_V_as_argv_steps_not_fn():
    code = _code("kaggle_ld1.ipynb")
    for ph in ("D", "T", "E", "V"):
        assert f'"name": "{ph}"' in code or f'"name": ph' in code
    tree = ast.parse(code)
    argv_steps = [n for n in ast.walk(tree) if isinstance(n, ast.Dict) and any(
        isinstance(k, ast.Constant) and k.value == "argv" for k in n.keys)]
    assert len(argv_steps) >= 1
    assert "s_controls" in code and "torch.cuda.is_available()" in code


def test_the_run_exit_is_written_in_a_finally_and_named_ld1():
    tree = ast.parse(_code("kaggle_ld1.ipynb"))
    finals = [n for n in ast.walk(tree) if isinstance(n, ast.Try) and n.finalbody]
    text = " ".join(ast.unparse(f) for t in finals for f in t.finalbody)
    assert "ld1_run_exit.json" in text


def test_the_pytest_is_an_argv_step_in_the_smoke_carnet():
    code = _code("kaggle_ld1_smoke.ipynb")
    assert '"pytest_heavy"' in code and "-m pytest" in code
    assert "CUDA" not in code or "cuda" not in code.split("STEPS")[0]


def test_timeouts_come_from_the_config_and_match_section_12():
    t = json.loads((ROOT / "config" / "ld1_line_detr.json").read_text())["robustness"]["timeouts_s"]
    assert (t["T"], t["E"], t["V"]) == (600, 4500, 900)
    assert B._timeouts()["T"] == 600 and B._timeouts()["E"] == 4500 and B._timeouts()["V"] == 900


def test_the_checkpoint_and_pages_are_the_run3_and_e2e5_mounts_with_the_sha_fixed():
    code = _code("kaggle_ld1.ipynb")
    assert "babaorum-e2e3-run3-fp-classes" in code
    assert "babaorum-e2e5-lambda/logs/checkpoint_last.pt" in code
    assert "ef5e826803c347f73e49c83c0f21f61399f9ea06ee985905bd17644dff3844fa" in code


# ---- kit : fichiers lus par data2_common et par le détecteur (§12 (4), (5)) ----

def test_the_kit_lists_data2_common_the_ld0_script_and_the_vendor_package():
    files = set(json.loads((LD1 / "kit_files.json").read_text()))
    assert "scripts/data2_common.py" in files and "scripts/ld0_zone_coverage.py" in files
    assert "scripts/ld0_vendor/axel/layout/chains.py" in files
    for rel in files:
        assert (ROOT / rel).is_file(), rel


def test_kaggle_install_is_unchanged():
    install = (ROOT / "scripts" / "stage_s_notebooks" / "KAGGLE_INSTALL.txt").read_bytes()
    assert hashlib.sha256(install).hexdigest().startswith("a0cda2416808a1ab")


def test_the_carnets_never_name_the_reserved_pages_or_list_a_directory():
    for name in ("kaggle_ld1.ipynb", "kaggle_ld1_smoke.ipynb"):
        code = _code(name)
        for bad in ("glob(", "rglob", "listdir", "scandir", "os.walk", "iterdir", "mf_005"):
            assert bad not in code, (name, bad)


# ---- §13 (N1, N3, N4) : kit complet, internet, pytest complet sans -k ----

def test_every_file_the_kit_reads_is_in_kit_files():
    files = set(json.loads((LD1 / "kit_files.json").read_text()))
    for rel in B.KIT_READS:
        assert rel in files and (ROOT / rel).is_file(), rel
    assert "config/data2_pages.json" in files                       # lu par data2_common (N1, voie « à défaut »)


def test_internet_is_on_for_D_and_the_kernel_metadata_says_so():
    smoke = json.loads((LD1 / "smoke_kernel" / "kernel-metadata.json").read_text())
    full = json.loads((LD1 / "full_kernel" / "kernel-metadata.json").read_text())
    assert smoke["enable_internet"] is True and full["enable_internet"] is True
    assert full["enable_gpu"] is True and smoke["enable_gpu"] is False


def test_the_smoke_pytest_runs_the_whole_file_without_k_and_does_not_block():
    code = _code("kaggle_ld1_smoke.ipynb")
    assert '"-k"' not in code and "tests/test_line_detr.py" in code
    assert "exit 0" in code and "ld1_pytest_rc.txt" in code


def test_offline_is_set_for_T_E_V_only_and_not_for_D():
    """§13 (3) : D a internet ; T, E, V sont hors ligne (env HF_HUB_OFFLINE=1), D exclu par la boucle du carnet."""
    code = _code("kaggle_ld1.ipynb")
    assert 'OFFLINE = lambda: dict(os.environ, HF_HUB_OFFLINE="1")' in code
    assert '{} if ph == "D" else {"env": OFFLINE}' in code


# ---- gel 7 : échéance globale, plafonds lus dans la config, rc des étapes, environnement (§15 (4), §16, §17) ----
# On exécute le VRAI corps du carnet complet (FULL_BODY) avec un faux run_step, un faux torch et une fausse horloge.

import importlib
import importlib.metadata
import importlib.util
import types

CFG_VALUES = {"deadline_s": 1234, "v_reserve_s": 111, "stop_margin_s": 22, "e_reserve_s": 77}
CFG_TIMEOUTS = {"D": 33, "T": 44, "E": 5555, "V": 66, "pytest_heavy": 99}


@pytest.fixture
def cfg_file(tmp_path, monkeypatch):
    cfg = json.loads((ROOT / "config" / "ld1_line_detr.json").read_text())
    cfg["robustness"].update(CFG_VALUES)
    cfg["robustness"]["timeouts_s"] = dict(CFG_TIMEOUTS)
    f = tmp_path / "cfg.json"
    f.write_text(json.dumps(cfg))
    monkeypatch.setattr(B, "CONFIG", f)
    return f


def _literal(tree, name):
    for n in tree.body:
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets):
            return ast.literal_eval(n.value)
    raise AssertionError(name)


def test_limits_and_caps_come_from_the_config_and_not_from_the_code(cfg_file):
    tree = ast.parse(B._cell(B.FULL_BODY, "cuda"))
    assert _literal(tree, "LIMITS") == CFG_VALUES                    # valeurs distinctes de celles du dépôt
    assert _literal(tree, "TIMEOUTS") == {k: CFG_TIMEOUTS[k] for k in ("D", "T", "E", "V", "pytest_heavy")}
    real = json.loads((ROOT / "config" / "ld1_line_detr.json").read_text())["robustness"]
    assert (real["deadline_s"], real["v_reserve_s"], real["stop_margin_s"], real["e_reserve_s"]) == (7200, 900, 120, 300)


def test_T_START_is_the_first_statement_of_the_cell():
    tree = ast.parse(B._cell(B.FULL_BODY, "cuda"))
    assert ast.unparse(tree.body[0]) == "import time" and ast.unparse(tree.body[1]) == "T_START = time.monotonic()"


def _data2_common():
    spec = importlib.util.spec_from_file_location("data2_common_ld1test", ROOT / "scripts" / "data2_common.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Run:
    """Exécute le vrai FULL_BODY. plan[étape] = {"rc", "dur", "exit_status"} ; l'horloge avance de dur à chaque étape."""

    def __init__(self, tmp_path, monkeypatch, plan, start=0.0, cuda=True, missing_pkg=None):
        self.calls, self.clock, self.plan = [], [start], plan
        self.out = tmp_path / "work"
        self.out.mkdir()
        self.C = _data2_common()
        monkeypatch.setattr(self.C, "run_step", self.run_step)
        tree = ast.parse(B._cell(B.FULL_BODY, "cuda"))
        body = ast.parse(B.FULL_BODY)
        ns = {"time": types.SimpleNamespace(monotonic=lambda: self.clock[0]), "json": json, "os": os, "sys": sys,
              "Path": Path, "OUT": self.out, "C": self.C, "T_START": 0.0, "CKPT": tmp_path / "ck.pt",
              "LIMITS": _literal(tree, "LIMITS"), "TIMEOUTS": _literal(tree, "TIMEOUTS"),
              "_utc": lambda: "00:00:00", "_argv": lambda phase, extra=(): ["fake", phase, *extra]}
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = types.SimpleNamespace(is_available=lambda: cuda)
        fake_torch.load = lambda *a, **k: {"model": {}, "step": 2000}
        fake_torch.__version__ = "9.9"
        fake_torch.version = types.SimpleNamespace(cuda="12.x")
        ld = types.ModuleType("babaorum.experimental.line_detr")
        ld.LineDetr = lambda: object()
        ld.load_enc_det_from_e2e5 = lambda *a, **k: None
        for name, mod in (("torch", fake_torch), ("babaorum", types.ModuleType("babaorum")),
                          ("babaorum.experimental", types.ModuleType("babaorum.experimental")),
                          ("babaorum.experimental.line_detr", ld)):
            monkeypatch.setitem(sys.modules, name, mod)

        def version(name):
            if name == missing_pkg:
                raise importlib.metadata.PackageNotFoundError(name)
            return f"v-{name}"

        monkeypatch.setattr(importlib.metadata, "version", version)
        exec(compile(body, "<FULL_BODY>", "exec"), ns)
        self.ns = ns

    def run_step(self, name, argv, timeout_s, logs_dir, cwd=None, env=None, kill_grace_s=10.0):
        p = {"rc": 0, "dur": 1.0, "exit_status": None, **self.plan.get(name, {})}
        self.calls.append((name, timeout_s))
        self.clock[0] += p["dur"]
        if name == "T" and p["rc"] == 0:
            run = self.out / "ld1"
            run.mkdir(parents=True, exist_ok=True)
            (run / "ld1_timing_manifest.json").write_text(json.dumps({"max_steps": 20000, "s_per_step": 2.0, "steps": 999}))
        if p["exit_status"]:
            ex = self.out / "exits"
            ex.mkdir(exist_ok=True)
            (ex / f"ld1_run_exit_{name}.json").write_text(json.dumps({"status": p["exit_status"]}))
        return {"step": name, "rc": p["rc"], "seconds": p["dur"], "stdout": "", "stderr": "", "stderr_tail": ""}

    @property
    def exit(self):
        return self.ns["EXIT"]

    @property
    def names(self):
        return [c[0] for c in self.calls]


def test_each_step_gets_max_0_min_cap_remaining_minus_reserve_and_S_is_recomputed(tmp_path, monkeypatch, cfg_file):
    r = Run(tmp_path, monkeypatch, {"D": {"dur": 10.0}, "T": {"dur": 20.0}, "E": {"dur": 5.0}})
    assert r.exit["status"] == "DONE" and r.names == ["D", "T", "E", "V"]
    t = dict(r.calls)
    assert t["D"] == 33 and t["T"] == 44                           # plafonds de la config (33, 44), pas ceux du dépôt
    # E : min(5555, 1234 - 30 - (111 + 22)) = 1071 ; V : min(66, 1234 - 30 - 5 ...) = 66
    assert t["E"] == 1234 - 30 - 133 and t["V"] == 66
    man = json.loads((r.out / "ld1" / "ld1_timing_manifest.json").read_text())
    assert man["steps"] == (1071 - 77) // 2 == 497                  # S = floor((budget_E - E_RESERVE_S) / s_par_pas)
    assert man["budget_s"] == 1071 and man["e_reserve_s"] == 77


def test_S_with_the_real_config_is_floor_of_4200_over_s_per_step(tmp_path, monkeypatch):
    r = Run(tmp_path, monkeypatch, {})                              # E : min(4500, 7200 - 2 - 1020) = 4500 ; s/pas = 2
    man = json.loads((r.out / "ld1" / "ld1_timing_manifest.json").read_text())
    assert man["steps"] == 2100 and man["budget_s"] == 4500 and man["e_reserve_s"] == 300


def test_a_zero_delay_gives_INCOMPLETE_without_launching_the_step(tmp_path, monkeypatch, cfg_file):
    r = Run(tmp_path, monkeypatch, {"D": {"dur": 1234.0}})           # D consomme tout : T reçoit 0
    assert r.names == ["D"] and r.exit["status"] == "INCOMPLETE" and r.exit["step"] == "T"
    (tmp_path / "b").mkdir()
    r = Run(tmp_path / "b", monkeypatch, {"T": {"dur": 1100.0}})
    assert "E" not in r.names and "V" not in r.names              # E : 1234 - 1100 - 133 < 0, donc pas de lancement
    assert r.exit["status"] == "INCOMPLETE" and r.exit["step"] == "s_budget"


def test_rc_zero_continues_rc_one_and_TIMEOUT_stop_with_INCOMPLETE_at_the_step(tmp_path, monkeypatch):
    r = Run(tmp_path, monkeypatch, {})
    assert r.exit["status"] == "DONE" and r.names == ["D", "T", "E", "V"]
    (tmp_path / "1").mkdir()
    r = Run(tmp_path / "1", monkeypatch, {"T": {"rc": 1}})
    assert r.names == ["D", "T"] and r.exit["status"] == "INCOMPLETE" and r.exit["step"] == "T"
    (tmp_path / "2").mkdir()
    r = Run(tmp_path / "2", monkeypatch, {"E": {"rc": "TIMEOUT"}})
    assert r.names == ["D", "T", "E"] and r.exit["status"] == "INCOMPLETE" and r.exit["step"] == "E"


def test_a_failed_exit_record_of_the_step_keeps_FAILED(tmp_path, monkeypatch):
    """§17 (4) : une erreur de device à T (T écrit FAILED dans OUT/exits) donne FAILED à l'étape T."""
    r = Run(tmp_path, monkeypatch, {"T": {"rc": 1, "exit_status": "FAILED"}})
    assert r.names == ["D", "T"] and r.exit["status"] == "FAILED" and r.exit["step"] == "T"
    (tmp_path / "1").mkdir()
    r = Run(tmp_path / "1", monkeypatch, {"T": {"rc": 1, "exit_status": "INCOMPLETE"}})   # OOM répété
    assert r.exit["status"] == "INCOMPLETE" and r.exit["step"] == "T"


def test_the_controls_step_archives_the_environment_and_fails_without_transformers(tmp_path, monkeypatch):
    r = Run(tmp_path, monkeypatch, {})
    env = json.loads((r.out / "ld1_env.json").read_text())
    assert set(env) == {"python", "torch", "cuda", "numpy", "PIL", "transformers", "huggingface_hub"}
    assert env["transformers"] == "v-transformers" and env["huggingface_hub"] == "v-huggingface_hub"
    assert not (r.out / "ld1").exists() or list((r.out / "ld1").glob("ld1_env*")) == []     # jamais dans RUN
    for i, pkg in enumerate(("transformers", "huggingface_hub")):
        d = tmp_path / f"m{i}"
        d.mkdir()
        r = Run(d, monkeypatch, {}, missing_pkg=pkg)
        assert r.names == [] and r.exit["status"] == "FAILED" and r.exit["step"] == "controls"
        assert pkg in r.exit["exception"]


def test_the_cell_uses_step_argv_and_the_cleanup_and_keeps_execute_pipeline():
    code = _code("kaggle_ld1.ipynb")
    assert "C.execute_pipeline(OUT, STEPS, heartbeat_interval_s=60.0, cleanup=ld1_cleanup)" in code
    assert "step_argv(ph," in code and "STEPS.insert(3," in code
