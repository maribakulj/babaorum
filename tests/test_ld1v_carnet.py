"""Carnet LD-1 « V seule » (docs/END_TO_END_LD1_LOT.txt §18), SANS torch."""
import ast
import builtins
import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "stage_s_notebooks"))
import build_kaggle_ld1 as B  # noqa: E402
import build_kaggle_ld1v as V  # noqa: E402

DIR = ROOT / "scripts" / "stage_s_notebooks" / "ld1v"


def _src():
    nb = json.loads((DIR / "notebook.ipynb").read_text())
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")


def test_notebook_and_metadata_on_disk_equal_build():
    assert json.loads((DIR / "notebook.ipynb").read_text()) == V.build_notebook()
    assert json.loads((DIR / "kernel-metadata.json").read_text()) == V.kernel_metadata()


def test_metadata():
    m = V.kernel_metadata()
    assert m["id"] == "mabaku/babaorum-ld1-v" and m["is_private"] is True
    assert m["enable_gpu"] is False and m["enable_internet"] is True
    assert m["dataset_sources"] == ["mabaku/babaorum-ld1-kit"]
    assert set(m["kernel_sources"]) == {"mabaku/babaorum-ld1-run", "mabaku/babaorum-e2e3-run3-fp-classes",
                                        "mabaku/babaorum-e2e5-lambda"}
    assert m["code_file"] == "notebook.ipynb"
    assert len((DIR / "notebook.ipynb").read_bytes()) < 300 * 1024


def test_run_candidates():
    assert V.RUN_CANDIDATES == ["/kaggle/input/babaorum-ld1-run/ld1",
                                "/kaggle/input/notebooks/mabaku/babaorum-ld1-run/ld1"]
    assert V.RUN_CANDIDATES[0] in _src() and V.RUN_CANDIDATES[1] in _src()


def test_global_names_defined():
    tree = ast.parse(_src())
    defined = set(dir(builtins))
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            defined.add(n.name)
        elif isinstance(n, ast.Import):
            defined |= {(a.asname or a.name).split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            defined |= {a.asname or a.name for a in n.names}
        elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            defined.add(n.id)
        elif isinstance(n, ast.arg):
            defined.add(n.arg)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            defined.add(n.name)
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    assert not (used - defined), used - defined


def test_no_listing_of_held_pages_and_no_recursive_search():
    s = _src()
    assert "mf_005" not in s and "mf_0050" not in s
    assert "rglob" not in s and "os.walk" not in s and "listdir" not in s and "iterdir" not in s
    assert "glob(" not in s.replace('d.glob("ld1_step*.pt")', "")      # seul le nettoyage de nos propres RUN


# ---- arbre factice ----

def _tree(tmp_path, s=1871, extra=True, with_step=True):
    kit = tmp_path / "kit"
    (kit / "scripts").mkdir(parents=True)
    (kit / "config").mkdir()
    shutil.copy(ROOT / "scripts" / "data2_common.py", kit / "scripts")
    shutil.copy(ROOT / "config" / "ld1_line_detr.json", kit / "config")
    shutil.copy(ROOT / "config" / "data2_pages.json", kit / "config")
    (kit / "scripts" / "train_line_detr.py").write_text("")
    pages = tmp_path / "pages"
    (pages / "pages").mkdir(parents=True)
    (pages / "pages" / "mf_0000.json").write_text("{}")
    ck = tmp_path / "e2e5" / "checkpoint_last.pt"
    ck.parent.mkdir()
    ck.write_bytes(b"init")
    run = tmp_path / "in_run" / "ld1"
    run.mkdir(parents=True)
    (run / "ld1_timing_manifest.json").write_text(json.dumps({"steps": s}))
    if with_step:
        (run / f"ld1_step{s:06d}.pt").write_bytes(b"final")
    if extra:
        (run / "ld1_step001000.pt").write_bytes(b"old")
    return kit, pages, ck, run


def _ns(tmp_path, tree, **kw):
    kit, pages, ck, run = tree
    src = V.cell(out=str(tmp_path / "out"), kit=[str(kit)], pages=[str(pages)], ckpt=[str(ck)],
                 run_src=["/nonexistent/x", str(run)], init_sha=hashlib.sha256(b"init").hexdigest())
    ns = {"LD1V_NO_RUN": True, "__name__": "ld1v_test"}
    exec(compile(src, "ld1v", "exec"), ns)
    return ns


def test_single_checkpoint_s_1871_and_ignores_other(tmp_path):
    ns = _ns(tmp_path, _tree(tmp_path))
    man, ck, S = ns["checkpoint_inputs"]()
    assert S == 1871 and ck.name == "ld1_step001871.pt"
    copied = []
    ns["s_copy"]()
    for d in (ns["RUN"], ns["PROBE"]):
        names = sorted(p.name for p in d.iterdir())
        assert names == ["ld1_step001871.pt", "ld1_timing_manifest.json"]


def test_wrong_s_or_missing_file_refused(tmp_path):
    ns = _ns(tmp_path, _tree(tmp_path, s=1000))
    with pytest.raises(RuntimeError, match="1871"):
        ns["checkpoint_inputs"]()
    p2 = tmp_path / "b"
    p2.mkdir()
    ns = _ns(p2, _tree(p2, with_step=False))
    with pytest.raises(RuntimeError, match="absent"):
        ns["checkpoint_inputs"]()


def test_two_run_candidates_refused(tmp_path):
    kit, pages, ck, run = _tree(tmp_path)
    src = V.cell(out=str(tmp_path / "o"), kit=[str(kit)], pages=[str(pages)], ckpt=[str(ck)],
                 run_src=[str(run), str(run)], init_sha=hashlib.sha256(b"init").hexdigest())
    with pytest.raises(RuntimeError):
        exec(compile(src, "x", "exec"), {"LD1V_NO_RUN": True})


class _Clock:
    def __init__(self):
        self.t = 0.0

    def monotonic(self):
        return self.base + self.t


def _fake(ns, probe_s, calls, armed=None):
    clock = _Clock()
    clock.base = ns["time"].monotonic()
    ns["time"] = clock

    def run_step(name, argv, timeout_s, logs_dir, cwd=None, env=None, kill_grace_s=10.0):
        argv = [str(a) for a in argv]
        calls.append((name, argv, timeout_s, env))
        if armed is not None:
            armed[name] = armed["state"]
        clock.t += probe_s if name == "probe" else 1.0
        if "--out" in argv:
            out = Path(argv[argv.index("--out") + 1])
            (out.parent / "exits").mkdir(parents=True, exist_ok=True)
            if name == "V":
                assert (out / "ld1_step001871.pt").is_file()       # V lit le checkpoint copié
                (out / "ld1_report.json").write_text("{}")
        return {"step": name, "rc": 0, "seconds": 1.0, "stdout": "", "stderr": "", "stderr_tail": ""}
    ns["C"].run_step = run_step


def test_margin_declared_and_applied(tmp_path):
    assert V.MARGIN_S == 300 and "MARGIN_S = 300" in _src()
    ns = _ns(tmp_path, _tree(tmp_path))
    assert ns["MARGIN_S"] == 300
    calls = []
    _fake(ns, 100.0, calls)
    ns["run"]()
    for name, _a, t, _e in calls[1:]:
        assert 0 < t <= 10800 - 300, (name, t)
        if name == "probe":
            assert t > 10800 - 300 - 10
    ns["DEADLINE_S"] = 300
    with pytest.raises(ns["Incomplete"]):
        ns["v_argv"]()


def test_faulthandler_timer_not_armed_during_probe_and_v(tmp_path, monkeypatch):
    import faulthandler
    armed = {"state": False}
    monkeypatch.setattr(faulthandler, "dump_traceback_later",
                        lambda *a, **k: armed.__setitem__("state", True))
    monkeypatch.setattr(faulthandler, "cancel_dump_traceback_later",
                        lambda *a, **k: armed.__setitem__("state", False))
    ns = _ns(tmp_path, _tree(tmp_path))
    calls = []
    _fake(ns, 100.0, calls, armed)
    ns["run"]()
    assert [c[0] for c in calls] == ["D", "probe", "V"]
    assert armed["D"] is False and armed["probe"] is False and armed["V"] is False


def test_probe_then_v_in_order_with_same_options(tmp_path):
    ns = _ns(tmp_path, _tree(tmp_path))
    calls = []
    _fake(ns, 100.0, calls)
    ex = ns["run"]()
    assert ex["status"] == "DONE"
    assert [c[0] for c in calls] == ["D", "probe", "V"]
    d, probe, v = (c[1] for c in calls)
    assert probe[probe.index("--pages") + 1:] == ["mf_0045"]
    assert "--pages" not in v
    base = lambda a: [x for x in a if not x.startswith(str(tmp_path / "out"))]
    assert probe[:probe.index("--out")] == v[:v.index("--out")]
    assert v[v.index("--phase") + 1] == "V" and v[v.index("--device") + 1] == "cpu"
    assert str(ns["PROBE"]) == probe[probe.index("--out") + 1] != v[v.index("--out") + 1] == str(ns["RUN"])
    assert calls[1][3]["HF_HUB_OFFLINE"] == "1" and calls[2][3]["HF_HUB_OFFLINE"] == "1"
    assert calls[0][3] is None                                       # D : internet
    for c in calls:
        assert 0 < c[2] <= 10800 - (0 if c[0] == "D" else 300)
    assert not list(ns["RUN"].glob("ld1_step*.pt"))                 # copie du checkpoint retirée de la sortie
    assert not ns["PROBE"].exists()
    assert not (ns["RUN"] / "exits").exists() and (ns["RUN"].parent / "exits").is_dir()


def test_estimate_too_large_is_incomplete_without_v(tmp_path, capfd):
    ns = _ns(tmp_path, _tree(tmp_path))
    calls = []
    _fake(ns, 2500.0, calls)           # 12 500 s > 90 % du restant
    ex = ns["run"]()
    assert [c[0] for c in calls] == ["D", "probe"]
    assert ex["status"] == "INCOMPLETE" and ex["step"] == "estimate"
    assert "ESTIMATION_V_S 12500" in capfd.readouterr().out


def test_deadline_zero_means_no_launch(tmp_path):
    ns = _ns(tmp_path, _tree(tmp_path))
    ns["DEADLINE_S"] = 0
    with pytest.raises(ns["Incomplete"]):
        ns["v_argv"]()
    assert ns["allowed"](900) == 0.0
