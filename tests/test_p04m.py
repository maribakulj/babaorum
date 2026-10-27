"""AV-P04 v1.4 (pA, 25/10): permanent tests of the pairing gate (the reference against ITSELF must open it) and of the Δ outcomes (p1)."""
import importlib.util
import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REF = Path.home() / "Documents/reviews/babaorum-pipeline/rappel_pages/runs_26/runs"
AXEL = Path.home() / "axel"
PY = Path.home() / "axel/.venv/bin/python"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


D = _load("p04m_delta")


def test_the_four_outcomes_are_exclusive_and_written_in_advance():
    assert D.outcome(0.010, 0.001).startswith("(a)")
    assert D.outcome(0.004, 0.001).startswith("(b)")
    assert D.outcome(0.006, -0.001).startswith("(c′)")
    assert D.outcome(0.004, -0.001).startswith("(c)") and not D.outcome(0.004, -0.001).startswith("(c′)")
    assert D.outcome(0.005, 0.0).startswith("(c′)")                         # lower bound must be STRICTLY > 0; TEM is inclusive
    assert D.outcome(0.005, 1e-12).startswith("(a)")
    assert D.TEM == 0.005


def test_frozen_pages_and_groups():
    assert len(D.GROUPS) == 24 and len(set(D.GROUPS.values())) == 21 and not set(D.APART) & set(D.GROUPS)


def _gate(tmp_path, pass1):
    out = tmp_path / "porte.json"
    r = subprocess.run([str(PY), str(ROOT / "scripts/p04m_porte.py"), "--ref", str(REF), "--pass1", str(pass1), "--axel", str(AXEL),
                        "--python-axel", str(PY), "--out", str(out)], env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
                       capture_output=True, text=True)
    return r.returncode, json.loads(out.read_text())


needs_ref = pytest.mark.skipif(not REF.exists() or not PY.exists(), reason="runs_26 reference or axel venv absent")


@needs_ref
def test_gate_reference_against_itself_is_open(tmp_path):
    rc, rep = _gate(tmp_path, REF)
    assert rc == 0 and rep["porte_ouverte"] is True and rep["pages_expected"] == 26
    assert all(v["verdict"] == "identique" for v in rep["pages"].values())
    assert all(v["portability"]["passed"] and v["portability"]["c1"]["found_share"] == 1.0 for v in rep["pages"].values() if "portability" in v)


def _copy_as_links(dst: Path):
    for p in REF.rglob("*"):
        q = dst / p.relative_to(REF)
        if p.is_dir():
            q.mkdir(parents=True, exist_ok=True)
        else:
            q.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(p, q)


@needs_ref
def test_gate_closes_on_a_one_sided_file_and_on_one_changed_letter(tmp_path):
    p1 = tmp_path / "pass1"
    _copy_as_links(p1)
    extra = p1 / "e32/dev/read/mccatmus_press/extra/L_OEuvre_4617562_T0000007.json"
    assert extra.is_symlink()
    extra.unlink()                                                             # present on the reference side only
    page = "La_Presse_0476474_T0000001"
    f = p1 / f"e12/read/ppdl3_z/{page}.json"
    d = json.loads(f.read_text())
    f.unlink()
    k = next(iter(d["texts"]))
    d["texts"][k] += "x"
    f.write_text(json.dumps(d))
    rc, rep = _gate(tmp_path, p1)
    assert rc == 1 and rep["porte_ouverte"] is False
    assert set(rep["pages_closing"]) == {"L_OEuvre_4617562_T0000007", page}
    assert rep["pages"]["L_OEuvre_4617562_T0000007"]["verdict"] == "anomalie" and rep["pages"][page]["verdict"] == "équivalent"


# pQ B3: the identity-then-coverage matching of threshold 3 is a BIJECTION (no pass-1 line reused)
P = _load("p04m_porte")


def _line(y, x0=0, x1=400):
    return {"baseline": [[x0, y], [x1, y]], "polygon": [[x0, y - 20], [x1, y - 20], [x1, y], [x0, y]], "text": "abc"}


def test_threshold3_matching_is_one_to_one():
    ref = [_line(30 * i) for i in range(100)]
    p1 = [dict(_line(30 * i + 1)) for i in range(98)]                         # 2 lines missing; none identical (shifted by 1 px)
    cover = lambda u, v: 1.0 if abs(u["baseline"][0][1] - v["baseline"][0][1]) <= 40 else 0.0   # noqa: E731  (lets one line cover 3)
    m = P.biject(ref, p1, cover)
    assert len(set(m.values())) == len(m) and len(m) == 98                    # never 100 by reuse
    assert len(m) / len(ref) < 0.99


def test_threshold3_identity_first_then_coverage():
    ref = [_line(0), _line(30)]
    p1 = [_line(30), _line(0)]
    assert P.biject(ref, p1, lambda u, v: 0.0) == {0: 1, 1: 0}                # identity, no coverage needed
    dup = [_line(0), _line(0)]
    assert P.biject(dup, [_line(0)], lambda u, v: 1.0) == {0: 0}              # one identical pass-1 line serves ONE reference line


# pQ B4 / M1: Δ always writes a structured outcome; nothing of pass 2 before an open gate; broken SHA chain = incomplete
def _run_delta(tmp_path, **files):
    args = {k: str(tmp_path / f"{k}.json") for k in ("porte", "report1", "report2", "link1", "link2", "bench1", "bench2", "reference")}
    for k, v in files.items():
        (tmp_path / f"{k}.json").write_text(json.dumps(v))
    out = tmp_path / "delta.json"
    r = subprocess.run([str(PY), str(ROOT / "scripts/p04m_delta.py"), *[x for k, v in args.items() for x in (f"--{k}", v)], "--out", str(out)],
                       env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"), capture_output=True, text=True)
    return r.returncode, json.loads(out.read_text())


@pytest.mark.skipif(not PY.exists(), reason="axel venv absent")
def test_delta_structured_outcomes(tmp_path):
    rc, rep = _run_delta(tmp_path)                                             # no file at all
    assert rc == 1 and rep["outcome"].startswith("PORTE ABSENTE")
    pages = {p: {"page_ok": True} for p in list(D.GROUPS) + list(D.APART)}
    t1 = tmp_path / "x"
    t1.mkdir()
    rc, rep = _run_delta(t1, porte={"porte_ouverte": False, "pages_closing": ["p"], "pages": pages})
    assert rc == 1 and rep["outcome"].startswith("PORTE FERMÉE") and "report2" not in rep.get("inputs", {})
    t2 = tmp_path / "y"
    t2.mkdir()
    rc, rep = _run_delta(t2, porte={"porte_ouverte": True, "pages_closing": [], "pages": pages},
                         report1={"passe": 1, "incomplete": [], "elapsed_hours": 9.0, "axel_commit": "c62abf9", "manifest_p04m": "0" * 64,
                                  "freeze": {"a": ["x==1"]}, "machine": {"cpu_model": "C"}, "runs_archive_sha256": "1" * 64,
                                  "gate": {"porte_ouverte": True, "sha256": "f" * 64}})
    assert rc == 1 and rep["outcome"].startswith("MESURE INCOMPLÈTE") and "THIS open gate" in rep["problems"][0]
    t3 = tmp_path / "z"
    t3.mkdir()
    rc, rep = _run_delta(t3, porte={"porte_ouverte": True, "pages_closing": [], "pages": pages}, report1={"passe": 1, "incomplete": None})
    assert rc == 1 and "missing or mistyped" in rep["problems"][0]


BENCH_REF = ROOT / "reports/bench_pipeline_p07b_26.json"


HEAD = "c62abf946ba69e5ffb522dc4e8d08b002b9a0f7b"
PRODUCERS = dict({k: "d" * 64 for k in D.REQUIRED_PRODUCERS}, **{"babaorum:src/babaorum/bench_metrics.py": "b" * 64, "axel:src/axel/layout/fill.py": "c" * 64})


def _chain(tmp_path):
    import copy
    import hashlib
    tmp_path.mkdir(parents=True, exist_ok=True)
    h = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()                  # noqa: E731
    pages = {p: {"page_ok": True} for p in list(D.GROUPS) + list(D.APART)}
    porte = tmp_path / "porte.json"
    porte.write_text(json.dumps({"porte_ouverte": True, "pages_closing": [], "pages": pages}))
    base = {"incomplete": [], "elapsed_hours": 9.0, "axel_commit": "c62abf9", "manifest_p04m": "0" * 64, "freeze": {"a": ["x==1"]},
            "machine": {"cpu_model": "C"}}
    r1 = tmp_path / "report1.json"
    r1.write_text(json.dumps(dict(base, passe=1, runs_archive_sha256="1" * 64, gate={"porte_ouverte": True, "pages_closing": [], "sha256": h(porte)})))
    r2 = tmp_path / "report2.json"
    r2.write_text(json.dumps(dict(base, passe=2, runs_archive_sha256="2" * 64,
                                  consumed={"porte_sha256": h(porte), "passe1_report_sha256": h(r1)})))
    b = json.loads(BENCH_REF.read_text())
    b["provenance"]["axel_head"] = HEAD                                       # pQ B4: the bench was produced at the report's commit
    b2 = copy.deepcopy(b)
    for r in b2["chains"]:
        if r["chain"] == "c1+ma":
            r["cer_reordered"] -= 0.01
    (tmp_path / "bench1.json").write_text(json.dumps(b))
    (tmp_path / "bench2.json").write_text(json.dumps(b2))
    for k, arch in ((1, "1" * 64), (2, "2" * 64)):
        (tmp_path / f"link{k}.json").write_text(json.dumps({
            "passe": k, "report_sha256": h(tmp_path / f"report{k}.json"), "archive_sha256": arch, "bench_sha256": h(tmp_path / f"bench{k}.json"),
            "truth_manifest_sha256": D.TRUTH_MANIFEST_SHA, "config_sha256": D.CONFIG_SHA, "axel_head": HEAD, "axel_clean": True,
            "producers": PRODUCERS, "producers_sha256": D.producers_sha(PRODUCERS), "script_sha256": "5" * 64,
            "bench_env": ["python==3.12.0", "numpy==2.0.0"]}))
    (tmp_path / "reference.json").write_text(json.dumps(b))
    return tmp_path


def _delta(d):
    out = d / "delta.json"
    if out.exists():
        out.unlink()
    args = [x for k in ("porte", "report1", "report2", "link1", "link2", "bench1", "bench2", "reference") for x in (f"--{k}", str(d / f"{k}.json"))]
    r = subprocess.run([str(PY), str(ROOT / "scripts/p04m_delta.py"), *args, "--out", str(out)], env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
                       capture_output=True, text=True)
    return r.returncode, json.loads(out.read_text())


def _relink(d, k, **change):
    lk = json.loads((d / f"link{k}.json").read_text())
    lk.update(change)
    (d / f"link{k}.json").write_text(json.dumps(lk))


@pytest.mark.skipif(not PY.exists() or not BENCH_REF.exists(), reason="axel venv or reference bench absent")
def test_delta_full_sha_chain_gives_an_outcome(tmp_path):
    rc, rep = _delta(_chain(tmp_path))
    assert rc == 0 and rep["outcome"].startswith("(a)") and abs(rep["delta"] - 0.01) < 1e-12
    assert rep["producers_sha256"] == D.producers_sha(PRODUCERS)


@pytest.mark.skipif(not PY.exists() or not BENCH_REF.exists(), reason="axel venv or reference bench absent")
def test_delta_refuses_head_mismatch_and_changed_producers(tmp_path):
    d = _chain(tmp_path / "a")
    b1 = json.loads((d / "bench1.json").read_text())
    b1["provenance"]["axel_head"] = "fc4d70ed14a21ef6cbbadd385706d89d075a2b99"   # pQ's counter-example: links c62abf9, benches fc4d70ed
    (d / "bench1.json").write_text(json.dumps(b1))
    import hashlib
    _relink(d, 1, bench_sha256=hashlib.sha256((d / "bench1.json").read_bytes()).hexdigest())
    rc, rep = _delta(d)
    assert rc == 1 and rep["outcome"].startswith("MESURE INCOMPLÈTE") and "axel head" in rep["problems"][0]
    d = _chain(tmp_path / "b")
    pr = dict(PRODUCERS, **{"babaorum:src/babaorum/bench_metrics.py": "e" * 64})   # same axel HEAD, a babaorum producer changed in between
    _relink(d, 2, producers=pr, producers_sha256=D.producers_sha(pr))
    rc, rep = _delta(d)
    assert rc == 1 and "producers differ" in rep["problems"][0] and "babaorum:src/babaorum/bench_metrics.py" in rep["problems"][0]
    d = _chain(tmp_path / "c")
    _relink(d, 1, axel_clean=False)
    rc, rep = _delta(d)
    assert rc == 1 and "not clean" in rep["problems"][0]
    d = _chain(tmp_path / "e")
    pr = {k: v for k, v in PRODUCERS.items() if not k.endswith(".xsd")}       # XSD absent from both links: refused, not "identical"
    for k in (1, 2):
        _relink(d, k, producers=pr, producers_sha256=D.producers_sha(pr))
    rc, rep = _delta(d)
    assert rc == 1 and "producers absent" in rep["problems"][0]
    d = _chain(tmp_path / "f")
    _relink(d, 2, script_sha256="6" * 64)
    rc, rep = _delta(d)
    assert rc == 1 and "p04m_bench.py differs" in rep["problems"][0]
    d = _chain(tmp_path / "g")
    _relink(d, 2, bench_env=["python==3.12.0", "numpy==2.1.0"])
    rc, rep = _delta(d)
    assert rc == 1 and "installed distributions" in rep["problems"][0]


@pytest.mark.skipif(not PY.exists() or not BENCH_REF.exists(), reason="axel venv or reference bench absent")
def test_delta_odd_json_still_gives_a_structured_outcome(tmp_path):
    d = _chain(tmp_path / "a")
    (d / "report1.json").write_text("[1, 2]")                                  # pQ minor: a list instead of an object
    rc, rep = _delta(d)
    assert rc == 1 and rep["outcome"].startswith("MESURE INCOMPLÈTE")
    d = _chain(tmp_path / "b")
    porte = json.loads((d / "porte.json").read_text())
    porte["pages"][next(iter(porte["pages"]))] = "ok"                          # a page entry that is not an object
    (d / "porte.json").write_text(json.dumps(porte))
    rc, rep = _delta(d)
    assert rc == 1 and rep["outcome"].startswith("PORTE FERMÉE")


def test_threshold3_has_no_geometric_prefilter():
    a = {"baseline": [[0, 0], [400, 0]], "polygon": [[0, -500], [400, -500], [400, 0], [0, 0]], "text": "a"}      # pQ's repro (filter-100)
    b = {"baseline": [[0, 151], [400, 151]], "polygon": [[0, 151], [400, 151], [400, 651], [0, 651]], "text": "a"}
    assert P.biject([a], [b], lambda u, v: 1.0) == {0: 0}


# pQ B1 (2nd audit): the deadline kills the WHOLE process group, writes the failure report, and leftovers of a finished step are killed too
def _notebook_clock(tmp_path, limit_s, reserve_s):
    import ast
    import time
    nb = json.loads((ROOT / "notebooks/kaggle_p04m.ipynb").read_text())
    src = "".join(next(c["source"] for c in nb["cells"] if c["cell_type"] == "code" and "def bounded" in "".join(c["source"])))
    keep = {"elapsed_h", "remaining", "fail", "guard", "_kill_group", "bounded", "_report_path", "_close_gate", "_write_failure", "_abort"}
    tree = ast.parse(src)
    body = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom)) or (isinstance(n, ast.FunctionDef) and n.name in keep)]
    ns = {"T0": time.monotonic(), "LIMIT_H": limit_s / 3600, "RESERVE_S": reserve_s, "W": tmp_path, "LOGS": tmp_path, "PASSE": 1}
    exec(compile(ast.Module(body=body, type_ignores=[]), "cell0", "exec"), ns)
    return ns

def test_deadline_kills_the_process_group(tmp_path):
    import time
    ns = _notebook_clock(tmp_path, limit_s=2.1, reserve_s=1.0)               # 1.1 s for the step, as in pQ's repro
    mark = tmp_path / "late.txt"
    child = f"import subprocess,sys,time;subprocess.Popen([sys.executable,'-c','import time;time.sleep(2);open(r\"{mark}\",\"w\").write(\"x\")']);time.sleep(60)"
    with pytest.raises(SystemExit):
        ns["bounded"]("etape", [os.sys.executable, "-c", child])
    rep = json.loads((tmp_path / "p04m_passe1_report.json").read_text())
    assert rep["incomplete"] and "groupe de processus tué" in rep["incomplete"][0]
    time.sleep(2.5)
    assert not mark.exists()                                                   # the descendant did not survive the deadline


def test_leftover_descendants_of_a_finished_step_are_killed(tmp_path):
    import time
    ns = _notebook_clock(tmp_path, limit_s=3600, reserve_s=1.0)
    mark = tmp_path / "late.txt"
    child = f"import subprocess,sys;subprocess.Popen([sys.executable,'-c','import time;time.sleep(1);open(r\"{mark}\",\"w\").write(\"x\")'])"
    rc, _, _ = ns["bounded"]("etape", [os.sys.executable, "-c", child])
    assert rc == 0
    time.sleep(1.5)
    assert not mark.exists()
    with pytest.raises(SystemExit):                                            # no subprocess may start inside the reserve
        _notebook_clock(tmp_path, limit_s=1.0, reserve_s=1.0)["guard"]("x")


def test_bench_producer_trace_sees_the_babaorum_imports(tmp_path):
    import sys
    B = _load("p04m_bench")
    reads = tmp_path / "reads.json"
    r = subprocess.run([sys.executable, "-c", B.TRACE, str(reads), str(ROOT), str(tmp_path), str(ROOT / "scripts/bench_pipeline.py"), "--help"],
                       env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"), capture_output=True, text=True)
    got = json.loads(reads.read_text())["babaorum"]
    assert r.returncode == 0 and "scripts/bench_pipeline.py" in got
    assert {f"src/babaorum/{m}.py" for m in ("bench_metrics", "bench_pipeline", "export_page", "page_graph", "words_from_ctc", "line_geometry")} <= set(got)
    assert B.producers_sha({"x": "1"}) == D.producers_sha({"x": "1"})          # same canonical form on both sides
    assert B.REQUIRED_PRODUCERS == D.REQUIRED_PRODUCERS and all((ROOT / k.split(":", 1)[1]).is_file() for k in B.REQUIRED_PRODUCERS)


def test_bench_producer_trace_sees_dynamic_axel_imports_and_reads(tmp_path):
    import sys
    B = _load("p04m_bench")
    ax = tmp_path / "axel"
    (ax / "src/axel").mkdir(parents=True)
    (ax / "scripts").mkdir()
    (ax / "src/axel/__init__.py").write_text("")
    (ax / "src/axel/fill.py").write_text("X = 1\n")
    (ax / "scripts/e28_fill.py").write_text("Y = 2\n")
    (ax / "data.txt").write_text("z")
    drv = tmp_path / "driver.py"                                               # like bench_pipeline: sys.path + importlib.import_module + a read
    drv.write_text(f"import importlib,sys\nsys.path[:0]=[{str(ax / 'src')!r},{str(ax / 'scripts')!r}]\nimport axel.fill\n"
                   f"importlib.import_module('e28_fill')\nopen({str(ax / 'data.txt')!r}).read()\n")
    reads = tmp_path / "reads.json"
    for _ in range(2):                                                         # 2nd run: modules served from .pyc must still be counted
        r = subprocess.run([sys.executable, "-c", B.TRACE, str(reads), str(ROOT), str(ax), str(drv)], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        assert {"src/axel/__init__.py", "src/axel/fill.py", "scripts/e28_fill.py", "data.txt"} <= set(json.loads(reads.read_text())["axel"])


def test_any_cell_failure_leaves_a_report_and_no_open_gate(tmp_path):
    ns = _notebook_clock(tmp_path, limit_s=3600, reserve_s=1.0)
    (tmp_path / "porte_pairage.json").write_text(json.dumps({"porte_ouverte": True, "pages_closing": []}))
    ns["_abort"](AssertionError("un import de la chaîne échoue"))
    rep = json.loads((tmp_path / "p04m_passe1_report.json").read_text())
    assert rep["incomplete"] == ["AssertionError: un import de la chaîne échoue"]
    assert json.loads((tmp_path / "porte_pairage.json").read_text())["porte_ouverte"] is False
    nb = json.loads((ROOT / "notebooks/kaggle_p04m.ipynb").read_text())
    code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    assert all(c.startswith("try:\n") and c.endswith("except BaseException as _e:\n    _abort(_e)\n    raise") for c in code[1:])
    ns2 = _notebook_clock(tmp_path / "p2", limit_s=3600, reserve_s=1.0)
    (tmp_path / "p2").mkdir()
    ns2["PASSE"] = 2
    ns2["_abort"](RuntimeError("journal du chargeur"))                         # pass 2: a foreign message is not written
    assert "journal" not in (tmp_path / "p2/p04m_passe2_report.json").read_text()



def test_an_interrupted_wait_still_kills_the_group(tmp_path):
    import subprocess as sp
    import time
    ns = _notebook_clock(tmp_path, limit_s=3600, reserve_s=1.0)              # pQ B1 (3rd audit): KeyboardInterrupt during wait()
    mark, alive = tmp_path / "late.txt", tmp_path / "alive.txt"
    grand = tmp_path / "grand.py"
    grand.write_text(f"import time\ntime.sleep(2)\nopen({str(mark)!r}, 'w').write('x')\n")
    step = tmp_path / "step.py"
    step.write_text(f"import subprocess, sys, time\nsubprocess.Popen([sys.executable, {str(grand)!r}])\n"
                    f"open({str(alive)!r}, 'w').write('1')\ntime.sleep(60)\n")

    class Interrupting(sp.Popen):
        def wait(self, timeout=None):
            if timeout is not None and timeout > 10:                           # the bounded wait of the step, not the reaping waits
                t = time.monotonic() + 5
                while not alive.exists() and time.monotonic() < t:
                    time.sleep(0.05)
                raise KeyboardInterrupt
            return super().wait(timeout)

    class FakeSubprocess:
        Popen, TimeoutExpired = Interrupting, sp.TimeoutExpired
    ns["subprocess"] = FakeSubprocess
    with pytest.raises(KeyboardInterrupt):                                     # the original exception is preserved
        ns["bounded"]("etape", [os.sys.executable, str(step)])
    time.sleep(2.5)
    assert alive.exists() and not mark.exists()                                # the step had started; its descendant did not survive


# note B (pA, 25/10): the pass-1 gate closed on 3 Mac pages; ONLY that exact file is accepted, with a sensitivity on 21 pages / 19 groups
NOTE_B = ROOT / "tests/fixtures/p04m_porte_note_b.json"                    # byte copy of the published porte_pairage.json (05c1bb52)


def test_note_b_gate_fixture_and_constants():
    import hashlib
    assert hashlib.sha256(NOTE_B.read_bytes()).hexdigest() == D.NOTE_B_GATE_SHA
    g = json.loads(NOTE_B.read_text())
    assert g["porte_ouverte"] is False and g["pages_closing"] == D.NOTE_B_CLOSING and set(g["pages"]) == set(D.GROUPS) | set(D.APART)
    assert set(D.NOTE_B_CLOSING) <= set(D.GROUPS)
    rest = {p: v for p, v in D.GROUPS.items() if p not in D.NOTE_B_CLOSING}
    assert len(rest) == 21 and len(set(rest.values())) == 19
    gen = (ROOT / "notebooks/make_kaggle_p04m.py").read_text()                # the notebook accepts the same file and the same 3 pages
    assert f"NOTE_B_GATE_SHA = '{D.NOTE_B_GATE_SHA}'" in gen and f"NOTE_B_CLOSING = {D.NOTE_B_CLOSING!r}" in gen


def _chain_note_b(d, gate_bytes=None, gate1=None):
    import hashlib
    d = _chain(d)
    (d / "porte.json").write_bytes(NOTE_B.read_bytes() if gate_bytes is None else gate_bytes)
    h = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()                  # noqa: E731
    r1 = json.loads((d / "report1.json").read_text())
    r1["gate"] = gate1 or {"porte_ouverte": False, "pages_closing": D.NOTE_B_CLOSING, "sha256": h(d / "porte.json")}
    (d / "report1.json").write_text(json.dumps(r1))
    r2 = json.loads((d / "report2.json").read_text())
    r2["consumed"] = {"porte_sha256": h(d / "porte.json"), "passe1_report_sha256": h(d / "report1.json"), "porte_note_b": True}
    (d / "report2.json").write_text(json.dumps(r2))
    _relink(d, 1, report_sha256=h(d / "report1.json"))
    _relink(d, 2, report_sha256=h(d / "report2.json"))
    return d


@pytest.mark.skipif(not PY.exists() or not BENCH_REF.exists(), reason="axel venv or reference bench absent")
def test_note_b_gate_gives_primary_and_concordant_sensitivity(tmp_path):
    rc, rep = _delta(_chain_note_b(tmp_path / "a"))
    assert rc == 0 and rep["gate_note_b"] is True and rep["outcome"].startswith("(a)") and abs(rep["delta"] - 0.01) < 1e-12
    s = rep["sensitivity_note_b"]
    assert s["pages"] == 21 and s["groups"] == 19 and s["excluded"] == D.NOTE_B_CLOSING and abs(s["delta"] - 0.01) < 1e-12
    assert rep["bootstrap"]["groups"] == 21 and rep["fragile"] is False and rep["robustness"] == "concordant"
    rc, rep = _delta(_chain(tmp_path / "b"))                                   # an open gate: no sensitivity, no robustness field
    assert rc == 0 and rep["gate_note_b"] is False and "sensitivity_note_b" not in rep and "fragile" not in rep


@pytest.mark.skipif(not PY.exists() or not BENCH_REF.exists(), reason="axel venv or reference bench absent")
def test_note_b_fragile_when_the_three_pages_carry_the_gain(tmp_path):
    import hashlib
    d = _chain_note_b(tmp_path / "a")
    b2 = json.loads((d / "bench1.json").read_text())                          # pass 2 = pass 1, except a large gain on the 3 closing pages
    for r in b2["chains"]:
        if r["chain"] == "c1+ma" and r["page"] in D.NOTE_B_CLOSING:
            r["cer_reordered"] -= 0.2
    (d / "bench2.json").write_text(json.dumps(b2))
    _relink(d, 2, bench_sha256=hashlib.sha256((d / "bench2.json").read_bytes()).hexdigest())
    rc, rep = _delta(d)
    assert rc == 0 and rep["delta"] > D.TEM and abs(rep["sensitivity_note_b"]["delta"]) < 1e-12
    assert rep["sensitivity_note_b"]["outcome"].startswith("(c)") and rep["outcome"] != rep["sensitivity_note_b"]["outcome"]
    assert rep["fragile"] is True and rep["robustness"].startswith("FRAGILE")


@pytest.mark.skipif(not PY.exists() or not BENCH_REF.exists(), reason="axel venv or reference bench absent")
def test_note_b_accepts_only_that_exact_closed_gate(tmp_path):
    raw = NOTE_B.read_bytes()
    rc, rep = _delta(_chain_note_b(tmp_path / "a", gate_bytes=raw + b"\n"))   # one byte more: another file, closed -> refused
    assert rc == 1 and rep["outcome"].startswith("PORTE FERMÉE")
    g = json.loads(raw)
    g["pages_closing"] = D.NOTE_B_CLOSING[:2]                                   # another closed gate with 2 pages: refused
    rc, rep = _delta(_chain_note_b(tmp_path / "b", gate_bytes=json.dumps(g).encode()))
    assert rc == 1 and rep["outcome"].startswith("PORTE FERMÉE")
    rc, rep = _delta(_chain_note_b(tmp_path / "c", gate1={"porte_ouverte": True, "pages_closing": [], "sha256": D.NOTE_B_GATE_SHA}))
    assert rc == 1 and "THIS open gate" in rep["problems"][0]                  # report 1 must record the gate as it was: closed, 3 pages


def test_note_b_notebook_accepts_only_that_exact_closed_gate():
    import ast
    nb = json.loads((ROOT / "notebooks/kaggle_p04m.ipynb").read_text())
    src = "".join(next(c["source"] for c in nb["cells"] if c["cell_type"] == "code" and "def porte_acceptee" in "".join(c["source"])))
    ns = {}
    body = [n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "porte_acceptee"]
    exec(compile(ast.Module(body=body, type_ignores=[]), "cell0", "exec"), ns)
    ok = ns["porte_acceptee"]
    g = json.loads(NOTE_B.read_text())
    assert ok(g, D.NOTE_B_GATE_SHA) and ok({"porte_ouverte": True, "pages_closing": []}, "0" * 64)
    assert not ok(g, "0" * 64)                                                  # any other closed file
    assert not ok(dict(g, pages_closing=D.NOTE_B_CLOSING[:2]), D.NOTE_B_GATE_SHA)
    assert not ok(dict(g, porte_ouverte=None), D.NOTE_B_GATE_SHA) and not ok([], D.NOTE_B_GATE_SHA)


@pytest.mark.skipif(not PY.exists() or not BENCH_REF.exists(), reason="axel venv or reference bench absent")
def test_note_b_report_bindings(tmp_path):
    import hashlib
    rc, rep = _delta(_chain_note_b(tmp_path / "a", gate1={"porte_ouverte": False, "pages_closing": [], "sha256": D.NOTE_B_GATE_SHA}))
    assert rc == 1 and "THIS open gate" in rep["problems"][0]                  # report 1 closed but without the 3 pages
    d = _chain_note_b(tmp_path / "b")
    r2 = json.loads((d / "report2.json").read_text())
    r2["consumed"]["porte_note_b"] = False                                     # the notebook did not record the note-B gate
    (d / "report2.json").write_text(json.dumps(r2))
    _relink(d, 2, report_sha256=hashlib.sha256((d / "report2.json").read_bytes()).hexdigest())
    rc, rep = _delta(d)
    assert rc == 1 and "porte_note_b" in rep["problems"][0]


@pytest.mark.skipif(not PY.exists() or not BENCH_REF.exists(), reason="axel venv or reference bench absent")
def test_note_b_fragile_success_is_treated_as_c_prime(tmp_path):
    import hashlib
    d = _chain_note_b(tmp_path / "a")
    b2 = json.loads((d / "bench1.json").read_text())                          # 0.4 point everywhere, a large gain on the 3 closing pages
    for r in b2["chains"]:
        if r["chain"] == "c1+ma":
            r["cer_reordered"] -= 0.004 + (0.2 if r["page"] in D.NOTE_B_CLOSING else 0)
    (d / "bench2.json").write_text(json.dumps(b2))
    _relink(d, 2, bench_sha256=hashlib.sha256((d / "bench2.json").read_bytes()).hexdigest())
    rc, rep = _delta(d)
    assert rc == 0 and rep["outcome"].startswith("(a)") and rep["sensitivity_note_b"]["outcome"].startswith("(b)") and rep["fragile"] is True
    assert rep["decision_note_b"].startswith("(a) FRAGILE traité comme (c′)")
    rc, rep = _delta(_chain_note_b(tmp_path / "b"))                            # concordant (a): no such decision
    assert rc == 0 and "decision_note_b" not in rep
