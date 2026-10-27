"""AV-P04 (pQ B4): the bench of ONE pass on the Mac, bound by SHA to the pass report, the repatriated runs archive and the frozen Mac truth.
Writes rappel_pages/runs_26_p04_passe<k>/ (refused if it exists), the bench JSON (--out, refused if it exists) and a LINK JSON next to it.

Checks, each a stop (exit 1, nothing benched): free disk >= 1.5 GB (pA); the report is pass <k>, not incomplete; sha256(archive) equals the
report's runs_archive_sha256; the archive holds only runs/{e12,e22,e28,e32}/... and read_rest_all/..., no XML, no absolute or '..' member;
the 26 truth XML (paths of runs_26/runs/e12/pages.json, 9fd1b347…) match the frozen truth manifest 7f2b90e2…; the bench axel worktree is at
the pass's axel commit and CLEAN (git status --porcelain empty: tracked and untracked). Then: extraction, runs/e12/pages.json = the runs_26
page list (truth paths, Mac only), bench_pipeline.py with the frozen config (c8216a7e), chain c1+ma, --no-controls.
Producers (pQ B4, 2nd audit): bench_pipeline.py runs under an audit hook that records EVERY file opened for reading under the babaorum repo
and the axel worktree (configs...; .pyc excluded) plus the source of every imported module under them (sys.modules at exit), plus every file of
src/babaorum/schemas (the XSD are read by libxml2 in C, unseen by the hook). The link carries their
sha256 ({"babaorum:<rel>" | "axel:<rel>": sha}) and producers_sha256 = sha256 of that map in canonical JSON; Δ requires the same map in
both passes.
    PYTHONDONTWRITEBYTECODE=1 ~/axel/.venv/bin/python scripts/p04m_bench.py --passe 1 --report p04m_passe1_report.json \
        --archive p04m_passe1_runs.tar.gz --axel <worktree at c62abf9> --out reports/bench_p04m_passe1.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RP = Path.home() / "Documents/reviews/babaorum-pipeline/rappel_pages"
REF_PAGES = RP / "runs_26/runs/e12/pages.json"
REF_PAGES_SHA = "9fd1b347c0c918ab6a2cc08c274d25aa4b02dc1c9b54f9ffd5a1919a1fe1e0a9"
TRUTH_MANIFEST_SHA = "7f2b90e23611750e02777b8dc84877c664bb6062687e65b32ac468ede7f0b16d"
CONFIG = ROOT / "config/bench_pipeline_v1.json"
CONFIG_SHA = "c8216a7ea475b5abc3ce1e4917e06e44288ec906d226a437bb65a2019f034389"
MIN_FREE = 1.5e9


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


# run under the audit hook: argv = [reads.json, babaorum root, axel root, script, *script args]; records files opened for READING under both roots
TRACE = r'''
import atexit, json, os, runpy, sys
out, roots, script = sys.argv[1], {"babaorum": os.path.realpath(sys.argv[2]), "axel": os.path.realpath(sys.argv[3])}, sys.argv[4]
seen = {k: set() for k in roots}
def hook(ev, args):
    if ev != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    mode = args[1] if len(args) > 1 and isinstance(args[1], str) else "r"
    if any(c in mode for c in "wax+"):
        return
    p = os.path.realpath(os.fsdecode(args[0]))
    if p.endswith(".pyc") or not os.path.isfile(p):
        return
    for k, r in roots.items():
        if p.startswith(r + os.sep):
            seen[k].add(os.path.relpath(p, r))
            return
def dump():
    for m in list(sys.modules.values()):                       # imports served from a valid .pyc open no source: count the module source
        f = getattr(m, "__file__", None)
        if f and f.endswith(".py"):
            hook("open", (f, "r"))
    open(out, "w").write(json.dumps({k: sorted(v) for k, v in seen.items()}))
atexit.register(dump)
sys.addaudithook(hook)
sys.argv = sys.argv[4:]
runpy.run_path(script, run_name="__main__")
'''


# a link without these is refused here and in Δ
REQUIRED_PRODUCERS = ("babaorum:scripts/bench_pipeline.py", "babaorum:config/bench_pipeline_v1.json", "babaorum:src/babaorum/schemas/alto-4-4.xsd",
                      "babaorum:src/babaorum/schemas/page-2019-07-15.xsd", "babaorum:src/babaorum/schemas/xlink.xsd")


def bench_env() -> list:
    """The interpreter and installed distributions that ran the bench (outside the two roots; Sonnet point 3): Δ requires the same list."""
    from importlib import metadata
    return [f"python=={sys.version.split()[0]}", f"executable=={sys.executable}"] + sorted(
        {f"{d.metadata['Name'].lower()}=={d.version}" for d in metadata.distributions() if d.metadata["Name"]})


def producers_sha(producers: dict) -> str:
    return hashlib.sha256(json.dumps(producers, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def truth_manifest() -> str:
    pages = json.loads(REF_PAGES.read_text())
    lines = sorted(f"{sha(Path(e['xml']))}  {e['page']}" for e in pages)
    return hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--passe", type=int, choices=(1, 2), required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--archive", required=True)
    ap.add_argument("--axel", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out, link = Path(a.out), Path(a.out).with_suffix(".link.json")
    dest = RP / f"runs_26_p04_passe{a.passe}"
    for p in (out, link, dest):
        if p.exists():
            raise SystemExit(f"exists, refused: {p}")
    if shutil.disk_usage(Path.home()).free < MIN_FREE:
        raise SystemExit("free disk < 1.5 GB: stop (pA)")
    report_p, archive_p = Path(a.report), Path(a.archive)
    rep = json.loads(report_p.read_text())
    if rep.get("passe") != a.passe or rep.get("incomplete") != []:
        raise SystemExit("report: wrong pass or incomplete")
    if sha(archive_p) != rep.get("runs_archive_sha256"):
        raise SystemExit("archive sha differs from the report")
    with tarfile.open(archive_p) as t:
        names = t.getnames()
        bad = [n for n in names if n.startswith("/") or ".." in Path(n).parts or n.lower().endswith(".xml")
               or not (n.split("/")[0] == "read_rest_all" or (n.split("/")[0] == "runs" and (len(n.split("/")) < 2 or n.split("/")[1] in ("e12", "e22", "e28", "e32"))))]
        if bad:
            raise SystemExit(f"archive members outside the expected layout: {bad[:3]}")
    if sha(REF_PAGES) != REF_PAGES_SHA or truth_manifest() != TRUTH_MANIFEST_SHA:
        raise SystemExit("Mac truth differs from the frozen manifest")
    if sha(CONFIG) != CONFIG_SHA:
        raise SystemExit("bench config differs from c8216a7e")
    head = subprocess.run(["git", "-C", a.axel, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    if not head.startswith(str(rep.get("axel_commit", ""))[:7]) or not str(rep.get("axel_commit", "")).startswith(head[:7]):
        raise SystemExit("bench axel worktree is not at the pass's axel commit")
    dirty = subprocess.run(["git", "-C", a.axel, "status", "--porcelain"], capture_output=True, text=True, check=True).stdout
    if dirty.strip():
        raise SystemExit(f"bench axel worktree is not clean: {dirty.splitlines()[:3]}")
    dest.mkdir(parents=True)
    with tarfile.open(archive_p) as t:
        t.extractall(dest)
    shutil.copy2(REF_PAGES, dest / "runs/e12/pages.json")                         # truth paths, on the Mac only
    env = dict(os.environ, AXEL_RUNS=str((dest / "runs").resolve()), PYTHONDONTWRITEBYTECODE="1")
    reads = dest / "producers_reads.json"
    r = subprocess.run([sys.executable, "-c", TRACE, str(reads), str(ROOT), str(Path(a.axel).resolve()), str(ROOT / "scripts/bench_pipeline.py"),
                        "--axel", a.axel, "--config", str(CONFIG), "--chains", "c1+ma", "--no-controls", "--label", f"AV-P04 passe {a.passe}",
                        "--out", str(out)], env=env, capture_output=True, text=True)
    if r.returncode:
        print(r.stdout[-2000:], r.stderr[-2000:])
        raise SystemExit("bench_pipeline failed")
    producers = {}
    for tag, base in (("babaorum", ROOT.resolve()), ("axel", Path(a.axel).resolve())):
        for rel in json.loads(reads.read_text())[tag]:
            producers[f"{tag}:{rel}"] = sha(base / rel)
    for x in sorted((ROOT / "src/babaorum/schemas").iterdir()):                 # read by libxml2 in C: invisible to the audit hook (Sonnet review)
        if x.is_file():
            producers[f"babaorum:{x.relative_to(ROOT).as_posix()}"] = sha(x)
    missing = [k for k in REQUIRED_PRODUCERS if k not in producers]
    if missing or not any(k.startswith("axel:src/") for k in producers):
        raise SystemExit(f"producer trace incomplete: {missing or 'no axel source'}")
    bench = json.loads(out.read_text())
    if bench["provenance"].get("config_sha256") != CONFIG_SHA or not str(bench["provenance"].get("axel_head", "")).startswith(head[:7]):
        raise SystemExit("bench provenance does not match")
    lk = {"passe": a.passe, "report_sha256": sha(report_p), "archive_sha256": sha(archive_p), "bench_sha256": sha(out),
          "truth_manifest_sha256": TRUTH_MANIFEST_SHA, "pages_json_sha256": REF_PAGES_SHA, "config_sha256": CONFIG_SHA, "axel_head": head,
          "runs_dir": str(dest), "script_sha256": sha(Path(__file__)), "axel_clean": True, "producers": producers,
          "producers_sha256": producers_sha(producers), "bench_env": bench_env()}
    with open(link, "x", encoding="utf-8") as f:
        f.write(json.dumps(lk, indent=1))
    print(json.dumps(lk))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
