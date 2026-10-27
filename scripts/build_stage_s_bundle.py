#!/usr/bin/env python3
"""Build the stage S code bundle for Colab (v4 layout: git bundle of the mille-feuilles source).

The mille-feuilles source ships as a git bundle pinned at a commit (default 23875ab0). The
bundle is cloned into a working folder here, the commit and the clean status are checked, and
the sha256 of every tracked mille-feuilles file is taken from that clone. The zip contains:
  babaorum/        code closure, scripts, pre-registered configs
  heritage-synth.gitbundle
  heritage-synth/runs/pilot-v0.2/config.json   (untracked at this commit; hashed in the manifest)
  bundle_manifest.json                        (commit, clean flag, sha256 of every file)
The heritage-synth repository itself is not modified.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

BABAORUM = Path(__file__).resolve().parents[1]
HERITAGE = Path("/Users/marcel/heritage-synth")
DEFAULT_DEST = Path("/Users/marcel/Library/CloudStorage/GoogleDrive-mrebg@hotmail.fr/Mon Drive/babaorum/stage_s/bundle")
DEFAULT_GITBUNDLE = Path("/private/tmp/claude-501/-Users-marcel/9da7e7f4-8da8-4aaa-bd41-2d774ff8c429/scratchpad/heritage-synth.gitbundle")
PINNED_COMMIT = "23875ab05425ce2aa07742bf314d02436c2f5642"
MIN_FREE_MB = 50

BABAORUM_FILES = [
    "src/babaorum/__init__.py", "src/babaorum/line_geometry.py",
    "src/babaorum/experimental/__init__.py", "src/babaorum/experimental/joint_curves.py",
    "src/babaorum/experimental/joint_geometry.py", "src/babaorum/experimental/joint_reader_bridge.py",
    "src/babaorum/experimental/stage_s_arms.py", "src/babaorum/experimental/stage_s_data.py",
    "src/babaorum/experimental/stage_s_model.py", "src/babaorum/experimental/stage_s_reader.py",
    "src/babaorum/experimental/stage_s_targets.py",
    "scripts/train_stage_s.py", "scripts/train_stage_s_v2.py", "scripts/colab_stage_s_calibration.py",
    "config/stage_s_gpu_calibration.json", "config/stage_s_smoke_v2.json",
]
PILOT_CONFIG = HERITAGE / "runs/pilot-v0.2/config.json"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git(args: list[str], cwd: Path) -> str:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()


def clone_and_check(gitbundle: Path, workdir: Path) -> tuple[Path, str, bool]:
    if workdir.exists():
        shutil.rmtree(workdir)
    subprocess.run(["git", "clone", "-q", str(gitbundle), str(workdir)], check=True)
    subprocess.run(["git", "-C", str(workdir), "checkout", "-q", PINNED_COMMIT], check=True)
    head = git(["rev-parse", "HEAD"], workdir)
    clean = git(["status", "--porcelain"], workdir) == ""
    if head != PINNED_COMMIT:
        raise SystemExit(f"clone HEAD {head} != pinned {PINNED_COMMIT}")
    if not clean:
        raise SystemExit("clone working tree is not clean")
    return workdir, head, clean


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="stage_s_bundle_v4.zip")
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--gitbundle", type=Path, default=DEFAULT_GITBUNDLE)
    parser.add_argument("--workdir", type=Path, default=DEFAULT_GITBUNDLE.parent / "bundle-check-clone")
    args = parser.parse_args(argv)

    args.dest.mkdir(parents=True, exist_ok=True)
    free_mb = shutil.disk_usage(args.dest).free / 2 ** 20
    if free_mb < MIN_FREE_MB:
        raise SystemExit(f"refused: {free_mb:.0f} MB free on the destination, need at least {MIN_FREE_MB} MB")
    if not args.gitbundle.exists():
        raise SystemExit(f"missing git bundle: {args.gitbundle}")

    clone, head, clean = clone_and_check(args.gitbundle, args.workdir)
    tracked = [f for f in git(["ls-files"], clone).splitlines()
               if f.startswith(("src/", "schemas/", "assets/", "profiles/")) or f == "pyproject.toml"]
    mf_sha = {f"heritage-synth/{f}": sha256_bytes((clone / f).read_bytes()) for f in tracked}
    babaorum_entries = [(BABAORUM / rel, f"babaorum/{rel}") for rel in BABAORUM_FILES]
    missing = [str(src) for src, _ in babaorum_entries if not src.exists()]
    if missing or not PILOT_CONFIG.exists():
        raise SystemExit(f"missing bundle inputs: {missing[:5]} {PILOT_CONFIG if not PILOT_CONFIG.exists() else ''}")

    pilot_bytes = PILOT_CONFIG.read_bytes()
    manifest = {
        "schema_version": "stage_s_bundle_manifest_v2",
        "built_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "heritage_synth_git_commit": head,
        "heritage_synth_git_clean": clean,
        "heritage_synth_gitbundle": "heritage-synth.gitbundle",
        "heritage_synth_gitbundle_sha256": sha256_bytes(args.gitbundle.read_bytes()),
        "mille_feuilles_sha256": mf_sha,
        "pilot_config_sha256": {"heritage-synth/runs/pilot-v0.2/config.json": sha256_bytes(pilot_bytes)},
        "babaorum_sha256": {arc: sha256_bytes(src.read_bytes()) for src, arc in babaorum_entries},
    }
    dest = args.dest / args.name
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for src, arc in babaorum_entries:
            z.write(src, arc)
        z.write(args.gitbundle, "heritage-synth.gitbundle")
        z.writestr("heritage-synth/runs/pilot-v0.2/config.json", pilot_bytes)
        z.writestr("bundle_manifest.json", json.dumps(manifest, indent=1, sort_keys=True) + "\n")

    print(json.dumps({"zip": str(dest), "bytes": dest.stat().st_size,
                      "sha256": sha256_bytes(dest.read_bytes()),
                      "gitbundle_bytes": args.gitbundle.stat().st_size,
                      "heritage_synth_git_commit": head, "heritage_synth_git_clean": clean,
                      "mille_feuilles_files": len(mf_sha)}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
