"""Check this source edition and its reachable history for excluded artifacts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_TOP = {"reports", "runs", "data", "corpus", "outputs", "checkpoints", "provenance"}
FIXTURES = {
    "empty_truth.json", "line.json", "pero_gray.json", "pero_red.json", "pero_long.json",
    "pero_gray.png", "pero_red.png", "pero_long.png", "g0_registry_complete.py",
}
CREDENTIALS = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})\b"),
    re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b"),
    re.compile(r"\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{35,}\b"),
    re.compile(r"\bhf_[A-Za-z0-9]{30,}\b"),
)


def path_allowed(name: str) -> bool:
    path = PurePosixPath(name)
    if not path.parts or path.parts[0] in FORBIDDEN_TOP:
        return False
    if name.startswith(("docs/reviews/", "docs/publication/", "docs/history/")):
        return False
    if path.suffix in {".pt", ".pth", ".safetensors", ".ckpt", ".gz", ".tar", ".zip", ".npz", ".npy"}:
        return False
    if name.startswith(("config/g0_keys/", "config/sealed/")):
        return False
    if name.startswith("config/g0_") and path.name not in {
        "g0_suivi_inputs_schema_v1.json", "g0_roots.local.example.json"
    }:
        return False
    if path.name in {"sealed_registry.json", "sealed_registry_v2.json"}:
        return False
    if name.startswith("tests/fixtures/") and path.name not in FIXTURES:
        return False
    if path.suffix == ".png" and name not in {f"tests/fixtures/{n}" for n in FIXTURES if n.endswith(".png")}:
        return False
    return not re.search(r"(^|/)mf_005[0-9]\.(json|png|jpg|xml)$", name)


def notebook_is_source_only(obj: dict) -> bool:
    return all(not c.get("outputs") and c.get("execution_count") is None for c in obj.get("cells", []))


def main() -> None:
    shallow = subprocess.check_output(
        ["git", "rev-parse", "--is-shallow-repository"], cwd=ROOT, text=True
    ).strip()
    if shallow != "false":
        raise SystemExit("PUBLIC_HISTORY_CHECK_REQUIRES_FULL_ANCESTRY")
    names = [n for n in subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0") if n]
    errors = []
    for name in names:
        if not path_allowed(name):
            errors.append("EXCLUDED_PATH:" + name)
            continue
        p = ROOT / name
        if p.is_symlink() or p.stat().st_size > 2 * 1024 * 1024:
            errors.append("UNEXPECTED_PAYLOAD:" + name)
            continue
        if p.suffix == ".png":
            continue  # only the explicit, generated fixture allowlist reaches here
        raw = p.read_bytes()
        try:
            text = raw.decode("utf-8")
        except UnicodeError:
            errors.append("NON_TEXT_PAYLOAD:" + name)
            continue
        if any(pattern.search(text) for pattern in CREDENTIALS):
            errors.append("CREDENTIAL_PATTERN:" + name)
        if p.suffix == ".ipynb" and not notebook_is_source_only(json.loads(text)):
            errors.append("NOTEBOOK_OUTPUT:" + name)

    manifest = json.loads((ROOT / "SOURCE-MANIFEST.json").read_text())
    for row in manifest["files"]:
        p = ROOT / row["path"]
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != row["sha256"]:
            errors.append("RETAINED_SOURCE_CHANGED:" + row["path"])

    for line in subprocess.check_output(["git", "rev-list", "--objects", "HEAD"], cwd=ROOT, text=True).splitlines():
        parts = line.split(" ", 1)
        if len(parts) == 2 and parts[1] and not path_allowed(parts[1]):
            errors.append("EXCLUDED_HISTORY_PATH:" + parts[1])
    if errors:
        raise SystemExit("\n".join(sorted(set(errors))))
    print(f"Source edition: {len(names)} tracked files checked; retained sources and reachable history conform.")


if __name__ == "__main__":
    main()
