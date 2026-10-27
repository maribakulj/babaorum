"""Load the pinned PERO OCR model from audited bytes without staging weights.

The native PERO constructor accepts a JSON path and ultimately calls
``torch.jit.load(checkpoint_path)``. This adapter gives it an exact private
copy of the small JSON config, while intercepting that one native load call
and supplying the already-hashed checkpoint bytes through ``BytesIO``.
Import of PERO/Torch is delayed until the explicit real-engine entry point.
"""
from __future__ import annotations

import hashlib
import importlib
import inspect
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from ..contracts import ContractError, require
from .pero import _checkpoint_from_config, _preflight_real, _code_revision

def _snapshot_config_checkpoint_path(config_bytes: bytes, config_path: Path) -> Path:
    """Resolve the checkpoint path the pinned BaseEngine will compute."""
    try:
        config = json.loads(config_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"invalid PERO config snapshot: {exc}") from exc
    require(isinstance(config, dict) and isinstance(config.get("checkpoint"), str)
            and bool(config["checkpoint"]), "PERO config snapshot has no checkpoint path")
    path = Path(config["checkpoint"])
    if not path.is_absolute():
        path = Path(os.path.realpath(config_path.parent / path))
    return Path(str(path) + ".cpu").resolve()


def _construct_from_bytes(*, engine_base_class, torch_module,
                          config_bytes: bytes, checkpoint_bytes: bytes,
                          config_sha256: str, checkpoint_sha256: str,
                          source_config_path: Path, original_checkpoint: Path, settings: dict,
                          expected: dict):
    """Private injection seam for byte-only tests; production uses pinned PERO."""
    require(hashlib.sha256(config_bytes).hexdigest() == config_sha256,
            "PERO config snapshot hash mismatch")
    require(hashlib.sha256(checkpoint_bytes).hexdigest() == checkpoint_sha256,
            "PERO checkpoint snapshot hash mismatch")
    require(settings.get("device") == "cpu", "PERO snapshot loader is CPU-only")
    require(_checkpoint_from_config(config_bytes, source_config_path, settings)
            == original_checkpoint.resolve(),
            "PERO config snapshot does not resolve to the captured checkpoint")

    with tempfile.TemporaryDirectory(prefix="babaorum-pero-") as temp_name:
        temp_root = Path(temp_name)
        os.chmod(temp_root, 0o700)
        config_file = temp_root / "ocr_engine.json"
        fd = os.open(config_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(config_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        require(config_file.read_bytes() == config_bytes,
                "PERO private config snapshot differs from captured bytes")
        expected_load_path = _snapshot_config_checkpoint_path(config_bytes, config_file)
        calls = []

        class SnapshotEngine(engine_base_class):
            def _load_exported_model(self):
                require(getattr(self.device, "type", None) == "cpu",
                        "PERO TorchScript load did not explicitly target CPU")
                # Mirror pinned PytorchEngineLineOCR, while routing its one
                # model load through the immutable bytes captured above.
                self.checkpoint += ".cpu"
                require(Path(self.checkpoint).resolve() == expected_load_path,
                        "PERO native constructor requested a checkpoint outside the snapshot mapping")
                require(not calls, "PERO native constructor attempted multiple checkpoint loads")
                calls.append(hashlib.sha256(checkpoint_bytes).hexdigest())
                stream = io.BytesIO(checkpoint_bytes)
                self.model = torch_module.jit.load(stream, map_location=self.device)
                require(stream.tell() >= 0, "invalid snapshot stream state")
                self.model = self.model.to(self.device)

        engine = SnapshotEngine(str(config_file), torch_module.device("cpu"),
                                batch_size=settings["batch_size"])
        require(len(calls) == 1 and calls[0] == checkpoint_sha256,
                "PERO constructor did not load exactly the audited checkpoint bytes")
        parsed_config = json.loads(config_bytes)
        require(getattr(engine, "config", None) == parsed_config,
                "PERO engine config differs from audited config snapshot")
        for attr, config_key in (("line_px_height", "line_px_height"),
                                 ("line_vertical_scale", "line_vertical_scale"),
                                 ("net_name", "net_name")):
            if config_key in parsed_config:
                require(getattr(engine, attr, None) == parsed_config[config_key],
                        f"PERO derived {attr} differs from audited config")
        expected_chars = list(parsed_config["characters"]) + ["\u200b"]
        require(getattr(engine, "characters", None) == expected_chars,
                "PERO loaded alphabet differs from audited config")
        require(Path(engine.checkpoint).resolve() == expected_load_path,
                "PERO loaded checkpoint path differs from private config resolution")
        require(getattr(getattr(engine, "device", None), "type", None) == "cpu",
                "PERO snapshot engine is not on CPU")
        require(getattr(engine, "batch_size", None) == settings["batch_size"],
                "PERO snapshot engine batch size differs from frozen settings")
        # Upstream records the synthetic resolved path on the object. Replace
        # it with the identity of the captured artifact for the bridge API.
        engine.checkpoint = str(original_checkpoint.resolve())
        engine.babaorum_provenance = dict(expected)
        engine.babaorum_provenance["loader"] = "torchscript_bytes_v1"
        return engine


def _verify_upstream_source_files(repository: Path, engine_class) -> None:
    """Bind the dynamically imported constructor and base implementation to HEAD."""
    seen = set()
    for cls in (engine_class, *engine_class.__mro__[1:]):
        if cls is object or cls in seen:
            continue
        seen.add(cls)
        try:
            source = Path(inspect.getfile(cls)).resolve()
            relative = source.relative_to(repository).as_posix()
            committed = subprocess.run(
                ["git", "-C", str(repository), "show", f"HEAD:{relative}"],
                check=True, capture_output=True,
            ).stdout
            actual = source.read_bytes()
        except (OSError, ValueError, subprocess.CalledProcessError, TypeError) as exc:
            raise ContractError(f"cannot verify pinned PERO source file for {cls.__name__}: {exc}") from exc
        require(actual == committed,
                f"PERO imported source differs from pinned checkout: {relative}")


def build_engine_from_snapshot(settings: dict, model: dict):
    """Instantiate pinned PERO OCR on CPU from captured, hash-verified bytes.

    This function imports Torch/PERO only when called. It does not write the
    checkpoint to disk. The checkpoint bytes passed to ``torch.jit.load`` are
    exactly the bytes whose SHA-256 matches the audited settings.
    """
    expected = _preflight_real(settings, model)
    config_path = Path(settings["config_path"])
    checkpoint_path = Path(settings["checkpoint_path"])
    config_bytes = config_path.read_bytes()
    checkpoint_bytes = checkpoint_path.read_bytes()
    require(hashlib.sha256(config_bytes).hexdigest() == expected["config_sha256"],
            "PERO config changed while capturing snapshot")
    require(hashlib.sha256(checkpoint_bytes).hexdigest() == expected["checkpoint_sha256"],
            "PERO checkpoint changed while capturing snapshot")
    require(_checkpoint_from_config(config_bytes, config_path, settings) == checkpoint_path.resolve(),
            "PERO config does not resolve to the captured checkpoint")
    require(_code_revision(settings) == expected["code_revision"],
            "PERO source revision changed before construction")

    repository = Path(settings["local_source_repository"]).resolve()
    if str(repository) not in sys.path:
        sys.path.insert(0, str(repository))
    try:
        pero_module = importlib.import_module("pero_ocr.ocr_engine.pytorch_ocr_engine")
        engine_class = pero_module.PytorchEngineLineOCR
        class_path = Path(inspect.getfile(engine_class)).resolve()
        class_path.relative_to(repository)
    except (ImportError, AttributeError, OSError, TypeError, ValueError) as exc:
        raise ContractError(f"cannot import pinned PERO OCR constructor: {exc}") from exc
    require(_code_revision(settings) == expected["code_revision"],
            "PERO source revision changed while importing constructor")
    _verify_upstream_source_files(repository, engine_class)
    import torch

    engine = _construct_from_bytes(
        engine_base_class=engine_class,
        torch_module=torch,
        config_bytes=config_bytes,
        checkpoint_bytes=checkpoint_bytes,
        config_sha256=expected["config_sha256"],
        checkpoint_sha256=expected["checkpoint_sha256"],
        source_config_path=config_path,
        original_checkpoint=checkpoint_path,
        settings=settings,
        expected=expected,
    )
    require(_code_revision(settings) == expected["code_revision"],
            "PERO source revision changed during construction")
    require(engine_class is pero_module.PytorchEngineLineOCR,
            "PERO constructor identity changed during construction")
    return engine
