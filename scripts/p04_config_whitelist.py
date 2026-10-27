"""AV-P04 v1.3 (decision of pA, 25/10): read ONLY the configuration keys of the fine-tuned blla that the legacy blla.segment needs, from the
safetensors HEADER (no weights loaded, no torch), with a strict WHITELIST. Nothing else is printed or written: no other key, no other key NAME,
no count of other keys (the file may hold validation metrics: `accuracy`, `metrics`, ...; reading them would reveal results before the measure).
The SHA-256 of the file is checked BEFORE and AFTER, and must equal the 18/10 archive (p04_report.json). Output: ONE small JSON (refused if it
exists). Exit 1 on any check failure, with a message that names no metadata key outside the whitelist.
    python p04_config_whitelist.py --model <best_0.6432.safetensors> --expected-sha <64 hex> --out p04_config.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

# keys of kraken_meta read by kraken 7.1.1 blla.segment (user_metadata): class_mapping (baselines, regions), topline, bounding_regions,
# hyper_params (only its 'padding'); plus model identity needed to check it is a segmentation model.
WHITELIST = ("class_mapping", "topline", "bounding_regions", "model_type", "_model", "one_channel_mode")
HYPER_PARAMS_SUBKEYS = ("padding",)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def header_meta(p: Path) -> dict:
    with p.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        if n <= 0 or n > 50_000_000:
            raise SystemExit("unexpected safetensors header length")
        hdr = json.loads(f.read(n))
    meta = hdr.get("__metadata__")
    if not isinstance(meta, dict) or "kraken_meta" not in meta:
        raise SystemExit("no kraken_meta in the safetensors metadata")
    km = json.loads(meta["kraken_meta"])
    if not isinstance(km, dict) or len(km) != 1:
        raise SystemExit("kraken_meta does not hold exactly one model")
    (entry,) = km.values()
    if not isinstance(entry, dict):
        raise SystemExit("kraken_meta entry is not a mapping")
    return entry


def _short_str(v) -> bool:
    return isinstance(v, str) and 0 < len(v) <= 40


def schema_ok(config: dict, hyper: dict) -> list[str]:
    """Strict schema of every whitelisted value (Sonnet review): a value outside it is NOT printed; only the KEY (whitelisted) is named."""
    bad = []
    cm = config.get("class_mapping")
    if cm is not None:
        if not (isinstance(cm, dict) and set(cm) <= {"aux", "baselines", "regions"}
                and all(isinstance(d, dict) and len(d) <= 64 and all(_short_str(k) and isinstance(v, int) and not isinstance(v, bool)
                                                                        for k, v in d.items()) for d in cm.values())):
            bad.append("class_mapping")
    mt = config.get("model_type")
    if mt is not None and not (isinstance(mt, list) and 0 < len(mt) <= 4 and all(_short_str(x) for x in mt)):
        bad.append("model_type")
    if "_model" in config and not _short_str(config["_model"]):
        bad.append("_model")
    if "topline" in config and not (config["topline"] is None or type(config["topline"]) is bool):   # strict: not 0/1/0.0 (pQ M1)
        bad.append("topline")
    if "one_channel_mode" in config and config["one_channel_mode"] is not None and not _short_str(config["one_channel_mode"]):
        bad.append("one_channel_mode")
    br = config.get("bounding_regions")
    if br is not None and not (isinstance(br, list) and len(br) <= 64 and all(_short_str(x) for x in br)):
        bad.append("bounding_regions")
    if "padding" in hyper:
        pd = hyper["padding"]
        if not (isinstance(pd, (list, tuple)) and len(pd) == 2 and all(isinstance(x, int) and not isinstance(x, bool) and 0 <= x <= 1000 for x in pd)):
            bad.append("hyper_params.padding")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--expected-sha", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    model, out = Path(a.model), Path(a.out)
    if out.exists():
        raise SystemExit(f"--out exists, refused: {out}")
    before = sha256(model)
    if before != a.expected_sha:
        print(f"SHA BEFORE differs: {before} != {a.expected_sha}", file=sys.stderr)
        return 1
    entry = header_meta(model)
    config = {k: entry[k] for k in WHITELIST if k in entry}
    absent = [k for k in WHITELIST if k not in entry]
    # presence = the KEY exists, as the loader defines absence (pQ B1); a present non-mapping value is a STOP, never "absent"
    hp_present = "hyper_params" in entry
    hp = entry.get("hyper_params")
    if hp_present and not isinstance(hp, dict):
        del entry, hp
        print("hyper_params is present but not a mapping: stop, value not printed, nothing written", file=sys.stderr)
        return 1
    hyper = {"present": hp_present}
    if hp_present:
        hyper.update({s: hp[s] for s in HYPER_PARAMS_SUBKEYS if s in hp})      # an empty mapping is PRESENT
    del entry, hp                                                                  # nothing else survives
    bad = schema_ok(config, hyper)
    if bad:
        print(f"whitelisted value(s) outside the expected schema, not printed: {bad}", file=sys.stderr)
        return 1
    after = sha256(model)
    if after != before:
        print("SHA AFTER differs from BEFORE", file=sys.stderr)
        return 1
    report = {"note": "AV-P04 v1.3 : configuration en liste blanche, en-tête seulement, aucune autre clé lue ni écrite",
              "model_file": model.name, "sha256_before": before, "sha256_after": after, "expected_sha256": a.expected_sha,
              "whitelist": list(WHITELIST), "hyper_params_subkeys": list(HYPER_PARAMS_SUBKEYS),
              "config": config, "whitelist_keys_absent": absent, "hyper_params": hyper,
              "script_sha256": sha256(Path(__file__))}
    with open(out, "x", encoding="utf-8") as f:                                   # exclusive creation: no race, no overwrite
        f.write(json.dumps(report, ensure_ascii=False, indent=1, sort_keys=True))
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
