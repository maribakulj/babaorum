"""G0-03 test fixture: the anchored registry v2.1 with EVERY ``possible`` collection's number key declared complete (n/n).

Built in memory only: nothing is written under config/. Used to exercise each defence alone (the real registry refuses everything through
INCOMPLETE_NUMBER_KEY). A collection with no number key gets an empty key set with coverage 1/1, as in p02_measure.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

from babaorum.g0_collections import _key_values

REGISTRY = Path(__file__).resolve().parents[2] / "config" / "g0_collections_v2_2.json"


def complete_registry() -> dict:
    reg = json.loads(REGISTRY.read_text(encoding="utf-8"))
    for name, entry in reg["collections"].items():
        if entry.get("gallica_scope", {}).get("value") != "possible":
            continue
        spec = entry["match_keys"]["issue"]
        if spec == "aucune clé à ce niveau":
            spec = entry["match_keys"]["issue"] = {"name": "issue", "values": []}
        n = len(_key_values(spec) or ())
        spec["coverage"] = {"keyed": max(n, 1), "total": max(n, 1)}
    return copy.deepcopy(reg)
