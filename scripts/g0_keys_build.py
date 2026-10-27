"""Thin wrapper: the key recipes now live in src/babaorum/g0_keys.py (LOT-G0-02). Prints the coverage per level."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from babaorum.g0_keys import *  # noqa: F401,F403,E402
from babaorum.g0_keys import LEVELS, compute_keys  # noqa: E402

if __name__ == "__main__":
    for name, (keys, reasons) in compute_keys().items():
        print(f"{name:26s}", {lv: (f"{len(keys[lv]['values'])} clés ({keys[lv]['coverage']['keyed']}/{keys[lv]['coverage']['total']})"
                                   if keys[lv] else "AUCUNE") for lv in LEVELS})
