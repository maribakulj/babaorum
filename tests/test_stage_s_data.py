"""Checks for the stage S loader on two canonical Mille Feuilles pages (read in place)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from babaorum.experimental.stage_s_data import BLANK, _band, build_alphabet, load_mf_page

DATA = Path("/Users/marcel/heritage-synth/runs/accept-layout-v2/reproduced")
PAGE_JSON = DATA / "pages/mf_0000.json"
IMAGE = DATA / "images/mf_0000.png"

if not (PAGE_JSON.exists() and IMAGE.exists()):
    pytest.skip(f"canonical page not found under {DATA}", allow_module_level=True)

PAGE = load_mf_page(PAGE_JSON, IMAGE)


def test_line_count_matches_canonical_json():
    assert len(PAGE["lines"]) == len(json.loads(PAGE_JSON.read_text())["lines"])


def test_known_line_band_matches_polygon_geometry():
    line = next(x for x in PAGE["lines"] if x["id"] == "mf_0000_l00000")
    assert line["baseline"][0][1] == pytest.approx(79, abs=1)
    assert line["normal_negative_px"] == pytest.approx(37, abs=1)
    assert line["normal_positive_px"] == pytest.approx(12, abs=1)


def test_all_distances_are_nonnegative_and_not_all_zero():
    negs = [x["normal_negative_px"] for x in PAGE["lines"]]
    poss = [x["normal_positive_px"] for x in PAGE["lines"]]
    assert all(v >= 0 for v in negs + poss)
    assert sum(negs) + sum(poss) > 0


def test_alphabet_has_no_duplicates_and_blank_at_index_zero():
    alphabet = build_alphabet([PAGE])
    assert alphabet[0] == BLANK
    assert len(alphabet) == len(set(alphabet))
    assert alphabet[1:] == sorted(alphabet[1:])


def test_image_matches_canonical_dimensions():
    meta = json.loads(PAGE_JSON.read_text())["image"]
    assert PAGE["image"].shape == (meta["height"], meta["width"])
    assert PAGE["image"].dtype.name == "uint8"


def test_right_to_left_chord_is_refused():
    with pytest.raises(ValueError, match="right-to-left"):
        _band([[10.0, 0.0], [0.0, 0.0]], [[0.0, -1.0], [10.0, 1.0]])
