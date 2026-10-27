"""Word boxes from CTC positions: conventions, no prorata, repairs declared (synthetic data only)."""
import pytest

from babaorum.words_from_ctc import (baseline_band, baseline_y, dedupe_polygon, distribution, line_extent,
                                      normalized, outside_share, repair_polygon, split_words, word_polygon,
                                      words_text)

TEXT = "ab cde f"
SPANS = [(10, 14), (16, 20), None, (30, 34), (36, 40), (42, 46), None, (60, 64)]
CONFS = [1.0, 0.5, 1.0, 0.9, 0.8, 0.7, 1.0, 0.6]


def test_words_are_runs_of_non_space_and_join_back_to_the_text():
    words = split_words(TEXT, SPANS, CONFS)
    assert [w.text for w in words] == ["ab", "cde", "f"]
    assert words_text(words) == TEXT == normalized(TEXT)


def test_x_extent_goes_from_the_first_to_the_last_cut_and_blank_cuts_are_ignored():
    words = split_words(TEXT, SPANS, CONFS)
    assert (words[0].x0, words[0].x1) == (10, 20)
    assert (words[1].x0, words[1].x1) == (30, 46)
    assert (words[2].x0, words[2].x1) == (60, 64)
    assert words[1].conf == pytest.approx((0.9 + 0.8 + 0.7) / 3)


def test_a_word_without_any_cut_has_no_coordinates_and_nothing_is_prorated():
    spans = [(10, 14), (16, 20), None, None, None, None, None, (60, 64)]
    words = split_words(TEXT, spans)
    assert words[1].has_box is False and words[1].x0 is None and words[1].x1 is None and words[1].cut_chars == 0
    assert words[1].conf is None
    # a line with no cuts at all: every word is uncut
    assert not any(w.has_box for w in split_words(TEXT, None))


def test_position_count_must_match_the_text():
    with pytest.raises(ValueError, match="positions"):
        split_words("abc", [(0, 1)])
    with pytest.raises(ValueError, match="confidences"):
        split_words("abc", None, [1.0])


def test_whitespace_runs_and_borders_are_normalized_for_the_round_trip():
    raw = "  ab   cd "
    words = split_words(raw)
    assert words_text(words) == normalized(raw) == "ab cd" and words_text(words) != raw


def test_baseline_height_is_interpolated_and_clamped():
    base = [(0.0, 10.0), (100.0, 20.0)]
    assert baseline_y(base, -5) == 10.0 and baseline_y(base, 50) == 15.0 and baseline_y(base, 500) == 20.0


def test_word_box_comes_from_cuts_baseline_and_height_not_from_the_polygon():
    base = [(0.0, 50.0), (100.0, 50.0)]
    words = split_words("ab", [(10, 14), (16, 20)])
    box, clamped = word_polygon(words[0], base, up=30.0, down=10.0, width=200, height=100)
    assert box == [(10, 20.0), (20, 20.0), (20, 60.0), (10, 60.0)] and clamped is False
    # outside the image: clamped and declared
    box, clamped = word_polygon(words[0], base, up=80.0, down=10.0, width=200, height=100)
    assert clamped is True and min(y for _, y in box) == 0.0
    assert word_polygon(split_words("ab")[0], base, 30.0, 10.0, 200, 100) == (None, False)  # no cut, no box


def test_outside_share_is_measured_not_corrected():
    line = [(0, 0), (100, 0), (100, 40), (0, 40)]
    inside = [(10, 10), (20, 10), (20, 30), (10, 30)]
    half_out = [(10, 20), (20, 20), (20, 60), (10, 60)]  # lower half below the polygon
    assert outside_share(inside, line) == 0.0
    assert outside_share(half_out, line) == pytest.approx(0.5, abs=0.1)


def test_distribution_reports_median_p95_and_count_over_ten_percent():
    d = distribution([0.0, 0.0, 0.05, 0.2, 0.5])
    assert d["n"] == 5 and d["median"] == 0.05 and d["over_10pct"] == 2 and d["p95"] == 0.5
    assert distribution([]) == {"n": 0, "median": None, "p95": None, "over_10pct": 0}


def test_line_extent_is_measured_around_the_mean_baseline():
    poly = [(0, 10), (100, 10), (100, 60), (0, 60)]
    assert line_extent(poly, [(0, 50), (100, 50)]) == (40.0, 10.0)


def valid_ring(points):
    """A stand-in validator: at least three distinct vertices, no repeated vertex, no repeated closing vertex."""
    return len(points) >= 3 and len(set(points)) == len(points) and points[0] != points[-1]


def test_dedupe_is_lossless_and_removes_a_repeated_closing_vertex():
    ring = [(0, 0), (0, 0), (10, 0), (10, 10), (0, 10), (0, 0), (0, 0)]
    assert dedupe_polygon(ring) == [(0, 0), (10, 0), (10, 10), (0, 10)]


def test_repair_declares_every_change_with_a_code():
    base = [(0.0, 50.0), (100.0, 50.0)]
    good = [(0, 0), (10, 0), (10, 10), (0, 10)]
    assert repair_polygon(good, valid_ring, base, 5, 5, 200, 100) == ([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)], None)
    dup = [(0, 0), (0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]
    fixed, code = repair_polygon(dup, valid_ring, base, 5, 5, 200, 100)
    assert code == "polygon_repaired" and fixed == [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    # a polygon that dedupe cannot save: replaced by a baseline band, declared
    two_points = [(0, 0), (5, 5)]
    band, code = repair_polygon(two_points, valid_ring, base, 20, 5, 200, 100)
    assert code == "polygon_from_baseline" and band is not None and valid_ring(band)
    # no baseline either: unrepairable, never silent
    assert repair_polygon(two_points, valid_ring, [], 20, 5, 200, 100) == (None, "polygon_unrepairable")


def test_baseline_band_follows_the_baseline_and_stays_in_the_image():
    band = baseline_band([(0.0, 5.0), (100.0, 5.0)], up=20.0, down=5.0, width=200, height=100)
    assert band is not None and min(y for _, y in band) == 0.0 and max(y for _, y in band) == 10.0
    assert baseline_band([(10.0, 5.0), (10.0, 5.0)], 1, 1, 100, 100) is None
