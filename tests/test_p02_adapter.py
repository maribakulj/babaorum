"""LOT-P02 adapter: round trip local -> page, geometric ids, median height, and no use of text or confidence for the choice."""
from __future__ import annotations

import random

import pytest

from babaorum.p02_adapter import baseline_length, geometric_id, line_height, rows_to_proposals

PAGE = {"page_key": "p", "document_key": "d", "width": 5000, "height": 7000, "image_sha256": "a" * 64}
ZC = {"x0": 100, "y0": 200, "x1": 1100, "y1": 700, "marge_px": 16, "canvas_h": 600, "input_h": 600, "complete_de_blanc": True, "facteur_echelle": 1.0}
ZC2 = {"x0": 100, "y0": 200, "x1": 1100, "y1": 3000, "marge_px": 16, "canvas_h": 2800, "input_h": 2400, "complete_de_blanc": False, "facteur_echelle": 2400 / 2800}


def row(i, bl, zone="z0", crop=ZC, text="t"):
    poly = [[bl[0][0], bl[0][1] - 20], [bl[-1][0], bl[-1][1] - 20], [bl[-1][0], bl[-1][1] + 10], [bl[0][0], bl[0][1] + 10]]
    return {"origin": "zone", "zone_selected_by_select_blocks": True, "zone_id": zone, "line_id": f"l{i}", "baseline": bl, "polygon": poly,
            "zone_crop": crop, "text": text, "conf": [0.5]}


def test_local_to_page_gives_back_the_page_baseline_for_both_crop_kinds():
    rows = [row(1, [[120.0, 300.0], [900.0, 305.5]]), row(2, [[130.0, 1500.0], [1000.0, 1490.25]], zone="z1", crop=ZC2)]
    props, row_of, h, skipped = rows_to_proposals(rows, PAGE)
    assert not skipped and len(props) == 2 and h > 0
    for p in props:
        back = [p.crop_transform.local_to_page(q) for q in p.baseline_xy]
        original = row_of[p.proposal_id]["baseline"]
        assert all(abs(a[0] - b[0]) < 1e-9 and abs(a[1] - b[1]) < 1e-9 for a, b in zip(back, original))


def test_ids_are_geometric_and_do_not_depend_on_the_input_order_or_on_text():
    rows = [row(i, [[100.0 + 10 * i, 300.0 + 40 * i], [900.0, 300.0 + 40 * i]], text=f"text {i}") for i in range(6)]
    a, _, _, _ = rows_to_proposals(rows, PAGE)
    shuffled = rows[:]
    random.Random(1).shuffle(shuffled)
    b, _, _, _ = rows_to_proposals([dict(r, text="other", conf=[0.99]) for r in shuffled], PAGE)
    assert sorted(p.proposal_id for p in a) == sorted(p.proposal_id for p in b)
    assert geometric_id(rows[0]).startswith("z0:") and len(geometric_id(rows[0]).split(":")) == 4


def test_longer_baselines_sort_first_and_a_collision_appends_the_blla_rank():
    long_, short = row(1, [[100.0, 300.0], [900.0, 300.0]]), row(2, [[100.0, 300.0], [200.0, 300.0]])
    assert geometric_id(long_) < geometric_id(short)
    twin_a, twin_b = row(3, [[100.0, 800.0], [500.0, 800.0]]), row(4, [[100.0, 800.0], [500.0, 800.0]])
    props, _, _, _ = rows_to_proposals([twin_a, twin_b], PAGE)
    ids = sorted(p.proposal_id for p in props)
    assert len(set(ids)) == 2 and ids[0].endswith(":0001") and ids[1].endswith(":0002")


def test_lines_without_geometry_are_reported_not_dropped_silently():
    ok = row(1, [[100.0, 300.0], [900.0, 300.0]])
    zero = row(2, [[100.0, 300.0], [100.0, 300.0]])
    nobase = dict(row(3, [[1.0, 1.0], [2.0, 2.0]]), baseline=[])
    props, _, _, skipped = rows_to_proposals([ok, zero, nobase], PAGE)
    assert len(props) == 1 and {s["line_id"] for s in skipped} == {"l2", "l3"}


def test_only_zone_lines_kept_by_select_blocks_enter():
    rows = [row(1, [[100.0, 300.0], [900.0, 300.0]]), dict(row(2, [[100.0, 400.0], [900.0, 400.0]]), zone_selected_by_select_blocks=False),
            dict(row(3, [[100.0, 500.0], [900.0, 500.0]]), origin="rest")]
    props, _, _, _ = rows_to_proposals(rows, PAGE)
    assert len(props) == 1


def test_line_height_matches_the_axel_definition():
    poly = [[0, 0], [100, 0], [100, 30], [0, 30]]
    assert line_height(poly, [[0, 25], [100, 25]]) == pytest.approx(30.0)
    assert line_height(poly, []) == pytest.approx(30.0)
    assert baseline_length([[0, 0], [3, 4]]) == pytest.approx(5.0)


# --- v15: collisions are broken by the geometry (and exact duplicates by a persistent source rank), never by the order of reception ---

def collision_pair():
    """pQ's counter-example: same length, same minimum x and y, same polygon; different baselines (they cross)."""
    poly = [[100, 80], [200, 80], [200, 110], [100, 110]]
    a = {"origin": "zone", "zone_selected_by_select_blocks": True, "zone_id": "z0", "line_id": "a", "baseline": [[100, 100], [200, 101]],
         "polygon": poly, "zone_crop": ZC, "text": "", "conf": [0.5]}
    b = dict(a, line_id="b", baseline=[[100, 101], [200, 100]])
    return a, b


def association(rows):
    props, row_of, _, _ = rows_to_proposals(rows, PAGE)
    return {row_of[p.proposal_id]["line_id"]: p.proposal_id for p in props}


def test_a_key_collision_gets_the_same_identifiers_in_both_orders():
    a, b = collision_pair()
    assert association([a, b]) == association([b, a])
    ids = association([a, b])
    assert len(set(ids.values())) == 2 and sorted(ids.values())[0].endswith(":0001") and sorted(ids.values())[1].endswith(":0002")


def test_the_suffix_follows_the_whole_geometry_not_the_reception_order():
    a, b = collision_pair()
    assert association([a, b])["a"].endswith(":0001") and association([b, a])["a"].endswith(":0001")


def test_an_exact_duplicate_is_broken_by_the_persistent_source_rank_when_it_is_present():
    a, _ = collision_pair()
    x = dict(a, line_id="x", _src=2)   # v16: the source rank is an integer (file index)
    y = dict(a, line_id="y", _src=1)
    assert association([x, y]) == association([y, x])
    assert association([x, y])["y"].endswith(":0001") and association([x, y])["x"].endswith(":0002")


def test_an_exact_duplicate_without_a_source_rank_is_broken_by_the_line_id():
    a, _ = collision_pair()
    x, y = dict(a, line_id="u-2"), dict(a, line_id="u-1")
    assert association([x, y]) == association([y, x])
    assert association([x, y])["u-1"].endswith(":0001") and association([x, y])["u-2"].endswith(":0002")


def test_the_adapter_association_is_stable_under_any_permutation_of_a_page_with_collisions():
    rng = random.Random(7)
    rows = []
    for i in range(10):
        bl = [[100.0 + (i % 3), 300.0 + 40 * i], [900.0, 300.0 + 40 * i + (i % 2)]]
        rows.append({"origin": "zone", "zone_selected_by_select_blocks": True, "zone_id": f"z{i % 2}", "line_id": f"l{i}", "baseline": bl,
                     "polygon": [[100, 280 + 40 * i], [900, 280 + 40 * i], [900, 310 + 40 * i], [100, 310 + 40 * i]], "zone_crop": ZC, "text": ""})
    a, b = collision_pair()
    rows += [dict(a, line_id="ca"), dict(b, line_id="cb")]
    reference = association(rows)
    for _ in range(10):
        shuffled = rows[:]
        rng.shuffle(shuffled)
        assert association(shuffled) == reference


def test_source_rank_from_lines_follows_the_file_order_block_after_block(tmp_path):
    import json

    from babaorum.p02_adapter import source_rank_from_lines

    f = tmp_path / "p.json"
    f.write_text(json.dumps({"blocks": [{"id": "b1", "lines": [{"id": "x"}, {"id": "y"}]}, {"id": "b0", "lines": [{"id": "a"}]}]}))
    assert source_rank_from_lines(f) == {"x": 0, "y": 1, "a": 2}   # file order, not id order


def test_source_key_compares_integers():
    from babaorum.p02_adapter import source_key

    assert source_key({"_src": 2, "line_id": "zz"}) < source_key({"_src": 10, "line_id": "aa"})
    assert source_key({"line_id": "a"}) > source_key({"_src": 10 ** 6, "line_id": "a"})
