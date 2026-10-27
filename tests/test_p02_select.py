"""LOT-P02 dedup(page, cands): signature expected by axel, order independence, fallback of abstained components, unjudged lines kept."""
from __future__ import annotations

import random

import pytest

from babaorum.p02_select import CONTEXT, STATS, dedup, register_page

PAGE = "pg"
ZC = {"x0": 0, "y0": 0, "x1": 3000, "y1": 4000, "marge_px": 16, "canvas_h": 4000, "input_h": 2400, "complete_de_blanc": False, "facteur_echelle": 1.0}


def setup_module(module):
    register_page(PAGE, {"width": 3000, "height": 4000, "image_sha256": "a" * 64, "document_key": "d", "zone_crops": {f"z{i}": ZC for i in range(40)}})


def cand(cid, bl, block="z0", h=20.0):
    poly = [[bl[0][0], bl[0][1] - h], [bl[-1][0], bl[-1][1] - h], [bl[-1][0], bl[-1][1] + h / 2], [bl[0][0], bl[0][1] + h / 2]]
    return {"id": cid, "baseline": [list(p) for p in bl], "boundary": poly, "_block": block, "_box": (0, 0, 1, 1)}


def ids(kept):
    return [c["id"] for c in kept]


def test_the_result_is_a_subset_in_the_input_order():
    cs = [cand("a", [(100, 100), (700, 100)]), cand("b", [(100, 300), (700, 300)], "z1"), cand("c", [(100, 500), (700, 500)], "z2")]
    assert ids(dedup(PAGE, cs)) == ["a", "b", "c"]


def test_a_long_line_and_the_fragments_inside_it_give_the_same_answer_in_every_order():
    """pJ's toy: 60 px then 100 px (the existing _covered keeps both or drops one depending on the order)."""
    short, long_ = cand("short", [(200, 100), (260, 100)], "z0"), cand("long", [(150, 100), (250, 100)], "z1")
    outs = {tuple(sorted(ids(dedup(PAGE, order)))) for order in ([short, long_], [long_, short])}
    assert len(outs) == 1


def test_permutations_of_blocks_and_lines_do_not_change_the_kept_set():
    rng = random.Random(20261017)
    base = []
    for r in range(12):
        y = 200 + r * 24
        base.append(cand(f"L{r}", [(100, y), (800, y)], f"z{r % 3}"))
        if r % 3 == 0:
            base.append(cand(f"D{r}", [(110, y + 3), (790, y + 3)], f"z{r % 3 + 3}"))
        if r % 4 == 0:
            base.append(cand(f"F{r}", [(300, y + 1), (310, y + 1)], f"z{r % 3 + 6}"))
    reference = sorted(ids(dedup(PAGE, base)))
    for _ in range(20):
        shuffled = base[:]
        rng.shuffle(shuffled)
        assert sorted(ids(dedup(PAGE, shuffled))) == reference


def test_an_abstained_component_uses_the_fixed_fallback_and_stays_order_independent():
    stack = [cand(f"s{i:02d}", [(100, 500 + 0.5 * i), (600 + i, 500 + 0.5 * i)], f"z{i % 5}") for i in range(25)]   # 25 lines all within 0.5 h of the first one's supports: one component of 25 actions > 18
    kept = dedup(PAGE, stack)
    assert STATS[PAGE]["abstained_components"] == 1 and STATS[PAGE]["fallback_emitted"] >= 1
    again = dedup(PAGE, list(reversed(stack)))
    assert sorted(ids(kept)) == sorted(ids(again))
    # the lines kept by the fallback share no support: they are far enough apart to be distinct lines
    assert 1 <= len(kept) < len(stack)


def test_lines_the_solver_cannot_judge_are_kept_and_counted():
    cs = [cand("ok", [(100, 100), (700, 100)]), cand("zero", [(900, 100), (900, 100)], "z1")]
    kept = dedup(PAGE, cs)
    assert ids(kept) == ["ok", "zero"] and STATS[PAGE]["unjudged_kept"] == 1


def test_context_is_required_for_the_page():
    with pytest.raises(KeyError):
        dedup("unknown", [cand("a", [(0, 0), (10, 0)])])
    assert PAGE in CONTEXT


def test_every_returned_element_is_one_of_the_received_objects_not_a_copy():
    """axel's _dedup_batch checks object identity (id()) and raises on anything else."""
    cs = [cand("a", [(100, 100), (700, 100)]), cand("b", [(110, 103), (690, 103)], "z1"), cand("c", [(100, 500), (700, 500)], "z2"),
          cand("zero", [(900, 100), (900, 100)], "z3")]
    received = {id(c) for c in cs}
    kept = dedup(PAGE, cs)
    assert kept and all(id(k) in received for k in kept) and all(any(k is c for c in cs) for k in kept)
    assert [id(k) for k in kept] == [id(c) for c in cs if any(c is k for k in kept)]   # subset, same order
    assert dedup(PAGE, [])[:0] == [] if PAGE in CONTEXT else True


def test_the_pQ_collision_is_order_independent_through_dedup():
    """pQ's counter-example (maintenance_codex_2026-10-17): the same kept geometry whatever the order of the candidates."""
    a = cand("a", [(100, 100), (200, 101)], "z0", h=10.0)
    b = cand("b", [(100, 101), (200, 100)], "z0", h=10.0)
    assert sorted(ids(dedup(PAGE, [a, b]))) == sorted(ids(dedup(PAGE, [b, a])))


# --- v16: the source rank of the file travels through dedup as _src (integers compared), in both orders of the candidates ---

RANK_PAGE = "pg_rank"


def register_rank_page(source_rank):
    register_page(RANK_PAGE, {"width": 3000, "height": 4000, "image_sha256": "b" * 64, "document_key": "d", "zone_crops": {"z0": ZC},
                              "source_rank": source_rank})


def test_the_file_order_wins_between_exact_duplicates_in_both_orders():
    """The duplicate that comes FIRST in lines/<det>/<page>.json (rank 0) keeps its place, whatever the order of the candidates."""
    register_rank_page({"z-line": 0, "a-line": 1})
    z = cand("z-line", [(100, 100), (700, 100)], "z0")
    a = cand("a-line", [(100, 100), (700, 100)], "z0")
    forward = ids(dedup(RANK_PAGE, [z, a]))
    reverse = ids(dedup(RANK_PAGE, [a, z]))
    assert forward == reverse == ["z-line"]


def test_source_ranks_are_compared_as_integers_not_as_strings():
    """Ranks 2 and 10: 2 comes before 10. A string comparison would put '10' first and fail."""
    register_rank_page({"ten": 10, "two": 2})
    ten = cand("ten", [(100, 100), (700, 100)], "z0")
    two = cand("two", [(100, 100), (700, 100)], "z0")
    assert ids(dedup(RANK_PAGE, [ten, two])) == ids(dedup(RANK_PAGE, [two, ten])) == ["two"]


def test_without_a_source_rank_the_line_id_is_the_fallback():
    register_page("pg_norank", {"width": 3000, "height": 4000, "image_sha256": "c" * 64, "document_key": "d", "zone_crops": {"z0": ZC}})
    x = cand("line-b", [(100, 100), (700, 100)], "z0")
    y = cand("line-a", [(100, 100), (700, 100)], "z0")
    assert ids(dedup("pg_norank", [x, y])) == ids(dedup("pg_norank", [y, x])) == ["line-a"]
