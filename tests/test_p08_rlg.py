"""LOT-P08 v10 : règle R-LG (note P-08 v2.3 § 1) sur des cas synthétiques, via rlg_resolve."""
from __future__ import annotations

import sys
from collections import namedtuple
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

from babaorum.p08_oracles import Group  # noqa: E402
import p08_decompose as P  # noqa: E402

M = namedtuple("M", "pred_id gt_id")


def run(groups, matches, pt, gt, missed=(), extra=(), partial=None):
    return P.rlg_resolve(list(groups), list(matches), pt, gt, list(missed), list(extra), dict(partial or {}))


def test_a_page_without_suspect_group_is_unchanged_from_v9():
    # (a) non-régression : sans groupe suspect, matched/missed/extra égaux à l'expression de v9
    groups = [Group("SPLIT", ("p1", "p2"), ("g1",))]
    matches = [M("p9", "g9")]
    pt = {"p1": "ab", "p2": "cd", "p9": "zz"}
    gt = {"g1": "xy", "g9": "qq"}  # CER élevé : aucun groupe suspect
    kept, matched, missed, extra, info = run(groups, matches, pt, gt, missed=["g5"], extra=["p7"])
    assert kept == groups and info["groups_dissolved"] == 0
    in_p = {p for g in groups for p in g.pred_ids}
    in_g = {x for g in groups for x in g.gt_ids}
    v9 = {m.pred_id: m.gt_id for m in matches if m.pred_id not in in_p and m.gt_id not in in_g}
    assert matched == v9 and missed == ["g5"] and extra == ["p7"]


def test_a_suspect_group_is_dissolved_whole_and_its_unpaired_predicted_line_goes_to_extra():
    # (b) groupe suspect (p1 apparié 1:1 à g1, CER 0) : dissous ; p2 sans partenaire -> en trop
    groups = [Group("SPLIT", ("p1", "p2"), ("g1",))]
    kept, matched, missed, extra, info = run(groups, [M("p1", "g1")], {"p1": "ab", "p2": "cd"}, {"g1": "ab"})
    assert kept == [] and info["groups_dissolved"] == 1
    assert matched == {"p1": "g1"}
    assert extra == ["p2"] and missed == []


def test_an_unpaired_true_line_of_a_dissolved_group_goes_to_missed():
    # (b) une V du groupe dissous sans partenaire 1:1 -> manquée
    groups = [Group("MERGE_COL", ("p1",), ("g1", "g2"))]
    kept, matched, missed, extra, info = run(groups, [M("p1", "g1")], {"p1": "ab"}, {"g1": "ab", "g2": "cd"})
    assert kept == [] and matched == {"p1": "g1"}
    assert missed == ["g2"] and extra == []


def test_a_group_is_kept_when_no_pair_inside_it_is_suspect():
    # (c) groupe non suspect : gardé tel quel (son unique paire a un CER > 0,10)
    groups = [Group("SPLIT", ("p3", "p4"), ("g3",))]
    kept, matched, missed, extra, info = run(groups, [M("p3", "g3")], {"p3": "abc", "p4": "zz"}, {"g3": "xyzuvw"})
    assert kept == groups and info["groups_dissolved"] == 0
    assert matched == {} and missed == [] and extra == []


def test_a_dissolved_line_whose_partner_sits_in_a_kept_group_goes_to_extra_or_missed():
    # (b) partenaire dans un groupe gardé : p2 (dissous) est apparié à g5 (groupe gardé) -> en trop
    dissolved = Group("SPLIT", ("p1", "p2"), ("g1",))
    kept_group = Group("SPLIT", ("p5", "p6"), ("g5",))
    matches = [M("p1", "g1"), M("p2", "g5"), M("p5", "g5")]
    pt = {"p1": "ab", "p2": "cd", "p5": "qq", "p6": "ww"}
    gt = {"g1": "ab", "g5": "ef"}
    kept, matched, missed, extra, info = run([dissolved, kept_group], matches, pt, gt)
    assert kept == [kept_group]
    assert matched == {"p1": "g1"} and extra == ["p2"] and missed == []


def test_a_dissolved_true_line_paired_with_a_kept_predicted_line_goes_to_missed():
    # (b) V dissoute dont le partenaire prédit est dans un groupe gardé -> manquée
    dissolved = Group("MERGE_VERT", ("p1",), ("g1", "g2"))
    kept_group = Group("SPLIT", ("p7", "p8"), ("g7",))
    matches = [M("p1", "g1"), M("p7", "g2")]
    pt = {"p1": "ab", "p7": "cd", "p8": "ef"}
    gt = {"g1": "ab", "g2": "xx", "g7": "yy"}
    kept, matched, missed, extra, info = run([dissolved, kept_group], matches, pt, gt)
    assert matched == {"p1": "g1"} and missed == ["g2"] and extra == []


def test_the_stop_guard_is_declared_and_no_valid_input_triggers_it():
    # (d) garde d'arrêt : construite par cas, aucune entrée valide ne la déclenche (ligne appariée ET manquée impossible)
    groups = [Group("SPLIT", ("p1", "p2"), ("g1",))]
    kept, matched, missed, extra, info = run(groups, [M("p1", "g1")], {"p1": "ab", "p2": "cd"}, {"g1": "ab"})
    assert not (set(info["to_missed_gt"]) & set(matched.values()))
    assert not (set(info["to_extra_pred"]) & set(matched))


def test_the_mass_report_counts_true_characters_and_predicted_characters_apart():
    # (§ 4.3) : caractères vrais vers matched (lus juste ou non) et vers missed ; caractères prédits vers extra
    groups = [Group("SPLIT", ("p1", "p2"), ("g1",))]
    # lecture juste : CER 0 -> suspect, p1 apparié à g1 ; p2 (4 caractères prédits) en trop
    _, _, _, _, info = run(groups, [M("p1", "g1")], {"p1": "ab", "p2": "cdef"}, {"g1": "ab"})
    assert info["chars_gt_to_matched"] == 2 and info["chars_gt_to_matched_read_right"] == 2
    assert info["chars_gt_to_missed"] == 0 and info["lines_pred_to_extra"] == 1 and info["chars_pred_to_extra"] == 4
    # lecture fausse mais CER 0,10 (1 erreur sur 10) : suspect, compté comme vrai apparié non bien lu
    _, _, _, _, info = run(groups, [M("p1", "g1")], {"p1": "abcdefghiX", "p2": "cd"}, {"g1": "abcdefghij"})
    assert info["chars_gt_to_matched"] == 10 and info["chars_gt_to_matched_read_right"] == 0


# ---- v10.1 (avenant v2.4, règle P-E27) ---------------------------------------------------------------
from babaorum.p08_oracles import decompose_page  # noqa: E402


def order_ok(pt, gt, matched, groups, missed, extra, partial):
    """contrôle d'ordre vrai : decompose_page lève si une ligne vraie n'est couverte par aucun oracle"""
    links = P.Links(matched, groups, missed, extra, partial)
    decompose_page(list(pt.items()), list(gt.items()), links, P.score)


def test_a_partner_already_matched_by_e27_refuses_the_pair_and_the_dissolved_line_goes_to_extra():
    # partenaire 'appariee' non dissous : la paire (p2, g5) est refusée ; p2 -> en trop ; (p5, g5) v9 inchangée
    groups = [Group("SPLIT", ("p1", "p2"), ("g1",))]
    matches = [M("p1", "g1"), M("p2", "g5"), M("p5", "g5")]
    pt = {"p1": "ab", "p2": "cd", "p5": "ef"}; gt = {"g1": "ab", "g5": "ef"}
    kept, matched, missed, extra, info = run(groups, matches, pt, gt)
    assert matched == {"p1": "g1", "p5": "g5"} and extra == ["p2"] and missed == []
    assert info["pairs_refused"] == 1 and info["pairs_added"] == 1   # p1-g1 : deux lignes dissoutes, admise
    order_ok(pt, gt, matched, kept, missed, extra, {})


def test_a_partial_line_as_key_refuses_the_pair_and_the_dissolved_true_line_goes_to_missed():
    # PARTIAL côté clé : p3 (partielle vers g6) ; la paire (p3, g2) avec g2 dissoute est refusée ; g2 -> manquée
    groups = [Group("MERGE_COL", ("p1", "p2"), ("g1", "g2"))]
    matches = [M("p1", "g1"), M("p3", "g2")]
    pt = {"p1": "ab", "p2": "cd", "p3": "zz"}; gt = {"g1": "ab", "g2": "zz", "g6": "zz"}
    partial = {"p3": "g6"}
    kept, matched, missed, extra, info = run(groups, matches, pt, gt, missed=[], extra=[], partial=partial)
    assert matched == {"p1": "g1"} and missed == ["g2"] and extra == ["p2"]
    assert info["pairs_refused"] == 1
    order_ok({"p1": "ab", "p2": "cd", "p3": "zz"}, {"g1": "ab", "g2": "zz", "g6": "zz"}, matched, kept, missed, extra, partial)


def test_a_partial_true_line_as_value_refuses_the_pair_and_keeps_the_partial_line():
    # PARTIAL côté valeur : g6 est la cible d'une partielle (p4) ; la paire (p2, g6) avec p2 dissoute est refusée
    groups = [Group("SPLIT", ("p1", "p2"), ("g1",))]
    matches = [M("p1", "g1"), M("p2", "g6")]
    pt = {"p1": "ab", "p2": "cd", "p4": "yy"}; gt = {"g1": "ab", "g6": "yy"}
    partial = {"p4": "g6"}
    kept, matched, missed, extra, info = run(groups, matches, pt, gt, partial=partial)
    assert matched == {"p1": "g1"} and extra == ["p2"] and missed == []
    assert info["pairs_refused"] == 1
    order_ok(pt, gt, matched, kept, missed, extra, partial)


def test_a_missed_true_line_refuses_the_pair_and_stays_missed():
    # MISSED : g7 est manquée d'origine ; la paire (p2, g7) est refusée ; p2 -> en trop ; g7 reste manquée
    groups = [Group("SPLIT", ("p1", "p2"), ("g1",))]
    matches = [M("p1", "g1"), M("p2", "g7")]
    pt = {"p1": "ab", "p2": "cd"}; gt = {"g1": "ab", "g7": "cd"}
    kept, matched, missed, extra, info = run(groups, matches, pt, gt, missed=["g7"], extra=[])
    assert matched == {"p1": "g1"} and missed == ["g7"] and extra == ["p2"]
    assert info["pairs_refused"] == 1
    order_ok(pt, gt, matched, kept, missed, extra, {})


def test_an_extra_predicted_line_refuses_the_pair_and_the_dissolved_true_line_goes_to_missed():
    # EXTRA : p8 est en trop d'origine ; la paire (p8, g2) avec g2 dissoute est refusée ; g2 -> manquée
    groups = [Group("MERGE_COL", ("p1",), ("g1", "g2"))]
    matches = [M("p1", "g1"), M("p8", "g2")]
    pt = {"p1": "ab", "p8": "zz"}; gt = {"g1": "ab", "g2": "zz"}
    kept, matched, missed, extra, info = run(groups, matches, pt, gt, missed=[], extra=["p8"])
    assert matched == {"p1": "g1"} and missed == ["g2"] and extra == ["p8"]
    assert info["pairs_refused"] == 1


def test_a_pair_between_two_lines_of_the_same_dissolved_group_is_admitted():
    # partenaire dissous, même groupe : la paire est admise et compte dans pairs_added
    groups = [Group("MERGE_VERT", ("p1", "p2"), ("g1", "g2"))]
    matches = [M("p1", "g1"), M("p2", "g2")]
    pt = {"p1": "ab", "p2": "cd"}; gt = {"g1": "ab", "g2": "cd"}
    kept, matched, missed, extra, info = run(groups, matches, pt, gt)
    assert matched == {"p1": "g1", "p2": "g2"} and missed == [] and extra == []
    assert info["pairs_added"] == 2 and info["pairs_refused"] == 0
    order_ok(pt, gt, matched, kept, missed, extra, {})


def test_a_pair_between_two_different_dissolved_groups_is_admitted():
    # partenaire dissous, autre groupe : p3 (groupe A) et g2 (groupe B) sont admis
    groups = [Group("SPLIT", ("p1", "p3"), ("g1",)), Group("MERGE_VERT", ("p2",), ("g2", "g3"))]
    matches = [M("p1", "g1"), M("p2", "g4"), M("p3", "g2")]   # p2 suspect (paire avec g4, hors groupe) -> groupe B dissous
    pt = {"p1": "ab", "p2": "xx", "p3": "cd"}; gt = {"g1": "ab", "g2": "cd", "g3": "ef", "g4": "xx"}
    kept, matched, missed, extra, info = run(groups, matches, pt, gt)
    assert matched == {"p1": "g1", "p3": "g2"} and missed == ["g3"] and extra == ["p2"]
    assert info["pairs_added"] == 2 and info["groups_dissolved"] == 2


def test_a_page_without_suspect_group_gives_v9_pairs_and_no_pair_added():
    # sans groupe suspect : paires v9 inchangées, pairs_added = 0
    groups = [Group("SPLIT", ("p3", "p4"), ("g3",))]
    matches = [M("p3", "g3"), M("p9", "g9")]
    pt = {"p3": "abc", "p4": "zz", "p9": "qq"}; gt = {"g3": "xyzuvw", "g9": "qq"}
    kept, matched, missed, extra, info = run(groups, matches, pt, gt)
    assert matched == {"p9": "g9"} and info["pairs_added"] == 0 and info["groups_dissolved"] == 0


def test_the_la_presse_case_in_small_refuses_the_partial_pair():
    # La_Presse_0514199 en petit : p3 partielle vers g1 ; line_geometry l'apparie à g2, dissoute ; la paire est refusée
    groups = [Group("MERGE_VERT", ("p1", "p2"), ("g2",))]
    matches = [M("p1", "g2"), M("p3", "g2")]
    pt = {"p1": "bb", "p2": "cc", "p3": "aa"}; gt = {"g1": "aa", "g2": "bb"}
    partial = {"p3": "g1"}
    kept, matched, missed, extra, info = run(groups, matches, pt, gt, partial=partial)
    assert "p3" not in matched and info["pairs_refused"] == 1
    assert extra == ["p2"] and missed == [] and matched == {"p1": "g2"}
    order_ok({"p1": "bb", "p2": "cc", "p3": "aa"}, {"g1": "aa", "g2": "bb"}, matched, kept, missed, extra, partial)


@pytest.mark.parametrize("partial,missed0,extra0", [
    ({"p2": "g9"}, [], []),          # clé de partial dissoute
    ({"p9": "g2"}, [], []),          # valeur de partial dissoute (g2 vrai dissoute)
    ({}, ["g2"], []),                # missed d'origine dissous
    ({}, [], ["p2"]),                # extra d'origine dissous
])
def test_the_guard_stops_when_a_dissolved_line_is_in_partial_missed_or_extra_of_origin(partial, missed0, extra0):
    # garde d'arrêt : une ligne dissoute dans un classement E27 d'origine (entrée incohérente) lève RuntimeError
    groups = [Group("MERGE_VERT", ("p1", "p2"), ("g1", "g2"))]
    with pytest.raises(RuntimeError):
        run(groups, [M("p1", "g1")], {"p1": "ab", "p2": "cd", "p9": "zz"}, {"g1": "ab", "g2": "cd", "g9": "zz"},
            missed=missed0, extra=extra0, partial=partial)


# ---- (3) bout en bout : page sans groupe suspect, decompose_page v9 et v10.1 sur les mêmes liens -----------
def test_a_page_without_suspect_group_gives_the_same_decomposition_as_v9_to_1e_9():
    groups = [Group("SPLIT", ("p3", "p4"), ("g3",))]           # non suspect : CER élevé sur sa seule paire
    matches = [M("p3", "g3"), M("p9", "g9")]
    pt = {"p3": "xy", "p4": "zuvw", "p9": "qq", "p5": "tt"}   # p3+p4 découpent g3 ; p5 en trop ; g5 manquée
    gt = {"g3": "xyzuvw", "g9": "qq", "g5": "tt"}
    missed, extra, partial = ["g5"], ["p5"], {}
    in_p = {p for g in groups for p in g.pred_ids}; in_g = {x for g in groups for x in g.gt_ids}
    v9_matched = {m.pred_id: m.gt_id for m in matches if m.pred_id not in in_p and m.gt_id not in in_g}
    kept, matched, miss10, extra10, info = run(groups, matches, pt, gt, missed=missed, extra=extra, partial=partial)
    assert info["groups_dissolved"] == 0 and info["pairs_added"] == 0 and matched == v9_matched
    pred = list(pt.items()); true = list(gt.items())
    d9 = decompose_page(pred, true, P.Links(v9_matched, groups, missed, extra, partial), P.score)
    d10 = decompose_page(pred, true, P.Links(matched, kept, miss10, extra10, partial), P.score)
    assert abs(d9.cer["0"] - d10.cer["0"]) < 1e-9 and abs(d9.cer["4"] - d10.cer["4"]) < 1e-9
    assert set(d9.parts) == set(d10.parts)
    for k in d9.parts:
        assert abs(d9.parts[k] - d10.parts[k]) < 1e-9, k
    assert abs(d9.true_order_cer - d10.true_order_cer) < 1e-9
