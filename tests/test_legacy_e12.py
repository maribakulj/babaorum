"""E12 : vérité PAGE, géométrie, appariement des lignes, ordre XY, CER de page."""

from __future__ import annotations

from pathlib import Path

from babaorum.legacy.layout.geom import baseline_cover, box_iou, mask_iou, rasterize
from babaorum.legacy.layout.gt import load_page
from babaorum.legacy.layout.match import kendall_tau, score_blocks, score_lines, seg, side_by_side
from babaorum.legacy.layout.order import xy_cut
from babaorum.legacy.layout.pagetext import page_text, score_page

PAGE = """<?xml version="1.0" encoding="UTF-8"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">
 <Page imageFilename="p.jpg" imageWidth="200" imageHeight="100">
  <ReadingOrder><OrderedGroup id="g">
   <RegionRefIndexed index="1" regionRef="r2"/>
   <RegionRefIndexed index="0" regionRef="r1"/>
  </OrderedGroup></ReadingOrder>
  <TextRegion id="r1"><Coords points="0,0 90,0 90,50 0,50"/>
   <TextLine id="l2" custom="readingOrder {index:1;}"><Coords points="0,25 90,25 90,45 0,45"/>
    <Baseline points="0,40 90,40"/><TextEquiv><Unicode>deux</Unicode></TextEquiv></TextLine>
   <TextLine id="l1" custom="readingOrder {index:0;}"><Coords points="0,0 90,0 90,20 0,20"/>
    <Baseline points="0,15 90,15"/><TextEquiv><Unicode>un</Unicode></TextEquiv></TextLine>
  </TextRegion>
  <TextRegion id="r2"><Coords points="100,0 200,0 200,50 100,50"/>
   <TextLine id="l3"><Coords points="100,0 200,0 200,20 100,20"/>
    <Baseline points="100,15 200,15"/><TextEquiv><Unicode>trois</Unicode></TextEquiv></TextLine>
  </TextRegion>
  <ImageRegion id="i1"><Coords points="0,60 200,60 200,100 0,100"/></ImageRegion>
 </Page>
</PcGts>
"""


def test_load_page(tmp_path: Path) -> None:
    f = tmp_path / "p.xml"
    f.write_text(PAGE)
    p = load_page(f)
    assert p.order[:2] == ["r1", "r2"]
    assert [ln.text for ln in p.lines()] == ["un", "deux", "trois"]
    assert len(p.text_regions) == 2 and p.has_line_geometry


def test_geometry() -> None:
    a = rasterize([(0, 0), (40, 0), (40, 40), (0, 40)], 1.0)
    b = rasterize([(20, 0), (60, 0), (60, 40), (20, 40)], 1.0)
    assert 0.3 < mask_iou(a, b) < 0.36
    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert baseline_cover([(0, 10), (100, 10)], [(0, 12), (50, 12)], tol=5) > 0.45
    assert baseline_cover([(0, 10), (100, 10)], [(0, 30), (100, 30)], tol=5) == 0.0


def test_score_lines_merge_and_order() -> None:
    gt = [
        seg("a", [(0, 0), (90, 0), (90, 20), (0, 20)], [(0, 15), (90, 15)]),
        seg("b", [(110, 0), (200, 0), (200, 20), (110, 20)], [(110, 15), (200, 15)]),
        seg("c", [(0, 30), (90, 30), (90, 50), (0, 50)], [(0, 45), (90, 45)]),
    ]
    merged = seg("m", [(0, 0), (200, 0), (200, 20), (0, 20)], [(0, 15), (200, 15)])
    c = seg("c2", [(0, 31), (90, 31), (90, 50), (0, 50)], [(0, 45), (90, 45)])
    s = score_lines(gt, [merged, c])
    assert len(s.pairs_bl) == 1 and s.merges == 1
    perfect = score_lines(gt, gt)
    assert perfect.bl == (1.0, 1.0, 1.0) and perfect.kendall == 1.0
    assert side_by_side(gt[0].box, gt[1].box) and not side_by_side(gt[0].box, gt[2].box)
    assert kendall_tau([0, 1, 2], [2, 1, 0]) == -1.0


def test_score_blocks_mixed() -> None:
    gt = [
        seg("a", [(0, 0), (90, 0), (90, 20), (0, 20)], [(0, 15), (90, 15)]),
        seg("b", [(110, 0), (200, 0), (200, 20), (110, 20)], [(110, 15), (200, 15)]),
    ]
    whole = [((0, 0), (200, 0), (200, 30), (0, 30))]
    bs = score_blocks(gt, [], whole)
    assert bs.covered == 1.0 and bs.mixed == 1


def test_xy_cut_headline_then_columns() -> None:
    boxes = [
        (0, 20, 45, 100),  # colonne gauche
        (55, 20, 100, 100),  # colonne droite
        (0, 0, 100, 15),  # titre sur toute la largeur
        (0, 105, 45, 120),  # bas gauche, après la bande des colonnes
    ]
    assert xy_cut(boxes) == [2, 0, 1, 3]


def test_page_text_dehyphen_and_long_s() -> None:
    assert page_text(["Les pro-", "scrits ſont là"]) == "les proscrits sont là"


def test_score_page_causes() -> None:
    ref = [
        "le chat dort sur le tapis rouge",
        "la pluie tombe sur la ville grise",
        "un homme marche seul dans la nuit",
    ]
    same = score_page(ref, ref)
    assert same.cer == 0 and same.cer_reordered == 0 and same.missed == 0
    swapped = score_page([ref[2], ref[0], ref[1]], ref)
    assert swapped.cer > 0.3 and swapped.cer_reordered == 0 and swapped.order > 0.3
    dup = score_page(ref + [ref[1]], ref)
    assert dup.duplicate > 0.3 and dup.missed == 0
    miss = score_page(ref[:2], ref)
    assert 0.3 < miss.missed < 0.36
    bad = score_page(ref + ["xqz wvk plm"], ref)
    assert bad.unanchored > 0


def test_reorder_keeps_short_lines_in_place() -> None:
    from babaorum.legacy.layout.pagetext import anchor_units, norm, reorder_runs

    ref_units = [
        "le conseil municipal a voté hier le budget",
        "de la ville",
        "suite",
        "on apprend que de la ville partent des trains",
        "pour la province",
    ]
    units = [norm(u) for u in ref_units]
    ref = " ".join(units)
    # « de la ville » est ambiguë (deux fois dans la page) : bien placée par la
    # chaîne, elle ne doit pas être déplacée
    assert reorder_runs(units, anchor_units(units, ref)) == ref
    assert score_page(ref_units, ref_units).cer_reordered == 0
