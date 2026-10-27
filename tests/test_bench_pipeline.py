"""Pipeline bench v1: adapter and reproduction controls, synthetic data only."""
import copy
import json

import pytest

from babaorum.bench_pipeline import (BenchError, ReproductionError, add_words, build_graph, check_controls,
                                      empty_metric_slots, install_polygon_cache, load_config, require_exact,
                                      uninstall_polygon_cache)
from babaorum.page_graph import PageGraphError, validate_page_graph

IDENTITY = {"page_key": "gallica:7000001/f1", "document_key": "gallica:7000001", "title_key": None}
IMAGE = {"sha256": "a" * 64, "width": 200, "height": 100}
PROVENANCE = {"run_id": "bench-test", "components": [{
    "name": "chain", "code_revision": "rev", "config_sha256": None, "checkpoint_sha256": None, "license": None}]}


def unit(uid, x0, y0, x1, y1, text="texte", block="b1", **extra):
    return {"id": uid, "text": text, "block": block,
            "polygon": [(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
            "baseline": [(x0, y1 - 2), (x1, y1 - 2)], **extra}


def graph(units, role="prediction"):
    return build_graph(units, role=role, identity=IDENTITY, image=IMAGE, provenance=PROVENANCE)


def test_adapter_builds_a_valid_prediction_graph_on_one_page():
    g, stats = graph([unit("l1", 10, 10, 90, 30), unit("l2", 10, 40, 90, 60)])
    validate_page_graph(g)
    assert g["graph_role"] == "prediction" and g["run_status"] == "ok"
    assert [n["id"] for n in g["lines"]] == ["l1", "l2"]
    assert [r["type"] for r in g["relations"]].count("precedes") == 1
    assert stats == {"units": 2, "invalid_polygons": 0, "invalid_baselines": 0,
                     "closing_vertex_dropped": 0, "partial_lines": 0}
    assert all(n["polygon"]["source"] == "predicted" for n in g["lines"])


def test_ground_truth_graph_uses_ground_truth_provenance():
    g, _ = graph([unit("l1", 10, 10, 90, 30)], role="ground_truth")
    assert g["graph_role"] == "ground_truth"
    assert g["lines"][0]["polygon"]["source"] == "ground_truth"


@pytest.mark.parametrize("source", ["ground_truth", "transferred"])
def test_a_truth_unit_is_refused_in_a_prediction(source):
    with pytest.raises(BenchError, match="cannot enter a prediction graph"):
        graph([unit("l1", 10, 10, 90, 30, source=source)])


def test_page_graph_itself_refuses_truth_provenance_in_a_prediction():
    g, _ = graph([unit("l1", 10, 10, 90, 30)])
    g["lines"][0]["polygon"]["source"] = "ground_truth"
    with pytest.raises(PageGraphError, match="prediction graph cannot claim"):
        validate_page_graph(g)


def test_invalid_geometry_is_explicit_error_not_repaired():
    out_of_bounds = unit("l2", 150, 40, 250, 60)  # beyond width 200
    g, stats = graph([unit("l1", 10, 10, 90, 30), out_of_bounds])
    assert g["lines"][1]["polygon"] == {"status": "error", "points": None, "source": None}
    assert g["lines"][1]["status"] == "partial" and g["run_status"] == "partial"
    assert stats["invalid_polygons"] == 1 and stats["partial_lines"] == 1
    assert g["coverage"]["lines"]["status"] == "partial"


def test_repeated_closing_vertex_is_removed_and_counted():
    u = unit("l1", 10, 10, 90, 30)
    u["polygon"] = u["polygon"] + [u["polygon"][0]]
    g, stats = graph([u])
    assert stats["closing_vertex_dropped"] == 1 and g["lines"][0]["polygon"]["status"] == "available"


def test_units_without_block_leave_blocks_unavailable():
    g, _ = graph([unit("l1", 10, 10, 90, 30, block=None)])
    assert g["blocks"] == [] and g["coverage"]["blocks"]["status"] == "unavailable"


def test_duplicate_line_ids_are_refused():
    with pytest.raises(PageGraphError):
        graph([unit("l1", 10, 10, 90, 30), unit("l1", 10, 40, 90, 60)])


# --- reproduction controls -----------------------------------------------------

CONFIG = {
    "schema_version": "babaorum.bench_pipeline.v1", "label": "dev exposé (D-600)", "tolerance": 1e-9,
    "forbidden_path_fragments": ["runs/e22/bench", "runs/e22/indep"],
    "controls": [
        {"id": "c1_base", "chain": "c1|base", "published_pct": 6.55,
         "expected": {"file": "runs/e32/dev.json", "path": ["aggregates", "c1|base", "reord"]}},
        {"id": "human", "chain": "human", "published_pct": 2.99,
         "expected": {"file": "runs/e27/decomp_dev.json", "path": ["aggregates", "human", "reord"]}},
    ],
    "metric_slots": {"line_recall_precision": "pending_review_pN"},
}


def axel_root(tmp_path):
    for rel, payload in (("runs/e32/dev.json", {"aggregates": {"c1|base": {"reord": 0.0654973139}}}),
                         ("runs/e27/decomp_dev.json", {"aggregates": {"human": {"reord": 0.0298553399}}})):
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
    return tmp_path


def test_controls_exact_when_measured_equals_published(tmp_path):
    root = axel_root(tmp_path)
    results = check_controls({"c1_base": 0.0654973139, "human": 0.0298553399}, CONFIG, root)
    assert [r["status"] for r in results] == ["exact", "exact"]
    require_exact(results)


def test_controls_fail_when_a_value_is_falsified(tmp_path):
    root = axel_root(tmp_path)
    results = check_controls({"c1_base": 0.0654973139 + 1e-6, "human": 0.0298553399}, CONFIG, root)
    assert results[0]["status"] == "deviation"
    with pytest.raises(ReproductionError, match="c1_base: deviation"):
        require_exact(results)


def test_controls_fail_when_a_measure_is_missing(tmp_path):
    root = axel_root(tmp_path)
    results = check_controls({"c1_base": 0.0654973139}, CONFIG, root)
    assert results[1]["status"] == "missing"
    with pytest.raises(ReproductionError):
        require_exact(results)


def test_published_percentage_must_match_the_json(tmp_path):
    root = axel_root(tmp_path)
    config = copy.deepcopy(CONFIG)
    config["controls"][0]["published_pct"] = 7.0
    results = check_controls({"c1_base": 0.0654973139, "human": 0.0298553399}, config, root)
    assert results[0]["status"] == "published_mismatch"
    with pytest.raises(ReproductionError):
        require_exact(results)


@pytest.mark.parametrize("source", ["runs/e22/bench/dev.json", "runs/e22/indep/x.json", "data/prepared/x.json",
                                    "../outside.json"])
def test_forbidden_or_foreign_sources_are_refused(tmp_path, source):
    root = axel_root(tmp_path)
    config = copy.deepcopy(CONFIG)
    config["controls"][0]["expected"]["file"] = source
    with pytest.raises(BenchError):
        check_controls({"c1_base": 0.1, "human": 0.1}, config, root)


def test_config_must_be_labelled_dev_exposed_and_have_tolerance(tmp_path):
    for change in ({"label": "test"}, {"tolerance": 1e-3}, {"controls": []}, {"schema_version": "x"}):
        path = tmp_path / "c.json"
        path.write_text(json.dumps({**CONFIG, **change}), encoding="utf-8")
        with pytest.raises(BenchError):
            load_config(path)


def test_shipped_config_loads_and_lists_the_five_controls():
    from pathlib import Path
    config = load_config(Path(__file__).resolve().parents[1] / "config" / "bench_pipeline_v1.json")
    assert [c["id"] for c in config["controls"]] == ["c1_base", "c1_ma", "ppdl3_z", "human", "e28_ppdl3_z_a_f"]
    assert all(v["value"] is None for v in empty_metric_slots(config).values())


def test_polygon_cache_changes_no_result_and_never_accepts_an_invalid_polygon():
    units = [unit("l1", 10, 10, 90, 30), unit("l2", 10, 40, 90, 60), unit("l3", 150, 40, 250, 60)]
    plain, plain_stats = graph(units)
    install_polygon_cache()
    try:
        install_polygon_cache()  # idempotent
        first, first_stats = graph(units)
        second, second_stats = graph(units)  # served from the cache
        assert first == plain == second and first_stats == plain_stats == second_stats
        assert second["lines"][2]["polygon"]["status"] == "error"  # invalid stays invalid on every call
        bad = copy.deepcopy(second)
        bad["lines"][0]["polygon"]["points"] = [[10, 10], [90, 10], [90, 30], [10, 30], [50, 50], [10, 10]]
        with pytest.raises(PageGraphError):
            validate_page_graph(bad)
    finally:
        uninstall_polygon_cache()


def test_metric_slots_hold_the_v3_columns_without_values():
    from pathlib import Path
    config = load_config(Path(__file__).resolve().parents[1] / "config" / "bench_pipeline_v1.json")
    slots = empty_metric_slots(config)
    assert config["tolerance"] == 1e-12
    assert slots["cer_columns"]["columns"] == ["cer_ordered", "cer_reordered_d217", "cer_local", "cer_bag_d556"]
    assert "flex" in slots["cer_columns"]["note"]  # the warning is written; D-217 is never called flex accuracy
    assert slots["cbad_baseline_rpf"]["status"] == "non calculé" and slots["cbad_baseline_rpf"]["value"] is None
    assert set(slots["reading_order_kendall_tau"]["not_applicable_pages"]) == {
        "Ce-soir_7633477_D0000007", "Regards_7635840_D0000005"}
    assert slots["alto_page_validity"]["checks"][1] == "baseline_present_on_every_line"
    assert all(v["value"] is None for v in slots.values())


def test_block_labels_map_to_exportable_categories_and_unknown_stays_autre():
    units = [unit("l1", 10, 10, 90, 30, block="b1"), unit("l2", 10, 40, 90, 60, block="b2")]
    g, _ = build_graph(units, role="prediction", identity=IDENTITY, image=IMAGE, provenance=PROVENANCE,
                       block_labels={"b1": "zone", "b2": "mystery"})
    assert {b["id"]: b["category"] for b in g["blocks"]} == {"block_b1": "texte", "block_b2": "autre"}
    g2, _ = graph(units)  # no labels at all
    assert {b["category"] for b in g2["blocks"]} == {"autre"}


def test_add_words_attaches_words_with_order_and_keeps_missing_boxes_missing():
    g, _ = graph([unit("l1", 10, 10, 90, 30), unit("l2", 10, 40, 90, 60, text="x")])
    box = [[10, 10], [40, 10], [40, 30], [10, 30]]
    stats = add_words(g, {"l1": [("ab", box), ("cd", None)], "l2": [("x", box)]})
    assert stats == {"words": 3, "words_without_box": 1, "lines_whitespace_normalized": 1, "lines_blank_source_kept": 0, "lines_without_words": 0}
    assert g["lines"][0]["text"]["value"] == "ab cd"  # line text now equals the words joined by one space
    assert [w["polygon"]["status"] for w in g["words"]] == ["available", "missing", "available"]
    assert sum(r["type"] == "precedes" for r in g["relations"]) == 1 + 1  # one line edge + one word edge
    assert g["coverage"]["words"]["available_count"] == 3


def test_add_words_counts_lines_without_words_and_stays_valid():
    g, _ = graph([unit("l1", 10, 10, 90, 30, text=""), unit("l2", 10, 40, 90, 60)])
    stats = add_words(g, {"l2": [("texte", [[10, 40], [50, 40], [50, 60], [10, 60]])]})
    assert stats["lines_without_words"] == 1 and stats["words"] == 1


def load_script():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("bench_script_under_test", Path(__file__).resolve().parents[1] / "scripts/bench_pipeline.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeChar:
    def __init__(self, x0, x1, conf):
        self.x0, self.x1, self.conf = x0, x1, conf


class FakeReading:
    def __init__(self, text, chars):
        self.text, self.chars = text, chars


def test_unit_spans_uses_direct_cuts_only_when_text_and_positions_align():
    script = load_script()
    cuts = [[0, 1], [2, 3], [4, 5]]
    ctc = {"l1": {"cuts": cuts, "conf": [1.0, 0.5, 0.25]}}
    spans, confs, how = script.unit_spans("abc", "l1", {"l1": "abc"}, ctc, reading_of=lambda *a: pytest.fail("not needed"))
    assert how == "cuts_direct" and spans == [(0.0, 1.0), (2.0, 3.0), (4.0, 5.0)] and confs == [1.0, 0.5, 0.25]


def test_unit_spans_falls_back_to_axel_reading_and_never_invents_positions():
    script = load_script()
    ctc = {"l1": {"cuts": [[0, 1], [2, 3], [4, 5]], "conf": [1.0, 1.0, 1.0]}}
    # chain text differs from the raw text (cleaned by axel): use the Reading whose text equals the chain text
    cleaned = FakeReading("ab", [FakeChar(2, 3, 0.9), FakeChar(4, 5, 0.8)])
    spans, confs, how = script.unit_spans("ab", "l1", {"l1": " ab"}, ctc, reading_of=lambda raw, c: cleaned)
    assert how == "reading_of" and spans == [(2, 3), (4, 5)] and confs == [0.9, 0.8]
    # a reading that does not reproduce the chain text gives no positions
    other = FakeReading("zz", [FakeChar(0, 1, 1.0), FakeChar(2, 3, 1.0)])
    assert script.unit_spans("ab", "l1", {"l1": " ab"}, ctc, reading_of=lambda raw, c: other) == (None, None, "misaligned")
    # no read, no ctc, or no cuts at all
    assert script.unit_spans("ab", "zz", {}, {}, reading_of=lambda *a: None)[2] == "no_cuts"
    assert script.unit_spans("ab", "l1", {"l1": "ab"}, {"l1": {"cuts": []}}, reading_of=lambda *a: None)[2] == "no_cuts"


def test_read_dirs_cover_only_chains_that_have_ctc_cuts():
    from pathlib import Path
    script = load_script()
    root = Path("/x")
    assert script.read_dirs(root, "c1") == [root / "runs/e22/read/ppdl3_z"]
    assert script.read_dirs(root, "c3") == [root / "runs/e27/read/gallica_t"]  # C3 keeps its cuts (E26/E27), dev pages only
    assert len(script.read_dirs(root, "ppdl3_z+a_f")) == 2
    # lines added by A and M+A are read in E32's extra dir first, then E28's a_z (as e32.read_dirs does)
    for label in ("c1+a", "c1+ma"):
        assert script.read_dirs(root, label) == [root / "runs/e22/read/ppdl3_z", root / "runs/e32/dev/read/mccatmus_press/extra",
                                                 root / "runs/e28/read/mccatmus_press/a_z"]
    assert script.read_dirs(root, "c1+m") == [root / "runs/e22/read/ppdl3_z"]  # M adds no line
    for label in ("tesseract", "gallica"):
        assert script.read_dirs(root, label) == []  # texts from external OCR: no cuts, no word coordinates



def test_add_words_keeps_a_whitespace_only_source_line_as_it_is():
    g, _ = graph([unit("l1", 10, 10, 90, 30, text="    "), unit("l2", 10, 40, 90, 60, text="")])
    stats = add_words(g, {})
    assert stats["lines_blank_source_kept"] == 1 and stats["lines_without_words"] == 2
    assert [n["text"]["value"] for n in g["lines"]] == ["    ", ""]  # not hidden as empty lines
