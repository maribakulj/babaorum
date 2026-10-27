"""Pipeline bench v1 (dev exposé, D-600): reproduction controls (part a) + descriptive metrics (part b).

Run with the existing axel environment, read-only on axel:

    PYTHONDONTWRITEBYTECODE=1 ~/axel/.venv/bin/python scripts/bench_pipeline.py

Part a recomputes the published D-217 numbers with axel's own functions and compares them
to axel's JSON at 1e-12. Part b adds, per chain and page: four CER columns, splits and
merges apart, line recall/precision, reading-order tau with its support, the "ligne réussie"
column, PAGE/ALTO export checks, inventions apart, cost, and paired intervals against C1.
Nothing is a criterion. cBAD R/P/F is "non calculé" while the reference tool is absent.
Only reports/ of this repository is written.
"""
from __future__ import annotations

import argparse
import copy
import importlib
import json
import os
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from babaorum.bench_metrics import (alto_checks, line_success, order_report, page_checks, texts_from_alto,
                                    texts_from_page)  # noqa: E402
from babaorum.bench_pipeline import (  # noqa: E402
    LABEL, add_words, BenchError, build_graph, check_controls, empty_metric_slots, file_sha256,
    install_polygon_cache, load_config,
)
from babaorum.export_page import ExportError, export_alto, export_pagexml, validate_export_xsd  # noqa: E402
from babaorum.line_geometry import (  # noqa: E402
    GroundTruthLine, PredictedLine, evaluate_line_geometry, evaluate_line_geometry_sensitivity,
)
from babaorum.page_graph import PageGraphError, _polygon  # noqa: E402
from babaorum.words_from_ctc import (distribution, line_extent, normalized, outside_share, repair_polygon,  # noqa: E402
                                     split_words, word_polygon, words_text)
from collections import Counter  # noqa: E402

#: chain whose line set the added-line chains start from (for the invention definitions of E28/E32)
BASE_OF = {"c1+m": "c1", "c1+a": "c1", "c1+ma": "c1", "ppdl3_z+a_f": "ppdl3_z"}
CI_METRICS = ("reord", "d212", "ordered", "local", "bag")
#: LOT-P01c v4 / S4: reference chains whose text comes from an external OCR without CTC cuts (outside the ALTO criterion)
REFERENCE_NO_CUTS = frozenset({"tesseract", "gallica"})


def axel_head(axel: Path) -> str:
    out = subprocess.run(["git", "-C", str(axel), "rev-parse", "HEAD"], capture_output=True, text=True)
    return out.stdout.strip() if out.returncode == 0 else "unknown"


def bbox_height(polygon) -> float | None:
    ys = [float(p[1]) for p in polygon or []]
    return (max(ys) - min(ys)) if ys else None


def export_report(graph, kind: str, image_filename: str, n_lines: int) -> dict:
    """PAGE or ALTO. Strict decides the outcome; best_effort is attempted if strict refuses and is described apart.

    outcome: "valid" (strict accepted, XSD valid, no line lost), "explicit_refusal" (strict refused: counted, with
    the loss codes), or "invalid_xsd". A best_effort export that is XSD-valid but lost lines is never called valid.
    """
    out: dict = {}
    for mode in ("strict", "best_effort"):
        try:
            if kind == "alto":
                result = export_alto(graph, image_filename=image_filename, physical_img_nr=1, mode=mode)
            else:
                result = export_pagexml(graph, image_filename=image_filename, mode=mode, quantization="nearest_pixel")
        except ExportError as exc:
            out[mode] = {"accepted": False, "refusal": str(exc)[:300]}
            continue
        codes: dict = {}
        for loss in result.losses:
            if loss.severity != "info":
                codes[f"{loss.severity}:{loss.code}"] = codes.get(f"{loss.severity}:{loss.code}", 0) + 1
        entry = {"accepted": result.accepted, "loss_codes": codes,
                 "losses": {sev: sum(l.severity == sev for l in result.losses) for sev in ("info", "warning", "error")}}
        if result.accepted:
            try:
                validate_export_xsd(result)
                entry["xsd"] = "valid"
            except ExportError as exc:
                entry["xsd"] = f"invalid: {str(exc)[:200]}"
            entry["checks"] = alto_checks(result.xml, n_lines) if kind == "alto" else page_checks(result.xml, n_lines)
        out[mode] = entry
        if result.accepted and mode == "strict":
            break
    strict = out.get("strict", {})
    if strict.get("accepted"):
        lost = strict.get("checks", {}).get("lines_lost", 0)
        out["outcome"] = ("valid" if strict.get("xsd") == "valid" and lost == 0
                          else "invalid_xsd" if strict.get("xsd") != "valid" else "valid_lines_lost")
    else:
        out["outcome"] = "explicit_refusal"
    best = out.get("best_effort")
    out["best_effort_summary"] = (None if not best or not best.get("accepted") else
                                  {"xsd": best.get("xsd"), "lines_lost": best["checks"].get("lines_lost"),
                                   "text_lines": best["checks"].get("text_lines")})
    return out


def runs_root(axel: Path) -> Path:
    """<axel>/runs, or the absolute AXEL_RUNS when it is set (LOT-P07: caches of the dev n° 2 live under rappel_pages/runs)."""
    env = os.environ.get("AXEL_RUNS", "").strip()
    return Path(env).expanduser().resolve() if env else axel / "runs"


def read_dirs(axel: Path, label: str) -> list[Path]:
    rr = runs_root(axel)
    e12, e22, e28 = rr / "e12/read", rr / "e22/read", rr / "e28/read"
    e32_extra = rr / "e32/dev/read"
    return {"human": [e12 / "human"], "ppdl3_z": [e12 / "ppdl3_z"],
            "ppdl3_z+a_f": [e12 / "ppdl3_z", e28 / "german_print_s/a_z"],
            "c1": [e22 / "ppdl3_z"], "c1+m": [e22 / "ppdl3_z"],
            # E32 reads the lines added by A and M+A in [its own extra dir, then E28's a_z] (e32.read_dirs); the extra dir first
            "c1+a": [e22 / "ppdl3_z", e32_extra / "mccatmus_press/extra", e28 / "mccatmus_press/a_z"],
            "c1+ma": [e22 / "ppdl3_z", e32_extra / "mccatmus_press/extra", e28 / "mccatmus_press/a_z"],
            # C3 reads the recalibrated Gallica lines with mccatmus_press and keeps its cuts (E26); dev pages only
            "c3": [rr / "e27/read/gallica_t"]}.get(label, [])


def read_index(dirs: list[Path], page: str) -> tuple[dict, dict]:
    texts: dict = {}
    ctc: dict = {}
    for d in dirs:
        f = d / f"{page}.json"
        if f.exists():
            j = json.loads(f.read_text())
            for k, v in j.get("texts", {}).items():
                texts.setdefault(k, v)
            for k, v in (j.get("ctc") or {}).items():
                ctc.setdefault(k, v)
    return texts, ctc


def unit_spans(unit_text: str, uid: str, texts: dict, ctc: dict, reading_of):
    """Per-character cuts aligned to the chain's own text, or None. Nothing is invented when they do not align."""
    raw, c = texts.get(uid), ctc.get(uid)
    if raw is None or not c or not c.get("cuts"):
        return None, None, "no_cuts"
    cuts, conf = c["cuts"], c.get("conf")
    if raw == unit_text and len(cuts) == len(raw):
        return ([(float(a), float(b)) for a, b in cuts], [float(x) for x in conf] if conf and len(conf) == len(raw) else None,
                "cuts_direct")
    r = reading_of(raw, c)
    if r.text == unit_text and r.chars and len(r.chars) == len(unit_text):
        return [(ch.x0, ch.x1) for ch in r.chars], [ch.conf for ch in r.chars], "reading_of"
    return None, None, "misaligned"


def strict_export(graph, kind: str, image_name: str):
    try:
        if kind == "alto":
            result = export_alto(graph, image_filename=image_name, physical_img_nr=1, mode="strict")
        else:
            result = export_pagexml(graph, image_filename=image_name, mode="strict", quantization="nearest_pixel")
    except ExportError as exc:
        return {"accepted": False, "refusal": str(exc)[:200]}, None
    codes: dict = {}
    declared: dict = {}  # info losses the lot counts apart: empty_line_omitted, word_geometry_unavailable
    for loss in result.losses:
        if loss.severity != "info":
            codes[f"{loss.severity}:{loss.code}"] = codes.get(f"{loss.severity}:{loss.code}", 0) + 1
        elif loss.code in ("empty_line_omitted", "blank_line_omitted", "word_geometry_unavailable"):
            declared[loss.code] = declared.get(loss.code, 0) + 1
    entry: dict = {"accepted": result.accepted, "loss_codes": codes, "declared_info_losses": declared}
    if not result.accepted:
        entry["refusal"] = "strict refused: " + ", ".join(k for k in codes if k.startswith("error:"))
        return entry, None
    try:
        validate_export_xsd(result)
        entry["xsd"] = "valid"
    except ExportError as exc:
        entry["xsd"] = f"invalid: {str(exc)[:160]}"
    return entry, result.xml


def run_p01c(*, axel, label, name, chain, page_info, image, provenance, gt_texts, original_cer, reading_of, score_page, pooled):
    """LOT-P01c: valid ALTO and PAGE (strict) for one chain and page, every repair declared, CER untouched."""
    width, height = int(page_info["width"]), int(page_info["height"])
    texts, ctc = read_index(read_dirs(axel, label), name)
    repairs: Counter = Counter()
    how: Counter = Counter()
    is_valid = lambda pts: _valid(pts, width, height)  # noqa: E731
    units, words_by_line, shares, uncut_words, all_words, join_mismatch = [], {}, [], 0, 0, 0
    closing = 0
    for u in chain.units:
        poly = [(float(x), float(y)) for x, y in u.polygon]
        if len(poly) > 3 and poly[0] == poly[-1]:
            poly = poly[:-1]
            closing += 1
        base = [(float(x), float(y)) for x, y in u.baseline]
        up, down = line_extent(poly, base) if len(base) >= 2 else (0.0, 0.0)
        fixed, code = repair_polygon(poly, is_valid, base, up, down, width, height)
        if code:
            repairs[code] += 1
        line_poly = fixed if fixed is not None else poly
        spans, confs, status = unit_spans(u.text, u.id, texts, ctc, reading_of)
        how[status] += 1
        words = split_words(u.text, spans, confs)
        if words_text(words) != normalized(u.text):
            join_mismatch += 1
        items = []
        for w in words:
            box, clamped = word_polygon(w, base, up, down, width, height) if len(base) >= 2 else (None, False)
            repairs["word_box_clamped_to_image"] += int(clamped)
            all_words += 1
            if box is None:
                uncut_words += 1
            else:
                shares.append(outside_share(box, line_poly))
            items.append((w.text, box))
        words_by_line[u.id] = items
        units.append({"id": u.id, "text": u.text, "polygon": fixed if fixed is not None else u.polygon, "baseline": u.baseline, "block": u.block})
    labels = {b.id: b.label for b in chain.blocks}
    for bid in {u["block"] for u in units if u["block"] is not None} - set(labels):
        labels[bid] = "TextBlock"
        repairs["block_label_defaulted"] += 1
    pooled.extend(shares)
    out: dict = {"cut_alignment": dict(how), "closing_vertex_convention": closing,
                 "words": {"total": all_words, "without_box": uncut_words,
                           "share_without_box": (uncut_words / all_words) if all_words else None,
                           "page_words_complete": (uncut_words / all_words <= 0.01) if all_words else None,
                           "join_mismatch_lines": join_mismatch},
                 "outside_polygon_share": distribution(shares)}
    try:
        graph, stats = build_graph(units, role="prediction", identity={"page_key": f"e12dev:{name}", "document_key": None, "title_key": None},
                                   image=image, provenance=provenance, block_labels=labels)
        graph_lines = copy.deepcopy(graph)  # lines-only twin, kept for the PAGE fallback
        wstats = add_words(graph, words_by_line)
    except (PageGraphError, BenchError) as exc:
        out.update(graph="refused", refusal=str(exc)[:240], repairs=dict(repairs))
        return out
    repairs["line_text_whitespace_normalized"] += wstats["lines_whitespace_normalized"]
    out["words"]["lines_without_words"] = wstats["lines_without_words"]
    out["graph"] = "valid"
    out["repairs"] = dict(repairs)
    image_name = Path(page_info["image"]).name
    source_empty = sum(1 for u in chain.units if u.text == "")
    source_blank = sum(1 for u in chain.units if u.text != "" and u.text.strip() == "")
    empty_omitted = blank_omitted = 0
    for kind in ("alto", "page"):
        entry, xml = strict_export(graph, kind, image_name)
        if kind == "alto":
            # S1 (v4, pJ precision 1): the exporter omits a line with empty source text and no word from the ALTO only,
            # declares empty_line_omitted, and keeps the line in the PAGE and in the companion graph
            empty_omitted = entry.get("declared_info_losses", {}).get("empty_line_omitted", 0)
            blank_omitted = entry.get("declared_info_losses", {}).get("blank_line_omitted", 0)
            entry["empty_line_omitted"] = empty_omitted
            entry["blank_line_omitted"] = blank_omitted
        if xml is not None:
            text_lines = len(units) - (empty_omitted + blank_omitted if kind == "alto" else 0)
            entry["checks"] = alto_checks(xml, text_lines) if kind == "alto" else page_checks(xml, len(units))
            back = texts_from_alto(xml) if kind == "alto" else texts_from_page(xml)
            cer = score_page(back, gt_texts).cer_reordered
            entry["round_trip"] = {"lines_read_back": len(back), "cer_reordered_from_export": cer,
                                   "cer_reordered_original_lines": original_cer, "equal": cer == original_cer}
        out[kind] = entry
    if not out["page"].get("accepted"):
        # PAGE does not need the word level: a lines-only PAGE is a legitimate, separately declared output
        entry, xml = strict_export(graph_lines, "page", image_name)
        if xml is not None:
            entry["checks"] = page_checks(xml, len(units))
            back = texts_from_page(xml)
            cer = score_page(back, gt_texts).cer_reordered
            entry["round_trip"] = {"lines_read_back": len(back), "cer_reordered_from_export": cer,
                                   "cer_reordered_original_lines": original_cer, "equal": cer == original_cer}
        out["page_lines_only"] = entry
    alto_ok = bool(out["alto"].get("accepted") and out["alto"].get("xsd") == "valid")
    page_words_ok = bool(out["page"].get("accepted") and out["page"].get("xsd") == "valid")
    page_lines_only_ok = bool(out.get("page_lines_only", {}).get("accepted") and out["page_lines_only"].get("xsd") == "valid")
    if not page_words_ok and page_lines_only_ok:
        repairs["page_lines_only_fallback"] += 1  # a PAGE Word needs Coords: the word level is left out, declared
        out["repairs"] = dict(repairs)
    valid = alto_ok and page_words_ok  # a lines-only fallback is declared apart and is never counted as a success
    out["page_with_words_valid"] = page_words_ok
    out["both_valid"] = valid
    out["valid_without_any_repair"] = valid and sum(repairs.values()) == 0
    out["valid_with_declared_repairs"] = valid and sum(repairs.values()) > 0
    out["page_lines_only_valid"] = bool(out.get("page_lines_only", {}).get("accepted") and out["page_lines_only"].get("xsd") == "valid")
    out["empty_line_omitted"] = empty_omitted
    alto_acc = bool(out["alto"].get("accepted"))
    out["blank_line_omitted"] = blank_omitted
    out["empty_line_check"] = {
        "source_empty_lines": source_empty, "source_blank_lines": source_blank, "lines_without_words": wstats["lines_without_words"],
        "empty_omitted_in_alto": empty_omitted if alto_acc else None, "blank_omitted_in_alto": blank_omitted if alto_acc else None,
        "equal": (empty_omitted == source_empty and blank_omitted == source_blank
                  and source_empty + source_blank == wstats["lines_without_words"]) if alto_acc else None}
    out["word_geometry_unavailable"] = out["alto"].get("declared_info_losses", {}).get("word_geometry_unavailable", 0)
    out["lines_lost"] = {k: out[k].get("checks", {}).get("lines_lost") for k in ("alto", "page")}  # ALTO: lines carrying text
    return out


def _valid(points, width, height) -> bool:
    try:
        _polygon([list(p) for p in points], "$", width, height)
        return True
    except PageGraphError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--axel", type=Path, default=Path.home() / "axel")
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "bench_pipeline_v1.json")
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "bench_pipeline_dev.json")
    parser.add_argument("--only-page", help="debug: keep pages whose name starts with this (skips the controls)")
    parser.add_argument("--chains", help="comma-separated chain labels to keep (default: all); LOT-P07 uses c1,c1+m,c1+a,c1+ma")
    parser.add_argument("--no-controls", action="store_true", help="skip the reproduction controls (they concern the 5 pages of the E12 dev)")
    parser.add_argument("--label", help="label written in the report (default: the dev exposé label)")
    parser.add_argument("--part-a-only", action="store_true", help="skip the part b metrics")
    parser.add_argument("--skip-part-c", action="store_true", help="skip the P-01c words/exports pass")
    args = parser.parse_args()
    axel = args.axel.resolve()
    if args.out.resolve().is_relative_to(axel):
        raise BenchError("the bench never writes inside axel")
    config = load_config(args.config)
    sys.path[:0] = [str(axel / "src"), str(axel / "scripts")]
    from axel.eval.doc_bootstrap import DocScore, micro, paired_difference
    from axel.layout.gt import load_page
    from axel.layout.pagetext import score_page
    from axel.pipeline.chain import reading_of
    e27 = importlib.import_module("e27_decompose")
    e28 = importlib.import_module("e28_fill")
    e32 = importlib.import_module("e32")

    install_polygon_cache()  # performance only; results unchanged (see bench_pipeline.py)
    head = axel_head(axel)
    config_sha = file_sha256(args.config)
    pages = e28.dev_pages()
    if args.only_page:
        pages = [p for p in pages if p["page"].startswith(args.only_page)]
    bench = e32.BENCHES["dev"]
    chains: dict[str, dict] = {}

    def add(label: str, page: str, chain) -> None:
        if chain is not None:
            chains.setdefault(label, {})[page] = chain

    for p in pages:
        name = p["page"]
        for label, key in (("human", "human"), ("tesseract", "tesseract"), ("ppdl3_z", "ppdl3_z"),
                           ("c1", "c1"), ("gallica", "gallica"), ("c3", "c3")):
            add(label, name, e27.load(key, p))
        arms = e32.build_arms(bench, "c1", name)
        if arms is not None:
            for arm in ("m", "a", "ma"):
                add(f"c1+{arm}", name, arms[0].get(arm))
        base = chains.get("ppdl3_z", {}).get(name)
        if base is not None:
            got = e28.build_arm(base, "ppdl3_z", "german_print_s", "a_z", name, True)
            add("ppdl3_z+a_f", name, got[0] if got else None)

    if args.chains:
        keep = {c.strip() for c in args.chains.split(",")}
        chains = {k: v for k, v in chains.items() if k in keep}
    rows, scores, metric_rows = [], {}, {}
    pooled_outside: dict[str, list] = {}
    page_info = {p["page"]: p for p in pages}
    gt_cache = {}
    for label, by_page in chains.items():
        docs = []
        for name, chain in by_page.items():
            p = page_info[name]
            t_page = time.perf_counter()
            if name not in gt_cache:
                gt_cache[name] = load_page(Path(p["xml"]))
            gt = gt_cache[name]
            gt_lines = gt.lines()
            t0 = time.perf_counter()
            ps = score_page([u.text for u in chain.units], [g.text for g in gt_lines])
            seconds = time.perf_counter() - t0
            docs.append(DocScore(name, ps.cer_reordered * ps.chars, ps.chars))
            image = {"sha256": file_sha256(p["image"]), "width": int(p["width"]), "height": int(p["height"])}
            units = [{"id": u.id, "text": u.text, "polygon": u.polygon, "baseline": u.baseline, "block": u.block}
                     for u in chain.units]
            provenance = {"run_id": f"bench-pipeline-v1:{label}:{name}", "components": [{
                "name": label, "code_revision": head, "config_sha256": config_sha,
                "checkpoint_sha256": None, "license": None}]}
            row = {"chain": label, "page": name, "units": len(units), "chars": ps.chars,
                   "cer_reordered": ps.cer_reordered, "score_seconds": round(seconds, 3)}
            graph = None
            try:
                graph, stats = build_graph(units, role="prediction", identity={
                    "page_key": f"e12dev:{name}", "document_key": None, "title_key": None},
                    image=image, provenance=provenance,
                    block_labels={b.id: b.label for b in chain.blocks})
                row.update(graph="valid", adapter=stats)
            except (PageGraphError, BenchError) as exc:
                row.update(graph="refused", refusal=str(exc)[:300])
            if not args.part_a_only:
                base_chain = chains.get(BASE_OF.get(label, label), {}).get(name, chain)
                zones = [blk.box for blk in chain.blocks]
                m = e28.page_metrics(chain, p, zones, {u.id for u in base_chain.units})
                metric_rows.setdefault(label, {})[name] = m
                row["cer_columns"] = {"cer_ordered": m["cer_ordered"], "cer_reordered_d217": m["cer_reordered"],
                                      "cer_reordered_d212": m["cer_reordered_d212"],
                                      "cer_local": m["local"] / m["gt_chars"] if m["gt_chars"] else None,
                                      "cer_bag_d556": m["bag"] / m["bag_chars"] if m["bag_chars"] else None}
                row["cer_columns_check"] = {"d217_equals_part_a": m["cer_reordered"] == ps.cer_reordered}
                row["splits_and_merges"] = {"split_cost_chars": m["split"], "split_gt_lines": m["split_lines"],
                                            "merge_tangle_cost_chars": m["tangle"], "missed_lines": m["missed_lines"],
                                            "missed_cost_chars": m["missed"]}
                row["inventions"] = {"added_lines": m["added"], "invented_out": m["added_invented_out"],
                                     "invented_in": m["added_invented_in"], "invented_chars": m["added_invented_chars"],
                                     "definition": "E28/E32 : lignes ajoutées hors de la base, classées EXTRA_OUT/EXTRA_IN"}
                # a truth line without a usable baseline is "missing" (contract: the page becomes non_evaluable, counts stay diagnostic)
                gt_geo = [GroundTruthLine(g.id, g.baseline, bbox_height(g.polygon), "ground_truth")
                          if g.baseline and len(g.baseline) >= 2 and bbox_height(g.polygon)
                          else GroundTruthLine(g.id, None, None, None, "missing") for g in gt_lines]
                pred_geo = [PredictedLine(u.id, [list(q) for q in u.baseline], "predicted")
                            if u.baseline and len(u.baseline) >= 2 else PredictedLine(u.id, None, None, "missing")
                            for u in chain.units]
                lg = evaluate_line_geometry(gt_geo, pred_geo)
                lg1 = evaluate_line_geometry_sensitivity(gt_geo, pred_geo)
                matches = [(x.gt_id, x.pred_id) for x in lg.matches]
                row["line_geometry"] = {
                    "status": lg.status, "recall": lg.recall, "precision": lg.precision, "f1": lg.f1,
                    "n_gt": lg.n_gt, "n_pred": lg.n_pred, "matched": lg.matched_count,
                    "gt_non_evaluable": lg.gt_non_evaluable_count,
                    "note": ("diagnostic : au moins une ligne vraie sans ligne de base, rappel et précision non publiés (contrat)"
                             if lg.status == "non_evaluable" else None),
                    "sensitivity_1h_matched": lg1.matched_count, "sensitivity_1h_recall": lg1.recall,
                    "split_candidate_gt": lg.split_candidate_gt_count, "merge_candidate_pred": lg.merge_candidate_pred_count,
                    "orientation_disagreements": lg.orientation_disagreement_count,
                    "height_px_basis": "hauteur de la boîte du polygone vrai (hypothèse du banc)"}
                gt_order = {g.id: i for i, g in enumerate(gt_lines)}
                pred_order = {u.id: i for i, u in enumerate(chain.units)}
                row["reading_order"] = order_report(name, matches, gt_order, pred_order, len(gt_lines))
                row["line_success"] = line_success(name, matches, {g.id: g.text for g in gt_lines},
                                                   {u.id: u.text for u in chain.units}, gt_order, pred_order, len(gt_lines))
                if graph is not None:
                    image_name = Path(p["image"]).name
                    row["alto"] = export_report(graph, "alto", image_name, len(units))
                    row["page_xml"] = export_report(graph, "page", image_name, len(units))
                else:
                    row["alto"] = row["page_xml"] = {"outcome": "explicit_refusal", "reason": "graph refused"}
                if not args.skip_part_c:
                    row["p01c"] = run_p01c(axel=axel, label=label, name=name, chain=chain, page_info=p, image=image, provenance=provenance,
                                           gt_texts=[g.text for g in gt_lines], original_cer=ps.cer_reordered, reading_of=reading_of,
                                           score_page=score_page, pooled=pooled_outside.setdefault(label, []))
                row["cost"] = {"seconds_page_total": round(time.perf_counter() - t_page, 2)}
                print(f"{label:12s} {name}: D-217 {100 * m['cer_reordered']:.2f} | recall {lg.recall} | alto {row['alto']['outcome']}",
                      flush=True)
            rows.append(row)
        scores[label] = micro(docs)

    measured = {"c1_base": scores.get("c1"), "c1_ma": scores.get("c1+ma"), "ppdl3_z": scores.get("ppdl3_z"),
                "human": scores.get("human"), "e28_ppdl3_z_a_f": scores.get("ppdl3_z+a_f")}
    controls = None if (args.only_page or args.no_controls) else check_controls(measured, config, axel)

    aggregates, comparisons = {}, {}
    if not args.part_a_only:
        for label, by_page in metric_rows.items():
            rs = list(by_page.values())
            lab_rows = [r for r in rows if r["chain"] == label]
            agg = {"pages": sorted(by_page)}
            for w in e28.METRICS:
                agg[w] = micro([e28.doc(r, w) for r in rs])
            ev = [r for r in lab_rows if r["line_geometry"]["status"] == "evaluated"]
            gt_n = sum(r["line_geometry"]["n_gt"] for r in ev)
            agg["line_geometry_pages_evaluated"] = [r["page"] for r in ev]
            agg["line_geometry_pages_non_evaluable"] = [r["page"] for r in lab_rows if r["line_geometry"]["status"] != "evaluated"]
            agg["line_recall_0.5h"] = sum(r["line_geometry"]["matched"] for r in ev) / gt_n if gt_n else None
            agg["line_precision_0.5h"] = (sum(r["line_geometry"]["matched"] for r in ev)
                                          / max(sum(r["line_geometry"]["n_pred"] for r in ev), 1)) if ev else None
            agg["line_recall_1h"] = sum(r["line_geometry"]["sensitivity_1h_matched"] for r in ev) / gt_n if gt_n else None
            all_gt = sum(r["line_geometry"]["n_gt"] for r in lab_rows)
            agg["line_recall_0.5h_all_pages_diagnostic"] = sum(r["line_geometry"]["matched"] for r in lab_rows) / all_gt if all_gt else None
            ls = [r["line_success"] for r in lab_rows]
            ls_gt = sum(x["gt_lines"] for x in ls)
            agg["share_matched_and_cer_under_15"] = sum(x["matched_and_cer_under_threshold"] for x in ls) / ls_gt if ls_gt else None
            app = [x for x in ls if x["order_condition"] == "applicable"]
            agg["share_success_with_order"] = (sum(x["matched_cer_and_same_rank"] for x in app) / sum(x["gt_lines"] for x in app)
                                               if app else None)
            agg["share_success_with_predecessor"] = (sum(x["matched_cer_and_same_predecessor"] for x in app) / sum(x["gt_lines"] for x in app)
                                                     if app else None)
            taus = [r["reading_order"]["tau"] for r in lab_rows if r["reading_order"]["tau"] is not None]
            agg["tau_mean_applicable_pages"] = sum(taus) / len(taus) if taus else None
            agg["tau_pages_applicable"] = len(taus)
            agg["invented_lines"] = sum(r["inventions"]["invented_out"] + r["inventions"]["invented_in"] for r in lab_rows)
            agg["invented_chars"] = sum(r["inventions"]["invented_chars"] for r in lab_rows)
            agg["alto_outcomes"] = {o: sum(r["alto"]["outcome"] == o for r in lab_rows) for o in ("valid", "valid_lines_lost", "invalid_xsd", "explicit_refusal")}
            agg["page_outcomes"] = {o: sum(r["page_xml"]["outcome"] == o for r in lab_rows) for o in ("valid", "valid_lines_lost", "invalid_xsd", "explicit_refusal")}
            agg["alto_best_effort_lines_lost"] = sum((r["alto"].get("best_effort_summary") or {}).get("lines_lost") or 0 for r in lab_rows)
            aggregates[label] = agg
        ref = metric_rows.get("c1", {})
        for label, by_page in metric_rows.items():
            if label == "c1":
                continue
            common = sorted(set(by_page) & set(ref))
            if len(common) < 2:
                continue
            entry = {"pages": common, "n_pages": len(common), "note": "IC apparié par page, 20 000 tirages, graine 0"}
            for w in CI_METRICS:
                diff, ci = paired_difference([e28.doc(by_page[q], w) for q in common], [e28.doc(ref[q], w) for q in common],
                                             n_iter=e28.N_ITER, seed=e28.SEED)
                entry[w] = {"diff_vs_c1": diff, "ci95": ci}
            comparisons[f"{label} − c1"] = entry

    p01c_aggregates, p01c_acceptance = {}, {}
    if not args.part_a_only and not args.skip_part_c:
        for label in metric_rows:
            cr = [r["p01c"] for r in rows if r["chain"] == label and "p01c" in r]
            ok_graph = [c for c in cr if c.get("graph") == "valid"]
            repairs: Counter = Counter()
            for c in ok_graph:
                repairs.update(c["repairs"])
            trips = [c[k]["round_trip"] for c in ok_graph for k in ("alto", "page") if c.get(k, {}).get("round_trip")]
            lost = [v for c in ok_graph for v in (c.get("lines_lost") or {}).values() if v is not None]
            tw = sum(c["words"]["total"] for c in ok_graph)
            uw = sum(c["words"]["without_box"] for c in ok_graph)
            refusals = Counter((c.get(k, {}).get("refusal") or "").replace("strict refused: ", "") for c in ok_graph for k in ("alto", "page")
                               if c.get(k) and not c[k].get("accepted"))
            agg = {"pages": len(cr), "graph_refused_pages": len(cr) - len(ok_graph),
                   "alto_strict_valid_pages": sum(c.get("alto", {}).get("xsd") == "valid" for c in ok_graph),
                   "page_strict_valid_pages": sum(c.get("page", {}).get("xsd") == "valid" for c in ok_graph),
                   "page_with_words_valid_pages": sum(bool(c.get("page_with_words_valid")) for c in ok_graph),
                   "page_lines_only_valid_pages": sum(bool(c.get("page_lines_only_valid")) for c in ok_graph),
                   "both_valid_pages": sum(bool(c.get("both_valid")) for c in ok_graph),
                   "valid_without_any_repair_pages": sum(bool(c.get("valid_without_any_repair")) for c in ok_graph),
                   "valid_with_declared_repairs_pages": sum(bool(c.get("valid_with_declared_repairs")) for c in ok_graph),
                   "lines_carrying_text_lost_total": sum(lost), "accepted_exports_checked_for_loss": len(lost),
                   "empty_line_omitted_total_in_alto": sum(c.get("empty_line_omitted", 0) for c in ok_graph),
                   "blank_line_omitted_total_in_alto": sum(c.get("blank_line_omitted", 0) for c in ok_graph),
                   "empty_line_count_equals_source_empty_lines_all_pages": all(c["empty_line_check"]["equal"] is not False for c in ok_graph),
                   "word_geometry_unavailable_total": sum(c.get("word_geometry_unavailable", 0) for c in ok_graph),
                   "repairs_by_code": dict(repairs), "words_total": tw, "words_without_box": uw,
                   "share_words_without_box": (uw / tw) if tw else None,
                   "pages_words_complete": sum(bool(c["words"]["page_words_complete"]) for c in ok_graph),
                   "pages_words_incomplete": sum(c["words"]["page_words_complete"] is False for c in ok_graph),
                   "words_join_mismatch_lines": sum(c["words"]["join_mismatch_lines"] for c in ok_graph),
                   "lines_without_words": sum(c["words"].get("lines_without_words", 0) for c in ok_graph),
                   "outside_polygon_share_pooled": distribution(pooled_outside.get(label, [])),
                   "round_trip": {"checked": len(trips), "equal": sum(t["equal"] for t in trips)},
                   "refusals": dict(refusals),
                   "cut_alignment": dict(sum((Counter(c["cut_alignment"]) for c in ok_graph), Counter()))}
            p01c_aggregates[label] = agg
            elements = {
                "alto_strict_valid_all_pages": agg["alto_strict_valid_pages"] == len(cr),
                "page_strict_valid_with_words_all_pages": agg["page_with_words_valid_pages"] == len(cr),
                "no_line_carrying_text_lost": agg["lines_carrying_text_lost_total"] == 0 and agg["accepted_exports_checked_for_loss"] > 0,
                "round_trip_equal_all": agg["round_trip"]["checked"] > 0 and agg["round_trip"]["checked"] == agg["round_trip"]["equal"],
                "words_complete_all_pages": agg["pages_words_incomplete"] == 0 and agg["pages_words_complete"] == len(cr),
                "empty_and_blank_counts_equal": agg["empty_line_count_equals_source_empty_lines_all_pages"],
            }
            met = bool(cr) and all(elements.values())
            if label in REFERENCE_NO_CUTS:
                p01c_acceptance[label] = "reference_chain_no_cuts"
            elif label == "human":
                p01c_acceptance[label] = "control (not a pipeline output)"
            else:
                p01c_acceptance[label] = met
            agg["criterion_elements"] = elements
            agg["criterion_met_if_evaluated"] = met
            agg["page_lines_only_fallback_pages_not_counted_as_success"] = agg["page_lines_only_valid_pages"]

    gt_rows = []
    for name, gt in gt_cache.items():
        p = page_info[name]
        units = [{"id": g.id, "text": g.text, "polygon": g.polygon, "baseline": g.baseline, "block": None}
                 for g in gt.lines()]
        image = {"sha256": file_sha256(p["image"]), "width": int(p["width"]), "height": int(p["height"])}
        provenance = {"run_id": f"bench-pipeline-v1:ground_truth:{name}", "components": [{
            "name": "newseye_page_xml", "code_revision": head, "config_sha256": config_sha,
            "checkpoint_sha256": None, "license": "CC BY 4.0"}]}
        try:
            _, stats = build_graph(units, role="ground_truth", identity={
                "page_key": f"e12dev:{name}", "document_key": None, "title_key": None},
                image=image, provenance=provenance)
            gt_rows.append({"page": name, "graph": "valid", "adapter": stats})
        except (PageGraphError, BenchError) as exc:
            gt_rows.append({"page": name, "graph": "refused", "refusal": str(exc)[:300]})

    slots = empty_metric_slots(config)
    if not args.part_a_only:
        for name in ("cer_columns", "segmentation_errors", "line_recall_precision", "reading_order_kendall_tau",
                     "alto_page_validity", "inventions", "cost", "paired_ci_vs_c1"):
            if name in slots:
                slots[name]["status"] = "computed" if name != "paired_ci_vs_c1" or comparisons else "no_common_pages"
                slots[name]["value"] = "voir chains[], chain_aggregates, comparisons_vs_c1"
        slots["line_success"] = {"value": "voir chains[].line_success et chain_aggregates", "status": "computed",
                                 "definition": "ligne vraie appariée par line_geometry.v1, CER de ligne < 15 %, même rang parmi les lignes appariées ; ordre non applicable sur Ce-soir f7 et Regards f5. Descriptif, jamais un critère."}
    report = {
        "schema_version": "babaorum.bench_pipeline_report.v2", "label": args.label or LABEL,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": {"axel_head": head, "axel_root": str(axel), "interpreter": sys.executable,
                       "config": str(args.config), "config_sha256": config_sha,
                       "forbidden": config["forbidden_path_fragments"], "axel_written": False,
                       "part_b": not args.part_a_only, "only_page": args.only_page},
        "controls": controls,
        "all_controls_exact": (all(c["status"] == "exact" for c in controls) if controls else None),
        "chains": rows,
        "chain_aggregates": aggregates,
        "comparisons_vs_c1": comparisons,
        "p01c_aggregates": p01c_aggregates,
        "p01c_accepted_chains": p01c_acceptance,
        "ground_truth_graphs": gt_rows,
        "adapter_summary": {
            "chain_pages": len(rows),
            "valid": sum(r["graph"] == "valid" for r in rows),
            "refused": sum(r["graph"] == "refused" for r in rows),
            "invalid_polygons": sum(r.get("adapter", {}).get("invalid_polygons", 0) for r in rows),
            "invalid_baselines": sum(r.get("adapter", {}).get("invalid_baselines", 0) for r in rows),
            "closing_vertex_dropped": sum(r.get("adapter", {}).get("closing_vertex_dropped", 0) for r in rows),
        },
        "cbad_baseline_rpf": slots.get("cbad_baseline_rpf"),
        "metric_slots": slots,
        "cost": {"process_peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024), 1)},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for c in controls or []:
        print(f"{c['id']:18s} {c['status']:10s} measured={c['measured']!r} expected={c['expected']!r} diff={c['diff']}")
    print("adapter:", report["adapter_summary"])
    return 0 if (controls is None or report["all_controls_exact"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
