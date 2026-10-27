#!/usr/bin/env python3
"""Reproduce the geometry-only global Kraken smoke on synthetic mf_0036."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

from babaorum.evidence import atomic_write
from babaorum.line_geometry import (
    GroundTruthLine, PredictedLine, _bounds, _bounds_distance_lower_bound,
    _pair_measure, _sample, _validated_gt, _validated_pred, evaluate_line_geometry,
)


ROOT = Path(__file__).resolve().parents[1]
PILOT = Path('/Users/marcel/heritage-synth/runs/pilot-v0.2')
MODEL = Path('/Users/marcel/outils-seg/modeles/blla-2026/blla.mlmodel')
EXPECTED = {
    'image': '43cdc8bca3b077f561607bd8091a798645cb6eb3ea73f73dcf2ea9c2af4ebaea',
    'annotation': '08ece81f878de97a549d0f66c7707294ce37161a930ecc46fbb1d48284012722',
    'model': 'a3ccc7309fe8df62d1cbd372706edc73db6a2a32806e3758522ed8528a734073',
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def gt_height(baseline, polygon) -> float:
    (x0, y0), (x1, y1) = baseline
    dx, dy = x1 - x0, y1 - y0
    norm = math.hypot(dx, dy)
    if norm == 0:
        raise ValueError('degenerate ground-truth baseline')
    nx, ny = -dy / norm, dx / norm  # contract normal, y-down coordinates
    signed = [(x - x0) * nx + (y - y0) * ny for x, y in polygon]
    height = max(signed) - min(signed)
    if not math.isfinite(height) or height <= 0:
        raise ValueError('ground-truth polygon gives no positive baseline band')
    return height


def candidate_parent_diagnostics(ground_truth, predicted, result, block_by_line, block_x_intervals):
    """Reproduce line_geometry.v1's permissive split/merge candidate edges."""
    gt_lines, incomplete = _validated_gt(ground_truth)
    pred_lines, _ = _validated_pred(predicted)
    if incomplete:
        raise ValueError('candidate parent analysis requires complete GT')
    gt_lines.sort(key=lambda row: row[0].id)
    pred_lines.sort(key=lambda row: row[0].id)
    pred_bounds = [_bounds(points) for _, points in pred_lines]
    matched_predictions = {match.pred_id for match in result.matches}
    gt_neighbors: dict[int, set[int]] = {i: set() for i in range(len(gt_lines))}
    pred_neighbors: dict[int, set[int]] = {j: set() for j in range(len(pred_lines))}
    for i, (gt_line, gt_points, height) in enumerate(gt_lines):
        gt_samples = _sample(gt_points, min(4.0, height / 4.0))
        gt_bounds = _bounds(gt_points)
        for j, (pred_line, pred_points) in enumerate(pred_lines):
            if _bounds_distance_lower_bound(gt_bounds, pred_bounds[j], height=height) > 0.5 * height:
                continue
            pred_samples = _sample(pred_points, min(4.0, height / 4.0))
            cover_gt, cover_pred, _, _ = _pair_measure(
                gt_points, pred_points, gt_samples, pred_samples, height, 0.5
            )
            if cover_gt >= 0.25 or cover_pred >= 0.50:
                gt_neighbors[i].add(j)
                pred_neighbors[j].add(i)
    merge_candidates = {j for j, neighbors in pred_neighbors.items() if len(neighbors) >= 2}
    touched_gt = {i for j in merge_candidates for i in pred_neighbors[j]}
    if len(merge_candidates) != result.merge_candidate_pred_count:
        raise ValueError('merge-candidate reconstruction differs from metric output')
    if len(touched_gt) != result.merge_candidate_gt_count:
        raise ValueError('touched-GT reconstruction differs from metric output')
    rows = []
    for j in sorted(merge_candidates, key=lambda index: pred_lines[index][0].id):
        line_ids = sorted(gt_lines[i][0].id for i in pred_neighbors[j])
        block_ids = sorted({block_by_line[line_id] for line_id in line_ids})
        intervals = sorted(block_x_intervals[block_id] for block_id in block_ids)
        x_bands = []
        for lo, hi in intervals:
            if not x_bands or lo > x_bands[-1][1]:
                x_bands.append([lo, hi])
            else:
                x_bands[-1][1] = max(x_bands[-1][1], hi)
        pred_x = [point[0] for point in pred_lines[j][1]]
        crosses_gap = (len(x_bands) > 1 and min(pred_x) <= x_bands[0][1]
                       and max(pred_x) >= x_bands[-1][0])
        rows.append({
            'prediction_id': pred_lines[j][0].id,
            'matched_by_primary_metric': pred_lines[j][0].id in matched_predictions,
            'candidate_gt_line_ids': line_ids,
            'candidate_parent_block_ids': block_ids,
            'candidate_parent_block_x_intervals': [
                {'block_id': block_id, 'min_x': block_x_intervals[block_id][0],
                 'max_x': block_x_intervals[block_id][1]}
                for block_id in block_ids
            ],
            'candidate_gt_line_count': len(line_ids),
            'candidate_parent_block_count': len(block_ids),
            'crosses_parent_blocks': len(block_ids) > 1,
            'candidate_parent_x_bands': x_bands,
            'predicted_baseline_x_span_px': [min(pred_x), max(pred_x)],
            'baseline_crosses_gap_between_parent_x_bands': crosses_gap,
            'interpretation': ('block_crop_could_separate_parent_regions' if len(block_ids) > 1
                               else 'same_parent_block_crop_does_not_structurally_split_sibling_lines'),
        })
    return {
        'rule': 'line_geometry.v1 diagnostic adjacency: cover_gt >= 0.25 OR cover_pred >= 0.50 at 0.5 hGT',
        'merge_candidate_predictions': rows,
        'summary': {
            'candidate_prediction_count': len(rows),
            'matched_candidate_predictions': sum(row['matched_by_primary_metric'] for row in rows),
            'unmatched_candidate_predictions': sum(not row['matched_by_primary_metric'] for row in rows),
            'cross_parent_block_predictions': sum(row['crosses_parent_blocks'] for row in rows),
            'baseline_crosses_parent_x_gap': sum(row['baseline_crosses_gap_between_parent_x_bands'] for row in rows),
            'parent_block_count_histogram': {
                str(count): sum(row['candidate_parent_block_count'] == count for row in rows)
                for count in sorted({row['candidate_parent_block_count'] for row in rows})
            },
            'single_parent_block_predictions': sum(not row['crosses_parent_blocks'] for row in rows),
            'unique_gt_lines_touched': len(touched_gt),
        },
        'limits': 'Adjacency candidates are geometric diagnostics, not verified merges or proof that local segmentation fixes them.',
    }


def crop_rect(block_polygon, width, height, margin=24):
    xs = [point[0] for point in block_polygon]
    ys = [point[1] for point in block_polygon]
    return (max(0, int(min(xs)) - margin), max(0, int(min(ys)) - margin),
            min(width, int(max(xs) + margin)), min(height, int(max(ys) + margin)))


def block_x_interval(block_polygon):
    xs = [point[0] for point in block_polygon]
    return min(xs), max(xs)


def segment_rect_fraction(a, b, rect):
    x0, y0, x1, y1 = rect
    dx, dy = b[0] - a[0], b[1] - a[1]
    lower, upper = 0.0, 1.0
    for p, q in ((-dx, a[0] - x0), (dx, x1 - a[0]),
                 (-dy, a[1] - y0), (dy, y1 - a[1])):
        if p == 0:
            if q < 0:
                return 0.0
            continue
        ratio = q / p
        if p < 0:
            lower = max(lower, ratio)
        else:
            upper = min(upper, ratio)
        if lower > upper:
            return 0.0
    return max(0.0, upper - lower)


def clip_polygon_to_rect(polygon, rect):
    x0, y0, x1, y1 = rect
    points = [tuple(point) for point in polygon]
    boundaries = (
        (lambda p: p[0] >= x0, lambda a, b: (x0, a[1] + (b[1] - a[1]) * (x0 - a[0]) / (b[0] - a[0]))),
        (lambda p: p[0] <= x1, lambda a, b: (x1, a[1] + (b[1] - a[1]) * (x1 - a[0]) / (b[0] - a[0]))),
        (lambda p: p[1] >= y0, lambda a, b: (a[0] + (b[0] - a[0]) * (y0 - a[1]) / (b[1] - a[1]), y0)),
        (lambda p: p[1] <= y1, lambda a, b: (a[0] + (b[0] - a[0]) * (y1 - a[1]) / (b[1] - a[1]), y1)),
    )
    for inside, intersect in boundaries:
        if not points:
            return []
        output = []
        previous = points[-1]
        previous_inside = inside(previous)
        for current in points:
            current_inside = inside(current)
            if current_inside:
                if not previous_inside:
                    output.append(intersect(previous, current))
                output.append(current)
            elif previous_inside:
                output.append(intersect(previous, current))
            previous, previous_inside = current, current_inside
        points = output
    return points


def polygon_area(polygon):
    if len(polygon) < 3:
        return 0.0
    return abs(math.fsum(
        polygon[i][0] * polygon[(i + 1) % len(polygon)][1]
        - polygon[(i + 1) % len(polygon)][0] * polygon[i][1]
        for i in range(len(polygon))
    )) / 2.0


def polygon_pixel_mask(polygon, rect):
    x0, y0, x1, y1 = rect
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    ix0, iy0 = max(x0, math.ceil(min(xs) - 0.5)), max(y0, math.ceil(min(ys) - 0.5))
    ix1, iy1 = min(x1, math.ceil(max(xs) - 0.5)), min(y1, math.ceil(max(ys) - 0.5))
    if ix0 >= ix1 or iy0 >= iy1:
        return (ix0, iy0, ix1, iy1), np.zeros((0, 0), dtype=bool)
    grid_x, grid_y = np.meshgrid(np.arange(ix0, ix1) + 0.5, np.arange(iy0, iy1) + 0.5)
    inside = np.zeros(grid_x.shape, dtype=bool)
    previous = polygon[-1]
    for current in polygon:
        x0p, y0p = previous
        x1p, y1p = current
        crossing = (y0p > grid_y) != (y1p > grid_y)
        if y1p != y0p:
            crossing_x = (x1p - x0p) * (grid_y - y0p) / (y1p - y0p) + x0p
            inside ^= crossing & (grid_x < crossing_x)
        previous = current
    return (ix0, iy0, ix1, iy1), inside


def polygon_ink_fractions(gray, polygon, rect, thresholds=(64, 128, 192)):
    bounds, mask = polygon_pixel_mask(polygon, (0, 0, gray.shape[1], gray.shape[0]))
    x0, y0, x1, y1 = bounds
    if x0 >= x1 or y0 >= y1:
        return {str(threshold): None for threshold in thresholds}
    pixels = gray[y0:y1, x0:x1]
    cx0, cy0, cx1, cy1 = rect
    overlap_x0, overlap_y0 = max(x0, cx0), max(y0, cy0)
    overlap_x1, overlap_y1 = min(x1, cx1), min(y1, cy1)
    crop_counts = {}
    for threshold in thresholds:
        total = int(np.count_nonzero(mask & (pixels < threshold)))
        if total == 0:
            crop_counts[str(threshold)] = None
            continue
        if overlap_x0 >= overlap_x1 or overlap_y0 >= overlap_y1:
            crop_counts[str(threshold)] = 0.0
            continue
        ys, xs = overlap_y0 - y0, overlap_x0 - x0
        crop_mask = mask[ys:ys + overlap_y1 - overlap_y0, xs:xs + overlap_x1 - overlap_x0]
        crop_pixels = pixels[ys:ys + overlap_y1 - overlap_y0, xs:xs + overlap_x1 - overlap_x0]
        crop_counts[str(threshold)] = int(np.count_nonzero(crop_mask & (crop_pixels < threshold))) / total
    return crop_counts


def parent_crop_overlap_diagnostics(candidate_rows, page, image_path, margin=24):
    width, height = page['image']['width'], page['image']['height']
    with Image.open(image_path) as image:
        gray = np.asarray(image.convert('L'))
    block_by_id = {block['id']: block for block in page['blocks']}
    lines_by_id = {line['id']: line for line in page['lines']}
    pairs = {}
    adjacent_pairs = {}
    for candidate in candidate_rows:
        ordered_blocks = sorted(candidate['candidate_parent_block_ids'],
                                key=lambda block_id: (block_x_interval(block_by_id[block_id]['polygon']), block_id))
        for left_id, right_id in zip(ordered_blocks, ordered_blocks[1:]):
            left_polygon = block_by_id[left_id]['polygon']
            right_polygon = block_by_id[right_id]['polygon']
            left_x = (min(point[0] for point in left_polygon), max(point[0] for point in left_polygon))
            right_x = (min(point[0] for point in right_polygon), max(point[0] for point in right_polygon))
            gap = right_x[0] - left_x[1]
            left_rect = crop_rect(left_polygon, width, height, margin)
            right_rect = crop_rect(right_polygon, width, height, margin)
            crop_overlap = min(left_rect[2], right_rect[2]) - max(left_rect[0], right_rect[0])
            key = tuple(sorted((left_id, right_id)))
            adjacent_pairs.setdefault(key, {
                'left_block_id': left_id, 'right_block_id': right_id,
                'parent_x_gap_px': gap,
                'dilated_crop_x_overlap_px': max(0, crop_overlap),
                'gap_after_global_scale_px': gap * 1800 / height,
            })
        for crop_block_id in candidate['candidate_parent_block_ids']:
            rect = crop_rect(block_by_id[crop_block_id]['polygon'], width, height, margin)
            for neighbor_line_id in candidate['candidate_gt_line_ids']:
                neighbor = lines_by_id[neighbor_line_id]
                if neighbor['block_id'] == crop_block_id:
                    continue
                baseline = neighbor['baseline']
                segment_lengths = [math.dist(a, b) for a, b in zip(baseline, baseline[1:])]
                baseline_length = math.fsum(segment_lengths)
                baseline_inside_length = math.fsum(
                    length * segment_rect_fraction(a, b, rect)
                    for a, b, length in zip(baseline, baseline[1:], segment_lengths)
                )
                baseline_fraction = (baseline_inside_length / baseline_length
                                     if baseline_length > 0 else None)
                original_area = polygon_area(neighbor['polygon'])
                overlap_area = polygon_area(clip_polygon_to_rect(neighbor['polygon'], rect))
                ink_fractions = polygon_ink_fractions(gray, neighbor['polygon'], rect)
                key = (crop_block_id, neighbor_line_id)
                row = {
                    'crop_parent_block_id': crop_block_id,
                    'crop_rect_xyxy': list(rect),
                    'neighbor_gt_line_id': neighbor_line_id,
                    'neighbor_parent_block_id': neighbor['block_id'],
                    'neighbor_baseline_intersects_crop': baseline_inside_length > 0,
                    'neighbor_baseline_length_inside_crop_px': baseline_inside_length,
                    'neighbor_baseline_length_px': baseline_length,
                    'neighbor_baseline_fraction_inside_crop': baseline_fraction,
                    'neighbor_polygon_overlap_area_px2': overlap_area,
                    'neighbor_polygon_area_px2': original_area,
                    'neighbor_polygon_overlap_fraction': (overlap_area / original_area if original_area else None),
                    'neighbor_ink_fraction_inside_crop_by_threshold': ink_fractions,
                }
                previous = pairs.get(key)
                if previous is None or row['neighbor_polygon_overlap_fraction'] > previous['neighbor_polygon_overlap_fraction']:
                    pairs[key] = row
    records = [pairs[key] for key in sorted(pairs)]
    adjacent_records = [adjacent_pairs[key] for key in sorted(adjacent_pairs)]
    baseline_fractions = [row['neighbor_baseline_fraction_inside_crop'] for row in records
                          if row['neighbor_baseline_fraction_inside_crop'] is not None]
    positive_polygon_fractions = [row['neighbor_polygon_overlap_fraction'] for row in records
                                  if row['neighbor_polygon_overlap_fraction'] > 0]
    ink_summaries = {}
    for threshold in ('64', '128', '192'):
        values = [row['neighbor_ink_fraction_inside_crop_by_threshold'][threshold]
                  for row in records
                  if row['neighbor_ink_fraction_inside_crop_by_threshold'][threshold] is not None]
        ink_summaries[threshold] = {
            'pairs_with_ink_pixels': len(values),
            'pairs_with_any_neighbor_ink_in_crop': sum(value > 0 for value in values),
            'mean_fraction': math.fsum(values) / len(values) if values else None,
            'median_fraction': statistics.median(values) if values else None,
            'max_fraction': max(values) if values else None,
        }
    return {
        'policy': 'Each GT block polygon bounding rectangle expanded by 24 px and clipped to image; no model inference or OCR; synthetic grayscale pixels read for fixed dark-pixel thresholds.',
        'adjacent_parent_block_pairs': {
            'candidate_adjacency_observations': sum(max(0, row['candidate_parent_block_count'] - 1) for row in candidate_rows),
            'unique_block_pairs': len(adjacent_records),
            'pairs': adjacent_records,
            'parent_x_gap_px_range': ([min(row['parent_x_gap_px'] for row in adjacent_records),
                                       max(row['parent_x_gap_px'] for row in adjacent_records)]
                                      if adjacent_records else None),
            'dilated_crop_x_overlap_px_range': ([min(row['dilated_crop_x_overlap_px'] for row in adjacent_records),
                                                max(row['dilated_crop_x_overlap_px'] for row in adjacent_records)]
                                               if adjacent_records else None),
            'gap_after_global_blla_scale_px_range': ([min(row['gap_after_global_scale_px'] for row in adjacent_records),
                                                      max(row['gap_after_global_scale_px'] for row in adjacent_records)]
                                                     if adjacent_records else None),
        },
        'unique_directed_crop_to_foreign_candidate_line_pairs': len(records),
        'pairs_with_neighbor_baseline_intersection': sum(row['neighbor_baseline_intersects_crop'] for row in records),
        'pairs_with_positive_neighbor_polygon_overlap': sum(row['neighbor_polygon_overlap_area_px2'] > 0 for row in records),
        'ink_threshold_intensity_lt': ink_summaries,
        'neighbor_baseline_fraction_inside_crop': {
            'pairs_with_valid_baseline': len(baseline_fractions),
            'pairs_with_any_baseline_inside': sum(value > 0 for value in baseline_fractions),
            'mean': (math.fsum(baseline_fractions) / len(baseline_fractions) if baseline_fractions else None),
            'median': statistics.median(baseline_fractions) if baseline_fractions else None,
            'max': max(baseline_fractions) if baseline_fractions else None,
        },
        'neighbor_polygon_area_fraction_inside_crop': {
            'pairs_with_positive_overlap': len(positive_polygon_fractions),
            'mean_positive': (math.fsum(positive_polygon_fractions) / len(positive_polygon_fractions)
                              if positive_polygon_fractions else None),
            'median_positive': statistics.median(positive_polygon_fractions) if positive_polygon_fractions else None,
            'max': max(positive_polygon_fractions) if positive_polygon_fractions else None,
        },
        'pairs': records,
        'limits': 'Computed only for GT lines in the same permissive merge-candidate neighborhoods; not all page lines/crops. Intensity threshold counts are dark-pixel diagnostics, not semantic ink segmentation.',
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction', type=Path,
                        default=ROOT / 'reports/g3_mf_0036_global_kraken.json')
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'reports/g3_global_smoke.json')
    args = parser.parse_args()

    image = PILOT / 'images/mf_0036.png'
    annotation = PILOT / 'pages/mf_0036.json'
    observed = {'image': sha256(image), 'annotation': sha256(annotation), 'model': sha256(MODEL)}
    if observed != EXPECTED:
        raise SystemExit(f'input hash mismatch: {observed}')
    page = json.loads(annotation.read_text(encoding='utf-8'))
    prediction = json.loads(args.prediction.read_text(encoding='utf-8'))
    if page.get('page_id') != 'mf_0036':
        raise SystemExit('unexpected synthetic page')
    if len(page['lines']) != 321 or len(page['blocks']) != 84:
        raise SystemExit('synthetic page inventory changed')
    parent_ids = {row['id'] for row in page['blocks']}
    if any(row.get('block_id') not in parent_ids for row in page['lines']):
        raise SystemExit('GT line missing a valid parent block')

    ground_truth = [GroundTruthLine(
        id=row['id'], baseline=row['baseline'],
        height_px=gt_height(row['baseline'], row['polygon']),
        baseline_source='ground_truth',
    ) for row in page['lines']]
    predicted = [PredictedLine(
        id=f'p{index:04d}',
        baseline=[(float(x) + 0.5, float(y) + 0.5) for x, y in row['baseline']],
        baseline_source='predicted',
    ) for index, row in enumerate(prediction['lines'])]
    baseline_lengths = [math.fsum(math.dist(a, b) for a, b in zip(line.baseline, line.baseline[1:]))
                        for line in predicted]
    result = evaluate_line_geometry(ground_truth, predicted)
    block_by_line = {row['id']: row['block_id'] for row in page['lines']}
    block_x_intervals = {
        row['id']: (min(point[0] for point in row['polygon']), max(point[0] for point in row['polygon']))
        for row in page['blocks']
    }
    topology = candidate_parent_diagnostics(
        ground_truth, predicted, result, block_by_line, block_x_intervals
    )
    topology['crop_neighbor_geometry'] = parent_crop_overlap_diagnostics(
        topology['merge_candidate_predictions'], page, image
    )
    output = {
        'schema_version': 'babaorum.g3_global_smoke.v1',
        'role': 'synthetic_global_only_diagnostic',
        'page_id': 'mf_0036',
        'input_sha256': observed,
        'prediction_sha256': sha256(args.prediction),
        'checkpoint_path': str(MODEL),
        'checkpoint_sha256': observed['model'],
        'detector': 'Kraken 7.1.1 BLLA; CPU; one thread; VGSL [1,1800,0,3]; internal Lanczos resize to height 1800',
        'global_internal_scale_factor': 1800 / 3698,
        'metric_policy': 'babaorum.line_geometry.v1; primary 0.5 hGT; fixed sensitivity 1 hGT',
        'ground_truth': {
            'lines': len(ground_truth), 'blocks': len(page['blocks']),
            'lines_with_valid_parent': len(ground_truth),
            'parented_blocks_with_lines': len({row['block_id'] for row in page['lines']}),
            'height': 'signed normal span of explicit line polygon about explicit baseline',
        },
        'prediction_count': len(predicted),
        'prediction_baseline_lengths_px': {
            'count_over_2000_px': sum(length > 2000 for length in baseline_lengths),
            'maximum': max(baseline_lengths) if baseline_lengths else None,
            'top_10_descending': sorted(baseline_lengths, reverse=True)[:10],
        },
        'result': asdict(result),
        'merge_parent_diagnostics': topology,
        'scope_limit': 'No local crops completed; no p, q_GT, q_pred, paired overlap, or oracle union.',
    }
    atomic_write(args.output, output)
    print(json.dumps({
        'output': str(args.output), 'n_gt': result.n_gt, 'n_pred': result.n_pred,
        'matched': result.matched_count, 'recall': result.recall,
        'precision': result.precision, 'f1': result.f1,
        'sensitivity_1h_matched': result.sensitivity_1h_matched_count,
    }, sort_keys=True))


if __name__ == '__main__':
    main()
