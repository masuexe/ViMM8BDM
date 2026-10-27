"""Estimate input-to-image response delay without changing dataset labels."""
from __future__ import annotations

import argparse
import bisect
import io
import json
import math
from pathlib import Path
import statistics

from .recording_review import review


def motion_sum(events, starts, begin, end, axis):
    left = bisect.bisect_left(starts, begin)
    right = bisect.bisect_left(starts, end)
    return sum(e['d' + axis] for e in events[left:right])


def regression(x, y, scale=None):
    denominator = sum(v * v for v in x)
    if not denominator:
        return None
    if scale is None:
        scale = sum(a * b for a, b in zip(x, y)) / denominator
    residual = sum((scale * a - b) ** 2 for a, b in zip(x, y))
    mean = statistics.fmean(y)
    variation = sum((v - mean) ** 2 for v in y)
    return {'scale': scale, 'rmse_px': math.sqrt(residual / len(y)),
            'r2': 1 - residual / variation if variation else None}


def fit_axis(pairs, events, axis, grid=range(-100, 151, 5)):
    motion = sorted((e for e in events if e['type'] == 'motion'), key=lambda e: e['time_ns'])
    grid = list(grid)
    if not grid:
        raise ValueError('The delay grid must contain at least one value')
    rows = []
    for pair in pairs:
        if pair['axis'] != axis:
            continue
        segment_events = [e for e in motion if e['segment'] == pair['segment']]
        segment_starts = [e['time_ns'] for e in segment_events]
        values = [motion_sum(segment_events, segment_starts,
                             pair['begin_ns'] - delay * 1_000_000,
                             pair['end_ns'] - delay * 1_000_000, axis) for delay in grid]
        # Select once for the whole search. Do not change samples for each candidate.
        if max(map(abs, values), default=0) >= 2 or abs(pair['shift_px']) >= 1:
            rows.append((pair, values))
    rows.sort(key=lambda item: item[0]['begin_ns'])
    report = {'axis': axis, 'status': 'insufficient_data', 'pairs': len(rows),
              'delay_definition': 'Positive delay means input receipt precedes visible response.',
              'automatic_label_shift_applied': False, 'holdout_guard_ms': 300,
              'delay_selection': 'training_rmse_only'}
    if len(rows) < 36:
        return report
    cut = len(rows) * 2 // 3
    train = rows[:cut]
    validation = [row for row in rows[cut:]
                  if row[0]['begin_ns'] - train[-1][0]['end_ns'] >= 300_000_000]
    if len(train) < 24 or len(validation) < 12:
        report['status'] = 'insufficient_holdout_after_guard_interval'
        return report
    candidates = []
    for column, delay in enumerate(grid):
        x = [row[1][column] for row in train]
        y = [row[0]['shift_px'] for row in train]
        fitted = regression(x, y)
        if fitted is None or fitted['scale'] >= 0:
            continue
        vx = [row[1][column] for row in validation]
        vy = [row[0]['shift_px'] for row in validation]
        checked = regression(vx, vy, fitted['scale'])
        if checked is None:
            continue
        candidates.append({'delay_ms': delay, 'scale_px_per_raw_count': fitted['scale'],
                           'train_rmse_px': fitted['rmse_px'],
                           'validation_rmse_px': checked['rmse_px'], 'validation_r2': checked['r2'],
                           'train_positive': sum(v >= 2 for v in x), 'train_negative': sum(v <= -2 for v in x),
                           'validation_positive': sum(v >= 2 for v in vx), 'validation_negative': sum(v <= -2 for v in vx)})
    if not candidates:
        report['status'] = 'no_response_fit'
        return report
    # Choose the delay using training only. Validation does not select the winner.
    best = min(candidates, key=lambda item: item['train_rmse_px'])
    tolerance = max(.25, best['train_rmse_px'] * .1)
    near = [c['delay_ms'] for c in candidates if c['train_rmse_px'] <= best['train_rmse_px'] + tolerance]
    low, high = min(near), max(near)
    reasons = []
    if min(best[name] for name in ('train_positive', 'train_negative', 'validation_positive', 'validation_negative')) < 3:
        reasons.append('too_few_both_direction_samples')
    if best['validation_r2'] is None or best['validation_r2'] < .7:
        reasons.append('weak_validation_fit')
    if high - low > 30:
        reasons.append('broad_training_delay_range')
    if low == min(grid) or high == max(grid):
        reasons.append('search_boundary')
    if best['validation_rmse_px'] > 4:
        reasons.append('large_validation_error')
    report.update(status='candidate' if not reasons else 'unresolved', best=best,
                  training_near_best_range_ms=[low, high], range_is_statistical_confidence_interval=False,
                  train_pairs=len(train), validation_pairs=len(validation), reasons=reasons, profile=candidates)
    return report


def translation(before, after, axis, radius=24):
    from PIL import ImageChops, ImageStat
    width, height = before.size
    box = (max(radius, width // 4), max(radius, height // 6),
           min(width - radius, width * 3 // 4), min(height - radius, height * 5 // 12))
    reference = before.crop(box)
    if ImageStat.Stat(reference).stddev[0] < 8:
        return None
    scores = {}
    for delta in range(-radius, radius + 1):
        dx, dy = (delta, 0) if axis == 'x' else (0, delta)
        shifted = after.crop((box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy))
        scores[delta] = ImageStat.Stat(ImageChops.difference(reference, shifted)).mean[0]
    best = min(scores, key=scores.get)
    second = min(score for delta, score in scores.items() if abs(delta - best) > 1)
    uniqueness = (second - scores[best]) / max(second, 1)
    if abs(best) == radius or scores[best] > 25 or uniqueness < .15:
        return None
    return {'shift': best, 'mean_error': scores[best], 'uniqueness': uniqueness}


def analyze(directory, window_ms=80):
    from PIL import Image
    integrity = review(directory)
    index = [json.loads(line) for line in (directory / 'frames.jsonl').read_text().splitlines()]
    events = [json.loads(line) for line in (directory / 'events.jsonl').read_text().splitlines()]
    phases = sorted((e for e in events if e['type'] == 'protocol_phase'), key=lambda e: e['time_ns'])
    phase_times = [e['time_ns'] for e in phases]
    keys = sorted(e['time_ns'] for e in events if e['type'] == 'key')
    pairs = []
    rejected = {'segment_boundary': 0, 'held_key': 0, 'near_key_event': 0, 'uncertain_match': 0, 'protocol_phase': 0, 'large_capture_gap': 0}
    previous = None
    with (directory / 'frames.vrec').open('rb') as archive:
        for row in index:
            archive.seek(row['offset'] + 4)
            raw = archive.read(row['png_bytes'])
            with Image.open(io.BytesIO(raw)) as image:
                gray = image.convert('L').resize((320, 240), Image.Resampling.BILINEAR)
            if previous is not None:
                before, old = previous
                begin = (old['capture_started_ns'] + old['capture_finished_ns']) // 2
                end = (row['capture_started_ns'] + row['capture_finished_ns']) // 2
                if old['segment'] == row['segment'] and end - begin < window_ms * 1_000_000:
                    continue
                reason = None
                if old['segment'] != row['segment']:
                    reason = 'segment_boundary'
                elif end - begin > max(window_ms * 2, 150) * 1_000_000:
                    reason = 'large_capture_gap'
                elif old['held_keys_at_capture_start'] or row['held_keys_at_capture_start']:
                    reason = 'held_key'
                elif bisect.bisect_left(keys, end + 100_000_000) > bisect.bisect_left(keys, begin - 300_000_000):
                    reason = 'near_key_event'
                if reason:
                    rejected[reason] += 1
                else:
                    for axis in ('x', 'y'):
                        if phases:
                            phase_index = bisect.bisect_right(phase_times, begin) - 1
                            if (phase_index < 0 or phases[phase_index].get('axis') != axis
                                    or phases[phase_index]['segment'] != row['segment']
                                    or begin - phase_times[phase_index] < 600_000_000
                                    or (phase_index + 1 < len(phases) and end + 150_000_000 >= phase_times[phase_index + 1])):
                                rejected['protocol_phase'] += 1
                                continue
                        match = translation(before, gray, axis)
                        if match is None:
                            rejected['uncertain_match'] += 1
                            continue
                        factor = old['width'] / 320 if axis == 'x' else old['height'] / 240
                        pairs.append({'axis': axis, 'begin_ns': begin, 'end_ns': end,
                                      'segment': row['segment'], 'before_frame': old['frame'],
                                      'after_frame': row['frame'], 'shift_px': match['shift'] * factor,
                                      'mean_error': match['mean_error'], 'uniqueness': match['uniqueness']})
            previous = (gray, row)
    return {'method': 'Upper central image translation and receipt-clock motion integration',
            'minimum_image_window_ms': window_ms,
            'protocol_phase_filter': bool(phases),
            'phase_start_guard_ms': 600, 'phase_end_guard_ms': 150,
            'guided_protocol_completed': json.loads((directory / 'summary.json').read_text()).get('exit_reason') == 'protocol_complete',
            'frame_time_reference': 'Midpoint of capture start and completion',
            'integrity': integrity, 'rejected': rejected,
            'axes': {axis: fit_axis(pairs, events, axis) for axis in ('x', 'y')},
            'accepted_image_pairs': pairs, 'training_labels_modified': False,
            'limits': ['The delay describes observed image response, not a hardware event timestamp.',
                       'A translation match approximates camera rotation and can fail on animated scenes.',
                       'The delay range is a search diagnostic, not a statistical confidence interval.',
                       'A fitted response delay does not justify shifting behavior-cloning labels to later images.']}


def main():
    parser = argparse.ArgumentParser(description='Estimate mouse input-to-image response delay')
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--window-ms', type=float, default=80)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('The output file already exists')
    if not math.isfinite(args.window_ms) or not 16 <= args.window_ms <= 200:
        parser.error('The image window must be between 16 and 200 milliseconds')
    result = analyze(args.run_dir, args.window_ms)
    with args.output.open('x', encoding='utf-8') as output:
        json.dump(result, output, indent=2)
    print(json.dumps({'axes': {axis: {k: v for k, v in report.items() if k != 'profile'}
                               for axis, report in result['axes'].items()}, 'rejected': result['rejected']}, indent=2))


if __name__ == '__main__':
    main()
