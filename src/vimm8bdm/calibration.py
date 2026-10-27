"""Measure small mouse pulses from displayed pixels. No engine state is read."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import struct
import sys
import time
import zlib

from .control import Action, Frame, blank_frame


def save_png(frame, path):
    """Save an RGB screenshot with exclusive file creation."""
    def chunk(kind, payload):
        return struct.pack('!I', len(payload)) + kind + payload + struct.pack('!I', zlib.crc32(kind + payload))
    rows = bytearray()
    for y in range(frame.height):
        rows.append(0)
        for x in range(frame.width):
            offset = (y * frame.width + x) * 4
            b, g, r = frame.bgra[offset:offset + 3]
            rows.extend((r, g, b))
    with Path(path).open('xb') as stream:
        stream.write(b'\x89PNG\r\n\x1a\n')
        stream.write(chunk(b'IHDR', struct.pack('!2I5B', frame.width, frame.height, 8, 2, 0, 0, 0)))
        stream.write(chunk(b'IDAT', zlib.compress(rows)))
        stream.write(chunk(b'IEND', b''))


def estimate_shift(before, after, axis, radius=24):
    """Search for image translation in an upper central region."""
    if axis not in ('x', 'y'):
        raise ValueError('The axis must be x or y')
    if (before.width, before.height) != (after.width, after.height):
        return {'valid': False, 'reason': 'size_changed'}
    width, height = before.width, before.height
    if width < radius * 4 or height < radius * 4:
        return {'valid': False, 'reason': 'image_too_small'}
    def gray(raw, x, y):
        offset = (y * width + x) * 4
        b, g, r = raw[offset:offset + 3]
        return (r * 77 + g * 150 + b * 29) / 256
    points = [(x, y) for y in range(max(radius, height // 6), height * 5 // 12, 5)
              for x in range(max(radius, width // 4), min(width - radius, width * 3 // 4), 5)]
    reference = [gray(before.bgra, x, y) for x, y in points]
    if not reference or statistics.pstdev(reference) < 8:
        return {'valid': False, 'reason': 'low_texture'}
    scores = {}
    for delta in range(-radius, radius + 1):
        scores[delta] = statistics.fmean(abs(value - gray(after.bgra, x + (delta if axis == 'x' else 0),
                                                         y + (delta if axis == 'y' else 0)))
                                          for (x, y), value in zip(points, reference))
    best = min(scores, key=scores.get)
    alternatives = [value for delta, value in scores.items() if abs(delta - best) > 2]
    uniqueness = (min(alternatives) - scores[best]) / max(min(alternatives), 1)
    improvement = (scores[0] - scores[best]) / max(scores[0], 1)
    valid = abs(best) < radius and uniqueness >= .15 and scores[best] <= 25
    return {'valid': valid, 'reason': 'matched' if valid else 'uncertain_match',
            'image_shift_px': best, 'mean_error': scores[best],
            'zero_shift_error': scores[0], 'improvement': improvement, 'uniqueness': uniqueness}


def measure_trial(baseline, before, after, axis, counts):
    still = estimate_shift(baseline, before, axis)
    moved = estimate_shift(before, after, axis)
    valid = (still['valid'] and abs(still['image_shift_px']) <= 1 and still['mean_error'] <= 5
             and moved['valid'] and abs(moved['image_shift_px']) >= 1 and moved['improvement'] >= .2)
    return {'axis': axis, 'counts': counts, 'valid': valid,
            'reason': 'measured' if valid else 'scene_or_match_rejected',
            'baseline_match': still, 'pulse_match': moved,
            'image_pixels_per_count': moved['image_shift_px'] / counts if valid else None}


def collect(window_id, directory, cycles, arm_timeout):
    """Collect pulse triplets. Analyze them only after desktop input closes."""
    from .desktop import WindowsWindowCapture
    from .input_worker import InputWorker
    directory.mkdir(parents=True, exist_ok=False)
    trials = [(axis, sign * count) for _ in range(cycles) for axis in ('x', 'y')
              for count in (4, 8) for sign in (1, -1)]
    pairs = []
    started = None
    index = 0
    baseline = before = None
    pulse_sent = False
    generation = None
    last_capture_end = None
    frame_count = 0
    max_gap_ms = 0
    accepted_actions = 0
    exit_reason = 'complete'
    waiting_until = time.perf_counter() + arm_timeout if arm_timeout else None
    with (directory / 'frames.local.jsonl').open('x', encoding='utf-8') as log:
        capture = WindowsWindowCapture(window_id)
        try:
            with InputWorker(window_id, trace_path=directory / 'worker.local.jsonl') as worker:
                print('Calibration ready. Select the game. F8: enable/pause. Esc: pause. F9: exit.', flush=True)
                print('Keep the player still. Do not move the real mouse during capture.', flush=True)
                while index < len(trials):
                    begin = time.perf_counter()
                    worker.refresh()
                    if worker.stopped:
                        exit_reason = worker.reason
                        break
                    if not worker.armed:
                        baseline = before = None
                        pulse_sent = False
                        generation = None
                        if started is None and waiting_until is not None and begin >= waiting_until:
                            exit_reason = 'arm_timeout'
                            break
                        if started is not None and begin - started >= cycles * 8 * 2 + 60:
                            exit_reason = 'duration_limit'
                            break
                        time.sleep(.02)
                        continue
                    if started is None:
                        started = begin
                    if generation != worker.generation:
                        generation = worker.generation
                        baseline = before = None
                        pulse_sent = False
                        trial_started = begin
                    capture_started = time.perf_counter()
                    frame = capture.grab()
                    captured = time.perf_counter()
                    if last_capture_end is not None:
                        max_gap_ms = max(max_gap_ms, (captured - last_capture_end) * 1000)
                    last_capture_end = captured
                    frame_count += 1
                    if blank_frame(frame):
                        worker.pause('blank_frame')
                        continue
                    axis, counts = trials[index]
                    elapsed = begin - trial_started
                    action = Action()
                    if elapsed >= .4 and baseline is None:
                        baseline = frame
                    elif elapsed >= .6 and not pulse_sent:
                        before = frame
                        action = Action(mouse_dx=counts if axis == 'x' else 0,
                                        mouse_dy=counts if axis == 'y' else 0)
                    applied = worker.submit(action, frame.captured_ns, generation)
                    accepted_actions += bool(applied)
                    log.write(json.dumps({'time_ns': frame.captured_ns, 'trial': index,
                                          'generation': generation, 'applied': applied,
                                          'capture_ms': (captured - capture_started) * 1000,
                                          'dx': action.mouse_dx, 'dy': action.mouse_dy}) + '\n')
                    if not applied:
                        baseline = before = None
                        pulse_sent = False
                        generation = None
                    elif action.mouse_dx or action.mouse_dy:
                        pulse_sent = True
                        print(f'Pulse {index + 1}/{len(trials)}: {axis} {counts:+d} counts', flush=True)
                    elif pulse_sent and elapsed >= 1.1:
                        pairs.append((index, axis, counts, baseline, before, frame))
                        index += 1
                        baseline = before = None
                        pulse_sent = False
                        trial_started = begin
                    time.sleep(max(0, .05 - (time.perf_counter() - begin)))
        finally:
            capture.close()
    active_elapsed = None if started is None else time.perf_counter() - started
    print('Desktop input closed. Analyze saved image pairs.', flush=True)
    results = []
    for index, axis, counts, baseline, before, after in pairs:
        for label, frame in [('baseline', baseline), ('before', before), ('after', after)]:
            save_png(frame, directory / f'{index:03d}-{label}.png')
        result = measure_trial(baseline, before, after, axis, counts)
        results.append({'trial': index, **result})
    summary = {'exit_reason': exit_reason, 'requested_trials': len(trials), 'completed_trials': len(results),
               'frames': frame_count, 'accepted_actions': accepted_actions, 'maximum_capture_gap_ms': max_gap_ms,
               'elapsed_after_first_enable_s': active_elapsed,
               'units': 'image pixels per injected mouse count', 'engine_angles_read': False,
               'visible_game_response_latency_measured': False, 'trials': results, 'axes': {}}
    for axis in ('x', 'y'):
        values = [r['image_pixels_per_count'] for r in results if r['axis'] == axis and r['valid']]
        summary['axes'][axis] = {'valid_trials': len(values),
                                 'median_image_pixels_per_count': statistics.median(values) if values else None}
    with (directory / 'summary.local.json').open('x', encoding='utf-8') as output:
        json.dump(summary, output, indent=2)
    print(json.dumps({key: value for key, value in summary.items() if key != 'trials'}, indent=2))
    print('Calibration stopped. F8 is inactive. Start a new run to enable control.', flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description='Measure mouse pulses from game images')
    parser.add_argument('--window-id', type=lambda value: int(value, 0), required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--cycles', type=int, default=3)
    parser.add_argument('--arm-timeout', type=float, default=0,
                        help='Initial waiting limit in seconds. Zero waits until F8 or F9.')
    args = parser.parse_args()
    if sys.platform != 'win32':
        parser.error('Live calibration requires Windows Python')
    if not 1 <= args.cycles <= 20:
        parser.error('The cycle count must be between 1 and 20')
    if not math.isfinite(args.arm_timeout) or args.arm_timeout < 0:
        parser.error('The waiting limit must be finite and nonnegative')
    collect(args.window_id, args.run_dir, args.cycles, args.arm_timeout)


if __name__ == '__main__':
    main()
