"""Measure action changes separately from persistent action windows."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

KEYS = ('w', 'a', 's', 'd')
VKS = (87, 65, 83, 68)


def binary_events(truth, predicted):
    if len(truth) != len(predicted):
        raise ValueError('Event targets and predictions require equal lengths')
    tp = sum(t and p for t, p in zip(truth, predicted))
    fp = sum(not t and p for t, p in zip(truth, predicted))
    fn = sum(t and not p for t, p in zip(truth, predicted))
    return {'true_windows': sum(truth), 'predicted_windows': sum(predicted),
            'tp': tp, 'fp': fp, 'fn': fn,
            'precision': tp / (tp + fp) if tp + fp else None,
            'recall': tp / (tp + fn) if tp + fn else None,
            'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None}


def contiguous(previous, sample):
    return (previous is not None and previous['segment'] == sample['segment']
            and previous['action_end_ns'] == sample['anchor_ns'])


def motion_phase(previous, current, threshold=1):
    if previous is None:
        return 'boundary'
    before, after = abs(previous) >= threshold, abs(current) >= threshold
    if not before:
        return 'start' if after else 'idle'
    if not after:
        return 'stop'
    return 'continue' if previous * current > 0 else 'reverse'


def evaluate_events(samples, predictions, key_threshold=.5, mouse_threshold=1):
    """Use only contiguous windows for history-dependent transition checks."""
    if not samples or len(samples) != len(predictions):
        raise ValueError('Provide nonempty aligned samples and predictions')
    if not 0 < key_threshold < 1 or not math.isfinite(mouse_threshold) or mouse_threshold <= 0:
        raise ValueError('Event thresholds must remain within their supported ranges')
    if any(len(row) != 6 or not all(math.isfinite(v) for v in row) for row in predictions):
        raise ValueError('Each prediction requires six finite values')
    if any(not 0 <= v <= 1 for row in predictions for v in row[:4]):
        raise ValueError('Predicted key fractions must remain between zero and one')
    key_report = {}
    for column, (name, vk) in enumerate(zip(KEYS, VKS)):
        truth = {'press': [], 'release': []}
        windowed = {'press': [], 'release': []}
        sequential_truth = {'press': [], 'release': []}
        sequential = {'press': [], 'release': []}
        short_taps = occupancy_end_mismatch = event_count = boundaries = 0
        previous = previous_on = None
        for sample, prediction in zip(samples, predictions):
            label = sample['action']
            edges = [e for e in label['events'] if e['type'] == 'key' and e['vk'] == vk]
            down, up = any(e['down'] for e in edges), any(not e['down'] for e in edges)
            short_taps += down and up
            event_count += len(edges)
            short_occupancy = label['key_held_fraction'].get(str(vk), 0) >= key_threshold
            occupancy_end_mismatch += short_occupancy != (vk in label['keys_at_end'])
            on = prediction[column] >= key_threshold
            before = vk in label['keys_at_start']
            truth['press'].append(down)
            truth['release'].append(up)
            windowed['press'].append(not before and on)
            windowed['release'].append(before and not on)
            if contiguous(previous, sample):
                sequential_truth['press'].append(down)
                sequential_truth['release'].append(up)
                sequential['press'].append(not previous_on and on)
                sequential['release'].append(previous_on and not on)
            else:
                boundaries += 1
            previous, previous_on = sample, on
        key_report[name] = {
            'recorded_events': event_count, 'windows_with_press_and_release': short_taps,
            'target_occupancy_end_state_disagreements': occupancy_end_mismatch,
            'recorded_start_state_reference': {edge: binary_events(truth[edge], windowed[edge])
                                              for edge in truth},
            'predicted_state_sequence': {edge: binary_events(sequential_truth[edge], sequential[edge])
                                         for edge in sequential},
            'sequence_boundary_windows_excluded': boundaries}
    mouse_report = {}
    for column, axis, field in [(4, 'x', 'mouse_dx_raw'), (5, 'y', 'mouse_dy_raw')]:
        groups = {}
        actual_phases, predicted_phases = [], []
        previous = None
        previous_actual = previous_prediction = None
        for sample, prediction in zip(samples, predictions):
            actual = sample['action'][field]
            if not contiguous(previous, sample):
                previous_actual = previous_prediction = None
            phase = motion_phase(previous_actual, actual, mouse_threshold)
            predicted = motion_phase(previous_prediction, prediction[column], mouse_threshold)
            groups.setdefault(phase, []).append((actual, prediction[column]))
            if phase != 'boundary':
                actual_phases.append(phase)
                predicted_phases.append(predicted)
            previous, previous_actual, previous_prediction = sample, actual, prediction[column]
        mouse_report[axis] = {
            'by_actual_phase': {phase: {'windows': len(rows),
                                      'mae_raw_counts': sum(abs(p - t) for t, p in rows) / len(rows),
                                      'predicted_nonzero_windows': sum(abs(p) >= mouse_threshold for _, p in rows)}
                                for phase, rows in groups.items()},
            'phase_detection': {phase: binary_events([p == phase for p in actual_phases],
                                                     [p == phase for p in predicted_phases])
                                for phase in ('start', 'continue', 'reverse', 'stop', 'idle')}}
    return {'samples': len(samples), 'key_threshold': key_threshold,
            'mouse_threshold_raw_counts': mouse_threshold, 'keys': key_report, 'mouse': mouse_report,
            'exact_window_matching': True, 'event_timing_error_measured': False,
            'independent_policy_rollout': False, 'input_sent': False}


def main():
    from .behavior_cloning import load_checkpoint, load_inputs, predict
    from .recording import file_digest
    import torch

    parser = argparse.ArgumentParser(description='Measure offline key events and mouse motion phases')
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--split', choices=('train', 'validation'), default='validation')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('The output file already exists. Select a new file')
    torch.set_num_threads(4)
    images, _, identity = load_inputs(args.dataset, args.split)
    model, mean, scale = load_checkpoint(args.checkpoint)
    predictions = predict(model, images, mean, scale, 64).tolist()
    samples = [json.loads(line) for line in (args.dataset / 'samples.jsonl').read_text().splitlines()]
    result = evaluate_events(samples, predictions)
    result.update(source=identity, checkpoint_sha256=file_digest(args.checkpoint), features=model.features)
    with args.output.open('x', encoding='utf-8') as output:
        json.dump(result, output, indent=2)
    print(json.dumps({'samples': len(samples), 'features': model.features, 'input_sent': False}))


if __name__ == '__main__':
    main()
