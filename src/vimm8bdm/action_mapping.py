"""Audit dataset labels against the current action sink. Never execute input."""
import argparse
import collections
import json
import math
from pathlib import Path

from .control import Action
from .dataset import verify_source

VK_KEYS = {ord(key.upper()): key for key in 'wasd123456789'} | {32: 'space', 16: 'shift'}


def inspect_label(label, scale_x=None, scale_y=None):
    if (scale_x is None) != (scale_y is None):
        raise ValueError('Supply both mouse scales or neither scale')
    if scale_x is not None and any(not math.isfinite(value) or value <= 0 for value in (scale_x, scale_y)):
        raise ValueError('Candidate mouse scales must be finite and positive')
    reasons = set()
    events = label['events']
    key_fractions = {int(key): value for key, value in label['key_held_fraction'].items()}
    button_fractions = label['button_held_fraction']
    all_keys = set(label['keys_at_start']) | set(label['keys_at_end']) | set(key_fractions)
    all_keys.update(event['vk'] for event in events if event['type'] == 'key')
    unsupported_keys = sorted(all_keys - VK_KEYS.keys())
    if unsupported_keys:
        reasons.add('unsupported_key')
    if 16 in all_keys:
        reasons.add('modifier_binding_and_side_unverified')
    all_buttons = set(label['buttons_at_start']) | set(label['buttons_at_end']) | set(button_fractions)
    all_buttons.update(event['button'] for event in events if event['type'] == 'button')
    unsupported_buttons = sorted(all_buttons - {'left'})
    if unsupported_buttons:
        reasons.add('unsupported_mouse_button')
    if label['wheel_vertical'] or label['wheel_horizontal']:
        reasons.add('wheel_not_supported')
    for fraction in [*key_fractions.values(), *button_fractions.values()]:
        if not math.isfinite(fraction) or not 0 <= fraction <= 1:
            raise ValueError('Held fractions must be finite and between zero and one')
        if fraction not in (0, 1):
            reasons.add('timed_transition_required')
    edges = collections.Counter()
    for event in events:
        if event['type'] in ('key', 'button'):
            if event['offset_ns'] != 0:
                reasons.add('timed_transition_required')
            identity = (event['type'], event.get('vk'), event.get('scan'), event.get('extended'), event.get('button'))
            edges[identity] += 1
    if any(count > 1 for count in edges.values()):
        reasons.add('multiple_edges_in_one_step')
    raw = (label['mouse_dx_raw'], label['mouse_dy_raw'])
    if not all(math.isfinite(value) for value in raw):
        raise ValueError('Raw mouse counts must be finite')
    if any(raw) and scale_x is None:
        reasons.add('mouse_scale_not_supplied')
    scaled = (raw[0] * (scale_x or 1), raw[1] * (scale_y or 1))
    if scale_x is not None and any(abs(value) > 24 for value in scaled):
        reasons.add('mouse_limit_exceeded')
    result = {'blocking_reasons': sorted(reasons), 'unsupported_virtual_keys': unsupported_keys,
              'unsupported_mouse_buttons': unsupported_buttons,
              'mouse_unit_conversion_verified': False, 'input_sent': False, 'candidate_action': None}
    if not reasons:
        keys = frozenset(VK_KEYS[key] for key, fraction in key_fractions.items() if fraction == 1)
        action = Action(keys, *scaled, button_fractions.get('left', 0) == 1)
        result['candidate_action'] = {'keys': sorted(action.keys), 'mouse_dx': action.mouse_dx,
                                      'mouse_dy': action.mouse_dy, 'attack': action.attack}
        result['rounded_mouse_counts'] = [round(action.mouse_dx), round(action.mouse_dy)]
        result['motion_rounded_to_zero'] = any(scaled) and not any(result['rounded_mouse_counts'])
    return result


def audit(directory, scale_x=None, scale_y=None):
    checked = verify_source(directory)
    samples = [json.loads(line) for line in (directory / 'samples.jsonl').read_text().splitlines()]
    reasons = collections.Counter()
    unsupported_keys, unsupported_buttons = collections.Counter(), collections.Counter()
    candidates = 0
    rounded_to_zero = 0
    for sample in samples:
        result = inspect_label(sample['action'], scale_x, scale_y)
        reasons.update(result['blocking_reasons'])
        unsupported_keys.update(map(str, result['unsupported_virtual_keys']))
        unsupported_buttons.update(result['unsupported_mouse_buttons'])
        candidates += result['candidate_action'] is not None
        rounded_to_zero += result.get('motion_rounded_to_zero', False)
    return {'samples': len(samples), 'candidate_actions': candidates, 'blocking_reason_counts': dict(reasons),
            'unsupported_virtual_keys': dict(unsupported_keys), 'unsupported_mouse_buttons': dict(unsupported_buttons),
            'source_hashes_match': checked['source_hashes_match'], 'candidate_mouse_scales': [scale_x, scale_y],
            'mouse_unit_conversion_verified': False, 'game_binding_review_required': True,
            'motion_rounded_to_zero_samples': rounded_to_zero,
            'input_sent': False, 'training_ready': False}


def main():
    parser = argparse.ArgumentParser(description='Audit labels against the current bounded action contract')
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--candidate-scale-x', type=float)
    parser.add_argument('--candidate-scale-y', type=float)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('The output file already exists')
    result = audit(args.dataset, args.candidate_scale_x, args.candidate_scale_y)
    with args.output.open('x', encoding='utf-8') as output:
        json.dump(result, output, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
