"""Encode keyboard changes and compile explicit targets without desktop input."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path

from .dataset import HeldState, action_window
from .timed_actions import compile_label

KEYS = ('w', 'a', 's', 'd')
VKS = (87, 65, 83, 68)
WINDOW_NS = 50_000_000
MAX_KEY_EVENTS = 2


@dataclass(frozen=True)
class EventTarget:
    key_offsets_ns: tuple[tuple[int, ...], ...]
    mouse_dx_raw: float = 0
    mouse_dy_raw: float = 0
    duration_ns: int = WINDOW_NS

    def validate(self):
        if type(self.duration_ns) is not int or self.duration_ns != WINDOW_NS:
            raise ValueError('Event targets require a 50-millisecond window')
        if not isinstance(self.key_offsets_ns, tuple) or len(self.key_offsets_ns) != 4:
            raise ValueError('Event targets require four chronological offset lists')
        for offsets in self.key_offsets_ns:
            if not isinstance(offsets, tuple) or len(offsets) > MAX_KEY_EVENTS:
                raise ValueError('Each key supports at most two changes per window')
            if any(type(t) is not int or not 0 <= t < self.duration_ns for t in offsets):
                raise ValueError('Key offsets must be integer times inside the action window')
            if tuple(sorted(offsets)) != offsets:
                raise ValueError('Key offsets must remain chronological')
        if not all(math.isfinite(v) for v in (self.mouse_dx_raw, self.mouse_dy_raw)):
            raise ValueError('Mouse totals must be finite')
        return self

    def training_fields(self):
        """Keep absent event times masked rather than treating zero as an event."""
        self.validate()
        return {'key_counts': [len(offsets) for offsets in self.key_offsets_ns],
                'key_offset_fraction': [[t / self.duration_ns for t in offsets]
                                        + [0.] * (MAX_KEY_EVENTS - len(offsets))
                                        for offsets in self.key_offsets_ns],
                'key_offset_mask': [[True] * len(offsets) + [False] * (MAX_KEY_EVENTS - len(offsets))
                                    for offsets in self.key_offsets_ns],
                'mouse_total_raw': [self.mouse_dx_raw, self.mouse_dy_raw],
                'mouse_nonzero_total': [self.mouse_dx_raw != 0, self.mouse_dy_raw != 0]}


def target_label(target, owned_keys):
    """Decode changes relative to caller-supplied keys, not recorded future state."""
    target.validate()
    owned_keys = frozenset(owned_keys)
    if not owned_keys <= set(KEYS):
        raise ValueError('The event decoder supports only owned W, A, S, and D keys')
    current = set(owned_keys)
    events = []
    changes = sorted((at, index) for index, offsets in enumerate(target.key_offsets_ns) for at in offsets)
    for at, index in changes:
        name, vk = KEYS[index], VKS[index]
        down = name not in current
        (current.add if down else current.discard)(name)
        events.append({'type': 'key', 'time_ns': at, 'vk': vk,
                       'scan': 0, 'extended': 0, 'down': down})
    if target.mouse_dx_raw or target.mouse_dy_raw:
        events.append({'type': 'motion', 'time_ns': target.duration_ns // 2,
                       'dx': target.mouse_dx_raw, 'dy': target.mouse_dy_raw})
    events.sort(key=lambda event: event['time_ns'])
    state = HeldState({'held_keys': [vk for name, vk in zip(KEYS, VKS) if name in owned_keys],
                       'held_buttons': []})
    return action_window(state, events, 0, target.duration_ns)


def encode_label(label, duration_ns=WINDOW_NS):
    """Retain alternating key edges and exact receipt offsets from a source label."""
    if (label['buttons_at_start'] or label['buttons_at_end'] or label['button_held_fraction']
            or label['wheel_vertical'] or label['wheel_horizontal']):
        raise ValueError('Navigation event targets do not support buttons or wheel input')
    all_keys = set(label['keys_at_start']) | set(label['keys_at_end']) | set(map(int, label['key_held_fraction']))
    if not all_keys <= set(VKS):
        raise ValueError('Navigation event targets support only W, A, S, and D')
    start = frozenset(name for name, vk in zip(KEYS, VKS) if vk in label['keys_at_start'])
    current = set(label['keys_at_start'])
    offsets = {vk: [] for vk in VKS}
    mouse_x = mouse_y = previous = 0
    for event in label['events']:
        at = event['offset_ns']
        if type(at) is not int or not previous <= at < duration_ns:
            raise ValueError('Source events must remain chronological inside the window')
        previous = at
        if event['type'] == 'key':
            vk, down = event['vk'], event['down']
            if vk not in offsets or type(down) is not bool:
                raise ValueError('The source key event is unsupported')
            if down == (vk in current):
                raise ValueError('Source key events must alternate from the recorded initial state')
            offsets[vk].append(at)
            (current.add if down else current.discard)(vk)
        elif event['type'] == 'motion':
            if not all(math.isfinite(v) for v in (event['dx'], event['dy'])):
                raise ValueError('Source mouse counts must be finite')
            mouse_x += event['dx']
            mouse_y += event['dy']
        else:
            raise ValueError('The source event type is unsupported')
    if (mouse_x, mouse_y) != (label['mouse_dx_raw'], label['mouse_dy_raw']):
        raise ValueError('Source mouse totals do not match their events')
    target = EventTarget(tuple(tuple(offsets[vk]) for vk in VKS), mouse_x, mouse_y, duration_ns).validate()
    decoded = target_label(target, start)
    if set(decoded['keys_at_end']) != set(label['keys_at_end']):
        raise ValueError('Source key events do not match their final state')
    for vk in VKS:
        expected = label['key_held_fraction'].get(str(vk), 0)
        actual = decoded['key_held_fraction'].get(str(vk), 0)
        if not math.isfinite(expected) or not math.isclose(expected, actual, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError('Source key offsets do not match their held fractions')
    return start, target


def compile_target(target, owned_keys, scale_x, scale_y):
    """Use the bounded timed executor with explicit candidate mouse conversion."""
    label = target_label(target, owned_keys)
    return compile_label(label, scale_x, scale_y, target.duration_ns)


def audit(directory, split, scale_x, scale_y):
    from .behavior_cloning import load_inputs
    # Reconstruct the source before auditing candidate labels.
    _, _, identity = load_inputs(directory, split)
    samples = [json.loads(line) for line in (directory / 'samples.jsonl').read_text().splitlines()]
    key_events = two_change = packets = cancellation = 0
    counts = {name: [0, 0, 0] for name in KEYS}
    for sample in samples:
        label = sample['action']
        start, target = encode_label(label, sample['action_end_ns'] - sample['anchor_ns'])
        window = compile_target(target, start, scale_x, scale_y)
        before = set(start)
        emitted = {vk: [] for vk in VKS}
        for packet in window.packets:
            for name, vk in zip(KEYS, VKS):
                if (name in before) != (name in packet.action.keys):
                    emitted[vk].append((packet.offset_ns, name in packet.action.keys))
            before = set(packet.action.keys)
        for vk in VKS:
            expected = [(e['offset_ns'], e['down']) for e in label['events']
                        if e['type'] == 'key' and e['vk'] == vk]
            if emitted[vk] != expected:
                raise ValueError('Compiled packets changed the recorded key transitions')
        for actual, expected in [
            (math.fsum(p.action.mouse_dx for p in window.packets), target.mouse_dx_raw * scale_x),
            (math.fsum(p.action.mouse_dy for p in window.packets), target.mouse_dy_raw * scale_y)]:
            if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-9):
                raise ValueError('Compiled packets changed the scaled mouse total')
        packets += len(window.packets)
        for name, offsets in zip(KEYS, target.key_offsets_ns):
            counts[name][len(offsets)] += 1
            key_events += len(offsets)
            two_change += len(offsets) == 2
        cancellation += any(
            sum(abs(e[field]) for e in label['events'] if e['type'] == 'motion')
            > abs(label[total]) for field, total in [('dx', 'mouse_dx_raw'), ('dy', 'mouse_dy_raw')])
    return {'samples': len(samples), 'source': identity, 'key_count_order': [0, 1, 2],
            'key_count_coverage': counts, 'key_events_retained': key_events,
            'two_change_key_windows_retained': two_change, 'timed_packets': packets,
            'source_reconstruction': 'pass', 'keyboard_round_trip': 'pass', 'mouse_total_round_trip': 'pass',
            'windows_with_mouse_cancellation': cancellation, 'mouse_trajectory_preserved': False,
            'mouse_placement': 'One aggregate event at the window midpoint',
            'candidate_mouse_scales': [scale_x, scale_y], 'input_sent': False, 'deployment_ready': False}


def main():
    parser = argparse.ArgumentParser(description='Audit explicit navigation targets without desktop input')
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--split', choices=('train', 'validation'), default='train')
    parser.add_argument('--scale-x', type=float, required=True)
    parser.add_argument('--scale-y', type=float, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('The output file already exists. Select a new file')
    report = audit(args.dataset, args.split, args.scale_x, args.scale_y)
    with args.output.open('x', encoding='utf-8') as output:
        json.dump(report, output, indent=2)
    print(json.dumps({key: report[key] for key in ('samples', 'keyboard_round_trip', 'mouse_total_round_trip',
                                                  'two_change_key_windows_retained', 'input_sent')}))


if __name__ == '__main__':
    main()
