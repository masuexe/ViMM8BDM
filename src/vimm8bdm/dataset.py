"""Build causal image histories and future human-action windows."""
from __future__ import annotations

import argparse
import bisect
import collections
import copy
import io
import json
import os
from pathlib import Path
import struct

from .recording import file_digest
from .recording_review import review


class HeldState:
    def __init__(self, start):
        self.snapshot_keys = set(start['held_keys'])
        self.physical_keys = set()
        self.buttons = set(start['held_buttons'])

    def keys(self):
        return self.snapshot_keys | {key[0] for key in self.physical_keys}

    def apply(self, event):
        if event['type'] == 'key':
            identity = (event['vk'], event['scan'], event['extended'])
            self.snapshot_keys.discard(event['vk'])
            (self.physical_keys.add if event['down'] else self.physical_keys.discard)(identity)
        elif event['type'] == 'button':
            (self.buttons.add if event['down'] else self.buttons.discard)(event['button'])


def action_window(state, events, begin, end):
    """Retain transitions and time held in the half-open interval [begin, end)."""
    if end <= begin:
        raise ValueError('The action interval must have positive duration')
    held = copy.deepcopy(state)
    key_time, button_time = collections.Counter(), collections.Counter()
    initial_keys, initial_buttons = sorted(held.keys()), sorted(held.buttons)
    result = {'keys_at_start': initial_keys, 'buttons_at_start': initial_buttons,
              'mouse_dx_raw': 0, 'mouse_dy_raw': 0, 'wheel_vertical': 0, 'wheel_horizontal': 0,
              'events': []}
    previous = begin
    for event in [*events, {'type': '_end', 'time_ns': end}]:
        at = event['time_ns']
        if not previous <= at <= end:
            raise ValueError('Action events must use ordered interval timestamps')
        duration = at - previous
        for key in held.keys():
            key_time[str(key)] += duration
        for button in held.buttons:
            button_time[button] += duration
        previous = at
        if event['type'] == '_end':
            break
        if at == end:
            raise ValueError('Events at the interval end belong to the next action window')
        held.apply(event)
        if event['type'] == 'motion':
            result['mouse_dx_raw'] += event['dx']
            result['mouse_dy_raw'] += event['dy']
        elif event['type'] == 'wheel':
            result['wheel_' + event['axis']] += event['delta']
        result['events'].append({k: v for k, v in event.items() if k not in ('time_ns', 'segment')}
                               | {'offset_ns': at - begin})
    result.update(keys_at_end=sorted(held.keys()), buttons_at_end=sorted(held.buttons),
                  key_held_fraction={key: value / (end - begin) for key, value in key_time.items() if value},
                  button_held_fraction={key: value / (end - begin) for key, value in button_time.items() if value})
    return result


def select_history(times, anchor, history, step_ns):
    limit = bisect.bisect_right(times, anchor)
    selected = []
    for distance in range(history):
        if limit == 0:
            return None
        target = anchor - distance * step_ns
        at = bisect.bisect_left(times, target, 0, limit)
        choices = [index for index in (at - 1, at) if 0 <= index < limit]
        chosen = min(choices, key=lambda index: (abs(times[index] - target), times[index]))
        selected.append(chosen)
        limit = chosen
    return list(reversed(selected))


def make_samples(frames, events, *, history=4, step_ns=50_000_000, max_age_ns=75_000_000,
                 max_gap_ns=100_000_000):
    if history < 1 or min(step_ns, max_age_ns, max_gap_ns) <= 0:
        raise ValueError('History and timing limits must be positive')
    allowed = {'segment_start', 'segment_end', 'recording_end', 'protocol_phase',
               'key', 'button', 'motion', 'wheel'}
    action_types = {'key', 'button', 'motion', 'wheel'}
    frame_groups, event_groups = collections.defaultdict(list), collections.defaultdict(list)
    for frame in frames:
        frame_groups[frame['segment']].append(frame)
    for event in events:
        event_groups[event['segment']].append(event)
    samples = []
    rejected = collections.Counter()
    for segment, rows in frame_groups.items():
        log = event_groups[segment]
        starts = [event for event in log if event['type'] == 'segment_start']
        ends = [event for event in log if event['type'] == 'segment_end']
        if len(starts) != 1 or len(ends) != 1 or any(e['type'] not in allowed for e in log):
            rejected['invalid_or_changed_segment'] += len(rows)
            continue
        start, end = starts[0], ends[0]
        if end['time_ns'] <= start['time_ns']:
            raise ValueError('A segment ends before it starts')
        times = [row['capture_finished_ns'] for row in rows]
        if any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError('Image completion timestamps must increase within each segment')
        if any(row['capture_started_ns'] < start.get('snapshot_ns', start['time_ns'])
               or row['capture_finished_ns'] > end['time_ns'] for row in rows):
            raise ValueError('An image crosses a segment boundary')
        actions = sorted((e for e in log if e['type'] in action_types), key=lambda e: (e['time_ns'], e['sequence']))
        if any(not start['time_ns'] <= e['time_ns'] < end['time_ns'] for e in actions):
            raise ValueError('An input event crosses a segment boundary')
        action_times = [e['time_ns'] for e in actions]
        state = HeldState(start)
        pointer = 0
        first_anchor = times[0] + (history - 1) * step_ns
        for anchor in range(first_anchor, end['time_ns'], step_ns):
            finish = anchor + step_ns
            if finish > end['time_ns']:
                rejected['incomplete_future_window'] += 1
                continue
            selected = select_history(times, anchor, history, step_ns)
            if selected is None:
                rejected['insufficient_history'] += 1
                continue
            if any(abs(anchor - distance * step_ns - times[index]) > max_age_ns
                   for distance, index in zip(range(history - 1, -1, -1), selected)):
                rejected['stale_history'] += 1
                continue
            after = bisect.bisect_left(times, finish)
            if after == len(rows):
                rejected['incomplete_future_capture_coverage'] += 1
                continue
            if any(times[j + 1] - times[j] > max_gap_ns for j in range(selected[0], after)):
                rejected['capture_gap'] += 1
                continue
            while pointer < len(actions) and actions[pointer]['time_ns'] < anchor:
                state.apply(actions[pointer])
                pointer += 1
            right = bisect.bisect_left(action_times, finish)
            future = actions[pointer:right]
            observation = [{key: rows[index][key] for key in
                            ('frame', 'offset', 'png_bytes', 'width', 'height',
                             'capture_started_ns', 'capture_finished_ns')} for index in selected]
            for distance, image in zip(range(history - 1, -1, -1), observation):
                image['target_time_ns'] = anchor - distance * step_ns
                image['target_offset_ns'] = image['capture_finished_ns'] - image['target_time_ns']
            samples.append({'sample': len(samples), 'segment': segment, 'anchor_ns': anchor,
                            'action_end_ns': finish, 'history': observation,
                            'latest_capture_age_ns': anchor - times[selected[-1]],
                            'action': action_window(state, future, anchor, finish)})
    return samples, dict(rejected)


def build(source, output, *, split='diagnostic', allow_interface_check=False):
    source, output = source.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError('The output directory already exists. Select a new directory')
    manifest = json.loads((source / 'session.json').read_text())
    diagnostic_source = manifest.get('purpose') != 'demonstration' or manifest.get('protocol', 'free') != 'free'
    if diagnostic_source and (not allow_interface_check or split != 'diagnostic'):
        raise ValueError('Interface or calibration data requires --allow-interface-check and the diagnostic split')
    if manifest.get('class_selection') != 'fixed' or not manifest.get('character_stable') or str(manifest.get('character', '')).strip().lower() in ('', 'unknown', 'none'):
        raise ValueError('The session needs one reported character with fixed selection')
    if not diagnostic_source and (manifest.get('task') not in {'navigation', 'aim', 'combat', 'class_skill'}
                                  or str(manifest.get('map', '')).strip().lower() in ('', 'unknown', 'none')):
        raise ValueError('A demonstration needs a known map and a task')
    integrity = review(source)
    frames = [json.loads(line) for line in (source / 'frames.jsonl').read_text().splitlines()]
    events = [json.loads(line) for line in (source / 'events.jsonl').read_text().splitlines()]
    if any(event['type'] == 'protocol_phase' for event in events):
        diagnostic_source = True
        if not allow_interface_check or split != 'diagnostic':
            raise ValueError('Guided protocol events require a diagnostic dataset')
    samples, rejected = make_samples(frames, events)
    hashes = {name: file_digest(source / name) for name in
              ('session.json', 'summary.json', 'events.jsonl', 'frames.jsonl', 'frames.vrec')}
    import hashlib
    session_id = hashlib.sha256(json.dumps({name: hashes[name] for name in
                                           ('events.jsonl', 'frames.jsonl', 'frames.vrec')},
                                          sort_keys=True).encode()).hexdigest()
    dataset = {'format': 'vimm8bdm-bc-v1', 'source_directory': Path(os.path.relpath(source, output)).as_posix(),
               'source_session_id': session_id, 'source_hashes': hashes, 'split': split,
               'character': manifest['character'], 'map': manifest.get('map'), 'task': manifest.get('task'),
               'source_purpose': manifest.get('purpose'), 'source_protocol': manifest.get('protocol', 'free'),
               'samples': len(samples), 'rejected': rejected, 'source_integrity': integrity,
               'history_frames': 4, 'history_spacing_ms': 50, 'future_action_ms': 50,
               'anchor_policy': 'Contiguous 50 millisecond grid after the first complete image history',
               'source_action_events': sum(event['type'] in {'key', 'button', 'motion', 'wheel'} for event in events),
               'labelled_action_events': sum(len(sample['action']['events']) for sample in samples),
               'frame_selection': 'Distinct chronological images nearest to history targets, all complete by the decision anchor',
               'input_units': 'Recorded Windows Raw Input counts; actuator conversion not applied',
               'training_ready': False, 'blockers': ['human_review_required', 'actuator_conversion_unverified'],
               'timestamps_changed': False, 'automatic_response_delay_correction': False}
    if diagnostic_source:
        dataset['blockers'].append('interface_or_calibration_source')
    if not samples:
        dataset['blockers'].append('no_samples')
    output.mkdir(parents=True, exist_ok=False)
    with (output / 'dataset.json').open('x', encoding='utf-8') as stream:
        json.dump(dataset, stream, indent=2)
    with (output / 'samples.jsonl').open('x', encoding='utf-8') as stream:
        for sample in samples:
            stream.write(json.dumps(sample) + '\n')
    return dataset


def load_history(directory, sample):
    """Load only causal referenced images. The source archive remains unchanged."""
    from PIL import Image
    manifest = json.loads((directory / 'dataset.json').read_text())
    source = directory / manifest['source_directory']
    images = []
    with (source / 'frames.vrec').open('rb') as archive:
        for frame in sample['history']:
            if frame['capture_finished_ns'] > sample['anchor_ns']:
                raise ValueError('The history includes an image from the future')
            archive.seek(frame['offset'])
            header = archive.read(4)
            if len(header) != 4 or struct.unpack('<I', header)[0] != frame['png_bytes']:
                raise ValueError('The referenced image length changed')
            with Image.open(io.BytesIO(archive.read(frame['png_bytes']))) as image:
                if image.size != (frame['width'], frame['height']):
                    raise ValueError('The referenced image dimensions changed')
                images.append(image.convert('RGB'))
    return images


def verify_source(directory):
    manifest = json.loads((directory / 'dataset.json').read_text())
    source = directory / manifest['source_directory']
    for name, digest in manifest['source_hashes'].items():
        if file_digest(source / name) != digest:
            raise ValueError('A source recording changed after the dataset build')
    return {'source_hashes_match': True, 'source_session_id': manifest['source_session_id']}


def check_splits(directories):
    seen = {}
    sessions = set()
    for directory in directories:
        manifest = json.loads((directory / 'dataset.json').read_text())
        split, session = manifest['split'], manifest['source_session_id']
        if split == 'diagnostic':
            continue
        identities = [('session', session)]
        if 'source_hashes' in manifest:
            identities.append(('archive', manifest['source_hashes']['frames.vrec']))
        for identity in identities:
            if identity in seen and seen[identity] != split:
                raise ValueError('One source session or image archive appears in different splits')
            seen[identity] = split
        sessions.add(session)
    return {'source_sessions': len(sessions), 'session_split_overlap': False}


def main():
    parser = argparse.ArgumentParser(description='Build causal behavior-cloning sample references')
    parser.add_argument('--session', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--split', choices=('diagnostic', 'train', 'validation', 'test'), default='diagnostic')
    parser.add_argument('--allow-interface-check', action='store_true')
    parser.add_argument('--check-splits', type=Path, nargs='+')
    parser.add_argument('--verify-source', type=Path)
    args = parser.parse_args()
    if args.verify_source:
        if args.check_splits or args.session or args.output:
            parser.error('Source verification cannot be combined with another mode')
        result = verify_source(args.verify_source)
    elif args.check_splits:
        if args.session or args.output:
            parser.error('Split checks cannot be combined with a dataset build')
        result = check_splits(args.check_splits)
    else:
        if args.session is None or args.output is None:
            parser.error('A dataset build requires --session and --output')
        result = build(args.session, args.output, split=args.split, allow_interface_check=args.allow_interface_check)
        result = {key: result[key] for key in ('samples', 'split', 'training_ready', 'blockers', 'rejected')}
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
