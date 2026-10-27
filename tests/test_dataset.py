"""Check causal histories, future transitions, boundaries, and source identity."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from vimm8bdm.dataset import HeldState, action_window, make_samples, build, load_history, check_splits, verify_source, filter_reviewed_samples

MS = 1_000_000


def start(segment=1, at=0):
    return dict(type='segment_start', time_ns=at, snapshot_ns=at, segment=segment, sequence=1,
                held_keys=[], held_buttons=[])


def frame(number, at, segment=1):
    return dict(frame=number, capture_started_ns=at * MS - MS,
                capture_finished_ns=at * MS, segment=segment,
                offset=0, png_bytes=1, width=2, height=2)


class DatasetTests(unittest.TestCase):
    def test_reviewed_exclusion_covers_history_and_future_without_changing_labels(self):
        frames = [frame(i, 50 + i * 50) for i in range(15)]
        events = [start(), dict(type='segment_end', segment=1, sequence=2, time_ns=800 * MS)]
        samples, _ = make_samples(frames, events)
        original = copy.deepcopy(samples)
        kept, count = filter_reviewed_samples(samples, frames,
                                             {'source_session_id': 'source', 'reason': 'Reviewed overlay',
                                              'frame_ranges': [[4, 4]]}, 'source')
        self.assertEqual(count, 5)
        # The first sample has a clean history, but its action window overlaps frame 4.
        self.assertEqual(samples[0]['anchor_ns'], 200 * MS)
        self.assertNotIn(samples[0], kept)
        self.assertEqual(kept[0]['anchor_ns'], 450 * MS)
        self.assertEqual(kept[0]['sample'], 5)
        self.assertEqual(samples, original)

    def test_reviewed_exclusion_rejects_wrong_source_and_invalid_ranges(self):
        frames = [frame(i, 50 + i * 50) for i in range(4)]
        valid = {'source_session_id': 'source', 'reason': 'Reviewed overlay', 'frame_ranges': [[0, 1]]}
        for patch in [{'source_session_id': 'other'}, {'reason': ''}, {'frame_ranges': []},
                      {'frame_ranges': [[2, 1]]}, {'frame_ranges': [[0, 4]]},
                      {'frame_ranges': [[0, 1], [1, 2]]}, {'frame_ranges': [[True, 2]]}]:
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                filter_reviewed_samples([], frames, {**valid, **patch}, 'source')

    def test_short_tap_and_mouse_counts_are_not_lost(self):
        events = [dict(type='button', time_ns=5, button='left', down=True, sequence=1),
                  dict(type='motion', time_ns=7, dx=36, dy=-9, sequence=2),
                  dict(type='button', time_ns=10, button='left', down=False, sequence=3)]
        result = action_window(HeldState(start()), events, 0, 50)
        self.assertEqual(result['buttons_at_start'], [])
        self.assertEqual(result['buttons_at_end'], [])
        self.assertEqual(result['button_held_fraction']['left'], .1)
        self.assertEqual(result['mouse_dx_raw'], 36)  # No actuator clipping.
        self.assertEqual(result['mouse_dy_raw'], -9)
        self.assertEqual(len(result['events']), 3)

    def test_interval_start_is_included_and_end_is_excluded(self):
        event = dict(type='key', time_ns=10, vk=87, scan=17, extended=0, down=True, sequence=1)
        result = action_window(HeldState(start()), [event], 10, 60)
        self.assertEqual(result['key_held_fraction']['87'], 1)
        with self.assertRaises(ValueError):
            action_window(HeldState(start()), [{**event, 'time_ns': 60}], 10, 60)

    def test_independent_modifiers_preserve_held_time(self):
        state = HeldState(start())
        left = dict(type='key', vk=16, scan=42, extended=0, down=True)
        state.apply(left)
        state.apply({**left, 'scan': 54})
        result = action_window(state, [{**left, 'time_ns': 20, 'down': False}], 0, 50)
        self.assertEqual(result['key_held_fraction']['16'], 1)
        self.assertEqual(result['keys_at_end'], [16])

    def test_histories_use_only_completed_past_images(self):
        frames = [frame(i, 50 + i * 50) for i in range(9)]
        events = [start(), dict(type='segment_end', segment=1, sequence=2, time_ns=500 * MS)]
        samples, rejected = make_samples(frames, events)
        self.assertTrue(samples)
        for sample in samples:
            self.assertEqual(len(sample['history']), 4)
            self.assertEqual(sample['action_end_ns'] - sample['anchor_ns'], 50 * MS)
            self.assertTrue(all(f['capture_finished_ns'] <= sample['anchor_ns'] for f in sample['history']))
        self.assertEqual(samples[0]['anchor_ns'], 200 * MS)
        # A key at the next window's start must not enter the previous label.
        events.insert(1, dict(type='key', vk=87, scan=17, extended=0, down=True,
                              segment=1, sequence=2, time_ns=250 * MS))
        events[-1]['sequence'] = 3
        samples, _ = make_samples(frames, events)
        self.assertFalse(samples[0]['action']['events'])
        self.assertEqual(samples[1]['action']['events'][0]['offset_ns'], 0)
        self.assertEqual(samples[2]['action']['keys_at_start'], [87])
        self.assertEqual(samples[2]['action']['key_held_fraction']['87'], 1)

    def test_jitter_does_not_create_gaps_between_future_windows(self):
        frames = [frame(i, at) for i, at in enumerate((5, 55, 104, 156, 205, 255, 304, 356, 405, 455))]
        events = [start(), dict(type='segment_end', segment=1, time_ns=500 * MS)]
        samples, _ = make_samples(frames, events)
        self.assertGreater(len(samples), 3)
        self.assertTrue(all(b['anchor_ns'] == a['action_end_ns'] for a, b in zip(samples, samples[1:])))
        for sample in samples:
            self.assertEqual(len({f['frame'] for f in sample['history']}), 4)
            self.assertTrue(all(f['capture_finished_ns'] <= sample['anchor_ns'] for f in sample['history']))

    def test_pause_and_character_marker_do_not_join_samples(self):
        frames = [frame(i, 50 + i * 50) for i in range(4)]
        frames += [frame(i + 4, 250 + i * 50, segment=2) for i in range(4)]
        events = [start(), dict(type='segment_end', segment=1, time_ns=225 * MS),
                  start(2, 225 * MS), dict(type='segment_end', segment=2, time_ns=425 * MS)]
        samples, _ = make_samples(frames, events)
        self.assertEqual(samples, [])  # Neither segment has enough future coverage.
        changed = [start(), dict(type='character_change', segment=1, time_ns=100 * MS),
                   dict(type='segment_end', segment=1, time_ns=500 * MS)]
        samples, rejected = make_samples(frames[:4], changed)
        self.assertEqual(samples, [])
        self.assertIn('invalid_or_changed_segment', rejected)

    def test_capture_gap_and_partial_future_window_are_rejected(self):
        frames = [frame(i, at) for i, at in enumerate((50, 100, 150, 200, 400, 450, 500))]
        events = [start(), dict(type='segment_end', segment=1, time_ns=520 * MS)]
        samples, rejected = make_samples(frames, events)
        self.assertTrue(rejected.get('capture_gap'))
        self.assertTrue(rejected.get('incomplete_future_window'))
        self.assertTrue(all(s['action_end_ns'] <= 520 * MS for s in samples))

    def test_stale_history_is_not_padded(self):
        frames = [frame(i, at) for i, at in enumerate((50, 200, 250, 300, 350))]
        events = [start(), dict(type='segment_end', segment=1, time_ns=400 * MS)]
        samples, rejected = make_samples(frames, events)
        self.assertFalse(samples)
        self.assertTrue(rejected)

    def test_identical_archive_cannot_cross_splits_with_different_session_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            datasets = []
            for number, split in enumerate(('train', 'test')):
                path = root / str(number);path.mkdir()
                manifest = dict(split=split, source_session_id=str(number),
                                source_hashes={'frames.vrec': 'same-image-payload'})
                (path / 'dataset.json').write_text(json.dumps(manifest))
                datasets.append(path)
            with self.assertRaises(ValueError):
                check_splits(datasets)

    def test_build_load_verify_and_split_guards(self):
        try:
            import PIL
        except ImportError:
            self.skipTest('Pillow is required for a dataset build')
        from vimm8bdm.control import Frame
        from vimm8bdm.recording import RecordingWriter
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source';source.mkdir()
            writer = RecordingWriter(source)
            for i in range(7):
                row = frame(i, 50 + i * 50)
                image = Frame(2, 2, bytes((0, 0, 255, 255)) * 4, row['capture_started_ns'])
                self.assertTrue(writer.put('frame', (image, row)))
            events = [start(), dict(type='button', button='left', down=True, time_ns=205 * MS, segment=1, sequence=2),
                      dict(type='button', button='left', down=False, time_ns=215 * MS, segment=1, sequence=3),
                      dict(type='segment_end', time_ns=400 * MS, segment=1, sequence=4),
                      dict(type='recording_end', time_ns=401 * MS, segment=1, sequence=5)]
            for event in events:
                writer.put('event', event)
            writer.close();self.assertIsNone(writer.error)
            manifest = dict(purpose='demonstration', task='navigation', map='MM1CUT', protocol='free',
                            class_selection='fixed', character_stable=True, character='grenademan',
                            training_ready=False, input_sent=False)
            (source / 'session.json').write_text(json.dumps(manifest))
            (source / 'summary.json').write_text(json.dumps(dict(complete=True, frames=7, events=5, segments=1, dropped_items=0)))
            output = root / 'dataset'
            result = build(source, output, split='train')
            self.assertFalse(result['training_ready'])
            self.assertTrue(verify_source(output)['source_hashes_match'])
            sample = json.loads((output / 'samples.jsonl').read_text().splitlines()[0])
            self.assertEqual(sample['action']['button_held_fraction']['left'], .2)
            self.assertEqual(load_history(output, sample)[0].getpixel((0, 0)), (255, 0, 0))
            corrupted = copy.deepcopy(sample)
            corrupted['history'][0]['capture_finished_ns'] = sample['anchor_ns'] + 1
            with self.assertRaises(ValueError):
                load_history(output, corrupted)
            with self.assertRaises(FileExistsError):
                build(source, output, split='train')
            other = root / 'validation';other.mkdir()
            (other / 'dataset.json').write_text(json.dumps({**result, 'split': 'validation'}))
            with self.assertRaises(ValueError):
                check_splits([output, other])
            (source / 'session.json').write_text(json.dumps({**manifest, 'task': 'aim'}))
            changed_metadata = build(source, root / 'changed-metadata', split='validation')
            self.assertEqual(changed_metadata['source_session_id'], result['source_session_id'])
            with self.assertRaises(ValueError):
                check_splits([output, root / 'changed-metadata'])
            (source / 'session.json').write_text(json.dumps({**manifest, 'purpose': 'interface_check'}))
            with self.assertRaises(ValueError):
                build(source, root / 'blocked')
            with self.assertRaises(ValueError):
                build(source, root / 'blocked-train', split='train', allow_interface_check=True)
            with self.assertRaises(ValueError):
                verify_source(output)
