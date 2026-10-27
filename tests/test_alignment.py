"""Check delay estimation against known timing, absent response, and images."""
import bisect
import random
import unittest

from vimm8bdm.alignment import fit_axis, translation


def synthetic(delay_ms=20, constant=False):
    rng = random.Random(94)
    events = []
    for at in range(0, 14_000, 3):
        value = 1 if constant else rng.choice((-3, -2, -1, 0, 1, 2, 3))
        events.append({'type': 'motion', 'segment': 1, 'time_ns': at * 1_000_000,
                       'dx': value, 'dy': 0})
    times = [e['time_ns'] for e in events]
    pairs = []
    for begin in range(200, 13_000, 80):
        end = begin + 80
        left = bisect.bisect_left(times, (begin - delay_ms) * 1_000_000)
        right = bisect.bisect_left(times, (end - delay_ms) * 1_000_000)
        value = sum(e['dx'] for e in events[left:right])
        pairs.append({'axis': 'x', 'segment': 1, 'begin_ns': begin * 1_000_000,
                      'end_ns': end * 1_000_000, 'shift_px': -1.25 * value})
    return pairs, events


class AlignmentTests(unittest.TestCase):
    def test_known_positive_and_negative_delays(self):
        for delay in (20, -15):
            with self.subTest(delay=delay):
                pairs, events = synthetic(delay)
                result = fit_axis(pairs, events, 'x')
                self.assertEqual(result['status'], 'candidate')
                self.assertEqual(result['best']['delay_ms'], delay)
                self.assertAlmostEqual(result['best']['scale_px_per_raw_count'], -1.25)
                self.assertAlmostEqual(result['best']['validation_r2'], 1)
                self.assertFalse(result['automatic_label_shift_applied'])

    def test_constant_pan_cannot_establish_delay(self):
        pairs, events = synthetic(constant=True)
        result = fit_axis(pairs, events, 'x')
        self.assertEqual(result['status'], 'unresolved')
        self.assertTrue('broad_training_delay_range' in result['reasons']
                        or 'too_few_both_direction_samples' in result['reasons'])

    def test_holdout_failure_is_not_a_successful_fit(self):
        pairs, events = synthetic()
        for pair in pairs[len(pairs) * 2 // 3:]:
            pair['shift_px'] = -pair['shift_px']
        result = fit_axis(pairs, events, 'x')
        self.assertEqual(result['status'], 'unresolved')
        self.assertIn('weak_validation_fit', result['reasons'])

    def test_uncorrelated_images_do_not_pass_validation(self):
        pairs, events = synthetic()
        rng = random.Random(17)
        for pair in pairs:
            pair['shift_px'] = rng.uniform(-20, 20)
        result = fit_axis(pairs, events, 'x')
        self.assertNotEqual(result['status'], 'candidate')

    def test_insufficient_data_has_no_delay_recommendation(self):
        pairs, events = synthetic()
        result = fit_axis(pairs[:12], events, 'x')
        self.assertEqual(result['status'], 'insufficient_data')
        self.assertNotIn('best', result)

    def test_archive_phase_guards_and_large_gap_rejection(self):
        try:
            import PIL
        except ImportError:
            self.skipTest('Pillow is required for archive analysis')
        import json
        from pathlib import Path
        import tempfile
        from vimm8bdm.alignment import analyze
        from vimm8bdm.control import Frame
        from vimm8bdm.recording import RecordingWriter
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            writer = RecordingWriter(root)
            events = [dict(type='protocol_phase', phase='horizontal', axis='x', time_ns=0, segment=1, sequence=1),
                      dict(type='protocol_phase', phase='vertical', axis='y', time_ns=1_300_000_000, segment=1, sequence=2),
                      dict(type='recording_end', time_ns=1_600_000_000, segment=1, sequence=3)]
            for event in events:
                writer.put('event', event)
            for i, at in enumerate((1_000_000_000, 1_100_000_000, 1_400_000_000, 1_500_000_000)):
                frame = Frame(320, 240, bytes((50, 50, 50, 255)) * 320 * 240, at - 1_000_000)
                writer.put('frame', (frame, dict(frame=i, segment=1, held_keys_at_capture_start=[],
                                                capture_started_ns=at - 1_000_000,
                                                capture_finished_ns=at + 1_000_000)))
            writer.close()
            self.assertIsNone(writer.error)
            (root / 'session.json').write_text(json.dumps(dict(input_sent=False, training_ready=False)))
            (root / 'summary.json').write_text(json.dumps(dict(complete=True, dropped_items=0, frames=4,
                                                              events=3, segments=1, exit_reason='duration_limit')))
            report = analyze(root)
            self.assertEqual(report['rejected']['large_capture_gap'], 1)
            self.assertEqual(report['rejected']['protocol_phase'], 3)
            self.assertFalse(report['training_labels_modified'])
            self.assertFalse(report['guided_protocol_completed'])

    def test_matcher_recovers_both_axes_and_rejects_flat_images(self):
        try:
            from PIL import Image, ImageChops
        except ImportError:
            self.skipTest('Pillow is required for image matching')
        rng = random.Random(91)
        before = Image.frombytes('L', (320, 240), bytes(rng.randrange(256) for _ in range(320 * 240)))
        for axis in ('x', 'y'):
            after = ImageChops.offset(before, 7 if axis == 'x' else 0, -4 if axis == 'y' else 0)
            result = translation(before, after, axis)
            self.assertEqual(result['shift'], 7 if axis == 'x' else -4)
        self.assertIsNone(translation(Image.new('L', (320, 240)), Image.new('L', (320, 240)), 'x'))
