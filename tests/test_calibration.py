"""Check visual matching and pulse dispatch without desktop input."""
import contextlib
import io
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from vimm8bdm.calibration import collect, estimate_shift, measure_trial, save_png
from vimm8bdm.control import Frame


def texture(width=160, height=120):
    rng = random.Random(82)
    return Frame(width, height, bytes(v for _ in range(width * height)
                                     for v in (rng.randrange(256), rng.randrange(256), rng.randrange(256), 255)), 0)


def shift(frame, dx=0, dy=0):
    raw = bytearray(len(frame.bgra))
    for y in range(frame.height):
        for x in range(frame.width):
            source = (((y - dy) % frame.height) * frame.width + (x - dx) % frame.width) * 4
            target = (y * frame.width + x) * 4
            raw[target:target + 4] = frame.bgra[source:source + 4]
    return Frame(frame.width, frame.height, bytes(raw), 0)


class CalibrationTests(unittest.TestCase):
    def test_both_axes_and_signs_match_known_translation(self):
        before = texture()
        for axis in ('x', 'y'):
            for delta in (-9, 5):
                with self.subTest(axis=axis, delta=delta):
                    after = shift(before, dx=delta if axis == 'x' else 0, dy=delta if axis == 'y' else 0)
                    result = measure_trial(before, before, after, axis, 4)
                    self.assertTrue(result['valid'])
                    self.assertEqual(result['image_pixels_per_count'], delta / 4)

    def test_moving_baseline_is_rejected(self):
        baseline = texture()
        before = shift(baseline, dx=3)
        self.assertFalse(measure_trial(baseline, before, shift(before, dx=7), 'x', 4)['valid'])

    def test_no_response_and_blank_scene_are_rejected(self):
        frame = texture()
        self.assertFalse(measure_trial(frame, frame, frame, 'x', 4)['valid'])
        blank = Frame(160, 120, bytes(160 * 120 * 4), 0)
        self.assertEqual(estimate_shift(blank, blank, 'x')['reason'], 'low_texture')

    def test_boundary_match_is_rejected(self):
        frame = texture()
        self.assertFalse(estimate_shift(frame, shift(frame, dx=24), 'x')['valid'])

    def test_png_is_exclusive_and_has_correct_dimensions(self):
        import struct
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'frame.png'
            save_png(texture(), path)
            raw = path.read_bytes()
            self.assertEqual(raw[:8], b'\x89PNG\r\n\x1a\n')
            self.assertEqual(struct.unpack('!2I', raw[16:24]), (160, 120))
            with self.assertRaises(FileExistsError):
                save_png(texture(), path)

    def test_initial_wait_does_not_expire_and_each_pulse_is_sent_once(self):
        class Clock:
            now = 0

            def read(self):
                return self.now

            def sleep(self, seconds):
                self.now += seconds

        clock = Clock()
        actions = []

        class Capture:
            def __init__(self, window):
                pass

            def grab(self):
                return Frame(160, 120, bytes((70, 70, 70, 255)) * 160 * 120, 0)

            def close(self):
                pass

        class Worker:
            armed = False
            stopped = False
            generation = 1

            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def refresh(self):
                self.armed = clock.now >= 181

            def submit(self, action, captured_ns, generation):
                actions.append(action)
                return True

        with tempfile.TemporaryDirectory() as directory:
            with (patch('vimm8bdm.desktop.WindowsWindowCapture', Capture),
                  patch('vimm8bdm.input_worker.InputWorker', Worker),
                  patch('vimm8bdm.calibration.time.perf_counter', clock.read),
                  patch('vimm8bdm.calibration.time.sleep', clock.sleep),
                  contextlib.redirect_stdout(io.StringIO())):
                result = collect(1, Path(directory) / 'run', 1, 0)
            self.assertGreaterEqual(clock.now, 181)
            self.assertLess(result['elapsed_after_first_enable_s'], 12)
            self.assertEqual(result['completed_trials'], 8)
            self.assertEqual(result['exit_reason'], 'complete')
            json.loads((Path(directory) / 'run' / 'summary.local.json').read_text())
        pulses = [(action.mouse_dx, action.mouse_dy) for action in actions
                  if action.mouse_dx or action.mouse_dy]
        self.assertEqual(pulses, [(4, 0), (-4, 0), (8, 0), (-8, 0),
                                  (0, 4), (0, -4), (0, 8), (0, -8)])
        self.assertTrue(all(not action.keys and not action.attack for action in actions))


if __name__ == '__main__':
    unittest.main()
