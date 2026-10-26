"""Check local input phases and invalid capture rejection."""
from dataclasses import replace
import unittest

from vimm8bdm.control import Action, Frame, InputCheckPolicy, blank_frame, synthetic_frame
from vimm8bdm.desktop import Focus
from test_windows import FakeWindows


class InputCheckTests(unittest.TestCase):
    def setUp(self):
        self.now = 0.0
        self.policy = InputCheckPolicy(lambda: self.now)

    def test_each_phase_has_the_expected_action(self):
        start = 0
        phases = set()
        for phase, duration, expected in self.policy.STEPS:
            self.now = start + duration / 2
            self.assertEqual(self.policy.predict(None), expected)
            self.assertEqual(self.policy.phase, phase)
            phases.add(phase)
            start += duration
        self.assertTrue({"pitch_up", "pitch_down", "attack", "release_attack"} <= phases)

    def test_finished_probe_releases_all_inputs(self):
        self.now = sum(step[1] for step in self.policy.STEPS) + 1
        self.assertEqual(self.policy.predict(None), Action())
        self.assertTrue(self.policy.complete)

    def test_reenable_starts_a_new_sequence(self):
        self.now = 100
        self.policy.predict(None)
        self.policy.reset()
        self.assertFalse(self.policy.complete)
        self.assertEqual(self.policy.predict(None), Action())
        self.assertEqual(self.policy.phase, "ready")

    def test_black_pixels_are_blank_despite_nonzero_alpha(self):
        frame = Frame(16, 16, bytes((0, 0, 0, 255)) * 256, 0)
        self.assertTrue(blank_frame(frame))
        self.assertFalse(blank_frame(synthetic_frame()))

    def test_resized_client_blocks_input(self):
        windows = FakeWindows()
        focus = Focus(100, windows)
        windows.info = replace(windows.info, width=800)
        self.assertFalse(focus())


if __name__ == "__main__":
    unittest.main()
