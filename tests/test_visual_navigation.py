"""Check visual navigation without a desktop or engine state."""
import unittest

from vimm8bdm.control import Action, Frame
from vimm8bdm.visual_navigation import VisualNavigationPolicy, image_change


def frame(seconds, shade=50, hud=False, width=160, height=120):
    pixels = bytearray(bytes((shade, shade, shade, 255)) * width * height)
    if hud:
        for y in range(height):
            for x in range(width):
                if x < width * .18 or y > height * .55:
                    at = (y * width + x) * 4
                    pixels[at:at + 3] = bytes((255, 0, 255))
    return Frame(width, height, bytes(pixels), int(seconds * 1e9))


class VisualNavigationTests(unittest.TestCase):
    def setUp(self):
        self.policy = VisualNavigationPolicy()

    def advance(self, until, shade=50, hud=False):
        start = (self.policy.last_frame_ns or 0) / 1e9
        action = Action()
        for tick in range(round(start * 20) + 1, round(until * 20) + 1):
            action = self.policy.predict(frame(tick / 20, shade, hud))
        return action

    def test_static_view_releases_forward_and_turns_by_exactly_256_counts(self):
        self.assertEqual(self.policy.predict(frame(0)), Action())
        self.assertEqual(self.advance(.3).keys, frozenset({'w'}))
        self.assertEqual(self.advance(.9), Action())
        total = 0
        for tick in range(19, 60):
            action = self.policy.predict(frame(tick / 20))
            total += action.mouse_dx
            self.assertFalse(action.keys)
            self.assertFalse(action.attack)
            self.assertLessEqual(action.mouse_dx, 8)
            if total == 256:
                break
        self.assertEqual(total, 256)
        self.assertEqual(self.policy.blocked_probes, 1)

    def test_repeated_blockage_increases_turn_budget_with_a_limit(self):
        self.policy.predict(frame(0))
        budgets = []
        previous = 0
        for tick in range(1, 1000):
            action = self.policy.predict(frame(tick / 20))
            if self.policy.blocked_probes != previous:
                previous = self.policy.blocked_probes
                budgets.append(self.policy.turn_budget)
            if self.policy.phase == 'turn':
                self.assertFalse(action.keys)
            self.assertLessEqual(action.mouse_dx, 8)
            if len(budgets) == 5:
                break
        self.assertEqual(budgets, [256, 512, 768, 1024, 1024])

    def test_one_changed_probe_does_not_clear_escape_history(self):
        self.policy.predict(frame(0))
        self.advance(2.85)
        self.assertEqual(self.policy.phase, 'forward')
        self.advance(3.45, shade=100)
        self.assertEqual(self.policy.changed_streak, 1)
        self.assertEqual(self.policy.escape_attempts, 1)
        self.advance(4.1, shade=100)
        self.assertEqual(self.policy.turn_budget, 512)

    def test_three_changed_probes_clear_escape_history(self):
        self.policy.predict(frame(0))
        self.advance(2.85)
        self.advance(3.5, shade=100)
        self.advance(4.15, shade=150)
        self.advance(4.8, shade=200)
        self.assertEqual(self.policy.changed_streak, 3)
        self.assertEqual(self.policy.escape_attempts, 0)

    def test_changed_view_keeps_forward_without_turning(self):
        self.policy.predict(frame(0))
        self.advance(.3)
        action = self.advance(.9, shade=100)
        self.assertEqual(action.keys, frozenset({'w'}))
        self.assertEqual(action.mouse_dx, 0)
        self.assertEqual(self.policy.successful_probes, 1)

    def test_hud_and_weapon_changes_do_not_count_as_world_movement(self):
        self.policy.predict(frame(0))
        self.advance(.3)
        self.advance(.9, hud=True)
        self.assertEqual(self.policy.motion_score, 0)
        self.assertEqual(self.policy.phase, 'turn')

    def test_capture_gap_and_resize_restart_without_an_action(self):
        self.policy.predict(frame(0))
        self.advance(.4)
        self.assertEqual(self.policy.predict(frame(1)), Action())
        self.assertEqual(self.policy.last_prediction['reset_reason'], 'capture_interval_invalid')
        self.assertEqual(self.policy.predict(frame(1.05, width=320, height=240)), Action())
        self.assertEqual(self.policy.last_prediction['reset_reason'], 'capture_size_changed')

    def test_manual_reenable_reset_discards_old_motion_reference(self):
        self.policy.predict(frame(0))
        self.advance(.9)
        self.policy.reset()
        self.assertEqual(self.policy.predict(frame(10)), Action())
        self.assertEqual(self.policy.phase, 'settle')
        self.assertIsNone(self.policy.motion_score)

    def test_isolated_animation_has_limited_effect_on_change_score(self):
        self.assertEqual(image_change([0] * 100, [255] * 5 + [0] * 95), 0)


if __name__ == '__main__':
    unittest.main()
