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
        self.assertEqual(self.advance(.5), Action())
        total = 0
        for tick in range(11, 60):
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
        self.advance(2.45)
        self.assertEqual(self.policy.phase, 'forward')
        self.advance(2.65, shade=100)
        self.assertEqual(self.policy.changed_streak, 1)
        self.assertEqual(self.policy.escape_attempts, 1)
        self.advance(2.85, shade=100)
        self.assertEqual(self.policy.turn_budget, 512)

    def test_sustained_changed_probes_clear_escape_history(self):
        self.policy.predict(frame(0))
        self.advance(2.45)
        for probe in range(9):
            self.advance(2.65 + .2 * probe, shade=100 if probe % 2 == 0 else 200)
        self.assertEqual(self.policy.changed_streak, 9)
        self.assertEqual(self.policy.escape_attempts, 0)

    def test_forward_probe_waits_200_ms_before_deciding(self):
        self.policy.predict(frame(0))
        self.advance(.3)
        self.assertEqual(self.advance(.45).keys, frozenset({'w'}))
        self.assertEqual(self.policy.blocked_probes, 0)
        self.assertEqual(self.advance(.5), Action())
        self.assertEqual(self.policy.phase, 'turn')
        self.assertGreater(self.advance(.55).mouse_dx, 0)

    def test_static_view_after_movement_starts_turn_within_450_ms(self):
        for stopped_at in [1.0, 1.05, 1.1, 1.15]:
            policy = VisualNavigationPolicy()
            turn_at = None
            for tick in range(40):
                now = tick / 20
                # Each moving frame changes, then the displayed scene stays static.
                shade = 20 + round(min(now, stopped_at) * 100)
                action = policy.predict(frame(now, shade=shade))
                if action.mouse_dx:
                    turn_at = now
                    break
            self.assertIsNotNone(turn_at)
            self.assertGreaterEqual(turn_at, stopped_at)
            self.assertLessEqual(turn_at - stopped_at, .450001)

    def motion_probes(self, changes):
        self.policy.predict(frame(0, shade=20))
        self.advance(.3, shade=20)
        shade = 20
        actions = []
        for probe, change in enumerate(changes):
            shade += change
            actions.append(self.advance(.5 + .2 * probe, shade=shade))
        return actions

    def test_sustained_reduction_triggers_turn_while_view_still_changes(self):
        actions = self.motion_probes([20, 20, 20, 4, 4])
        self.assertEqual(actions[-2].keys, frozenset({'w'}))
        self.assertEqual(actions[-1], Action())
        self.assertEqual(self.policy.phase, 'turn')
        self.assertEqual(self.policy.slow_probes, 1)
        self.assertEqual(self.policy.static_probes, 0)
        self.assertEqual(self.policy.blockage_reason, 'reduced_image_change')
        self.assertEqual(self.policy.motion_baseline, 20)

    def test_one_slow_probe_does_not_turn_and_recovery_clears_streak(self):
        actions = self.motion_probes([20, 20, 20, 4, 20, 4, 20])
        self.assertTrue(all(a.keys == frozenset({'w'}) for a in actions))
        self.assertEqual(self.policy.slow_streak, 0)
        self.assertEqual(self.policy.blocked_probes, 0)

    def test_single_large_change_does_not_set_baseline(self):
        self.motion_probes([80, 10, 10, 10, 10])
        self.assertEqual(self.policy.phase, 'forward')
        self.assertEqual(self.policy.motion_baseline, 10)

    def test_low_but_consistent_motion_does_not_trigger_relative_rule(self):
        self.motion_probes([4] * 10)
        self.assertEqual(self.policy.phase, 'forward')
        self.assertEqual(self.policy.slow_probes, 0)

    def test_pause_reset_discards_speed_reference(self):
        self.motion_probes([20, 20, 20, 4])
        self.policy.reset()
        self.assertFalse(self.policy.motion_history)
        self.assertIsNone(self.policy.motion_baseline)
        self.assertEqual(self.policy.slow_streak, 0)

    def test_changed_view_keeps_forward_without_turning(self):
        self.policy.predict(frame(0))
        self.advance(.3)
        action = self.advance(.5, shade=100)
        self.assertEqual(action.keys, frozenset({'w'}))
        self.assertEqual(action.mouse_dx, 0)
        self.assertEqual(self.policy.successful_probes, 1)

    def test_hud_and_weapon_changes_do_not_count_as_world_movement(self):
        self.policy.predict(frame(0))
        self.advance(.3)
        self.advance(.5, hud=True)
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
        self.advance(.5)
        self.policy.reset()
        self.assertEqual(self.policy.predict(frame(10)), Action())
        self.assertEqual(self.policy.phase, 'settle')
        self.assertIsNone(self.policy.motion_score)

    def test_isolated_animation_has_limited_effect_on_change_score(self):
        self.assertEqual(image_change([0] * 100, [255] * 5 + [0] * 95), 0)


if __name__ == '__main__':
    unittest.main()
