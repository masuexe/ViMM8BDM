"""Offline checks for input ownership, takeover, and observation freshness."""
import unittest

from vimm8bdm.control import Action, Frame, InputLease, PixelProbe, synthetic_frame


class Sink:
    def __init__(self):
        self.held = Action()
        self.events = []
        self.fail = False

    def apply(self, action):
        self.held = action
        self.events.append(action)
        if self.fail:
            raise OSError("injected input failure")

    def release(self):
        self.held = Action()


class ControlTests(unittest.TestCase):
    def setUp(self):
        self.now = 1_000_000_000
        self.focus = True
        self.sink = Sink()
        self.lease = InputLease(self.sink, lambda: self.focus, lambda: self.now)

    def submit(self, action=Action(frozenset({"w"}), attack=True)):
        return self.lease.submit(action, self.now)

    def test_default_pause_prevents_input(self):
        self.assertFalse(self.submit())
        self.assertEqual(self.sink.events, [])

    def test_takeover_releases_movement_and_attack(self):
        self.lease.toggle()
        self.assertTrue(self.submit())
        self.lease.pause()
        self.assertEqual(self.sink.held, Action())
        self.assertFalse(self.submit())

    def test_late_inference_cannot_reapply_after_takeover(self):
        self.lease.toggle()
        captured = self.now
        self.lease.pause()
        self.assertFalse(self.lease.submit(Action(attack=True), captured))

    def test_focus_loss_releases_and_requires_rearm(self):
        self.lease.toggle()
        self.submit()
        self.focus = False
        self.lease.poll()
        self.assertEqual(self.sink.held, Action())
        self.focus = True
        self.assertFalse(self.submit())

    def test_cannot_arm_on_another_window(self):
        self.focus = False
        self.lease.toggle()
        self.assertFalse(self.lease.armed)

    def test_stale_and_future_frames_disarm(self):
        for offset in (-150_000_001, 1):
            with self.subTest(offset=offset):
                self.lease.toggle()
                self.assertFalse(self.lease.submit(Action(attack=True), self.now + offset))
                self.assertFalse(self.lease.armed)

    def test_watchdog_releases_without_policy_heartbeat(self):
        self.lease.toggle()
        self.submit()
        self.now += 200_000_000
        self.lease.poll()
        self.assertEqual(self.sink.held, Action())
        self.assertFalse(self.lease.armed)

    def test_late_submit_cannot_revive_expired_lease(self):
        self.lease.toggle()
        self.now += 200_000_001
        self.assertFalse(self.submit())

    def test_partial_input_failure_releases_and_disarms(self):
        self.lease.toggle()
        self.sink.fail = True
        with self.assertRaises(OSError):
            self.submit()
        self.assertEqual(self.sink.held, Action())
        self.assertFalse(self.lease.armed)

    def test_invalid_model_action_disarms(self):
        for action in (Action(frozenset({"enter"})), Action(mouse_dx=float("nan"))):
            with self.subTest(action=action):
                self.lease.toggle()
                with self.assertRaises(ValueError):
                    self.submit(action)
                self.assertFalse(self.lease.armed)

    def test_marker_drives_both_axes_and_attack(self):
        probe = PixelProbe()
        center, right, left = [probe.predict(synthetic_frame(i)) for i in range(3)]
        self.assertTrue(center.attack)
        self.assertGreater(right.mouse_dx, 0)
        self.assertLess(left.mouse_dx, 0)
        self.assertFalse(right.attack)
        self.assertIn("w", center.keys)
        high = bytearray(160 * 120 * 4)
        for y in range(16, 40):
            for x in range(68, 92):
                high[(y * 160 + x) * 4:(y * 160 + x) * 4 + 4] = bytes((0, 0, 255, 255))
        self.assertLess(probe.predict(Frame(160, 120, bytes(high), self.now)).mouse_dy, 0)

    def test_corrupt_frame_rejected(self):
        with self.assertRaises(ValueError):
            Frame(2, 2, b"", self.now)

    def test_capture_and_default_lease_use_the_same_clock(self):
        sink = Sink()
        lease = InputLease(sink, lambda: True, max_age_ns=5_000_000_000, lease_ns=5_000_000_000)
        frame = synthetic_frame()
        lease.toggle()
        self.assertTrue(lease.submit(Action(), frame.captured_ns))


if __name__ == "__main__":
    unittest.main()
