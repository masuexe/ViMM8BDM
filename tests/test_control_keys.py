"""Check takeover keys and the distinction between pause and exit."""
import threading
import unittest

from vimm8bdm.control import Action, ControlKeys, InputLease


class Sink:
    def __init__(self):
        self.last = Action()

    def apply(self, action):
        self.last = action

    def release(self):
        self.last = Action()


class ControlKeyTests(unittest.TestCase):
    def setUp(self):
        self.sink = Sink()
        self.lease = InputLease(self.sink, lambda: True)
        self.stop = threading.Event()
        self.keys = ControlKeys(self.lease, self.stop)

    def test_escape_pauses_without_ending_the_process(self):
        self.keys.press("f8")
        self.assertTrue(self.lease.submit(Action(frozenset({"w"}), attack=True), self.lease.clock()))
        self.assertEqual(self.keys.press("esc"), "escape_pause")
        self.assertFalse(self.stop.is_set())
        self.assertFalse(self.lease.armed)
        self.assertEqual(self.sink.last, Action())
        self.keys.release("f8")
        self.keys.press("f8")
        self.assertTrue(self.lease.armed)

    def test_key_repeat_does_not_toggle_twice(self):
        self.keys.press("f8")
        self.assertIsNone(self.keys.press("f8"))
        self.assertTrue(self.lease.armed)
        self.keys.release("f8")
        self.keys.press("f8")
        self.assertFalse(self.lease.armed)

    def test_exit_releases_and_prevents_reenabling(self):
        self.keys.press("f8")
        self.keys.press("f9")
        self.assertTrue(self.stop.is_set())
        self.assertEqual(self.lease.reason, "quit_key")
        self.keys.release("f8")
        self.assertIsNone(self.keys.press("f8"))
        self.assertFalse(self.lease.armed)

    def test_other_keys_do_not_control_the_lease(self):
        self.assertIsNone(self.keys.press(None))
        self.assertIsNone(self.keys.press("space"))
        self.assertFalse(self.lease.armed)

    def test_blocked_focus_has_an_explicit_state(self):
        self.lease.focused = lambda: False
        self.assertEqual(self.keys.press("f8"), "focus_blocked")
        self.assertFalse(self.lease.armed)


if __name__ == "__main__":
    unittest.main()
