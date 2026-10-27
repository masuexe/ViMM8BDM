"""Check Windows mouse structure layout and event contents without OS input."""
import ctypes
import unittest

from vimm8bdm.winmouse import InputEvent, WindowsRelativeMouse, query_mouse_settings


class WindowsMouseTests(unittest.TestCase):
    def test_input_layout_matches_windows_pointer_size(self):
        self.assertEqual(ctypes.sizeof(InputEvent), 40 if ctypes.sizeof(ctypes.c_size_t) == 8 else 28)

    def test_move_uses_relative_input_without_absolute_flag(self):
        events = []
        def send(count, pointer, size):
            event = pointer._obj
            events.append((count, size, event.kind, event.data.mouse.dx, event.data.mouse.dy, event.data.mouse.flags))
            return 1
        mouse = WindowsRelativeMouse(send)
        mouse.move(-4, 2)
        self.assertEqual(events, [(1, ctypes.sizeof(InputEvent), 0, -4, 2, 1)])

    def test_zero_delta_does_not_send_an_event(self):
        def unexpected(*args):
            self.fail("Zero movement must not send an OS event")
        WindowsRelativeMouse(unexpected).move(0, 0)

    def test_pointer_query_uses_only_get_operations(self):
        calls = []
        def query(action, parameter, pointer, flags):
            calls.append((action, parameter, flags))
            if action == 3:
                pointer._obj[:] = (6, 10, 1)
            else:
                pointer._obj.value = 5
            return 1
        result = query_mouse_settings(query)
        self.assertEqual(calls, [(3, 0, 0), (0x70, 0, 0)])
        self.assertEqual(result['pointer_speed'], 5)
        self.assertFalse(result['settings_changed'])

    def test_pointer_query_failure_is_reported(self):
        with self.assertRaises(OSError):
            query_mouse_settings(lambda *args: 0)

    def test_rejected_event_is_reported(self):
        with self.assertRaises(OSError):
            WindowsRelativeMouse(lambda *args: 0).move(1, 0)


if __name__ == "__main__":
    unittest.main()
