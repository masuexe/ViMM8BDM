"""Check timed input with a deterministic clock and an isolated process."""
import math
from pathlib import Path
import tempfile
import time
import unittest

from vimm8bdm.control import Action
from vimm8bdm.dataset import HeldState, action_window
from vimm8bdm.input_worker import InputOwner, InputWorker
from vimm8bdm.timed_actions import MouseRemainder, TimedPacket, TimedWindow, compile_label
from vimm8bdm.timed_check import check_windows
from test_input_worker import AuditFactory


def example_label():
    state = HeldState({'held_keys': [], 'held_buttons': []})
    events = [
        {'type': 'key', 'time_ns': 5_000_000, 'vk': 65, 'scan': 30, 'extended': False, 'down': True},
        {'type': 'motion', 'time_ns': 10_000_000, 'dx': 413, 'dy': -37},
        {'type': 'key', 'time_ns': 15_000_000, 'vk': 65, 'scan': 30, 'extended': False, 'down': False},
    ]
    return action_window(state, events, 0, 50_000_000)


class Clock:
    def __init__(self):
        self.now = 1_000_000_000

    def __call__(self):
        return self.now


class Sink:
    def __init__(self):
        self.actions = []
        self.held = Action()

    def apply(self, action):
        self.actions.append(action)
        self.held = action

    def release(self):
        self.held = Action()


class TimedTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.focused = True
        self.sink = Sink()
        self.owner = InputOwner(self.sink, lambda: self.focused, clock=self.clock)
        self.owner.press('f8')
        self.owner.release_key('f8')
        self.window = compile_label(example_label(), 1.06, .92)

    def submit(self):
        return self.owner.submit_timed(self.window, self.clock(), self.owner.generation)

    def advance(self, milliseconds):
        self.clock.now += int(milliseconds * 1_000_000)
        self.owner.poll()

    def test_short_tap_preserves_edges_and_large_motion_preserves_total(self):
        packets = self.window.packets
        self.assertEqual(packets[1].offset_ns, 5_000_000)
        self.assertEqual(packets[1].action.keys, frozenset({'a'}))
        self.assertEqual(packets[-2].offset_ns, 15_000_000)
        self.assertEqual(packets[-2].action.keys, frozenset())
        self.assertAlmostEqual(math.fsum(p.action.mouse_dx for p in packets), 413 * 1.06)
        self.assertAlmostEqual(math.fsum(p.action.mouse_dy for p in packets), -37 * .92)
        self.assertTrue(all(abs(p.action.mouse_dx) <= 23 for p in packets))
        self.assertTrue(self.submit())
        self.advance(5)
        self.assertEqual(self.sink.held.keys, frozenset({'a'}))
        self.advance(5)
        self.advance(5)
        self.assertEqual(self.sink.held.keys, frozenset())
        self.advance(35)
        self.assertIsNone(self.owner.pending)
        self.assertEqual(sum(a.mouse_dx for a in self.sink.actions), round(413 * 1.06))
        self.assertEqual(sum(a.mouse_dy for a in self.sink.actions), round(-37 * .92))

    def test_fractional_motion_accumulates_across_windows(self):
        remainder = MouseRemainder()
        outputs = [remainder.convert(Action(mouse_dx=.1, mouse_dy=-.1)) for _ in range(100)]
        self.assertEqual(sum(a.mouse_dx for a in outputs), 10)
        self.assertEqual(sum(a.mouse_dy for a in outputs), -10)
        self.assertAlmostEqual(remainder.x, 0)

    def test_packets_and_status_do_not_renew_lease(self):
        self.assertTrue(self.submit())
        deadline = self.owner.lease.deadline
        for _ in range(50):
            self.advance(1)
            self.owner.state()
            self.assertEqual(self.owner.lease.deadline, deadline)
        self.advance(150)
        self.assertFalse(self.owner.lease.armed)
        self.assertIsNone(self.owner.pending)

    def test_pause_cancels_and_old_generation_is_rejected(self):
        self.submit()
        self.advance(5)
        generation = self.owner.generation
        self.owner.press('esc')
        self.owner.release_key('esc')
        count = len(self.sink.actions)
        self.owner.press('f8')
        self.owner.release_key('f8')
        self.advance(10)
        self.assertEqual(len(self.sink.actions), count)
        self.assertFalse(self.owner.submit_timed(self.window, self.clock(), generation))
        self.assertTrue(self.submit())

    def test_focus_loss_cancels_without_resume(self):
        self.submit()
        self.advance(5)
        self.focused = False
        self.advance(1)
        self.assertIsNone(self.owner.pending)
        self.assertFalse(self.sink.held.keys)
        self.focused = True
        self.advance(10)
        self.assertFalse(self.owner.lease.armed)

    def test_scheduler_stall_cancels_instead_of_replaying_a_burst(self):
        self.submit()
        self.advance(30)
        self.assertEqual(self.owner.lease.reason, 'timing_overrun')
        self.assertEqual(len(self.sink.actions), 1)
        self.assertIsNone(self.owner.pending)

    def test_busy_window_rejects_replacement_without_renewing_lease(self):
        self.submit()
        pending, deadline = self.owner.pending, self.owner.lease.deadline
        self.advance(1)
        self.assertFalse(self.submit())
        self.assertEqual(self.owner.pending, pending)
        self.assertEqual(self.owner.lease.deadline, deadline)

    def test_stale_frame_disarms(self):
        self.assertFalse(self.owner.submit_timed(self.window, self.clock() - 151_000_000,
                                                self.owner.generation))
        self.assertEqual(self.owner.lease.reason, 'stale_frame')
        self.assertIsNone(self.owner.pending)

    def test_ordinary_action_cancels_scheduled_work(self):
        self.submit()
        self.owner.submit(Action(frozenset({'d'})), self.clock(), self.owner.generation)
        self.advance(10)
        self.assertIsNone(self.owner.pending)
        self.assertEqual(self.sink.held.keys, frozenset({'d'}))

    def test_malformed_labels_are_rejected(self):
        for field, value in [('wheel_vertical', 1), ('mouse_dx_raw', 0), ('keys_at_end', [87])]:
            label = example_label()
            label[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                compile_label(label, 1, 1)
        label = example_label()
        label['events'][0]['vk'] = 27
        with self.assertRaises(ValueError):
            compile_label(label, 1, 1)
        for scale in [0, -1, math.inf, math.nan]:
            with self.assertRaises(ValueError):
                compile_label(example_label(), scale, 1)

    def test_bad_windows_cannot_bypass_packet_or_duration_limits(self):
        for window in [
            TimedWindow((TimedPacket(0, Action()), TimedPacket(50_000_000, Action(mouse_dx=25)))),
            TimedWindow((TimedPacket(0, Action()), TimedPacket(60_000_000, Action())), 60_000_000),
            TimedWindow((TimedPacket(0, Action()),) * 1025),
        ]:
            with self.subTest(window=window), self.assertRaises(ValueError):
                window.validate()

    def test_live_sequence_releases_every_window_and_never_attacks(self):
        sequence = check_windows()
        for _, window in sequence:
            window.validate()
            self.assertEqual(window.packets[-1].action, Action())
            self.assertTrue(all(not p.action.attack for p in window.packets))
        self.assertEqual(sum(p.action.mouse_dx for _, w in sequence for p in w.packets), 0)
        self.assertEqual(sum(p.action.mouse_dy for _, w in sequence for p in w.packets), 0)

    def test_real_worker_executes_a_tap_and_expires_without_heartbeat(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'events.jsonl'
            # Wider edges tolerate normal CI scheduling. Unit tests check exact offsets.
            window = TimedWindow((TimedPacket(0, Action(frozenset({'a'}))),
                                  TimedPacket(50_000_000, Action())))
            with InputWorker(backend_factory=AuditFactory(str(path))) as worker:
                worker._request('test_key', key='f8')
                self.assertTrue(worker.submit_timed(window, time.perf_counter_ns(), worker.generation))
                time.sleep(.3)
                worker.refresh()
                self.assertFalse(worker.armed)
                self.assertFalse(worker.snapshot['held_keys'])
            import json
            events = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertTrue(any(e['event'] == 'apply' and e['keys'] == ['a'] for e in events))
            self.assertTrue(any(e['event'] == 'apply' and e['keys'] == [] for e in events))


if __name__ == '__main__':
    unittest.main()
