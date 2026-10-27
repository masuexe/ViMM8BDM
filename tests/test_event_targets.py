"""Check explicit targets and their bounded timed output without desktop input."""
import copy
import math
import unittest

from vimm8bdm.dataset import HeldState, action_window
from vimm8bdm.event_targets import EventTarget, compile_target, encode_label, target_label

MS = 1_000_000


def label(keys=(), changes=(), motion=()):
    events = [{'type': 'key', 'time_ns': at, 'vk': vk, 'down': down, 'scan': 0, 'extended': 0}
              for at, vk, down in changes]
    events.extend({'type': 'motion', 'time_ns': at, 'dx': dx, 'dy': dy} for at, dx, dy in motion)
    events.sort(key=lambda e: e['time_ns'])
    return action_window(HeldState({'held_keys': list(keys), 'held_buttons': []}), events, 0, 50 * MS)


class EventTargetTests(unittest.TestCase):
    def test_late_press_and_release_keep_their_timing(self):
        for keys, down in [((), True), ((87,), False)]:
            original = label(keys, [(40 * MS, 87, down)])
            start, target = encode_label(original)
            self.assertEqual(target.key_offsets_ns[0], (40 * MS,))
            decoded = target_label(target, start)
            self.assertEqual(decoded['keys_at_end'], original['keys_at_end'])
            self.assertEqual(decoded['key_held_fraction'], original['key_held_fraction'])
            window = compile_target(target, start, 1, 1)
            self.assertEqual(window.packets[1].offset_ns, 40 * MS)
            self.assertEqual('w' in window.packets[1].action.keys, down)

    def test_brief_release_and_press_are_retained_separately(self):
        original = label((65, 87), [(5_044_700, 65, False), (5_363_500, 65, True)])
        start, target = encode_label(original)
        self.assertEqual(target.key_offsets_ns[1], (5_044_700, 5_363_500))
        self.assertEqual(target.training_fields()['key_counts'], [0, 2, 0, 0])
        decoded = target_label(target, start)
        self.assertAlmostEqual(decoded['key_held_fraction']['65'], .993624)
        self.assertEqual(decoded['keys_at_end'], [65, 87])

    def test_short_press_release_and_simultaneous_keys_round_trip(self):
        original = label(changes=[(10 * MS, 65, True), (10 * MS, 87, True),
                                  (20 * MS, 87, False), (30 * MS, 65, False)])
        start, target = encode_label(original)
        self.assertEqual(target.training_fields()['key_counts'], [2, 2, 0, 0])
        decoded = target_label(target, start)
        self.assertEqual(decoded['keys_at_end'], [])
        self.assertEqual(decoded['key_held_fraction'], original['key_held_fraction'])

    def test_zero_offset_is_an_event_and_absent_offsets_are_masked(self):
        start, target = encode_label(label(changes=[(0, 87, True)]))
        fields = target.training_fields()
        self.assertEqual(fields['key_offset_fraction'][0], [0., 0.])
        self.assertEqual(fields['key_offset_mask'][0], [True, False])
        self.assertEqual(fields['key_offset_mask'][1], [False, False])
        self.assertEqual(target_label(target, start)['keys_at_end'], [87])

    def test_large_mouse_total_is_split_without_clipping(self):
        original = label(motion=[(3 * MS, 300, -30), (40 * MS, 113, -7)])
        before = copy.deepcopy(original)
        start, target = encode_label(original)
        packets = compile_target(target, start, 1.06, .92).packets
        self.assertAlmostEqual(math.fsum(p.action.mouse_dx for p in packets), 413 * 1.06)
        self.assertAlmostEqual(math.fsum(p.action.mouse_dy for p in packets), -37 * .92)
        self.assertTrue(all(abs(p.action.mouse_dx) <= 23 for p in packets))
        self.assertEqual(original, before)
        decoded = target_label(target, start)
        motion = [e for e in decoded['events'] if e['type'] == 'motion']
        self.assertEqual(motion[0]['offset_ns'], 25 * MS)

    def test_mouse_total_does_not_claim_to_preserve_opposing_motion(self):
        original = label(motion=[(MS, 10, 0), (2 * MS, -10, 0)])
        before = copy.deepcopy(original)
        start, target = encode_label(original)
        self.assertEqual(target.mouse_dx_raw, 0)
        self.assertEqual(target_label(target, start)['events'], [])
        self.assertEqual(original, before)

    def test_caller_owned_state_controls_the_direction_of_each_change(self):
        target = EventTarget(((10 * MS,), (), (), ()))
        self.assertTrue(target_label(target, ())['events'][0]['down'])
        self.assertFalse(target_label(target, ('w',))['events'][0]['down'])
        with self.assertRaises(ValueError):
            target_label(target, ('space',))

    def test_corrupt_labels_and_extra_changes_are_rejected(self):
        for patch in [{'mouse_dx_raw': 10}, {'keys_at_end': [87]}, {'key_held_fraction': {'87': .5}}]:
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                encode_label({**label(), **patch})
        with self.assertRaises(ValueError):
            encode_label(label(changes=[(MS, 87, True), (2 * MS, 87, False), (3 * MS, 87, True)]))
        with self.assertRaises(ValueError):
            encode_label(label(changes=[(MS, 87, True), (2 * MS, 87, True)]))
        with self.assertRaises(ValueError):
            encode_label({**label(), 'wheel_vertical': 120})

    def test_invalid_offsets_totals_and_packet_overflow_are_rejected(self):
        for target in [EventTarget(((-1,), (), (), ())), EventTarget(((50 * MS,), (), (), ())),
                       EventTarget(((2, 1), (), (), ())), EventTarget(((), (), ())),
                       EventTarget(((), (), (), ()), float('nan'))]:
            with self.subTest(target=target), self.assertRaises(ValueError):
                target.validate()
        with self.assertRaises(ValueError):
            compile_target(EventTarget(((), (), (), ()), 1000000), (), 1, 1)
        with self.assertRaises(ValueError):
            compile_target(EventTarget(((), (), (), ())), (), 0, 1)


if __name__ == '__main__':
    unittest.main()
