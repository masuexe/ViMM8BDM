"""Check action conversion without suppressing unsupported labels."""
import unittest
from vimm8bdm.action_mapping import inspect_label


def empty():
    return dict(keys_at_start=[], keys_at_end=[], buttons_at_start=[], buttons_at_end=[],
                key_held_fraction={}, button_held_fraction={}, mouse_dx_raw=0, mouse_dy_raw=0,
                wheel_vertical=0, wheel_horizontal=0, events=[])


class ActionMappingTests(unittest.TestCase):
    def test_combined_movement_and_attack_remain_independent(self):
        label = empty()
        label.update(keys_at_start=[87, 68], keys_at_end=[87, 68], key_held_fraction={'87': 1, '68': 1},
                     buttons_at_start=['left'], buttons_at_end=['left'], button_held_fraction={'left': 1})
        result = inspect_label(label)
        self.assertEqual(result['candidate_action']['keys'], ['d', 'w'])
        self.assertTrue(result['candidate_action']['attack'])
        self.assertFalse(result['input_sent'])

    def test_key_press_at_anchor_can_be_one_step(self):
        label = empty();label['key_held_fraction'] = {'32': 1}
        label['events'] = [dict(type='key', vk=32, scan=57, extended=0, down=True, offset_ns=0)]
        self.assertEqual(inspect_label(label)['candidate_action']['keys'], ['space'])

    def test_short_tap_requires_timed_execution(self):
        label = empty();label['button_held_fraction'] = {'left': .2}
        label['events'] = [dict(type='button', button='left', down=True, offset_ns=5_000_000),
                           dict(type='button', button='left', down=False, offset_ns=15_000_000)]
        result = inspect_label(label)
        self.assertIsNone(result['candidate_action'])
        self.assertIn('timed_transition_required', result['blocking_reasons'])

    def test_unsupported_labels_are_not_dropped(self):
        label = empty();label.update(key_held_fraction={'82': 1}, button_held_fraction={'right': 1}, wheel_vertical=120)
        result = inspect_label(label)
        self.assertIsNone(result['candidate_action'])
        self.assertEqual(result['unsupported_virtual_keys'], [82])
        self.assertEqual(result['unsupported_mouse_buttons'], ['right'])
        self.assertIn('wheel_not_supported', result['blocking_reasons'])

    def test_unknown_scale_and_clipping_are_explicit(self):
        label = empty();label['mouse_dx_raw'] = 30
        self.assertIn('mouse_scale_not_supplied', inspect_label(label)['blocking_reasons'])
        self.assertIn('mouse_limit_exceeded', inspect_label(label, 1, 1)['blocking_reasons'])
        result = inspect_label(label, .5, .5)
        self.assertEqual(result['candidate_action']['mouse_dx'], 15)
        self.assertFalse(result['mouse_unit_conversion_verified'])
        label['mouse_dx_raw'] = 1
        self.assertTrue(inspect_label(label, .25, .25)['motion_rounded_to_zero'])

    def test_invalid_scales_and_modifier_semantics_do_not_pass(self):
        label = empty()
        for scales in [(1, None), (0, 1), (float('nan'), 1)]:
            with self.assertRaises(ValueError):inspect_label(label, *scales)
        label['key_held_fraction'] = {'16': 1}
        self.assertIn('modifier_binding_and_side_unverified', inspect_label(label)['blocking_reasons'])
