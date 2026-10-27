"""Check event metrics, short taps, and temporal boundaries."""
import unittest

from vimm8bdm.action_events import binary_events, evaluate_events, motion_phase


def sample(index, before=(), after=(), events=(), fraction=0, mouse=0, segment=1):
    return {'segment': segment, 'anchor_ns': index * 50, 'action_end_ns': (index + 1) * 50,
            'action': {'keys_at_start': list(before), 'keys_at_end': list(after),
                       'events': list(events), 'key_held_fraction': {'87': fraction},
                       'mouse_dx_raw': mouse, 'mouse_dy_raw': 0}}


class ActionEventTests(unittest.TestCase):
    def test_counts_missing_and_false_events(self):
        result = binary_events([True, True, False], [True, False, True])
        self.assertEqual((result['tp'], result['fp'], result['fn']), (1, 1, 1))
        self.assertEqual(result['f1'], .5)
        self.assertIsNone(binary_events([], [])['f1'])

    def test_two_changes_remain_two_events_when_end_state_matches_start(self):
        for before, fraction in [((), .2), ((87,), .9936)]:
            down_first = not before
            rows = [sample(0, before=before, after=before,
                           events=[{'type': 'key', 'vk': 87, 'down': down_first},
                                   {'type': 'key', 'vk': 87, 'down': not down_first}], fraction=fraction)]
            result = evaluate_events(rows, [[fraction, 0, 0, 0, 0, 0]])['keys']['w']
            self.assertEqual(result['recorded_events'], 2)
            self.assertEqual(result['windows_with_press_and_release'], 1)
            self.assertEqual(result['recorded_start_state_reference']['press']['fn'], 1)
            self.assertEqual(result['recorded_start_state_reference']['release']['fn'], 1)

    def test_predicted_sequence_is_distinct_from_recorded_start_state(self):
        rows = [sample(0), sample(1, after=(87,), events=[{'type': 'key', 'vk': 87, 'down': True}], fraction=.4),
                sample(2, before=(87,), after=(87,), fraction=1)]
        result = evaluate_events(rows, [[0, 0, 0, 0, 0, 0], [.2, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]])
        w = result['keys']['w']
        self.assertEqual(w['target_occupancy_end_state_disagreements'], 1)
        self.assertEqual(w['predicted_state_sequence']['press']['fp'], 1)
        self.assertEqual(w['predicted_state_sequence']['press']['fn'], 1)
        self.assertEqual(w['recorded_start_state_reference']['press']['fp'], 0)

    def test_gaps_and_segments_do_not_create_synthetic_transitions(self):
        rows = [sample(0, mouse=10), sample(3, mouse=0), sample(4, mouse=10, segment=2)]
        result = evaluate_events(rows, [[1, 0, 0, 0, 10, 0], [0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 10, 0]])
        self.assertEqual(result['keys']['w']['sequence_boundary_windows_excluded'], 3)
        self.assertEqual(result['keys']['w']['predicted_state_sequence']['press']['predicted_windows'], 0)
        self.assertEqual(result['mouse']['x']['by_actual_phase']['boundary']['windows'], 3)
        self.assertEqual(result['mouse']['x']['phase_detection']['start']['true_windows'], 0)

    def test_mouse_starts_stops_reversals_and_idle_are_separate(self):
        values = [0, 3, 4, -3, 0, 0]
        rows = [sample(i, mouse=value) for i, value in enumerate(values)]
        result = evaluate_events(rows, [[0, 0, 0, 0, value, 0] for value in values])
        for phase in ('start', 'continue', 'reverse', 'stop', 'idle'):
            self.assertEqual(result['mouse']['x']['phase_detection'][phase]['f1'], 1)
        self.assertEqual(motion_phase(None, 0), 'boundary')

    def test_invalid_predictions_and_thresholds_are_rejected(self):
        for predictions in [[], [[0] * 5], [[float('nan')] * 6], [[2, 0, 0, 0, 0, 0]]]:
            with self.assertRaises(ValueError):
                evaluate_events([sample(0)], predictions)
        with self.assertRaises(ValueError):
            evaluate_events([sample(0)], [[0] * 6], mouse_threshold=0)


if __name__ == '__main__':
    unittest.main()
