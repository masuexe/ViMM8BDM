"""Check the model-to-game action contract without optional model packages."""
import unittest
from vimm8bdm.continuous import map_prediction


class ContinuousActionTests(unittest.TestCase):
    def test_six_choices_match_the_pinned_training_config(self):
        expected = [(set(), False), (set(), True), ({'d'}, False),
                    ({'a'}, False), ({'w'}, False), ({'s'}, False)]
        for choice, (keys, attack) in enumerate(expected):
            with self.subTest(choice=choice):
                action = map_prediction(choice, -.25, 2)
                self.assertEqual(action.keys, keys)
                self.assertEqual(action.attack, attack)
                self.assertEqual(action.mouse_dx, -.5)
                self.assertEqual(action.mouse_dy, 0)

    def test_invalid_outputs_do_not_become_idle_actions(self):
        for choice, yaw, scale in [(-1, 0, 2), (6, 0, 2), (1, float('nan'), 2),
                                    (1, 0, float('inf')), (1, 0, 0)]:
            with self.subTest(choice=choice, yaw=yaw, scale=scale):
                with self.assertRaises(ValueError):
                    map_prediction(choice, yaw, scale)
