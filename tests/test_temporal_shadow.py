"""Check runtime image equivalence, causal anchors, and empty actions."""
from pathlib import Path
import tempfile
import unittest

try:
    import torch
    from vimm8bdm.temporal_shadow import TemporalShadowPolicy
    from vimm8bdm.temporal_steering import FORMAT, TemporalSteering, encode, predict
    from vimm8bdm.behavior_cloning import IMAGE_SIZE, load_inputs
    from vimm8bdm.steering import CLASSES, SteeringModel
except ImportError:
    torch = None

from vimm8bdm.control import Action, Frame
from test_behavior_cloning import fixture


class Clock:
    now = 0

    def __call__(self):
        return self.now


@unittest.skipIf(torch is None, 'Install the optional training packages')
class TemporalShadowTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        torch.set_num_threads(1)
        torch.manual_seed(0)
        path = self.root / 'model.pt'
        torch.save({'format': FORMAT, 'context_steps': 4, 'classes': CLASSES,
                    'image_size': list(IMAGE_SIZE), 'features': 'motion', 'action_window_ms': 50,
                    'history_spacing_ms': 50, 'feature_mean': torch.zeros(128),
                    'feature_scale': torch.ones(128),
                    'encoder_state_dict': SteeringModel().layers[:-1].state_dict(),
                    'temporal_state_dict': TemporalSteering().state_dict()}, path)
        self.clock = Clock()
        self.policy = TemporalShadowPolicy(path, self.clock)

    def tick(self, milliseconds, width=8, height=6):
        self.clock.now = int(milliseconds * 1_000_000)
        frame = Frame(width, height, bytes((35, 60, 120, 255)) * width * height,
                      self.clock.now - 1_000_000)
        action = self.policy.predict(frame)
        self.assertEqual(action, Action())
        return self.policy.last_prediction

    def test_recorded_and_runtime_image_paths_produce_equal_probabilities(self):
        dataset = fixture(self.root, 'demo', 'train', 100)
        images, _, _ = load_inputs(dataset, 'train')
        color = sum(map(ord, 'demo')) % 128
        for i in range(4):
            self.clock.now = (50 + i * 50) * 1_000_000
            frame = Frame(2, 2, bytes((i * 20, color, 255, 255)) * 4,
                          self.clock.now - 1_000_000)
            self.assertEqual(self.policy.predict(frame), Action())
        feature = encode(self.policy.encoder, images[:1])
        expected = predict(self.policy.model, feature, [[0]], 1).softmax(1)[0]
        actual = torch.tensor(self.policy.last_prediction['probabilities'])
        self.assertTrue(torch.allclose(expected, actual, atol=1e-6))
        self.assertEqual(self.policy.last_prediction['context_steps'], 1)

    def test_capture_gap_resize_and_manual_reset_discard_context(self):
        for time_ms in range(50, 551, 50):
            result = self.tick(time_ms)
        self.assertEqual(result['context_steps'], 4)
        result = self.tick(1000)
        self.assertFalse(result['inference_performed'])
        self.assertEqual(result['reset_reason'], 'capture_interval_invalid')
        for time_ms in (1050, 1100, 1150):
            result = self.tick(time_ms)
        self.assertEqual(result['context_steps'], 1)
        result = self.tick(1200, width=10)
        self.assertFalse(result['inference_performed'])
        self.assertEqual(result['reset_reason'], 'capture_size_changed')
        self.policy.reset()
        result = self.tick(1250, width=10)
        self.assertEqual(result['context_steps'], 0)

    def test_skipped_anchor_reconstructs_causal_features_without_old_actions(self):
        for time_ms in (50, 100, 150, 200, 250):
            result = self.tick(time_ms)
        self.assertEqual(result['context_steps'], 2)
        result = self.tick(350)
        self.assertIsNone(result['reset_reason'])
        self.assertEqual(result['context_steps'], 4)
        self.assertEqual(result['backfilled_feature_steps'], 1)
        self.assertEqual(result['feature_anchor_ns'], [200_000_000, 250_000_000, 300_000_000, 350_000_000])
        for history in result['new_feature_histories']:
            self.assertTrue(all(t <= history['anchor_ns'] for t in history['image_complete_ns']))

    def test_frame_complete_after_grid_anchor_does_not_enter_that_prediction(self):
        for time_ms in (50, 101, 152, 203, 254):
            result = self.tick(time_ms)
        self.assertTrue(result['inference_performed'])
        self.assertEqual(result['anchor_ns'], 250_000_000)
        self.assertTrue(all(t <= result['anchor_ns'] for t in result['history_image_complete_ns']))
        self.assertNotIn(254_000_000, result['history_image_complete_ns'])


if __name__ == '__main__':
    unittest.main()
