"""Check empty actions, image colors, and rejection of old body results."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from vimm8bdm.control import Action, Frame
try:
    import torch
    from vimm8bdm.body_shadow import BodyShadowPolicy
except ImportError:
    torch = None


class Clock:
    now = 10_000_000
    def __call__(self):
        return self.now


@unittest.skipIf(torch is None, 'Install the optional detector packages')
class BodyShadowTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        checkpoint = Path(self.directory.name)/'model.pt'
        checkpoint.write_bytes(b'local test checkpoint')
        self.clock = Clock()
        self.inputs = []
        self.delay_ns = 0
        def model(inputs):
            self.inputs.append(inputs[0].clone())
            self.clock.now += self.delay_ns
            return [dict(boxes=torch.tensor([[0., 0., 2., 2.]]),
                         labels=torch.tensor([1]), scores=torch.tensor([.9]))]
        with patch('vimm8bdm.body_shadow.load_checkpoint', return_value=(model, {})):
            self.policy = BodyShadowPolicy(checkpoint, self.clock)
        self.inputs.clear()

    def frame(self, captured_ns=9_000_000):
        return Frame(8, 6, bytes((10, 20, 30, 255))*48, captured_ns)

    def test_current_frame_returns_empty_action_and_preserves_rgb_colors(self):
        self.assertEqual(self.policy.predict(self.frame()), Action())
        result = self.policy.last_prediction
        self.assertEqual(len(result['candidates']), 1)
        self.assertFalse(result['stale'])
        self.assertFalse(result['input_sent'])
        self.assertTrue(torch.allclose(self.inputs[0][:, 0, 0], torch.tensor([30, 20, 10])/255))

    def test_old_future_and_repeated_frames_do_not_run_the_model(self):
        for timestamp in [-100_000_000, 11_000_000]:
            self.policy.reset()
            self.assertEqual(self.policy.predict(self.frame(timestamp)), Action())
            self.assertFalse(self.policy.last_prediction['inference_performed'])
        self.policy.reset()
        self.policy.predict(self.frame())
        self.inputs.clear()
        self.policy.predict(self.frame())
        self.assertEqual(self.inputs, [])
        self.assertEqual(self.policy.last_prediction['reset_reason'], 'capture_timestamp_invalid')

    def test_rejected_future_timestamp_does_not_replace_last_valid_timestamp(self):
        self.policy.predict(self.frame())
        self.policy.predict(self.frame(11_000_000))
        self.assertFalse(self.policy.last_prediction['inference_performed'])
        self.policy.predict(self.frame(9_500_000))
        self.assertTrue(self.policy.last_prediction['inference_performed'])

    def test_late_result_discards_boxes_and_reset_removes_previous_output(self):
        self.delay_ns = 160_000_000
        self.assertEqual(self.policy.predict(self.frame()), Action())
        result = self.policy.last_prediction
        self.assertTrue(result['stale'])
        self.assertEqual(result['candidates'], [])
        self.assertEqual(result['discarded_candidates'], 1)
        self.assertEqual(result['reset_reason'], 'result_age_invalid')
        self.policy.reset()
        self.assertIsNone(self.policy.last_prediction)

    def test_cli_rejects_shadow_mode_before_desktop_setup(self):
        from types import SimpleNamespace
        from vimm8bdm.__main__ import run
        for observe in [False, True]:
            args = SimpleNamespace(live=False, policy='body_shadow', observe_only=observe, checkpoint='unused')
            with self.assertRaisesRegex(ValueError, 'shadow policy requires'):
                run(args)

        args = SimpleNamespace(live=True, window_id=1, capture='window',
                               policy='body_shadow', observe_only=False, checkpoint='unused')
        with self.assertRaisesRegex(ValueError, 'shadow policy requires'):
            run(args)

    def test_runtime_summary_reports_body_inference_only_when_it_ran(self):
        import contextlib
        import io
        import json
        from vimm8bdm.__main__ import summarize
        for performed in [True, False]:
            row = dict(applied=True, observe_only=True, capture_ms=1, policy_ms=2,
                       input_ms=0, loop_ms=3, model_output=dict(inference_performed=performed))
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                summarize([row], 'desktop', 'body_shadow', 'duration_limit')
            result = json.loads(output.getvalue())
            self.assertEqual(result['neural_inference_measured'], performed)
            self.assertEqual(result['dispatched_policy_actions'], 0)
