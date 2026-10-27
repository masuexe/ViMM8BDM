"""Check explicit event training, context isolation, and checkpoint reload."""
import contextlib
import copy
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

try:
    import torch
    from vimm8bdm.event_cloning import (EventModel, decode_raw, event_loss, feedback_labels,
                                       infer_raw, load_checkpoint, save_checkpoint, training_statistics, train, unpack)
except ImportError:
    torch = None

from test_behavior_cloning import fixture
from vimm8bdm.action_events import evaluate_explicit_events
from vimm8bdm.event_targets import EventTarget, target_label


@unittest.skipIf(torch is None, 'Install the optional training packages')
class EventCloningTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def empty_data(self):
        return {'counts': torch.zeros(2, 4, dtype=torch.long), 'times': torch.zeros(2, 4, 2),
                'time_mask': torch.zeros(2, 4, 2, dtype=torch.bool), 'mouse': torch.zeros(2, 2),
                'mouse_class': torch.ones(2, 2, dtype=torch.long)}

    def test_absent_times_and_magnitudes_have_no_loss_gradient(self):
        data = self.empty_data()
        raw = torch.zeros(2, 28, requires_grad=True)
        loss = event_loss(raw, data, training_statistics(data))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertEqual(raw.grad[:, 12:20].abs().sum().item(), 0)
        self.assertEqual(raw.grad[:, 26:28].abs().sum().item(), 0)
        data['counts'][0, 0] = 1
        data['time_mask'][0, 0, 0] = True
        data['times'][0, 0, 0] = .1
        raw = torch.zeros(2, 28, requires_grad=True)
        event_loss(raw, data, training_statistics(data)).backward()
        self.assertNotEqual(raw.grad[0, 12].item(), 0)
        self.assertEqual(raw.grad[0, 13].item(), 0)

    def test_decoder_keeps_two_ordered_changes_and_exact_zero_mouse_gate(self):
        raw = torch.zeros(1, 28)
        raw[0, 2] = 10  # Two W changes.
        raw[0, 21] = raw[0, 24] = 10  # Zero mouse classes.
        target = decode_raw(raw, torch.ones(2))
        self.assertEqual(target.key_offsets_ns[0], (25_000_000, 37_500_000))
        self.assertEqual((target.mouse_dx_raw, target.mouse_dy_raw), (0, 0))
        with self.assertRaises(ValueError):
            decode_raw(raw, torch.zeros(2))
        raw[0, 0] = float('nan')
        with self.assertRaises(ValueError):
            decode_raw(raw, torch.ones(2))

    def test_feedback_uses_predictions_and_resets_at_gaps_without_reading_human_keys(self):
        model = EventModel()
        with torch.no_grad():
            model.head.weight.zero_()
            model.head.bias.zero_()
            model.head.bias[1] = 10  # One W change every window.
            model.head.bias[21] = model.head.bias[24] = 10
        images = torch.zeros(3, 4, 84, 112, dtype=torch.uint8)
        samples = [{'segment': 1, 'anchor_ns': at, 'action_end_ns': at + 50,
                    'action': {'keys_at_start': [65]}} for at in [0, 50, 200]]
        labels, resets = feedback_labels(model, images, samples, torch.ones(2))
        self.assertEqual(resets, 2)
        self.assertEqual([p['events'][0]['down'] for p in labels], [True, False, True])
        changed = copy.deepcopy(samples)
        for row in changed:
            row['action'] = {'keys_at_start': [87], 'keys_at_end': [87], 'events': 'must not be read'}
        self.assertEqual(feedback_labels(model, images, changed, torch.ones(2))[0], labels)

    def test_explicit_event_metrics_do_not_threshold_away_a_late_press(self):
        label = target_label(EventTarget(((40_000_000,), (), (), ())), ())
        sample = {'action': label, 'segment': 1, 'anchor_ns': 0, 'action_end_ns': 50_000_000}
        report = evaluate_explicit_events([sample], [label])
        self.assertEqual(report['keys']['w']['press']['tp'], 1)
        self.assertEqual(report['keys']['w']['timing_matched_events'], 1)
        self.assertEqual(report['keys']['w']['timing_mae_ms_on_matched_events'], 0)

    def test_checkpoint_preserves_context_model_and_rejects_wrong_scales(self):
        model = EventModel()
        images = torch.zeros(2, 4, 84, 112, dtype=torch.uint8)
        states = torch.tensor([[0., 0, 0, 0], [1., 0, 1, 0]])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pt'
            save_checkpoint(path, model, torch.tensor([2., 3.]))
            restored, scale = load_checkpoint(path)
            self.assertTrue(torch.equal(infer_raw(model, images, states), infer_raw(restored, images, states)))
            self.assertEqual(scale.tolist(), [2, 3])
            payload = torch.load(path, weights_only=True)
            payload['mouse_scale'] = torch.zeros(2)
            invalid = Path(directory) / 'invalid.pt'
            torch.save(payload, invalid)
            with self.assertRaises(ValueError):
                load_checkpoint(invalid)
        with self.assertRaises(ValueError):
            model(images.float(), torch.full((2, 4), .5))

    def test_changed_validation_labels_do_not_change_updates_or_training_statistics(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training = fixture(root, 'train', 'train', 100)
            validations = [fixture(root, 'validation-a', 'validation', 200),
                           fixture(root, 'validation-b', 'validation', 1000000)]
            review = root / 'review.json'
            review.write_text(json.dumps({'operator_report': 'Synthetic unit fixture', 'source_files_changed': False}))
            reports = []
            for index, validation in enumerate(validations):
                args = SimpleNamespace(train=training, validation=validation, train_review=review,
                                       validation_review=review, output=root / f'run-{index}', experimental=True,
                                       epochs=1, seed=0, batch_size=2, threads=1, learning_rate=.001)
                with contextlib.redirect_stdout(io.StringIO()):
                    reports.append(train(args))
            a, b = [torch.load(root / f'run-{i}' / 'model.pt', weights_only=True) for i in range(2)]
            self.assertTrue(all(torch.equal(a['state_dict'][key], b['state_dict'][key]) for key in a['state_dict']))
            self.assertEqual(reports[0]['training_statistics'], reports[1]['training_statistics'])
            self.assertTrue(all(r['checkpoint_reload_matches'] for r in reports))
            self.assertTrue(all(not r['input_sent'] for r in reports))


if __name__ == '__main__':
    unittest.main()
