"""Check causal context, packed sequences, and frozen-encoder training."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

try:
    import torch
    from vimm8bdm.temporal_steering import (TemporalSteering, batch_windows, causal_windows,
                                            fit, load_checkpoint, predict, train)
    from vimm8bdm.steering import CLASSES, FORMAT as ENCODER_FORMAT, SteeringModel
    from vimm8bdm.behavior_cloning import IMAGE_SIZE, load_inputs, load_training_inputs
except ImportError:
    torch = None

from test_behavior_cloning import fixture


def samples(count, start=0):
    return [dict(segment=1, anchor_ns=(i + start) * 50_000_000,
                 action_end_ns=(i + start + 1) * 50_000_000) for i in range(count)]


@unittest.skipIf(torch is None, 'Install the optional training packages')
class TemporalSteeringTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(0)

    def test_history_stays_causal_bounded_and_within_source(self):
        groups = [samples(5), samples(3, start=5)]
        windows = causal_windows(groups, 3)
        self.assertEqual(windows[4], [2, 3, 4])
        self.assertEqual(windows[5], [5])
        self.assertTrue(all(max(w) == i and len(w) <= 3 for i, w in enumerate(windows)))
        with self.assertRaises(ValueError):
            causal_windows(groups, 0)

    def test_gap_and_segment_change_clear_history(self):
        rows = samples(3) + samples(2, start=4)
        rows[-1]['segment'] = 2
        windows = causal_windows([rows], 32)
        self.assertEqual(windows[3], [3])
        self.assertEqual(windows[4], [4])

    def test_packed_batch_matches_each_variable_length_sequence(self):
        model = TemporalSteering().eval()
        features = torch.randn(6, 128)
        windows = causal_windows([samples(6)], 4)
        together = predict(model, features, windows)
        separate = torch.cat([predict(model, features, [window]) for window in windows])
        self.assertTrue(torch.allclose(together, separate, atol=1e-6))

    def test_future_feature_changes_do_not_affect_earlier_outputs(self):
        model, features = TemporalSteering().eval(), torch.randn(8, 128)
        windows = causal_windows([samples(8)], 4)
        original = predict(model, features, windows)
        changed = features.clone()
        changed[4:] += 1000
        self.assertTrue(torch.allclose(original[:4], predict(model, changed, windows)[:4], atol=1e-6))

    def test_validation_labels_do_not_change_updates_or_normalization(self):
        features, val_features = torch.randn(12, 128), torch.randn(7, 128)
        labels = torch.arange(12) % 3
        windows, val_windows = causal_windows([samples(12)], 4), causal_windows([samples(7)], 4)
        args = SimpleNamespace(seed=0, epochs=1, batch_size=4)
        with contextlib.redirect_stdout(io.StringIO()):
            a = fit(features, labels, windows, val_features, torch.zeros(7, dtype=torch.long), val_windows, args)
            b = fit(features, labels, windows, val_features, torch.full((7,), 2), val_windows, args)
        self.assertTrue(all(torch.equal(v, b[0].state_dict()[k]) for k, v in a[0].state_dict().items()))
        self.assertTrue(torch.equal(a[1], b[1]))
        self.assertTrue(torch.equal(a[2], b[2]))

    def test_full_training_preserves_encoder_and_reloads_complete_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training = fixture(root, 'train', 'train', 100)
            validation = fixture(root, 'validation', 'validation', -200)
            review = root / 'review.json'
            review.write_text('{"operator_report":"Synthetic fixture","source_files_changed":false}')
            _, _, identities = load_training_inputs([training])
            _, _, val_id = load_inputs(validation, 'validation')
            encoder = SteeringModel()
            checkpoint = root / 'encoder.pt'
            torch.save({'format': ENCODER_FORMAT, 'classes': CLASSES,
                        'image_size': list(IMAGE_SIZE), 'features': 'motion',
                        'action_window_ms': 50, 'state_dict': encoder.state_dict()}, checkpoint)
            (root / 'report.local.json').write_text(json.dumps({'train_sources': identities,
                                                             'validation_source': val_id}))
            before = checkpoint.read_bytes()
            args = SimpleNamespace(train=[training], train_review=[review], validation=validation,
                                   validation_review=review, encoder=checkpoint, output=root / 'output',
                                   experimental=True, context_steps=3, threads=1, seed=0, epochs=1, batch_size=2)
            with contextlib.redirect_stdout(io.StringIO()):
                report = train(args)
            self.assertTrue(report['encoder_unchanged'])
            self.assertEqual(before, checkpoint.read_bytes())
            self.assertTrue(report['checkpoint_reload_matches'])
            self.assertEqual(report['checkpoint_reload_checked_samples'], val_id['samples'])
            self.assertFalse(report['input_sent'])
            restored, _, _, _, steps = load_checkpoint(args.output / 'model.pt')
            self.assertEqual(steps, 3)
            self.assertTrue(all(torch.equal(v, restored.state_dict()[k])
                                for k, v in encoder.layers[:-1].state_dict().items()))
            value = torch.load(args.output / 'model.pt', weights_only=True)
            value['feature_scale'][0] = 0
            torch.save(value, root / 'invalid.pt')
            with self.assertRaises(ValueError):
                load_checkpoint(root / 'invalid.pt')
            args.output = root / 'wrong-source'
            (root / 'report.local.json').write_text(json.dumps({'train_sources': [val_id],
                                                             'validation_source': val_id}))
            with self.assertRaisesRegex(ValueError, 'encoder source report'):
                train(args)


if __name__ == '__main__':
    unittest.main()
