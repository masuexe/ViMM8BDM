"""Check steering classes, class metrics, and training isolation."""
import contextlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

try:
    import torch
    from vimm8bdm.steering import (SteeringModel, candidate_conditions, class_statistics,
                                   direction_metrics, direction_targets, load_checkpoint,
                                   predict, train, transition_metrics, tolerant_transitions)
except ImportError:
    torch = None

from test_behavior_cloning import fixture


@unittest.skipIf(torch is None, 'Install the optional training packages')
class SteeringTests(unittest.TestCase):
    def test_targets_preserve_raw_direction_without_clipping(self):
        targets = torch.zeros(3, 6)
        targets[:, 4] = torch.tensor([-413, 0, 413])
        self.assertEqual(direction_targets(targets).tolist(), [0, 1, 2])
        self.assertEqual(targets[:, 4].tolist(), [-413, 0, 413])
        targets[0, 4] = float('nan')
        with self.assertRaises(ValueError):
            direction_targets(targets)

    def test_confusion_counts_all_classes_including_zero(self):
        labels = torch.tensor([0, 0, 1, 1, 2, 2])
        result = direction_metrics(labels, torch.tensor([0, 1, 1, 2, 2, 0]))
        self.assertEqual(result['confusion_rows_actual_columns_predicted'],
                         [[1, 1, 0], [0, 1, 1], [1, 0, 1]])
        self.assertEqual(result['macro_f1'], .5)
        self.assertEqual(result['zero_window_false_turn_rate'], .5)
        self.assertFalse(candidate_conditions(result, {'macro_f1': .3})['zero_false_turn_rate_at_most_0_1'])
        with self.assertRaises(ValueError):
            direction_metrics(labels.float(), labels)

    def test_weights_use_supplied_training_counts_and_handle_missing_class(self):
        labels = torch.tensor([0, 0, 1])
        counts, weights = class_statistics(labels, True)
        self.assertEqual(counts.tolist(), [2, 1, 0])
        self.assertEqual(weights[2], 0)
        self.assertTrue(torch.isfinite(weights).all())
        self.assertAlmostEqual(float((counts * weights).sum() / len(labels)), 1, places=6)
        self.assertTrue(torch.equal(class_statistics(labels, False)[1], torch.ones(3)))

    def test_training_and_checkpoint_do_not_depend_on_validation_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a = fixture(root, 'train-a', 'train', 100)
            b = fixture(root, 'train-b', 'train', -50)
            va = fixture(root, 'validation-a', 'validation', 200)
            vb = fixture(root, 'validation-b', 'validation', -200)
            review = root / 'review.json'
            review.write_text('{"operator_report":"Synthetic fixture","source_files_changed":false}')
            reports = []
            for i, validation in enumerate((va, vb)):
                args = SimpleNamespace(train=[a, b], train_review=[review, review],
                                       validation=validation, validation_review=review,
                                       output=root / str(i), experimental=True, weighted=True,
                                       epochs=1, seed=0, batch_size=2, threads=1)
                with contextlib.redirect_stdout(io.StringIO()):
                    reports.append(train(args))
            first = load_checkpoint(root / '0/model.pt')
            second = load_checkpoint(root / '1/model.pt')
            self.assertTrue(all(torch.equal(v, second.state_dict()[k])
                                for k, v in first.state_dict().items()))
            self.assertEqual(reports[0]['train_class_weights'], reports[1]['train_class_weights'])
            self.assertEqual(reports[0]['train_class_counts'], reports[1]['train_class_counts'])
            self.assertTrue(all(r['checkpoint_reload_matches'] and not r['input_sent'] for r in reports))
            checkpoint = torch.load(root / '0/model.pt', weights_only=True)
            checkpoint['action_window_ms'] = 200
            torch.save(checkpoint, root / 'incompatible.pt')
            with self.assertRaises(ValueError):
                load_checkpoint(root / 'incompatible.pt')

    def test_prediction_returns_only_three_logits(self):
        model = SteeringModel()
        result = predict(model, torch.zeros(2, 4, 84, 112, dtype=torch.uint8), 1)
        self.assertEqual(tuple(result.shape), (2, 3))
        self.assertTrue(torch.isfinite(result).all())

    def test_activity_transitions_count_starts_stops_and_reversals(self):
        samples = [dict(segment=1, anchor_ns=i * 50, action_end_ns=(i + 1) * 50) for i in range(6)]
        labels = torch.tensor([1, 0, 2, 1, 2, 1])
        result = transition_metrics(samples, labels, labels)
        self.assertEqual(result['boundary_windows_excluded'], 1)
        self.assertEqual(result['transitions']['start']['tp'], 2)
        self.assertEqual(result['transitions']['stop']['tp'], 2)
        self.assertEqual(result['transitions']['reversal']['tp'], 1)

    def test_source_gap_does_not_count_as_a_start_or_stop(self):
        samples = [dict(segment=1, anchor_ns=0, action_end_ns=50),
                   dict(segment=1, anchor_ns=100, action_end_ns=150)]
        labels = torch.tensor([1, 0])
        result = transition_metrics(samples, labels, labels)
        self.assertEqual(result['boundary_windows_excluded'], 2)
        self.assertEqual(result['transitions']['start']['true_windows'], 0)

    def test_tolerant_matching_allows_delay_but_requires_direction(self):
        samples = [dict(segment=1, anchor_ns=i * 50, action_end_ns=(i + 1) * 50) for i in range(5)]
        labels = torch.tensor([1, 0, 0, 1, 1])
        prediction = torch.tensor([1, 1, 0, 0, 1])
        self.assertEqual(tolerant_transitions(samples, labels, prediction, 0)['transitions']['start']['tp'], 0)
        delayed = tolerant_transitions(samples, labels, prediction, 50)['transitions']
        self.assertEqual(delayed['start']['tp'], 1)
        self.assertEqual(delayed['stop']['tp'], 1)
        with self.assertRaises(ValueError):
            tolerant_transitions(samples, labels, prediction, float('nan'))
        wrong_direction = torch.where(prediction == 0, 2, prediction)
        self.assertEqual(tolerant_transitions(samples, labels, wrong_direction, 50)['transitions']['start']['tp'], 0)

    def test_tolerant_matching_does_not_reuse_a_true_start(self):
        samples = [dict(segment=1, anchor_ns=i * 50, action_end_ns=(i + 1) * 50) for i in range(5)]
        labels, prediction = torch.tensor([1, 0, 0, 0, 1]), torch.tensor([1, 0, 1, 0, 1])
        result = tolerant_transitions(samples, labels, prediction, 100)['transitions']['start']
        self.assertEqual(result['tp'], 1)
        self.assertEqual(result['predicted_events'], 2)
        self.assertEqual(result['precision'], .5)


if __name__ == '__main__':
    unittest.main()
