"""Check causal loading, target units, and validation isolation."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

try:
    import torch
    from vimm8bdm.behavior_cloning import (IMAGE_SIZE, KEYS, NavigationModel, image_features,
                                         label_target, load_checkpoint, load_inputs, metrics,
                                         normalization, predict, train)
except ImportError:
    torch = None

from vimm8bdm.control import Frame
from vimm8bdm.dataset import build
from vimm8bdm.recording import RecordingWriter
from test_dataset import frame, start, MS


def fixture(root, name, split, mouse):
    source = root / (name + '-source')
    source.mkdir()
    writer = RecordingWriter(source)
    for i in range(7):
        row = frame(i, 50 + i * 50)
        image = Frame(2, 2, bytes((i * 20, sum(map(ord, name)) % 128, 255, 255)) * 4,
                      row['capture_started_ns'])
        writer.put('frame', (image, row))
    events = [start(),
              dict(type='key', time_ns=205 * MS, segment=1, sequence=2, vk=87, scan=17, extended=0, down=True),
              dict(type='key', time_ns=215 * MS, segment=1, sequence=3, vk=87, scan=17, extended=0, down=False),
              dict(type='motion', time_ns=225 * MS, segment=1, sequence=4, dx=mouse, dy=-4),
              dict(type='segment_end', time_ns=400 * MS, segment=1, sequence=5),
              dict(type='recording_end', time_ns=401 * MS, segment=1, sequence=6)]
    for event in events:
        writer.put('event', event)
    writer.close()
    if writer.error:
        raise RuntimeError(writer.error)
    manifest = dict(purpose='demonstration', task='navigation', map='MM1CUT', protocol='free',
                    class_selection='fixed', character_stable=True, character='grenademan',
                    training_ready=False, input_sent=False)
    (source / 'session.json').write_text(json.dumps(manifest))
    (source / 'summary.json').write_text(json.dumps(dict(complete=True, frames=7, events=6, segments=1, dropped_items=0)))
    output = root / name
    build(source, output, split=split)
    return output


@unittest.skipIf(torch is None, 'Install the optional training packages')
class BehaviorCloningTests(unittest.TestCase):
    def test_motion_features_keep_signed_chronological_differences(self):
        images = torch.tensor([0., .25, .5, 1.]).reshape(1, 4, 1, 1)
        original = images.clone()
        self.assertTrue(torch.equal(image_features(images, 'motion').flatten(),
                                    torch.tensor([1., .5, .5, 1.])))
        self.assertTrue(torch.equal(image_features(images, 'centered'), images * 2 - 1))
        self.assertTrue(torch.equal(image_features(images, 'raw'), original))
        self.assertTrue(torch.equal(images, original))
        repeated = images[:, -1:].expand(-1, 4, -1, -1)
        self.assertEqual(image_features(repeated, 'motion')[:, 1:].abs().sum().item(), 0)
        with self.assertRaises(ValueError):
            image_features(images, 'unknown')
        with self.assertRaises(ValueError):
            image_features(images[:, :3], 'motion')

    def test_versioned_checkpoint_restores_features_and_accepts_legacy_raw(self):
        torch.set_num_threads(1)
        inputs = torch.rand(2, 4, 84, 112)
        with tempfile.TemporaryDirectory() as directory:
            for mode, version in [('raw', 'v1'), ('centered', 'v2'), ('motion', 'v2')]:
                model = NavigationModel(mode)
                checkpoint = {'format': 'vimm8bdm-navigation-bc-' + version,
                              'image_size': list(IMAGE_SIZE), 'keys': list(KEYS),
                              'mouse_mean': torch.zeros(2), 'mouse_scale': torch.ones(2),
                              'state_dict': model.state_dict()}
                if version == 'v2':
                    checkpoint['features'] = mode
                path = Path(directory) / (mode + '.pt')
                torch.save(checkpoint, path)
                restored, mean, scale = load_checkpoint(path)
                self.assertEqual(restored.features, mode)
                self.assertTrue(torch.allclose(predict(model, inputs, mean, scale, 2),
                                               predict(restored, inputs, mean, scale, 2)))
                self.assertTrue(torch.allclose(
                    predict(restored, inputs, mean, scale, 2, repeat_latest=True),
                    predict(restored, inputs[:, -1:].expand(-1, 4, -1, -1), mean, scale, 2)))
                checkpoint['mouse_scale'] = torch.zeros(2)
                torch.save(checkpoint, path)
                with self.assertRaises(ValueError):
                    load_checkpoint(path)

    def test_loader_reconstructs_reviewed_exclusions_and_rejects_changed_ranges(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            full = fixture(root, 'train', 'train', 100)
            metadata = json.loads((full / 'dataset.json').read_text())
            exclusions = {'source_session_id': metadata['source_session_id'],
                          'reason': 'Synthetic reviewed overlay', 'frame_ranges': [[0, 0]]}
            clean = root / 'clean'
            result = build(root / 'train-source', clean, split='train', exclusions=exclusions)
            self.assertEqual(result['samples'], metadata['samples'] - 1)
            images, _, _ = load_inputs(clean, 'train')
            self.assertEqual(len(images), result['samples'])
            result['reviewed_frame_exclusions']['frame_ranges'] = [[0, 1]]
            (clean / 'dataset.json').write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError, 'causal source reconstruction'):
                load_inputs(clean, 'train')

    def test_targets_retain_fractional_tap_and_large_raw_mouse_count(self):
        with tempfile.TemporaryDirectory() as directory:
            data = fixture(Path(directory), 'train', 'train', 413)
            samples = [json.loads(line) for line in (data / 'samples.jsonl').read_text().splitlines()]
            target = label_target(samples[0])
            self.assertAlmostEqual(target[0], .2)
            self.assertEqual(target[4:], [413, -4])
            samples[0]['action']['keys_at_end'] = [32]
            with self.assertRaises(ValueError):
                label_target(samples[0])

    def test_loader_checks_split_and_reconstructs_source_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            data = fixture(Path(directory), 'train', 'train', 100)
            x, y, identity = load_inputs(data, 'train')
            self.assertEqual(tuple(x.shape[1:]), (4, 84, 112))
            self.assertEqual(x.dtype, torch.uint8)
            self.assertEqual(y.shape[1], 6)
            self.assertEqual(identity['split'], 'train')
            with self.assertRaises(ValueError):
                load_inputs(data, 'validation')
            path = data / 'samples.jsonl'
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[0]['action']['mouse_dx_raw'] += 1
            path.write_text('\n'.join(json.dumps(row) for row in rows) + '\n')
            with self.assertRaisesRegex(ValueError, 'causal source reconstruction'):
                load_inputs(data, 'train')

    def test_normalization_keeps_constant_axis_finite(self):
        targets = torch.tensor([[0, 0, 0, 0, 10, 0], [1, 0, 0, 0, 30, 0]], dtype=torch.float32)
        mean, scale = normalization(targets)
        self.assertEqual(mean.tolist(), [20, 0])
        self.assertEqual(scale.tolist(), [10, 1])

    def test_metrics_report_raw_units_and_absent_motion(self):
        targets = torch.tensor([[.25, 0, 0, 0, 100, 0], [.75, 0, 0, 0, -100, 0]])
        predictions = torch.tensor([[.25, 0, 0, 0, 80, 2], [.75, 0, 0, 0, -80, -2]])
        report = metrics(predictions, targets, torch.tensor([10, 1]))
        self.assertEqual(report['mouse_mae_raw_counts'], {'x': 20, 'y': 2})
        self.assertEqual(report['mouse_sign']['x']['accuracy'], 1)
        self.assertIsNone(report['mouse_sign']['y']['accuracy'])
        self.assertEqual(report['key_held_fraction_mae']['w'], 0)

    def test_validation_targets_do_not_change_training_updates_or_normalization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_data = fixture(root, 'train', 'train', 100)
            validation_a = fixture(root, 'validation-a', 'validation', 200)
            validation_b = fixture(root, 'validation-b', 'validation', 1000000)
            review = root / 'review.json'
            review.write_text(json.dumps({'operator_report': 'Synthetic unit fixture', 'source_files_changed': False}))
            reports = []
            for i, validation in enumerate((validation_a, validation_b)):
                args = SimpleNamespace(train=train_data, validation=validation, train_review=review,
                                       validation_review=review, output=root / f'run-{i}', experimental=True,
                                       seed=0, epochs=1, batch_size=2, threads=1, learning_rate=.001)
                with contextlib.redirect_stdout(io.StringIO()):
                    reports.append(train(args))
            a, b = [torch.load(root / f'run-{i}' / 'model.pt', weights_only=True) for i in range(2)]
            self.assertTrue(all(torch.equal(a['state_dict'][key], b['state_dict'][key]) for key in a['state_dict']))
            self.assertEqual(reports[0]['mouse_mean_train_only'], reports[1]['mouse_mean_train_only'])
            self.assertEqual(reports[0]['mouse_scale_train_only'], reports[1]['mouse_scale_train_only'])
            self.assertTrue(all(report['checkpoint_reload_matches'] for report in reports))
            self.assertTrue(all(not report['input_sent'] for report in reports))
            args.experimental = False
            with self.assertRaises(ValueError):
                train(args)


if __name__ == '__main__':
    unittest.main()
