"""Keep reserved recordings out of experimental real-image training."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

try:
    import torch
    from vimm8bdm.real_body_data import build, FORMAT, REVIEW_FORMAT
    from vimm8bdm.sprite_synthesis import build_bank, build_dataset
    from vimm8bdm.sprite_transfer import SyntheticImages, train
except ImportError:
    torch = None

from test_behavior_cloning import fixture
from test_sprite_transfer import packages
from test_visual_labels import labels
from vimm8bdm.recording import file_digest
from vimm8bdm.visual_labels import export, source_identity


@unittest.skipIf(torch is None, 'Install the optional detector packages')
class RealBodyDataTests(unittest.TestCase):
    def prepare(self, root, split='calibration'):
        fixture(root,'background','train',1)
        build_bank(packages(root),root/'bank')
        build_dataset(root/'bank',root/'background-source',[0,1,2,3],root/'synthetic',train_count=4,validation_count=2)
        fixture(root,'real','train',3)
        identity,_ = source_identity(root/'real-source')
        rows = labels(identity)
        rows['split'] = split
        review = root/'boxes.json'
        review.write_text(json.dumps(rows))
        export(root/'real-source',review,root/'real-images')
        permission = dict(format=REVIEW_FORMAT,reviewer='synthetic_test',source_session_id=identity,
                          image_gradient_training_allowed=True,action_training_allowed=False,
                          exports={'real-images':dict(dataset_sha256=file_digest(root/'real-images/dataset.json'),
                                                      annotations_sha256=file_digest(root/'real-images/annotations.coco.json'))})
        (root/'permission.json').write_text(json.dumps(permission))

    def generate(self, root):
        return build(root/'synthetic',[root/'real-images'],root/'bank',root/'permission.json',root/'mixed',
                     positive_repeats=2,negative_repeats=2)

    def test_mixture_retains_both_recording_sources_and_leaves_originals_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.prepare(root)
            before = source_identity(root/'real-source')
            metadata = self.generate(root)
            self.assertEqual(metadata['counts'],dict(train=8,validation=2))
            self.assertEqual(len(metadata['training_source_session_ids']),2)
            self.assertEqual(len(metadata['training_source_archives']),2)
            self.assertEqual(source_identity(root/'real-source'),before)
            self.assertFalse(metadata['action_training_eligible'])
            self.assertEqual({row['kind'] for row in metadata['origins']},
                             {'existing_synthetic','reviewed_real_positive','sprite_on_reviewed_empty_real_image'})
            images = SyntheticImages(root/'mixed','train')
            self.assertEqual(len(images),8)
            for i in range(len(images)):
                image,target = images[i]
                self.assertEqual(tuple(image.shape),(3,240,320))
                self.assertEqual(target['labels'].tolist(),[1])

    def test_reserved_validation_is_rejected_even_with_an_image_training_review(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.prepare(root,split='validation')
            with self.assertRaisesRegex(ValueError,'Reserved validation'):
                self.generate(root)
            self.assertFalse((root/'mixed').exists())

    def test_changed_real_annotations_fail_the_explicit_review_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.prepare(root)
            (root/'real-images/annotations.coco.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'explicit training review'):
                self.generate(root)
            self.assertFalse((root/'mixed').exists())

    def test_training_rejects_inherited_recordings_missing_from_new_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.prepare(root)
            self.generate(root)
            with patch('vimm8bdm.sprite_transfer.load_checkpoint',return_value=(torch.nn.Linear(1,1),
                        {'training_source_session_id':'missing_recording'})), self.assertRaisesRegex(ValueError,'inherited training'):
                train(root/'mixed',root/'initial.pt',root/'training',epochs=1,initial_transfer=True)
            self.assertFalse((root/'training').exists())

    def test_mixed_loader_requires_explicit_image_gradient_eligibility(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'dataset.json').write_text(json.dumps(dict(format=FORMAT,image_gradient_training_allowed=False)))
            with self.assertRaises(ValueError):
                SyntheticImages(root,'train')
