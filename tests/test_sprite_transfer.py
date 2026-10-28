"""Check synthetic labels, split provenance, and the reused detector loss."""
import io
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
import zipfile

try:
    import numpy as np
    from PIL import Image
    import torch
    from torchvision.models.detection.ssd import SSD
    from torchvision.models.detection._utils import BoxCoder
    from vimm8bdm.sprite_synthesis import POSES, build_bank, build_dataset, composite, augment_scene
    from vimm8bdm.sprite_transfer import SyntheticImages, load_checkpoint
except ImportError:
    torch = None

from test_behavior_cloning import fixture
from vimm8bdm.visual_labels import source_identity


def packages(root):
    path = root / 'base.pk3'
    rgba = np.zeros((10, 8, 4), dtype=np.uint8)
    rgba[1:9, 1:7] = [180, 20, 90, 255]
    rgba[2:4, 2:3] = [255, 255, 255, 255]
    out = io.BytesIO()
    Image.fromarray(rgba).save(out, format='PNG')
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('PLAYPAL.pal', bytes(768))
        for pose in POSES + 'HU':
            for suffix in (pose+'1',pose+'5',pose+'6'+pose+'4',pose+'7'+pose+'3',pose+'8'+pose+'2'):
                archive.writestr('sprites/body/CUTM'+suffix+'.png', out.getvalue())
    return [path]


@unittest.skipIf(torch is None, 'Install the optional detector packages')
class SpriteTransferTests(unittest.TestCase):
    def test_training_bank_contains_normal_poses_and_correct_mirrored_views(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            assets = packages(root)
            original = assets[0].read_bytes()
            bank = root / 'bank'
            meta = build_bank(assets, bank)
            self.assertEqual(len(meta['templates']), 136)
            self.assertEqual(set(row['pose'] for row in meta['templates']), set(POSES))
            self.assertFalse({'H', 'U'} & set(meta['upright_frames']))
            with Image.open(bank/'Q-6.png') as a, Image.open(bank/'Q-4.png') as b:
                self.assertTrue(np.array_equal(np.asarray(a)[:,::-1],np.asarray(b)))
            self.assertEqual(assets[0].read_bytes(),original)

    def test_clipped_and_occluded_boxes_match_actual_visible_pixels(self):
        background = Image.new('RGB',(320,240),(35,55,75))
        sprite = Image.new('RGBA',(20,30),(200,10,80,255))
        rng = random.Random(7)
        for number in range(100):
            image, expected = composite(background,sprite,rng,profile="baseline" if number % 2 else "combat")
            changed = np.any(np.asarray(image) != np.asarray(background),axis=2)
            ys,xs = np.nonzero(changed)
            self.assertEqual(expected,[int(xs.min()),int(ys.min()),int(xs.max()-xs.min()+1),int(ys.max()-ys.min()+1)])
            self.assertLessEqual(expected[1]+expected[3],200)

    def test_scene_augmentation_keeps_geometry_and_is_reproducible(self):
        image = Image.new('RGB', (320, 240), (20, 60, 90))
        image.paste((230, 0, 80), (100, 50, 140, 110))
        first = augment_scene(image, random.Random(7))
        second = augment_scene(image, random.Random(7))
        self.assertTrue(np.array_equal(np.asarray(first), np.asarray(second)))
        pixels = np.asarray(first)
        changed = np.any(pixels != pixels[0, 0], axis=2)
        ys, xs = np.nonzero(changed)
        self.assertEqual([int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())], [100, 50, 139, 109])
        self.assertEqual(first.size, image.size)

    def test_unknown_generation_profile_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'profile is unsupported'):
            composite(Image.new('RGB',(320,240)), Image.new('RGBA',(20,30)), random.Random(0), profile='other')

    def test_dataset_reproduces_images_and_keeps_source_and_splits_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root,'background','train',1)
            source = root/'background-source'
            identity = source_identity(source)
            build_bank(packages(root),root/'bank')
            a = build_dataset(root/'bank',source,[0,1,2,3],root/'a',train_count=4,validation_count=2)
            b = build_dataset(root/'bank',source,[0,1,2,3],root/'b',train_count=4,validation_count=2)
            self.assertEqual(a['image_sha256'],b['image_sha256'])
            self.assertFalse(set(a['train_background_frames']) & set(a['validation_background_frames']))
            self.assertEqual(source_identity(source),identity)
            data = SyntheticImages(root/'a','train')
            image,target = data[0]
            self.assertEqual(tuple(image.shape),(3,240,320))
            self.assertEqual(target['labels'].tolist(),[1])
            path = root/'a'/data.rows[0]['file_name']
            path.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'dataset hash'):
                data[0]
            path = root/'a/train.coco.json'
            path.write_text('{}')
            with self.assertRaisesRegex(ValueError,'annotations changed'):
                SyntheticImages(root/'a','train')

    def test_reserved_real_validation_export_cannot_enter_synthetic_training(self):
        from test_visual_labels import labels
        from vimm8bdm.visual_labels import export
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root, 'example', 'train', 1)
            source = root / 'example-source'
            identity, _ = source_identity(source)
            rows = labels(identity)
            rows['split'] = 'validation'
            review = root / 'review.json'
            review.write_text(json.dumps(rows))
            export(source, review, root / 'validation')
            with self.assertRaisesRegex(ValueError, 'synthetic body dataset'):
                SyntheticImages(root / 'validation', 'train')

    def test_transfer_checkpoint_rejects_an_unknown_training_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'fake.pt'
            torch.save(dict(format='other'),path)
            with self.assertRaisesRegex(ValueError,'unsupported identity'):
                load_checkpoint(path)

    def test_reused_ssd_loss_gives_no_signal_for_an_entirely_empty_image(self):
        context = SimpleNamespace(box_coder=BoxCoder((10.,10.,5.,5.)),neg_to_pos_ratio=3.)
        losses = SSD.compute_loss(context,
            [dict(boxes=torch.zeros(0,4),labels=torch.zeros(0,dtype=torch.int64))],
            dict(bbox_regression=torch.zeros(1,2,4),cls_logits=torch.ones(1,2,91)),
            [torch.tensor([[0.,0.,10.,10.],[10.,10.,20.,20.]])],
            [torch.tensor([-1,-1])])
        self.assertEqual(float(sum(losses.values())),0)
