"""Check version-agnostic PK3 extraction, foreground occlusion synthesis, and Colab bundles."""
import io
import json
from pathlib import Path
import random
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import zipfile

try:
    import numpy as np
    from PIL import Image
    import torch
    from torchvision.models.detection import ssdlite320_mobilenet_v3_large
    from vimm8bdm.colab_bundle import BUNDLE_FORMAT, export_bundle, train_from_bundle
    from vimm8bdm.pretrained_detection import CHECKPOINT_SHA256
    from vimm8bdm.sprite_synthesis import (
        POSES,
        build_bank,
        build_dataset,
        build_occluder_bank,
        composite,
    )
except ImportError:
    torch = None

from test_behavior_cloning import fixture


def _make_png(width, height, rgba_color):
    arr = np.zeros((height, width, 4), dtype=np.uint8)
    arr[1:height - 1, 1:width - 1] = rgba_color
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format='PNG')
    return buf.getvalue()


def _build_multi_version_pk3_stack(root):
    base_pk3 = root / 'mm8bdm-v6b.pk3'
    addon_pk3 = root / 'gondola-addon.pk3'
    body_png = _make_png(10, 14, [180, 20, 90, 255])
    occ_png = _make_png(8, 8, [250, 240, 120, 255])

    base_poses = 'ABCDEFGIJ'
    addon_poses = 'KLMQRSTVW'
    with zipfile.ZipFile(base_pk3, 'w') as zf:
        zf.writestr('PLAYPAL.pal', bytes(768))
        for pose in base_poses:
            prefix = 'CUT3' if pose in 'ABCDEFGH' else 'CUTM'
            for suffix in (pose + '1', pose + '5', pose + '6' + pose + '4', pose + '7' + pose + '3', pose + '8' + pose + '2'):
                zf.writestr(f'sprites/skins/MM1/cutman/{prefix}{suffix}.PNG', body_png)
        # Place SKININFO.txt after sprites/ to verify two-pass archive prefix discovery
        zf.writestr('SKININFO.txt', '{\nname = "CutMan"\nsprite = CUT3\nclass = Megaman\n}\n')
        zf.writestr('sprites/effects/explosions/MXP1A0.png', occ_png)
        zf.writestr('sprites/weapons/buster/BUSTA0.png', occ_png)

    with zipfile.ZipFile(addon_pk3, 'w') as zf:
        zf.writestr('actors/Megaman/Cutman.txt', 'actor Cutman : ClassBase {}')
        for pose in addon_poses:
            for suffix in (pose + '1', pose + '5', pose + '6' + pose + '4', pose + '7' + pose + '3', pose + '8' + pose + '2'):
                zf.writestr(f'sprites/MM1/Cutman/CUTM{suffix}.png', body_png)
        zf.writestr('sprites/weapons/gondola/W_0BB0.png', occ_png)
        zf.writestr('sprites/effects/NEDRU0.png', occ_png)
    return [base_pk3, addon_pk3]


@unittest.skipIf(torch is None, 'Install the optional detector packages')
class ColabBundleTests(unittest.TestCase):
    def test_v6b_dollar_skin_prefix_and_partial_pose_extraction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base_pk3, addon_pk3 = _build_multi_version_pk3_stack(root)
            # Base v6b alone has only base poses; strict mode rejects it, partial mode accepts it
            with self.assertRaisesRegex(ValueError, 'all reviewed upright views'):
                build_bank([base_pk3], root / 'strict-base')
            partial = build_bank([base_pk3], root / 'partial-base', allow_partial_poses=True)
            self.assertEqual(len(partial['upright_frames']), 8)
            self.assertEqual(len(partial['templates']), 64)
            # Combined v6b + addon stack supplies all 17 upright poses (136 views)
            full = build_bank([base_pk3, addon_pk3], root / 'full-bank')
            self.assertEqual(full['upright_frames'], list(POSES))
            self.assertEqual(len(full['templates']), 136)

    def test_occluder_bank_and_occlusion_composite_preserve_visible_body_bounds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stack = _build_multi_version_pk3_stack(root)
            occ_meta = build_occluder_bank(stack, root / 'occluders')
            self.assertEqual(occ_meta['prefixes'], ['BUST', 'MXP1', 'NEDR', 'W_0B'])
            self.assertEqual(len(occ_meta['templates']), 4)

            background = Image.new('RGB', (320, 240), (30, 45, 60))
            sprite = Image.new('RGBA', (24, 36), (220, 20, 100, 255))
            occluder = Image.new('RGBA', (14, 14), (255, 255, 0, 255))
            rng = random.Random(11)
            for _ in range(40):
                image, box = composite(background, sprite, rng, profile='occlusion', occluders=[occluder])
                x, y, w, h = box
                self.assertGreaterEqual(x, 0)
                self.assertGreaterEqual(y, 0)
                self.assertGreater(w, 0)
                self.assertGreater(h, 0)
                self.assertLessEqual(x + w, 320)
                self.assertLessEqual(y + h, 200)
                arr = np.asarray(image)
                crop = arr[y:y + h, x:x + w]
                body_mask = (crop[:, :, 0] > 100) & (crop[:, :, 1] < 80)
                self.assertTrue(np.any(body_mask))

    def test_export_and_train_from_portable_tar_gz_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stack = _build_multi_version_pk3_stack(root)
            fixture(root, 'bg', 'train', 1)
            recording = root / 'bg-source'
            initial = root / 'initial.pth'
            initial.write_bytes(b'pretrained fixture bytes')
            archive = root / 'bundle.tar.gz'

            def fake_load_model(_path):
                return ssdlite320_mobilenet_v3_large(weights=None, weights_backbone=None)

            with patch('vimm8bdm.colab_bundle.load_model', side_effect=fake_load_model), \
                 patch('vimm8bdm.sprite_transfer.load_model', side_effect=fake_load_model), \
                 patch('vimm8bdm.sprite_transfer.CHECKPOINT_SHA256', 'fixture-sha'):
                manifest = export_bundle(
                    stack,
                    initial,
                    archive,
                    recording=recording,
                    reviewed_frames=[0, 1, 2, 3],
                    train_count=4,
                    validation_count=2,
                )
                self.assertEqual(manifest['format'], BUNDLE_FORMAT)
                self.assertEqual(manifest['upright_views'], 136)
                self.assertEqual(manifest['occluder_count'], 4)
                self.assertTrue(archive.is_file())

                out_dir = root / 'trained'
                result = train_from_bundle(
                    archive,
                    out_dir,
                    epochs=1,
                    batch_size=2,
                    learning_rate=0.001,
                    device='cpu',
                )
                self.assertTrue((out_dir / 'model.pt').is_file())
                prov = json.loads((out_dir / 'colab_bundle_provenance.local.json').read_text())
                self.assertEqual(prov['checkpoint_sha256'], result['checkpoint_sha256'])
                self.assertEqual(prov['occluder_prefixes'], ['BUST', 'MXP1', 'NEDR', 'W_0B'])

                # Exporting from an existing dataset augments train split with foreground occluders
                existing_ds = root / 'existing-ds'
                build_bank(stack, root / 'ds-bank')
                build_dataset(root / 'ds-bank', recording, [0, 1, 2, 3], existing_ds,
                              train_count=4, validation_count=2)
                aug_bundle = root / 'aug-bundle'
                aug_manifest = export_bundle(
                    stack,
                    initial,
                    aug_bundle,
                    dataset=existing_ds,
                    augment_occluders=2,
                )
                self.assertEqual(aug_manifest['train_images'], 6)
                self.assertEqual(aug_manifest['validation_images'], 2)

            # Initial transfer checkpoint provenance mismatch is caught during export_bundle
            with patch('vimm8bdm.colab_bundle.load_checkpoint',
                       return_value=(None, {'training_source_session_id': 'other-session',
                                            'training_source_session_ids': ['other-session']})):
                with self.assertRaisesRegex(ValueError, 'inherited training recording'):
                    export_bundle(
                        stack,
                        initial,
                        root / 'bad-transfer-bundle',
                        dataset=existing_ds,
                        initial_transfer=True,
                    )

            # Unsafe tar member is rejected before extraction
            bad_tar = root / 'unsafe.tar.gz'
            with tarfile.open(bad_tar, 'w:gz') as tar:
                info = tarfile.TarInfo('../escape.txt')
                payload = b'bad'
                info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))
            with self.assertRaisesRegex(ValueError, 'Unsafe relative path'):
                train_from_bundle(bad_tar, root / 'bad-out', epochs=1, batch_size=2)
