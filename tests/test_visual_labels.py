"""Check image provenance and prevent unreviewed action-training exports."""
import copy
import json
from pathlib import Path
import struct
import tempfile
import unittest

from test_behavior_cloning import fixture
from vimm8bdm.visual_labels import FORMAT, export, source_identity, validate_labels


def labels(session):
    return dict(format=FORMAT, source_session_id=session, split='calibration',
                annotator='synthetic_test', human_box_reviewed=False, frames=[
                    dict(frame=0, status='usable', visible_presence='present', view_state='ordinary_play',
                         boxes=[dict(xywh=[0, 0, 2, 2], occluded=False, truncated=False,
                                     **{'class': 'cutman_body'})]),
                    dict(frame=1, status='usable', visible_presence='absent', boxes=[], view_state='ordinary_play'),
                    dict(frame=2, status='exclude', visible_presence='uncertain', boxes=[], reason='Unclear body')])


class VisualLabelsTests(unittest.TestCase):
    def test_export_retains_original_png_and_source_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root, 'example', 'train', 1)
            source = root / 'example-source'
            identity, hashes = source_identity(source)
            review = root / 'review.json'
            review.write_text(json.dumps(labels(identity)))
            result = export(source, review, root / 'export')
            self.assertEqual(source_identity(source), (identity, hashes))
            self.assertEqual((result['images'], result['boxes'], result['negative_images'],
                              result['excluded_reviewed_frames']), (2, 1, 1, 1))
            self.assertFalse(result['training_ready'])
            self.assertFalse(result['action_training_eligible'])
            with (source / 'frames.vrec').open('rb') as archive:
                first = json.loads((source / 'frames.jsonl').read_text().splitlines()[0])
                archive.seek(first['offset'])
                raw = archive.read(struct.unpack('<I', archive.read(4))[0])
            self.assertEqual(raw, (root / 'export/images/000000.png').read_bytes())
            coco = json.loads((root / 'export/annotations.coco.json').read_text())
            self.assertEqual([row['id'] for row in coco['images']], [0, 1])
            self.assertEqual(coco['annotations'][0]['bbox'], [0, 0, 2, 2])
            with self.assertRaises(FileExistsError):
                export(source, review, root / 'export')

    def test_wrong_source_is_rejected_before_output_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root, 'example', 'train', 1)
            review = root / 'review.json'
            review.write_text(json.dumps(labels('wrong')))
            with self.assertRaisesRegex(ValueError, 'different recording'):
                export(root / 'example-source', review, root / 'export')
            self.assertFalse((root / 'export').exists())

    def test_outside_boxes_and_boolean_coordinates_are_rejected(self):
        for rect in ([0, 0, 3, 2], [-1, 0, 1, 1], [0, 0, 0, 1], [True, 0, 1, 1]):
            sample = labels('test')
            sample['frames'][0]['boxes'][0]['xywh'] = rect
            with self.subTest(rect=rect), self.assertRaises(ValueError):
                validate_labels(sample, [{'width': 2, 'height': 2}] * 3)

    def test_duplicate_frames_and_unreviewed_presence_are_rejected(self):
        original = labels('test')
        changes = [lambda row: row.update(frame=0),
                   lambda row: row.update(visible_presence='uncertain'),
                   lambda row: row.update(visible_presence='present')]
        for change in changes:
            sample = copy.deepcopy(original)
            change(sample['frames'][1])
            with self.assertRaises(ValueError):
                validate_labels(sample, [{'width': 2, 'height': 2}] * 3)

    def test_exclusion_requires_reason_and_cannot_supply_a_box(self):
        for patch in ({'reason': ''}, {'boxes': labels('test')['frames'][0]['boxes']}):
            sample = labels('test')
            sample['frames'][2].update(patch)
            with self.assertRaises(ValueError):
                validate_labels(sample, [{'width': 2, 'height': 2}] * 3)
