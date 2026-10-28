"""Check detector output filtering without downloads or game state."""
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

from vimm8bdm.pretrained_detection import load_model, person_boxes
from vimm8bdm.visual_metrics import box_iou


def array(values):
    return SimpleNamespace(tolist=lambda: values)


class PretrainedDetectionTests(unittest.TestCase):
    def test_only_finite_person_predictions_pass_the_fixed_score_filter(self):
        result = person_boxes(dict(boxes=array([[1, 2, 5, 8]] * 6 + [[0, 0, float('nan'), 8]]),
                                   labels=array([1, 2, 1, 1, 1, 1, 1]),
                                   scores=array([.8, .9, .49, float('nan'), float('inf'), -.1, .9])), 20, 20)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['box'], [1, 2, 4, 6])
        self.assertFalse(result[0]['enemy_identification_verified'])

    def test_boxes_use_image_coordinates_and_reject_zero_area(self):
        result = person_boxes(dict(boxes=array([[-2, -1, 25, 40], [4, 4, 3, 8], [21, 0, 30, 2]]),
                                   labels=array([1, 1, 1]),scores=array([.5, .9, .9])), 20, 30)
        self.assertEqual([row['box'] for row in result], [[0, 0, 20, 30]])
        with self.assertRaises(ValueError):
            person_boxes({}, 20, 30, threshold=1.1)

    def test_unknown_checkpoint_is_rejected_before_optional_library_import(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'unknown.pt'
            checkpoint.write_bytes(b'not the official weights')
            with self.assertRaisesRegex(ValueError, 'expected hash'):
                load_model(checkpoint)

    def test_iou_distinguishes_partial_overlap_from_point_contact(self):
        self.assertAlmostEqual(box_iou([0, 0, 4, 4], [2, 0, 4, 4]), 1/3)
        self.assertEqual(box_iou([0, 0, 2, 2], [2, 2, 2, 2]), 0)
        self.assertEqual(box_iou([2, 3, 5, 6], [2, 3, 5, 6]), 1)
        with self.assertRaises(ValueError):
            box_iou([0, 0, 0, 4], [0, 0, 2, 2])
