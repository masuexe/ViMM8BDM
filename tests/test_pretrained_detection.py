"""Check detector output filtering without downloads or game state."""
from pathlib import Path
import tempfile
import copy
import unittest
from types import SimpleNamespace

from vimm8bdm.pretrained_detection import load_model, person_boxes, check_transfer_evaluation, complete_synthetic_provenance
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

    def test_reserved_evaluation_checks_both_recording_identity_and_archive(self):
        metadata = dict(format='vimm8bdm-visible-coco-v1',split='validation',reserved_for_validation=True,
                        source_session_id='validation_recording',source_hashes={'frames.vrec':'validation_archive'})
        checkpoint = dict(training_source_session_id='real_training',
                          training_source_session_ids=['real_training','synthetic_background'],
                          training_source_archives={'real_training':'real_archive','synthetic_background':'background_archive'})
        check_transfer_evaluation(metadata,checkpoint,validation=True)
        for source,archive in [('real_training','validation_archive'),
                               ('validation_recording','background_archive')]:
            changed = copy.deepcopy(metadata)
            changed.update(source_session_id=source,source_hashes={'frames.vrec':archive})
            with self.assertRaisesRegex(ValueError,'supplied training data'):
                check_transfer_evaluation(changed,checkpoint,validation=True)

    def test_reserved_evaluation_rejects_missing_history_and_implicit_split_selection(self):
        metadata = dict(format='vimm8bdm-visible-coco-v1',split='validation',reserved_for_validation=True,
                        source_session_id='validation_recording',source_hashes={'frames.vrec':'validation_archive'})
        checkpoint = dict(training_source_session_id='real_training')
        with self.assertRaisesRegex(ValueError,'complete project recording provenance'):
            check_transfer_evaluation(metadata,checkpoint,validation=True)
        with self.assertRaisesRegex(ValueError,'split explicitly'):
            check_transfer_evaluation(metadata,checkpoint)

    def test_legacy_provenance_requires_the_exact_original_synthetic_metadata(self):
        checkpoint = dict(synthetic_dataset_sha256='recorded_digest',training_source_session_id='background')
        dataset = dict(format='vimm8bdm-synthetic-body-coco-v1',source_session_id='background',
                       source_hashes={'frames.vrec':'background_archive'})
        full = complete_synthetic_provenance(checkpoint,dataset,'recorded_digest')
        self.assertEqual(full['training_source_archives'],{'background':'background_archive'})
        self.assertNotIn('training_source_archives',checkpoint)
        with self.assertRaisesRegex(ValueError,'does not match'):
            complete_synthetic_provenance(checkpoint,dataset,'changed_digest')
        with self.assertRaisesRegex(ValueError,'does not match'):
            complete_synthetic_provenance(checkpoint,{**dataset,'source_session_id':'other'},'recorded_digest')

    def test_iou_distinguishes_partial_overlap_from_point_contact(self):
        self.assertAlmostEqual(box_iou([0, 0, 4, 4], [2, 0, 4, 4]), 1/3)
        self.assertEqual(box_iou([0, 0, 2, 2], [2, 2, 2, 2]), 0)
        self.assertEqual(box_iou([2, 3, 5, 6], [2, 3, 5, 6]), 1)
        with self.assertRaises(ValueError):
            box_iou([0, 0, 0, 4], [0, 0, 2, 2])
