"""Check crop boundaries and original-image coordinates without model weights."""
import unittest
from types import SimpleNamespace
from vimm8bdm.local_zoom import crop_window, restore_boxes


def prediction(box, score=.3):
    def array(rows):
        return SimpleNamespace(tolist=lambda: rows)
    return dict(boxes=array([box]), labels=array([1]), scores=array([score]))


class LocalZoomTests(unittest.TestCase):
    def test_crop_reaches_each_edge_without_leaving_the_image(self):
        for box, expected in [([0, 0, 10, 10], (0, 0, 320, 320)),
                              ([630, 470, 640, 480], (320, 160, 640, 480)),
                              ([310, 230, 330, 250], (160, 80, 480, 400))]:
            self.assertEqual(crop_window(prediction(box), 640, 480), expected)

    def test_accepted_and_below_trigger_scores_do_not_request_another_call(self):
        for score in [.5, .9, .199, float('nan')]:
            self.assertIsNone(crop_window(prediction([20, 30, 40, 60], score), 640, 480))
        self.assertIsNotNone(crop_window(prediction([20, 30, 40, 60], .2), 640, 480))

    def test_small_image_uses_a_square_within_its_dimensions(self):
        self.assertEqual(crop_window(prediction([70, 30, 80, 40]), 80, 40), (40, 0, 80, 40))
        with self.assertRaises(ValueError):
            crop_window(prediction([0, 0, 1, 1]), 0, 40)

    def test_remap_preserves_size_score_and_unverified_identity(self):
        source = dict(box=[5, 7, 20, 30], score=.8, enemy_identification_verified=False)
        mapped = restore_boxes([source], (320, 160, 640, 480))
        self.assertEqual(mapped[0]['box'], [325, 167, 20, 30])
        self.assertEqual(source['box'], [5, 7, 20, 30])
        self.assertEqual(mapped[0]['score'], .8)
        self.assertFalse(mapped[0]['enemy_identification_verified'])
