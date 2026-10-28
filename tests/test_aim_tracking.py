"""Check aim sign, target loss, duplicate boxes, and projectile appearance."""
import unittest
try:
    from PIL import Image, ImageDraw
except ImportError:
    Image = None
from vimm8bdm.aim_tracking import StableAim, filter_candidates
from vimm8bdm.control import Action


def candidate(box, score=.99):
    return dict(box=box, score=score)


class StableAimTests(unittest.TestCase):
    def test_three_matching_frames_then_bounded_mouse_without_keys_or_attack(self):
        tracker = StableAim()
        box = [candidate([450,100,40,80])]
        for i in range(2):
            self.assertEqual(tracker.step(box,640,480,i*100_000_000)[0],Action())
        action,_ = tracker.step(box,640,480,200_000_000)
        self.assertEqual(action.keys,frozenset())
        self.assertFalse(action.attack)
        self.assertEqual(action.mouse_dx,6)
        self.assertEqual(action.mouse_dy,-4)

    def test_loss_ambiguity_gap_and_target_switch_require_new_confirmation(self):
        for interruption, timestamp in [([],300_000_000),
                                        ([candidate([450,100,40,80])]*2,300_000_000),
                                        ([candidate([450,100,40,80])],500_000_000),
                                        ([candidate([10,350,40,80])],300_000_000)]:
            tracker=StableAim();box=[candidate([450,100,40,80])]
            for i in range(3):tracker.step(box,640,480,i*100_000_000)
            self.assertEqual(tracker.step(interruption,640,480,timestamp)[0],Action())
            self.assertEqual(tracker.step(box,640,480,timestamp+100_000_000)[0],Action())

    def test_center_and_uncalibrated_sizes_do_not_turn(self):
        tracker=StableAim();box=[candidate([300,200,40,80])]
        for i in range(3):self.assertEqual(tracker.step(box,640,480,i*100_000_000)[0],Action())
        self.assertEqual(tracker.step(box,800,600,300_000_000)[1]['reason'],'uncalibrated_image_size')


@unittest.skipIf(Image is None,'Install Pillow')
class AimAppearanceTests(unittest.TestCase):
    def test_magenta_body_passes_and_blue_projectile_is_rejected(self):
        image=Image.new('RGB',(640,480),'black');draw=ImageDraw.Draw(image)
        draw.rectangle((100,100,139,179),fill=(255,0,120))
        draw.rectangle((200,100,239,179),fill=(50,70,140))
        good,bad=filter_candidates(image,[candidate([100,100,40,80]),candidate([200,100,40,80])])
        self.assertEqual(len(good),1)
        self.assertEqual(bad[0]['reason'],'cutman_appearance')

    def test_duplicate_box_is_removed_and_large_foreground_box_is_rejected(self):
        image=Image.new('RGB',(640,480),(255,0,120))
        good,bad=filter_candidates(image,[candidate([100,100,40,80]),candidate([101,101,40,80],.9),candidate([0,0,600,450],.95)])
        self.assertEqual(len(good),1)
        self.assertEqual({r['reason'] for r in bad},{'duplicate_box','box_extent'})

    def test_small_magenta_strip_does_not_validate_a_large_projectile_box(self):
        image=Image.new('RGB',(640,480),'black');ImageDraw.Draw(image).rectangle((100,100,139,119),fill=(255,0,120))
        good,bad=filter_candidates(image,[candidate([100,100,40,100])])
        self.assertFalse(good)
        self.assertEqual(bad[0]['reason'],'cutman_appearance')

    def test_wide_magenta_foreground_is_not_an_upright_body(self):
        image=Image.new('RGB',(640,480),(255,0,120))
        good,bad=filter_candidates(image,[candidate([100,100,240,180])])
        self.assertFalse(good)
        self.assertEqual(bad[0]['reason'],'box_extent')
