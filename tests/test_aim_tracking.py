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

    def test_gated_attack_pulses_only_inside_confirmed_target_window(self):
        tracker = StableAim(enable_attack=True)
        far_box = [candidate([450, 100, 40, 80])]
        for i in range(3):
            action, detail = tracker.step(far_box, 640, 480, i * 100_000_000)
            self.assertFalse(action.attack)
        self.assertFalse(detail['in_attack_window'])
        # Move into hit window: box=[290, 180, 80, 120] -> center=(330, 240), err=(10, 0), window=(28, 42)
        aligned_box = [candidate([290, 180, 80, 120])]
        for i in range(2):
            action, _ = tracker.step(aligned_box, 640, 480, (3 + i) * 100_000_000)
            self.assertFalse(action.attack)
        first, detail1 = tracker.step(aligned_box, 640, 480, 500_000_000)
        self.assertTrue(detail1['in_attack_window'])
        self.assertTrue(first.attack)
        self.assertEqual(first.keys, frozenset())
        # Next aligned frame releases attack (tap-fire) so charge-on-hold weapons fire
        second, detail2 = tracker.step(aligned_box, 640, 480, 600_000_000)
        self.assertTrue(detail2['in_attack_window'])
        self.assertFalse(second.attack)
        # Third aligned frame pulses attack again
        third, detail3 = tracker.step(aligned_box, 640, 480, 700_000_000)
        self.assertTrue(detail3['in_attack_window'])
        self.assertTrue(third.attack)
        # Single post-attack occlusion frame sends no input but retains confirmation
        occluded, detail_occ = tracker.step([], 640, 480, 840_000_000)
        self.assertEqual(occluded, Action())
        self.assertEqual(detail_occ['reason'], 'post_attack_occlusion')
        self.assertEqual(detail_occ['matches'], 5)
        # Immediate re-detection after 2-frame motion (85.6px shift, IoU=0 across 280ms) resumes tracking
        moved_box = [candidate([375, 190, 80, 120])]
        resumed, detail_res = tracker.step(moved_box, 640, 480, 980_000_000)
        self.assertEqual(detail_res['reason'], 'track_target')
        self.assertEqual(detail_res['matches'], 6)
        # Step back into the attack window and verify 2 consecutive missing frames reset confirmation
        tracker.step([candidate([335, 185, 80, 120])], 640, 480, 1_080_000_000)
        realigned, _ = tracker.step(aligned_box, 640, 480, 1_180_000_000)
        self.assertTrue(realigned.attack)
        self.assertEqual(tracker.step([], 640, 480, 1_280_000_000)[1]['reason'], 'post_attack_occlusion')
        lost, detail_lost = tracker.step([], 640, 480, 1_380_000_000)
        self.assertEqual(lost, Action())
        self.assertEqual(detail_lost['reason'], 'missing_target')
        self.assertEqual(detail_lost['matches'], 0)


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
