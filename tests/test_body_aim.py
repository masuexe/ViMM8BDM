"""Check image-to-mouse proposals and reset on rejected visual evidence."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from vimm8bdm.control import Frame,Action
try:
    import torch
    from PIL import Image,ImageDraw
    from vimm8bdm.body_aim import BodyAimPolicy
except ImportError:
    torch=None


@unittest.skipIf(torch is None,'Install the optional detector packages')
class BodyAimTests(unittest.TestCase):
    def test_detector_filter_tracker_returns_mouse_only_and_stops_on_loss(self):
        clock=SimpleNamespace(now=1_000_000_000)
        def model(inputs):
            return [dict(boxes=torch.tensor([[450.,100.,490.,180.]]),
                         labels=torch.tensor([1]),scores=torch.tensor([.99]))]
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'model.pt';path.write_bytes(b'fixture')
            with patch('vimm8bdm.body_shadow.load_checkpoint',return_value=(model,{})):
                policy=BodyAimPolicy(path,lambda:clock.now)
        image=Image.new('RGB',(640,480),'black')
        ImageDraw.Draw(image).rectangle((450,100,489,179),fill=(255,0,120))
        pixels=image.convert('RGBA').tobytes('raw','BGRA')
        for i in range(3):
            clock.now+=100_000_000
            action=policy.predict(Frame(640,480,pixels,clock.now-1_000_000))
            self.assertFalse(action.keys)
            self.assertFalse(action.attack)
            if i<2:self.assertEqual(action,Action())
        self.assertEqual((action.mouse_dx,action.mouse_dy),(6,-4))
        clock.now+=100_000_000
        blank=bytes((0,0,0,255))*640*480
        self.assertEqual(policy.predict(Frame(640,480,blank,clock.now-1_000_000)),Action())
        self.assertEqual(policy.last_prediction['aim']['matches'],0)
        policy.tracker.matches=3
        self.assertEqual(policy.predict(Frame(640,480,pixels,clock.now-200_000_000)),Action())
        self.assertEqual(policy.tracker.matches,0)

    def test_cli_requires_live_mode_and_checkpoint_before_desktop_setup(self):
        from vimm8bdm.__main__ import run, summarize
        import io, json, contextlib
        args=SimpleNamespace(live=False,policy='body_aim',checkpoint='fixture')
        with self.assertRaisesRegex(ValueError,'Body aim requires'):
            run(args)
        for bad_args in (
            SimpleNamespace(live=True,window_id=1,capture='window',policy='body_shadow',
                            observe_only=True,checkpoint='fixture',enable_attack=True),
            SimpleNamespace(live=True,window_id=1,capture='window',policy='body_aim',
                            observe_only=True,checkpoint='fixture',enable_attack=True),
        ):
            with self.assertRaisesRegex(ValueError,'--enable-attack requires'):
                run(bad_args)
        for bad_search_args in (
            SimpleNamespace(live=True,window_id=1,capture='window',policy='body_shadow',
                            observe_only=True,checkpoint='fixture',enable_search=True),
            SimpleNamespace(live=True,window_id=1,capture='window',policy='body_aim',
                            observe_only=True,checkpoint='fixture',enable_search=True),
        ):
            with self.assertRaisesRegex(ValueError,'--enable-search requires'):
                run(bad_search_args)
        for bad_strafe_args in (
            SimpleNamespace(live=True,window_id=1,capture='window',policy='body_shadow',
                            observe_only=True,checkpoint='fixture',enable_strafe=True),
            SimpleNamespace(live=True,window_id=1,capture='window',policy='body_aim',
                            observe_only=True,checkpoint='fixture',enable_strafe=True),
        ):
            with self.assertRaisesRegex(ValueError,'--enable-strafe requires'):
                run(bad_strafe_args)
        for bad_patrol_args in (
            SimpleNamespace(live=True,window_id=1,capture='window',policy='body_shadow',
                            observe_only=True,checkpoint='fixture',enable_patrol=True),
            SimpleNamespace(live=True,window_id=1,capture='window',policy='body_aim',
                            observe_only=True,checkpoint='fixture',enable_patrol=True),
        ):
            with self.assertRaisesRegex(ValueError,'--enable-patrol requires'):
                run(bad_patrol_args)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            summarize([
                {"applied": True, "observe_only": False, "capture_ms": 1., "policy_ms": 2., "input_ms": 1., "loop_ms": 4., "action": {"attack": True, "keys": ["a"]}},
                {"applied": True, "observe_only": False, "capture_ms": 1., "policy_ms": 2., "input_ms": 1., "loop_ms": 4., "action": {"attack": False, "keys": ["w"]}},
                {"applied": False, "observe_only": False, "capture_ms": 1., "policy_ms": 2., "input_ms": 1., "loop_ms": 4., "action": {"attack": True, "keys": ["d"]}},
            ], "desktop", "body_aim", "frame_limit")
        summary = json.loads(buf.getvalue())
        self.assertEqual(summary["dispatched_attack_actions"], 1)
        self.assertEqual(summary["dispatched_strafe_actions"], 1)
        self.assertEqual(summary["dispatched_patrol_actions"], 1)

    def test_gated_attack_policy_pulses_attack_without_movement_keys(self):
        clock=SimpleNamespace(now=1_000_000_000)
        def model(inputs):
            return [dict(boxes=torch.tensor([[290.,180.,350.,300.]]),
                         labels=torch.tensor([1]),scores=torch.tensor([.99]))]
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'model.pt';path.write_bytes(b'fixture')
            with patch('vimm8bdm.body_shadow.load_checkpoint',return_value=(model,{})):
                policy=BodyAimPolicy(path,lambda:clock.now,enable_attack=True)
        self.assertEqual(policy.phase,'aim_and_attack')
        image=Image.new('RGB',(640,480),'black')
        ImageDraw.Draw(image).rectangle((290,180,349,299),fill=(255,0,120))
        pixels=image.convert('RGBA').tobytes('raw','BGRA')
        attacks=[]
        for _ in range(5):
            clock.now+=100_000_000
            action=policy.predict(Frame(640,480,pixels,clock.now-1_000_000))
            self.assertFalse(action.keys)
            attacks.append(action.attack)
        self.assertEqual(attacks,[False,False,True,False,True])
        self.assertTrue(policy.last_prediction['aim']['attack_enabled'])
        self.assertFalse(policy.last_prediction['aim']['search_enabled'])

    def test_search_enabled_policy_scans_toward_last_seen_side_on_target_loss(self):
        clock=SimpleNamespace(now=1_000_000_000)
        visible=SimpleNamespace(flag=True)
        def model(inputs):
            if not visible.flag:
                return [dict(boxes=torch.zeros((0,4)),labels=torch.zeros((0,),dtype=torch.int64),scores=torch.zeros((0,)))]
            return [dict(boxes=torch.tensor([[40.,180.,100.,280.]]),
                         labels=torch.tensor([1]),scores=torch.tensor([.99]))]
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'model.pt';path.write_bytes(b'fixture')
            with patch('vimm8bdm.body_shadow.load_checkpoint',return_value=(model,{})):
                policy=BodyAimPolicy(path,lambda:clock.now,enable_attack=True,enable_search=True)
        image=Image.new('RGB',(640,480),'black')
        ImageDraw.Draw(image).rectangle((40,180,99,279),fill=(255,0,120))
        pixels=image.convert('RGBA').tobytes('raw','BGRA')
        clock.now+=100_000_000
        self.assertEqual(policy.predict(Frame(640,480,pixels,clock.now-1_000_000)),Action())
        self.assertTrue(policy.last_prediction['aim']['search_enabled'])
        visible.flag=False
        clock.now+=100_000_000
        self.assertEqual(policy.predict(Frame(640,480,pixels,clock.now-1_000_000)),Action())
        self.assertEqual(policy.last_prediction['aim']['reason'],'missing_target')
        clock.now+=100_000_000
        search_action=policy.predict(Frame(640,480,pixels,clock.now-1_000_000))
        self.assertEqual(search_action,Action(mouse_dx=-16,mouse_dy=0,attack=False))
        self.assertEqual(policy.last_prediction['aim']['reason'],'search_target')
        self.assertEqual(policy.last_prediction['aim']['search_direction'],-1)

    def test_search_patrol_engage_cycle_walks_escapes_wall_brakes_tracks_and_resets_on_stale(self):
        clock=SimpleNamespace(now=1_000_000_000)
        state=SimpleNamespace(box=None)
        def model(inputs):
            if state.box is None:
                return [dict(boxes=torch.zeros((0,4)),labels=torch.zeros((0,),dtype=torch.int64),scores=torch.zeros((0,)))]
            return [dict(boxes=torch.tensor([state.box]),labels=torch.tensor([1]),scores=torch.tensor([.99]))]
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'model.pt';path.write_bytes(b'fixture')
            with patch('vimm8bdm.body_shadow.load_checkpoint',return_value=(model,{})):
                policy=BodyAimPolicy(path,lambda:clock.now,enable_attack=True,enable_search=True,
                                     enable_strafe=True,enable_patrol=True)
        def make_frame(shade=40,box=None):
            img=Image.new('RGB',(640,480),(shade,shade,shade))
            if box is not None:
                x1,y1,x2,y2=[int(v) for v in box]
                ImageDraw.Draw(img).rectangle((x1,y1,x2-1,y2-1),fill=(255,0,120))
            clock.now+=100_000_000
            return Frame(640,480,img.convert('RGBA').tobytes('raw','BGRA'),clock.now-1_000_000)
        # 1 missing frame + 7 search_target frames
        self.assertEqual(policy.predict(make_frame()),Action())
        self.assertEqual(policy.last_prediction['aim']['reason'],'missing_target')
        for _ in range(7):
            act=policy.predict(make_frame())
            self.assertEqual(act,Action(mouse_dx=16))
            self.assertEqual(policy.last_prediction['aim']['reason'],'search_target')
        # 9th..11th missing frames enter patrol_target settle (0.0s, 0.1s, 0.2s)
        for _ in range(3):
            act=policy.predict(make_frame(shade=40))
            self.assertEqual(act,Action())
            self.assertEqual(policy.last_prediction['aim']['reason'],'patrol_target')
            self.assertEqual(policy.last_prediction['aim']['patrol_phase'],'settle')
        # 16th missing frame (0.3s elapsed) starts forward 'w' probe
        fwd1=policy.predict(make_frame(shade=40))
        self.assertEqual(fwd1,Action(keys=frozenset({'w'})))
        self.assertEqual(policy.last_prediction['aim']['patrol_phase'],'forward')
        # Moving scene (shade changes 40 -> 90 after 0.2s) keeps holding 'w'
        self.assertEqual(policy.predict(make_frame(shade=65)),Action(keys=frozenset({'w'})))
        self.assertEqual(policy.predict(make_frame(shade=90)),Action(keys=frozenset({'w'})))
        self.assertEqual(policy.navigator.successful_probes,1)
        # Static wall (shade stays 90 for 0.2s) releases 'w' and enters turn phase
        self.assertEqual(policy.predict(make_frame(shade=90)),Action(keys=frozenset({'w'})))
        wall_act=policy.predict(make_frame(shade=90))
        self.assertEqual(wall_act,Action())
        self.assertEqual(policy.last_prediction['aim']['patrol_phase'],'turn')
        turn_act=policy.predict(make_frame(shade=90))
        self.assertEqual(turn_act.keys,frozenset())
        self.assertEqual(turn_act.mouse_dx,8)
        # Target appears off-center during patrol: immediately releases movement, resets navigator, and applies search_brake
        state.box=[380.,180.,440.,280.]
        brake_act=policy.predict(make_frame(shade=90,box=state.box))
        self.assertEqual(brake_act,Action(mouse_dx=13.5))
        self.assertEqual(brake_act.keys,frozenset())
        self.assertEqual(policy.last_prediction['aim']['reason'],'confirm_target')
        self.assertTrue(policy.last_prediction['aim']['search_brake'])
        self.assertEqual(policy.navigator.phase,'settle')
        # 2nd target frame locks track_target with lateral strafe 'a'; 3rd frame fires attack=True
        state.box=[290.,180.,350.,280.]
        track_act=policy.predict(make_frame(shade=90,box=state.box))
        self.assertEqual(policy.last_prediction['aim']['reason'],'track_target')
        self.assertEqual(track_act.keys,frozenset({'a'}))
        self.assertFalse(track_act.attack)
        fire_act=policy.predict(make_frame(shade=90,box=state.box))
        self.assertEqual(policy.last_prediction['aim']['reason'],'track_target')
        self.assertEqual(fire_act.keys,frozenset({'a'}))
        self.assertTrue(fire_act.attack)
        # Stale frame immediately resets tracker and navigator and outputs empty Action()
        clock.now+=100_000_000
        stale_frame=Frame(640,480,bytes((0,0,0,255))*640*480,clock.now-200_000_000)
        self.assertEqual(policy.predict(stale_frame),Action())
        self.assertEqual(policy.tracker.matches,0)
        self.assertEqual(policy.navigator.phase,'settle')
