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
        from vimm8bdm.__main__ import run
        args=SimpleNamespace(live=False,policy='body_aim',checkpoint='fixture')
        with self.assertRaisesRegex(ValueError,'Body aim requires'):
            run(args)
