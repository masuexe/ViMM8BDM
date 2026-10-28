"""Propose mouse motion without movement keys or attack."""
from PIL import Image

from .aim_tracking import StableAim, filter_candidates
from .body_shadow import BodyShadowPolicy
from .control import Action


class BodyAimPolicy(BodyShadowPolicy):
    """Use current visual candidates for a limited Cutman aim experiment."""

    def reset(self):
        super().reset()
        self.tracker = StableAim()
        self.phase = 'aim_only'

    def predict(self, frame):
        super().predict(frame)
        result = self.last_prediction
        result.pop('input_sent', None)
        if result['stale']:
            self.tracker.reset()
            result['aim'] = dict(reason='invalid_detection_result', matches=0)
            return Action()
        image = Image.frombytes('RGB', (frame.width,frame.height), frame.bgra, 'raw','BGRX')
        candidates, rejected = filter_candidates(image, result['candidates'])
        action, detail = self.tracker.step(candidates,frame.width,frame.height,frame.captured_ns)
        result['aim'] = dict(**detail, eligible_candidates=candidates, rejected=rejected,
                             movement_keys_enabled=False, attack_enabled=False)
        return action
