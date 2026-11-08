"""Propose mouse motion without movement keys, with optional gated tap-fire."""
import time
from PIL import Image

from .aim_tracking import StableAim, filter_candidates
from .body_shadow import BodyShadowPolicy
from .control import Action
from .visual_navigation import VisualNavigationPolicy


class BodyAimPolicy(BodyShadowPolicy):
    """Use current visual candidates for a limited Cutman aim experiment."""

    def __init__(self, checkpoint, clock=time.perf_counter_ns, device=None,
                 enable_attack=False, enable_search=False, enable_strafe=False, enable_patrol=False):
        self.enable_attack = bool(enable_attack)
        self.enable_search = bool(enable_search)
        self.enable_strafe = bool(enable_strafe)
        self.enable_patrol = bool(enable_patrol)
        super().__init__(checkpoint, clock=clock, device=device)

    def reset(self):
        super().reset()
        self.tracker = StableAim(enable_attack=self.enable_attack, enable_search=self.enable_search,
                                 enable_strafe=self.enable_strafe, enable_patrol=self.enable_patrol)
        self.navigator = VisualNavigationPolicy()
        self.phase = 'aim_and_attack' if self.enable_attack else 'aim_only'

    def predict(self, frame):
        super().predict(frame)
        result = self.last_prediction
        result.pop('input_sent', None)
        if result['stale']:
            self.tracker.reset()
            self.navigator.reset()
            result['aim'] = dict(reason='invalid_detection_result', matches=0)
            return Action()
        image = self.last_image or Image.frombytes('RGB', (frame.width,frame.height), frame.bgra, 'raw','BGRX')
        candidates, rejected = filter_candidates(image, result['candidates'])
        action, detail = self.tracker.step(candidates,frame.width,frame.height,frame.captured_ns)
        if detail['reason'] == 'patrol_target':
            action = self.navigator.predict(frame)
            nav = self.navigator.last_prediction
            detail.update(patrol_phase=nav['phase'], navigation=nav)
        else:
            self.navigator.reset()
        result['aim'] = dict(**detail, eligible_candidates=candidates, rejected=rejected,
                             movement_keys_enabled=self.enable_strafe or self.enable_patrol,
                             attack_enabled=self.enable_attack,
                             search_enabled=self.enable_search, strafe_enabled=self.enable_strafe,
                             patrol_enabled=self.enable_patrol)
        return action
