"""Filter Cutman appearance and produce bounded mouse-only proposals."""
import math

from .control import Action
from .visual_metrics import box_iou


def filter_candidates(image, candidates):
    """Retain compact magenta bodies and remove overlapping duplicate boxes."""
    accepted, rejected = [], []
    for row in sorted(candidates, key=lambda item: item['score'], reverse=True):
        x, y, w, h = row['box']
        reason = None
        if not all(math.isfinite(v) for v in [x, y, w, h, row['score']]):
            reason = 'invalid_box'
        elif row['score'] < .8:
            reason = 'low_aim_score'
        elif (x < 0 or y < 0 or w < 8 or h < 12 or x+w > image.width+.01
              or y+h > image.height+.01 or w*h > image.width*image.height*.35
              or not .2 <= w/h <= 1.1):
            reason = 'box_extent'
        else:
            crop = image.crop((int(x), int(y), math.ceil(x+w), math.ceil(y+h))).convert('RGB')
            crop.thumbnail((96, 96))
            raw = crop.tobytes()
            pixels = list(zip(raw[0::3],raw[1::3],raw[2::3]))
            pink = [(i % crop.width, i // crop.width) for i, (r,g,b) in enumerate(pixels)
                    if r >= 100 and b >= 60 and g < r*.6 and g < b*.6]
            if (not pink or len(pink)/len(pixels) < .03
                    or (max(p[1] for p in pink)-min(p[1] for p in pink)+1)/crop.height < .45):
                reason = 'cutman_appearance'
            elif any(box_iou(row['box'], previous['box']) > .3 for previous in accepted):
                reason = 'duplicate_box'
        if reason:
            rejected.append(dict(box=row['box'], reason=reason))
        else:
            accepted.append(row)
    return accepted, rejected


class StableAim:
    """Require three matching observations and stop on every missing target."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.previous = None
        self.timestamp = None
        self.matches = 0

    def step(self, candidates, width, height, captured_ns):
        reason = None
        if (width, height) != (640, 480):
            reason = 'uncalibrated_image_size'
        elif len(candidates) != 1:
            reason = 'missing_target' if not candidates else 'ambiguous_targets'
        if reason:
            self.reset()
            return Action(), dict(reason=reason, matches=0)
        box = candidates[0]['box']
        interval = None if self.timestamp is None else (captured_ns-self.timestamp)/1e9
        matched = False
        if self.previous is not None and interval is not None and 0 < interval <= .2:
            x,y,w,h = box
            px,py,pw,ph = self.previous
            distance = math.hypot(x+w/2-px-pw/2, y+h/2-py-ph/2)
            matched = .5 <= w*h/(pw*ph) <= 2 and (box_iou(box,self.previous) >= .1 or distance <= 48)
        self.matches = self.matches+1 if matched else 1
        self.previous, self.timestamp = list(box), captured_ns
        detail = dict(reason='confirm_target', matches=self.matches, selected_box=list(box))
        if self.matches < 3:
            return Action(), detail
        x,y,w,h = box
        error_x, error_y = x+w/2-width/2, y+h/2-height/2
        dt = min(interval, .1)
        dx = 0 if abs(error_x) <= 8 else max(-60*dt,min(60*dt,error_x*.15))
        dy = 0 if abs(error_y) <= 6 else max(-40*dt,min(40*dt,error_y*.15/.625))
        detail.update(reason='track_target', error_pixels=[error_x,error_y])
        return Action(mouse_dx=dx, mouse_dy=dy), detail
