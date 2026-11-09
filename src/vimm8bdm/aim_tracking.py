"""Filter Cutman appearance and produce bounded mouse-only proposals."""
import math

from .control import Action
from .visual_metrics import box_iou


def measure_damage_cues(image):
    """Measure MM8BDM yellow hit-flash intensity and HUD health-bar pixels from RGB image."""
    if (image.width, image.height) != (640, 480):
        return 0, None
    if max(image.getpixel((47, 30))) > 8 or max(image.getpixel((47, 120))) > 8:
        return 0, None
    ks = [(r + g) // 2 for y in range(320, 470, 8) for x in range(400, 600, 8)
          for r, g, b in (image.getpixel((x, y)),) if b <= 4 and abs(r - g) <= 4]
    flash_k = min(ks) if ks else 0
    if flash_k > 160:
        flash_k = 0
    hp_pixels = sum(1 for y in range(20, 151) for x in (50, 52, 58, 60)
                    if image.getpixel((x, y))[0] > 180 and image.getpixel((x, y))[1] > 150)
    return flash_k, hp_pixels


def filter_candidates(image, candidates, flash_k=0):
    """Retain compact magenta bodies and remove overlapping duplicate boxes."""
    accepted, rejected = [], []
    unblend = flash_k >= 12
    scale = 255. / max(1, 255 - flash_k) if unblend else 1.
    for row in sorted(candidates, key=lambda item: item['score'], reverse=True):
        x, y, w, h = row['box']
        reason = None
        if not all(math.isfinite(v) for v in [x, y, w, h, row['score']]):
            reason = 'invalid_box'
        elif row['score'] < .8:
            reason = 'low_aim_score'
        elif (x < 0 or y < 0 or w < 8 or h < 12 or x+w > image.width+.01
              or y+h > image.height+.01 or w*h > image.width*image.height*.68
              or not .2 <= w/h <= 1.25):
            reason = 'box_extent'
        else:
            crop = image.crop((int(x), int(y), math.ceil(x+w), math.ceil(y+h))).convert('RGB')
            crop.thumbnail((96, 96))
            raw = crop.tobytes()
            if unblend:
                pink_y = [i // crop.width for i, (r,g,b) in enumerate(zip(raw[0::3],raw[1::3],raw[2::3]))
                          for ur, ug, ub in ((min(255., max(0., (r - flash_k) * scale)),
                                              max(0., (g - flash_k) * scale),
                                              min(255., b * scale)),)
                          if ur >= 100 and ub >= 60 and ug < ur*.6 and ug < ub*.6]
            else:
                pink_y = [i // crop.width for i, (r,g,b) in enumerate(zip(raw[0::3],raw[1::3],raw[2::3]))
                          if r >= 100 and b >= 60 and g < r*.6 and g < b*.6]
            if (not pink_y or len(pink_y)/(crop.width*crop.height) < .03
                    or (max(pink_y)-min(pink_y)+1)/crop.height < .45):
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

    def __init__(self, enable_attack=False, enable_search=False, enable_strafe=False, enable_patrol=False):
        self.enable_attack = bool(enable_attack)
        self.enable_search = bool(enable_search)
        self.enable_strafe = bool(enable_strafe)
        self.enable_patrol = bool(enable_patrol)
        self.reset()

    def reset(self):
        self.previous = self.timestamp = self.prev_hp_pixels = None
        self.matches = self.missing_streak = self.strafe_step = 0
        self.prev_flash_k = self.counter_search_frames = 0
        self.attack_pulsed = self.occluded_after_attack = self.braked_from_search = False
        self.search_direction = 1

    def step(self, candidates, width, height, captured_ns, flash_k=0, hp_pixels=None):
        hp_drop = (self.prev_hp_pixels is not None and hp_pixels is not None
                   and self.prev_hp_pixels - hp_pixels >= 3)
        flash_spike = flash_k >= 25 and flash_k - self.prev_flash_k >= 15
        damaged = bool(hp_drop or flash_spike)
        self.prev_flash_k = flash_k
        if hp_pixels is not None:
            self.prev_hp_pixels = hp_pixels
        reason = None
        if (width, height) != (640, 480):
            reason = 'uncalibrated_image_size'
        elif len(candidates) != 1:
            reason = 'missing_target' if not candidates else 'ambiguous_targets'
        if reason:
            interval = None if self.timestamp is None else (captured_ns - self.timestamp) / 1e9
            if (reason == 'missing_target' and self.enable_attack and self.attack_pulsed
                    and self.previous is not None and interval is not None and 0 < interval <= .2):
                self.attack_pulsed = False
                self.occluded_after_attack = True
                self.timestamp = captured_ns
                return Action(), dict(reason='post_attack_occlusion', matches=self.matches,
                                      selected_box=list(self.previous))
            if reason == 'missing_target':
                self.previous, self.matches = None, 0
                self.attack_pulsed = self.occluded_after_attack = self.braked_from_search = False
                self.missing_streak += 1
                self.timestamp = captured_ns
                if damaged and (self.enable_search or self.enable_patrol):
                    self.counter_search_frames = 32
                if self.counter_search_frames > 0:
                    self.counter_search_frames -= 1
                    return Action(mouse_dx=24 * self.search_direction), dict(
                        reason='counter_search', matches=0,
                        search_direction=self.search_direction,
                        counter_search_remaining=self.counter_search_frames,
                        missing_streak=self.missing_streak, damaged=damaged)
                search_limit = 8 if self.enable_patrol else 120
                if self.enable_search and 2 <= self.missing_streak <= search_limit:
                    return Action(mouse_dx=16 * self.search_direction), dict(
                        reason='search_target', matches=0,
                        search_direction=self.search_direction, missing_streak=self.missing_streak)
                if self.enable_patrol and self.missing_streak > (search_limit if self.enable_search else 1):
                    return Action(), dict(reason='patrol_target', matches=0,
                                          missing_streak=self.missing_streak)
                return Action(), dict(reason='missing_target', matches=0)
            self.reset()
            return Action(), dict(reason=reason, matches=0)
        box = candidates[0]['box']
        x, y, w, h = box
        was_searching = (self.counter_search_frames > 0
                         or (self.enable_search and 2 <= self.missing_streak <= 120)
                         or (self.enable_patrol and self.missing_streak >= 2))
        self.counter_search_frames = 0
        self.search_direction = -1 if x + w / 2 < width / 2 else 1
        self.missing_streak = 0
        interval = None if self.timestamp is None else (captured_ns-self.timestamp)/1e9
        matched = False
        if self.previous is not None and interval is not None and 0 < interval <= .2:
            px,py,pw,ph = self.previous
            wide_assoc = self.occluded_after_attack or self.braked_from_search
            max_distance = max(144, 2.0 * max(w, pw)) if wide_assoc else max(96, 1.35 * max(w, pw))
            distance = math.hypot(x+w/2-px-pw/2, y+h/2-py-ph/2)
            area_ratio = w*h/(pw*ph)
            size_ok = 1/3 <= area_ratio <= 3 or (.5 < h/ph < 2 and .18 <= area_ratio <= 4)
            matched = size_ok and (box_iou(box,self.previous) >= .1 or distance <= max_distance)
        self.occluded_after_attack = self.braked_from_search = False
        self.matches = self.matches+1 if matched else 1
        self.previous, self.timestamp = list(box), captured_ns
        error_x, error_y = x+w/2-width/2, y+h/2-height/2
        dt = min(interval, .1) if interval is not None and 0 < interval <= .2 else .1
        limit_x = (240 if self.enable_patrol else 160 if (self.enable_search or self.enable_strafe) else 60) * dt
        dx = 0 if abs(error_x) <= 8 else max(-limit_x,min(limit_x,error_x*.15))
        detail = dict(reason='confirm_target', matches=self.matches, selected_box=list(box))
        if damaged:
            detail['damaged'] = True
        if self.matches < (2 if (self.enable_search or self.enable_patrol) else 3):
            self.attack_pulsed = False
            if was_searching:
                self.braked_from_search = True
                detail['search_brake'] = True
                return Action(mouse_dx=dx), detail
            return Action(), detail
        dy = 0 if abs(error_y) <= 6 else max(-40*dt,min(40*dt,error_y*.15/.625))
        window_x, window_y = max(8., w * .35), max(6., h * .35)
        in_attack_window = abs(error_x) <= window_x and abs(error_y) <= window_y
        attack = self.enable_attack and self.matches >= 3 and in_attack_window and not self.attack_pulsed
        self.attack_pulsed = attack
        keys = set()
        if self.enable_strafe:
            strafe_key = 'a' if (self.strafe_step // 4) % 2 == 0 else 'd'
            self.strafe_step += 1
            keys.add(strafe_key)
            detail['strafe_key'] = strafe_key
        if self.enable_patrol:
            area = w * h
            if area < 4500:
                keys.add('w')
                detail['range_key'] = 'w'
            elif area > 16000:
                keys.add('s')
                detail['range_key'] = 's'
        detail.update(reason='track_target', error_pixels=[error_x,error_y],
                      attack_window_pixels=[window_x,window_y],
                      in_attack_window=in_attack_window, attack=attack)
        return Action(keys=frozenset(keys), mouse_dx=dx, mouse_dy=dy, attack=attack), detail
