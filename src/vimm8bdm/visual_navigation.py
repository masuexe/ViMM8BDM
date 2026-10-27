"""Use displayed image change as a movement proxy for a navigation baseline."""
from .control import Action


def view_sample(frame):
    """Sample above the weapon and to the right of the current profile's HUD."""
    values = []
    for row in range(16):
        y = min(frame.height - 1, int(frame.height * (.14 + .34 * row / 15)))
        for column in range(32):
            x = min(frame.width - 1, int(frame.width * (.22 + .58 * column / 31)))
            at = (y * frame.width + x) * 4
            b, g, r = frame.bgra[at:at + 3]
            values.append((r * 77 + g * 150 + b * 29) / 256)
    return tuple(values)


def image_change(previous, current):
    differences = sorted(abs(a - b) for a, b in zip(previous, current))
    # Remove the largest ten percent to reduce isolated animation effects.
    retained = differences[:max(1, len(differences) * 9 // 10)]
    return sum(retained) / len(retained)


class VisualNavigationPolicy:
    """Probe forward motion, then increase the turn budget after repeated static views."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.phase = 'settle'
        self.since_ns = None
        self.last_frame_ns = None
        self.shape = None
        self.reference = None
        self.turn_sent = 0.
        self.turn_budget = 256
        self.escape_attempts = 0
        self.changed_streak = 0
        self.blocked_probes = 0
        self.successful_probes = 0
        self.motion_score = None
        self.last_prediction = None

    def predict(self, frame):
        shape = (frame.width, frame.height)
        now = frame.captured_ns
        reset_reason = None
        if self.shape is not None and shape != self.shape:
            reset_reason = 'capture_size_changed'
        elif self.last_frame_ns is not None and not 0 < now - self.last_frame_ns <= 150_000_000:
            reset_reason = 'capture_interval_invalid'
        if reset_reason:
            self.reset()
        self.shape, self.last_frame_ns = shape, now
        if self.since_ns is None:
            self.since_ns = now
        view = view_sample(frame)
        elapsed = (now - self.since_ns) / 1e9
        action = Action()
        if self.phase == 'settle':
            if elapsed >= .3:
                self.phase = 'forward'
                self.since_ns, self.reference = now, view
                action = Action(frozenset({'w'}))
        elif self.phase == 'forward':
            action = Action(frozenset({'w'}))
            if elapsed >= .6:
                self.motion_score = image_change(self.reference, view)
                self.since_ns, self.reference = now, view
                if self.motion_score < 2:
                    self.blocked_probes += 1
                    self.changed_streak = 0
                    self.escape_attempts += 1
                    self.turn_budget = min(1024, 256 * self.escape_attempts)
                    self.phase, self.turn_sent = 'turn', 0.
                    action = Action()
                else:
                    self.successful_probes += 1
                    self.changed_streak += 1
                    if self.changed_streak >= 3:
                        self.escape_attempts = 0
        elif self.phase == 'turn':
            desired = min(self.turn_budget, int(elapsed * 160))
            # Use integer packets so the backend cannot lose fractional requests.
            delta = min(8, max(0, int(desired - self.turn_sent)))
            self.turn_sent += delta
            action = Action(mouse_dx=delta)
            if self.turn_sent >= self.turn_budget:
                # The last turn packet still executes before the settling interval.
                self.phase, self.since_ns = 'settle', now
        self.last_prediction = {'type': 'visual_navigation_rule', 'phase': self.phase,
                                'image_change_score': self.motion_score,
                                'turn_budget_counts': self.turn_budget,
                                'turn_sent_counts': self.turn_sent,
                                'escape_attempts': self.escape_attempts,
                                'changed_view_streak': self.changed_streak,
                                'static_view_probes': self.blocked_probes,
                                'changed_view_probes': self.successful_probes,
                                'reset_reason': reset_reason}
        return action
