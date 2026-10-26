"""Bounded visual probe and input lease. No engine state is consumed."""
from __future__ import annotations

from dataclasses import dataclass
import math
import threading
import time


@dataclass(frozen=True)
class Frame:
    width: int
    height: int
    bgra: bytes
    captured_ns: int  # Monotonic capture START, conservatively includes capture cost.

    def __post_init__(self):
        if self.width <= 0 or self.height <= 0 or len(self.bgra) != self.width * self.height * 4:
            raise ValueError("Frame must contain width * height * 4 BGRA bytes")


@dataclass(frozen=True)
class Action:
    keys: frozenset[str] = frozenset()
    mouse_dx: float = 0
    mouse_dy: float = 0
    attack: bool = False

    def bounded(self):
        allowed = set("wasd123456789") | {"space", "shift"}
        if not self.keys <= allowed:
            raise ValueError("Action contains an unsupported gameplay key")
        if not all(math.isfinite(v) for v in (self.mouse_dx, self.mouse_dy)):
            raise ValueError("Mouse deltas must be finite")
        return Action(frozenset(self.keys), max(-24, min(24, self.mouse_dx)),
                      max(-24, min(24, self.mouse_dy)), bool(self.attack))


class PixelProbe:
    """Find a red calibration marker. This is NOT an MM8BDM enemy detector."""

    def predict(self, frame: Frame) -> Action:
        sx = sy = count = 0
        # Exclude HUD/lower weapon region; sample pixels to keep this stdlib-only.
        for y in range(0, max(1, int(frame.height * .72)), 8):
            for x in range(0, frame.width, 8):
                i = (y * frame.width + x) * 4
                b, g, r = frame.bgra[i:i + 3]
                if r > 180 and r > g * 1.7 and r > b * 1.7:
                    sx += x
                    sy += y
                    count += 1
        if count < 3:
            return Action(frozenset({"w"}), mouse_dx=3)
        dx = (sx / count / frame.width - .5) * 32
        dy = (sy / count / frame.height - .5) * 32
        return Action(frozenset({"w"}), dx, dy, abs(dx) < 2 and abs(dy) < 2)


def blank_frame(frame):
    """Reject an image when every sampled RGB pixel is almost black."""
    step = max(1, frame.width * frame.height // 4096)
    return all(max(frame.bgra[i:i + 3]) <= 3
               for i in range(0, len(frame.bgra), step * 4))


class InputCheckPolicy:
    """Exercise each input separately in a short local sequence."""

    STEPS = (
        ("ready", .5, Action()),
        ("forward", .5, Action(frozenset({"w"}))),
        ("release_movement", .3, Action()),
        ("strafe_right", .5, Action(frozenset({"d"}))),
        ("release_strafe", .3, Action()),
        ("yaw_right", .5, Action(mouse_dx=4)),
        ("yaw_left", .5, Action(mouse_dx=-4)),
        ("pitch_up", .5, Action(mouse_dy=-2)),
        ("pitch_down", .5, Action(mouse_dy=2)),
        ("release_aim", .3, Action()),
        ("attack", .6, Action(attack=True)),
        ("release_attack", .5, Action()),
        ("jump", .1, Action(frozenset({"space"}))),
        ("release_jump", .5, Action()),
        ("weapon_slot_2", .1, Action(frozenset({"2"}))),
        ("release_weapon", .5, Action()),
    )

    def __init__(self, clock=time.perf_counter):
        self.clock = clock
        self.reset()

    def reset(self):
        self.started = self.clock()
        self.phase = "ready"
        self.complete = False

    def predict(self, frame):
        elapsed = max(0, self.clock() - self.started)
        end = 0
        for phase, duration, action in self.STEPS:
            end += duration
            if elapsed < end:
                self.phase = phase
                return action
        self.phase = "complete"
        self.complete = True
        return Action()


class InputLease:
    """Disarm on takeover, stale frames, lost focus, or an expired heartbeat.

    sink.apply sets the full held-key/button state; sink.release releases only
    inputs owned by this process. Desktop calls are serialized with takeover.
    """

    def __init__(self, sink, focused, clock=time.perf_counter_ns,
                 max_age_ns=150_000_000, lease_ns=200_000_000):
        self.sink, self.focused, self.clock = sink, focused, clock
        self.max_age_ns, self.lease_ns = max_age_ns, lease_ns
        self.armed = False
        self.deadline = 0
        self.reason = "manual"
        self.lock = threading.RLock()

    def pause(self, reason="manual"):
        with self.lock:
            self.armed = False
            self.deadline = 0
            self.reason = reason
            self.sink.release()

    def toggle(self):
        with self.lock:
            if self.armed:
                self.pause()
            elif self.focused():
                self.sink.release()
                self.deadline = self.clock() + self.lease_ns
                self.reason = "armed"
                self.armed = True
            else:
                self.reason = "focus_blocked"

    def submit(self, action, captured_ns):
        with self.lock:
            if not self.armed:
                return False
            now = self.clock()
            if now >= self.deadline:
                self.pause("watchdog")
                return False
            if not self.focused():
                self.pause("focus")
                return False
            if not 0 <= now - captured_ns <= self.max_age_ns:
                self.pause("stale_frame")
                return False
            try:
                self.sink.apply(action.bounded())
            except BaseException:
                self.pause("input_error")
                raise
            self.deadline = now + self.lease_ns
            return True

    def poll(self):
        with self.lock:
            if self.armed and (self.clock() >= self.deadline or not self.focused()):
                self.pause("watchdog_or_focus")


class ControlKeys:
    """Handle control keys without sending them to the input receiver."""

    def __init__(self, lease, stop):
        self.lease, self.stop = lease, stop
        self.down = set()

    def press(self, name):
        if name not in {"f8", "f9", "esc"} or name in self.down or self.stop.is_set():
            return None
        self.down.add(name)
        if name == "f8":
            self.lease.toggle()
        elif name == "esc":
            self.lease.pause("escape_pause")
        else:
            self.lease.pause("quit_key")
            self.stop.set()
        return self.lease.reason

    def release(self, name):
        self.down.discard(name)


def synthetic_frame(index=0, width=160, height=120):
    raw = bytearray(bytes((35, 35, 35, 255)) * (width * height))
    cx = (width // 2, width * 3 // 4, width // 4)[index % 3]
    for y in range(height // 2 - 12, height // 2 + 12):
        for x in range(cx - 12, cx + 12):
            i = (y * width + x) * 4
            raw[i:i + 4] = bytes((0, 0, 255, 255))
    return Frame(width, height, bytes(raw), time.perf_counter_ns())
