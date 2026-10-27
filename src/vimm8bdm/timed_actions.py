"""Compile recorded action windows without sending desktop input."""
from dataclasses import dataclass
import math

from .action_mapping import VK_KEYS
from .control import Action


@dataclass(frozen=True)
class TimedPacket:
    offset_ns: int
    action: Action


@dataclass(frozen=True)
class TimedWindow:
    packets: tuple[TimedPacket, ...]
    duration_ns: int = 50_000_000

    def validate(self):
        if type(self.duration_ns) is not int or not 0 < self.duration_ns <= 50_000_000:
            raise ValueError('The action window must not exceed 50 milliseconds')
        if not isinstance(self.packets, tuple) or not 2 <= len(self.packets) <= 1024:
            raise ValueError('The action window requires 2 through 1024 packets')
        previous = 0
        for packet in self.packets:
            if type(packet.offset_ns) is not int or not previous <= packet.offset_ns <= self.duration_ns:
                raise ValueError('Packet times must remain ordered inside the action window')
            if packet.action != packet.action.bounded():
                raise ValueError('Each mouse packet must remain within 24 counts per axis')
            previous = packet.offset_ns
        if self.packets[0].offset_ns != 0 or self.packets[-1].offset_ns != self.duration_ns:
            raise ValueError('The action window requires start and end packets')
        if self.packets[0].action.mouse_dx or self.packets[0].action.mouse_dy:
            raise ValueError('The start packet must contain no mouse motion')
        if self.packets[-1].action.mouse_dx or self.packets[-1].action.mouse_dy:
            raise ValueError('The end packet must contain no mouse motion')
        return self


class MouseRemainder:
    """Retain fractional counts until integer mouse output can represent them."""

    def __init__(self):
        self.x = self.y = 0.0

    def convert(self, action):
        x, y = action.mouse_dx + self.x, action.mouse_dy + self.y
        ix, iy = round(x), round(y)
        if abs(ix) > 24 or abs(iy) > 24:
            raise ValueError('Rounded mouse output exceeds the packet limit')
        self.x, self.y = x - ix, y - iy
        return Action(action.keys, ix, iy, action.attack)


def compile_label(label, scale_x, scale_y, duration_ns=50_000_000):
    """Preserve event order and receipt times with explicit experimental scales."""
    if any(not math.isfinite(value) or value <= 0 for value in (scale_x, scale_y)):
        raise ValueError('Both mouse scales must be finite and positive')
    if label['wheel_vertical'] or label['wheel_horizontal']:
        raise ValueError('Wheel input has no timed mapping')
    keys, buttons = set(label['keys_at_start']), set(label['buttons_at_start'])
    allowed = VK_KEYS.keys() - {16}
    packets = []

    def append(at, dx=0, dy=0):
        if not keys <= allowed or not buttons <= {'left'}:
            raise ValueError('The label contains an unsupported key or mouse button')
        # Leave one count for fractional rounding carried from an earlier packet.
        count = max(1, math.ceil(max(abs(dx), abs(dy)) / 23))
        if len(packets) + count > 1024:
            raise ValueError('The action window exceeds the packet limit')
        for _ in range(count):
            packets.append(TimedPacket(at, Action(frozenset(VK_KEYS[key] for key in keys),
                                                 dx / count, dy / count, 'left' in buttons)))

    append(0)
    previous = 0
    raw_x = raw_y = 0
    for event in label['events']:
        at = event['offset_ns']
        if type(at) is not int or not previous <= at < duration_ns:
            raise ValueError('Recorded event times must remain ordered inside the window')
        previous = at
        kind = event['type']
        if kind == 'key':
            if event['vk'] not in allowed:
                raise ValueError('The label contains an unsupported key')
            (keys.add if event['down'] else keys.discard)(event['vk'])
            append(at)
        elif kind == 'button':
            if event['button'] != 'left':
                raise ValueError('The label contains an unsupported mouse button')
            (buttons.add if event['down'] else buttons.discard)(event['button'])
            append(at)
        elif kind == 'motion':
            dx, dy = event['dx'], event['dy']
            if not all(math.isfinite(value) for value in (dx, dy, dx * scale_x, dy * scale_y)):
                raise ValueError('Mouse counts must be finite')
            raw_x += dx
            raw_y += dy
            append(at, dx * scale_x, dy * scale_y)
        else:
            raise ValueError('The label contains an unsupported event')
    if (raw_x, raw_y) != (label['mouse_dx_raw'], label['mouse_dy_raw']):
        raise ValueError('Mouse events do not match the recorded totals')
    if keys != set(label['keys_at_end']) or buttons != set(label['buttons_at_end']):
        raise ValueError('Input transitions do not match the recorded end state')
    append(duration_ns)
    return TimedWindow(tuple(packets), duration_ns).validate()
