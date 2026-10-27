"""Capture game images and send normal input on the same desktop."""
from __future__ import annotations

import os
import sys


class Focus:
    def __init__(self, window_id, windows=None):
        from .windows import open_windows
        self.window_id = window_id
        self.windows = windows if windows is not None else open_windows()
        self.identity = self.windows.describe(window_id) if window_id else None
        if window_id and self.identity is None:
            self.windows.close()
            raise ValueError("The selected window does not exist")

    def current(self):
        return self.windows.current()

    def __call__(self):
        from .windows import focus_matches
        try:
            info = self.windows.describe(self.window_id)
            if (self.identity is None or info is None or not info.visible or
                    info.width < 160 or info.height < 120):
                return False
            if info.process_id != self.identity.process_id or info.class_name != self.identity.class_name:
                return False
            if (info.width, info.height) != (self.identity.width, self.identity.height):
                return False
            return focus_matches(self.window_id, self.current(), self.windows.parent_of)
        except Exception:
            return False

    def close(self):
        self.windows.close()


class DesktopSink:
    def __init__(self):
        if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
            raise RuntimeError("This backend requires an X11 desktop")
        from pynput import keyboard, mouse
        self.keyboard, self.mouse = keyboard.Controller(), mouse.Controller()
        self.key_module, self.mouse_module = keyboard, mouse
        self.held = set()
        self.attacking = False
        self.relative_mouse = None
        if sys.platform == "win32":
            from .winmouse import WindowsRelativeMouse
            self.relative_mouse = WindowsRelativeMouse()

    def key(self, name):
        return getattr(self.key_module.Key, name) if name in {"space", "shift"} else name

    def apply(self, action):
        for name in list(self.held - action.keys):
            self.keyboard.release(self.key(name))
            self.held.remove(name)
        for name in sorted(action.keys - self.held):
            # Track first so a failed press is still released during cleanup.
            self.held.add(name)
            self.keyboard.press(self.key(name))
        if action.attack != self.attacking:
            self.attacking = action.attack
            method = self.mouse.press if action.attack else self.mouse.release
            method(self.mouse_module.Button.left)
        dx, dy = round(action.mouse_dx), round(action.mouse_dy)
        if dx or dy:
            if self.relative_mouse:
                self.relative_mouse.move(dx, dy)
            else:
                self.mouse.move(dx, dy)

    def release(self):
        errors = []
        for name in list(self.held):
            try:
                self.keyboard.release(self.key(name))
                self.held.remove(name)
            except Exception as exc:
                errors.append(exc)
        if self.attacking:
            try:
                self.mouse.release(self.mouse_module.Button.left)
                self.attacking = False
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise RuntimeError("Some owned inputs could not be released") from errors[0]

    def owned_state(self):
        return {"held_keys": sorted(self.held), "attack_held": self.attacking}


class X11WindowCapture:
    """Read displayed window pixels when a rootless X server has a black root."""

    def __init__(self, window_id):
        from Xlib import display
        self.connection = display.Display()
        self.window = self.connection.create_resource_object("window", window_id)

    def grab(self):
        from Xlib import X
        from .control import Frame
        import time
        started = time.perf_counter_ns()
        geometry = self.window.get_geometry()
        visual = self.window.get_attributes().visual
        # The prototype supports the tested 24-bit TrueColor visual only.
        screen = self.connection.screen()
        info = next((v for d in screen.allowed_depths for v in d.visuals
                     if v.visual_id == visual), None)
        if (geometry.depth != 24 or info is None or
                (info.red_mask, info.green_mask, info.blue_mask) != (0xff0000, 0xff00, 0xff)):
            raise RuntimeError("Window capture requires a 24-bit RGB TrueColor visual")
        result = self.window.get_image(0, 0, geometry.width, geometry.height, X.ZPixmap, 0xffffffff)
        # X11 byte order zero means least significant byte first.
        if self.connection.display.info.image_byte_order != 0:
            raise RuntimeError("Window capture requires least significant byte first")
        return Frame(geometry.width, geometry.height, bytes(result.data), started)

    def close(self):
        self.connection.close()


class WindowsWindowCapture:
    """Capture the client area only while the selected window has focus."""

    def __init__(self, window_id):
        import mss
        from .windows import WindowsDesktop
        self.window_id = window_id
        self.windows = WindowsDesktop()
        self.capture = mss.MSS()

    def grab(self):
        import time
        from .control import Frame
        from .windows import focus_matches
        started = time.perf_counter_ns()
        info = self.windows.describe(self.window_id)
        if info is None or not info.visible or info.width <= 0 or info.height <= 0:
            raise RuntimeError("The selected window is unavailable for capture")
        if not focus_matches(self.window_id, self.windows.current(), self.windows.parent_of):
            return Frame(info.width, info.height, bytes(info.width * info.height * 4), started)
        region = {"left": info.left, "top": info.top, "width": info.width, "height": info.height}
        shot = self.capture.grab(region)
        return Frame(shot.width, shot.height, bytes(shot.bgra), started)

    def close(self):
        self.capture.close()
        self.windows.close()
