"""Inspect OS window metadata. Do not send input or read game state."""
from dataclasses import asdict, dataclass
import ctypes
import sys


@dataclass(frozen=True)
class WindowInfo:
    window_id: int
    process_id: int | None
    class_name: str
    title: str
    width: int
    height: int
    visible: bool

    def diagnostic_view(self, include_title=False):
        # This output is local diagnostic data, not public experiment evidence.
        data = asdict(self)
        data["window_id"] = hex(self.window_id)
        if not include_title:
            data.pop("title")
        return data


def focus_matches(target, focused, parent_of):
    """Accept the target or its descendants. Reject unrelated or unknown focus."""
    visited = set()
    for _ in range(64):
        if focused in (None, 0, 1) or focused in visited:
            return False
        if focused == target:
            return True
        visited.add(focused)
        focused = parent_of(focused)
    return False


class X11Windows:
    def __init__(self):
        from Xlib import display
        self.connection = display.Display()
        self.root = self.connection.screen().root

    def current(self):
        focus = self.connection.get_input_focus().focus
        return getattr(focus, "id", focus)

    def parent_of(self, window_id):
        try:
            window = self.connection.create_resource_object("window", window_id)
            return window.query_tree().parent.id
        except Exception:
            return None

    def describe(self, window_id):
        from Xlib import X
        try:
            window = self.connection.create_resource_object("window", window_id)
            geometry = window.get_geometry()
            attributes = window.get_attributes()
            pid = window.get_full_property(self.connection.intern_atom("_NET_WM_PID"), X.AnyPropertyType)
            classes = window.get_wm_class() or ()
            title = window.get_wm_name() or ""
            if isinstance(title, bytes):
                title = title.decode("utf-8", errors="replace")
            return WindowInfo(window_id, int(pid.value[0]) if pid is not None and len(pid.value) else None,
                              "/".join(classes), title, geometry.width, geometry.height,
                              attributes.map_state == X.IsViewable)
        except Exception:
            return None

    def enumerate(self):
        pending = list(self.root.query_tree().children)
        visited = set()
        result = []
        while pending and len(visited) < 4096:
            window = pending.pop()
            if window.id in visited:
                continue
            visited.add(window.id)
            info = self.describe(window.id)
            if info:
                result.append(info)
            try:
                pending.extend(window.query_tree().children)
            except Exception:
                pass
        return result

    def close(self):
        self.connection.close()


class WindowsDesktop:
    def __init__(self):
        from ctypes import wintypes
        self.wintypes = wintypes
        self.lib = ctypes.WinDLL("user32", use_last_error=True)
        for name in ("GetForegroundWindow", "GetAncestor"):
            getattr(self.lib, name).restype = ctypes.c_void_p
        self.lib.GetAncestor.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        self.lib.GetForegroundWindow.argtypes = []
        for name in ("IsWindow", "IsWindowVisible"):
            getattr(self.lib, name).argtypes = [ctypes.c_void_p]
            getattr(self.lib, name).restype = wintypes.BOOL
        self.lib.GetWindowThreadProcessId.argtypes = [ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD)]
        self.lib.GetWindowThreadProcessId.restype = wintypes.DWORD
        self.lib.GetClientRect.argtypes = [ctypes.c_void_p, ctypes.POINTER(wintypes.RECT)]
        self.lib.GetClientRect.restype = wintypes.BOOL
        for name in ("GetClassNameW", "GetWindowTextW"):
            getattr(self.lib, name).argtypes = [ctypes.c_void_p, wintypes.LPWSTR, ctypes.c_int]
            getattr(self.lib, name).restype = ctypes.c_int
        self.callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, ctypes.c_void_p, ctypes.c_void_p)
        self.lib.EnumWindows.argtypes = [self.callback_type, ctypes.c_void_p]
        self.lib.EnumWindows.restype = wintypes.BOOL

    def current(self):
        return self.lib.GetForegroundWindow() or 0

    def parent_of(self, window_id):
        # GA_PARENT excludes the owner relationship of a popup window.
        return self.lib.GetAncestor(window_id, 1) or None

    def describe(self, window_id):
        if not self.lib.IsWindow(window_id):
            return None
        pid = self.wintypes.DWORD()
        self.lib.GetWindowThreadProcessId(window_id, ctypes.byref(pid))
        rectangle = self.wintypes.RECT()
        if not self.lib.GetClientRect(window_id, ctypes.byref(rectangle)):
            return None
        title, class_name = ctypes.create_unicode_buffer(1024), ctypes.create_unicode_buffer(256)
        self.lib.GetWindowTextW(window_id, title, len(title))
        self.lib.GetClassNameW(window_id, class_name, len(class_name))
        return WindowInfo(window_id, pid.value or None, class_name.value, title.value,
                          rectangle.right - rectangle.left, rectangle.bottom - rectangle.top,
                          bool(self.lib.IsWindowVisible(window_id)))

    def enumerate(self):
        result = []

        @self.callback_type
        def collect(window_id, unused):
            info = self.describe(window_id)
            if info:
                result.append(info)
            return True

        if not self.lib.EnumWindows(collect, None):
            raise OSError("Cannot enumerate desktop windows")
        return result

    def close(self):
        pass


def open_windows():
    if sys.platform == "win32":
        return WindowsDesktop()
    if sys.platform.startswith("linux"):
        return X11Windows()
    raise RuntimeError("Window inspection requires Windows or Linux X11")


def candidates(windows, all_windows=False):
    game_pids = {w.process_id for w in windows if w.process_id is not None and
                 "zandronum" in (w.class_name + " " + w.title).lower()}
    return [w for w in windows if w.visible and w.width >= 160 and w.height >= 120 and
            (all_windows or w.process_id in game_pids or
             "zandronum" in (w.class_name + " " + w.title).lower())]
