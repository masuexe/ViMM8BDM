"""Send normal relative mouse events through the Windows input API."""
import ctypes


class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_int32), ("dy", ctypes.c_int32),
                ("mouse_data", ctypes.c_uint32), ("flags", ctypes.c_uint32),
                ("time", ctypes.c_uint32), ("extra_info", ctypes.c_size_t)]


class InputData(ctypes.Union):
    _fields_ = [("mouse", MouseInput)]


class InputEvent(ctypes.Structure):
    _fields_ = [("kind", ctypes.c_uint32), ("data", InputData)]


class WindowsRelativeMouse:
    def __init__(self, sender=None):
        if sender is None:
            library = ctypes.WinDLL("user32", use_last_error=True)
            sender = library.SendInput
            sender.argtypes = [ctypes.c_uint32, ctypes.POINTER(InputEvent), ctypes.c_int]
            sender.restype = ctypes.c_uint32
        self.sender = sender

    def move(self, dx, dy):
        dx, dy = int(dx), int(dy)
        if not (dx or dy):
            return
        event = InputEvent(kind=0, data=InputData(mouse=MouseInput(dx=dx, dy=dy, flags=1)))
        if self.sender(1, ctypes.byref(event), ctypes.sizeof(event)) != 1:
            error = ctypes.get_last_error() if hasattr(ctypes, "get_last_error") else 0
            raise OSError(error, "Windows rejected the relative mouse event")
