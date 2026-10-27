"""Receive Windows Raw Input in a separate observer process. Never send input."""
import ctypes as C
import sys
import threading
import time


class Header(C.Structure):
    _fields_ = [('kind', C.c_uint32), ('size', C.c_uint32),
                ('device', C.c_void_p), ('wparam', C.c_size_t)]


class ButtonBits(C.Structure):
    _fields_ = [('flags', C.c_uint16), ('data', C.c_uint16)]


class Buttons(C.Union):
    _fields_ = [('value', C.c_uint32), ('bits', ButtonBits)]


class Mouse(C.Structure):
    _fields_ = [('flags', C.c_uint16), ('buttons', Buttons), ('raw_buttons', C.c_uint32),
                ('dx', C.c_int32), ('dy', C.c_int32), ('extra', C.c_uint32)]


class Keyboard(C.Structure):
    _fields_ = [('scan', C.c_uint16), ('flags', C.c_uint16), ('reserved', C.c_uint16),
                ('vk', C.c_uint16), ('message', C.c_uint32), ('extra', C.c_uint32)]


def decode_packet(packet):
    if len(packet) < C.sizeof(Header):
        raise ValueError('The Raw Input header is incomplete')
    header = Header.from_buffer_copy(packet)
    structure = {0: Mouse, 1: Keyboard}.get(header.kind)
    if structure is None:
        return None
    needed = C.sizeof(Header) + C.sizeof(structure)
    if header.size < needed or header.size > len(packet):
        raise ValueError('The Raw Input packet has an invalid size')
    data = structure.from_buffer_copy(packet[C.sizeof(Header):])
    if header.kind == 1:
        if data.scan == 0xFF:
            raise ValueError('The keyboard reported an input overrun')
        if data.vk == 255:  # Invalid virtual key, documented by RAWKEYBOARD.
            return None
        return {'type': 'key', 'vk': data.vk, 'scan': data.scan,
                'extended': data.flags & 6, 'down': not bool(data.flags & 1)}
    return {'type': 'mouse', 'absolute': bool(data.flags & 1), 'dx': data.dx, 'dy': data.dy,
            'button_flags': data.buttons.bits.flags,
            'wheel_delta': C.c_int16(data.buttons.bits.data).value}


class RawInput:
    def __init__(self, callback):
        if sys.platform != 'win32':
            raise RuntimeError('Raw Input requires Windows Python')
        self.callback = callback
        self.error = None
        self.stop = threading.Event()
        self.ready = threading.Event()
        self.thread = threading.Thread(target=self._run, name='raw-input', daemon=True)
        self.thread.start()
        if not self.ready.wait(15):
            self.close()
            raise RuntimeError('The Raw Input observer did not become ready')
        if self.error:
            self.close()
            raise RuntimeError('The Raw Input observer failed') from self.error

    def _run(self):
        window = atom = None
        try:
            user = C.WinDLL('user32', use_last_error=True)
            kernel = C.WinDLL('kernel32', use_last_error=True)
            callback_type = C.WINFUNCTYPE(C.c_ssize_t, C.c_void_p, C.c_uint, C.c_size_t, C.c_ssize_t)

            class WindowClass(C.Structure):
                _fields_ = [('size', C.c_uint32), ('style', C.c_uint32), ('proc', callback_type),
                            ('cls_extra', C.c_int), ('win_extra', C.c_int), ('instance', C.c_void_p),
                            ('icon', C.c_void_p), ('cursor', C.c_void_p), ('background', C.c_void_p),
                            ('menu', C.c_wchar_p), ('name', C.c_wchar_p), ('small_icon', C.c_void_p)]

            class Device(C.Structure):
                _fields_ = [('page', C.c_uint16), ('usage', C.c_uint16),
                            ('flags', C.c_uint32), ('target', C.c_void_p)]

            class Point(C.Structure):
                _fields_ = [('x', C.c_int32), ('y', C.c_int32)]

            class Message(C.Structure):
                _fields_ = [('window', C.c_void_p), ('message', C.c_uint32), ('wparam', C.c_size_t),
                            ('lparam', C.c_ssize_t), ('time', C.c_uint32), ('point', Point), ('private', C.c_uint32)]

            kernel.GetModuleHandleW.argtypes = [C.c_wchar_p]
            kernel.GetModuleHandleW.restype = C.c_void_p
            user.DefWindowProcW.argtypes = [C.c_void_p, C.c_uint, C.c_size_t, C.c_ssize_t]
            user.DefWindowProcW.restype = C.c_ssize_t
            user.GetRawInputData.argtypes = [C.c_void_p, C.c_uint, C.c_void_p, C.POINTER(C.c_uint32), C.c_uint]
            user.GetRawInputData.restype = C.c_uint
            user.RegisterClassExW.argtypes = [C.POINTER(WindowClass)]
            user.RegisterClassExW.restype = C.c_uint16
            user.CreateWindowExW.argtypes = [C.c_uint32, C.c_wchar_p, C.c_wchar_p, C.c_uint32,
                                            C.c_int, C.c_int, C.c_int, C.c_int,
                                            C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p]
            user.CreateWindowExW.restype = C.c_void_p
            user.RegisterRawInputDevices.argtypes = [C.POINTER(Device), C.c_uint, C.c_uint]
            user.RegisterRawInputDevices.restype = C.c_int
            user.PeekMessageW.argtypes = [C.POINTER(Message), C.c_void_p, C.c_uint, C.c_uint, C.c_uint]
            user.PeekMessageW.restype = C.c_int
            user.DispatchMessageW.argtypes = [C.POINTER(Message)]
            user.DispatchMessageW.restype = C.c_ssize_t
            user.DestroyWindow.argtypes = [C.c_void_p]
            user.DestroyWindow.restype = C.c_int
            user.UnregisterClassW.argtypes = [C.c_wchar_p, C.c_void_p]
            user.UnregisterClassW.restype = C.c_int

            @callback_type
            def procedure(hwnd, message, wparam, lparam):
                if message == 0xFF:  # WM_INPUT
                    try:
                        received = time.perf_counter_ns()
                        size = C.c_uint32()
                        result = user.GetRawInputData(lparam, 0x10000003, None, C.byref(size), C.sizeof(Header))
                        if result == 0xFFFFFFFF or not C.sizeof(Header) <= size.value <= 65536:
                            raise OSError('Cannot obtain the Raw Input packet size')
                        buffer = C.create_string_buffer(size.value)
                        result = user.GetRawInputData(lparam, 0x10000003, buffer, C.byref(size), C.sizeof(Header))
                        if result == 0xFFFFFFFF or result != size.value:
                            raise OSError('Cannot read a complete Raw Input packet')
                        event = decode_packet(buffer.raw[:size.value])
                        if event is not None:
                            self.callback(event, received)
                    except BaseException as exc:
                        self.error = exc
                        self.stop.set()
                return user.DefWindowProcW(hwnd, message, wparam, lparam)

            instance = kernel.GetModuleHandleW(None)
            name = 'ViMM8BDMRawInput' + str(threading.get_ident())
            definition = WindowClass(size=C.sizeof(WindowClass), proc=procedure, instance=instance, name=name)
            atom = user.RegisterClassExW(C.byref(definition))
            if not atom:
                raise C.WinError(C.get_last_error())
            window = user.CreateWindowExW(0, name, '', 0, 0, 0, 0, 0, C.c_void_p(-3), None, instance, None)
            if not window:
                raise C.WinError(C.get_last_error())
            # INPUTSINK permits observation while the game remains foreground.
            # Do not set NOLEGACY, CAPTUREMOUSE, or any exclusive-input flags.
            devices = (Device * 2)(Device(1, 2, 0x100, window), Device(1, 6, 0x100, window))
            if not user.RegisterRawInputDevices(devices, 2, C.sizeof(Device)):
                raise C.WinError(C.get_last_error())
            self.ready.set()
            message = Message()
            while not self.stop.is_set():
                while user.PeekMessageW(C.byref(message), None, 0, 0, 1):
                    user.DispatchMessageW(C.byref(message))
                    if self.stop.is_set():
                        break
                self.stop.wait(.002)
        except BaseException as exc:
            self.error = exc
        finally:
            if window:
                removed = (Device * 2)(Device(1, 2, 1, None), Device(1, 6, 1, None))
                user.RegisterRawInputDevices(removed, 2, C.sizeof(Device))
                user.DestroyWindow(window)
            if atom:
                user.UnregisterClassW(name, instance)
            self.ready.set()

    def close(self):
        self.stop.set()
        self.thread.join(timeout=5)
        if self.thread.is_alive():
            raise RuntimeError('The Raw Input observer did not stop')
