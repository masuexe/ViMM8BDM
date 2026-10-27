"""Check Windows ABI packet decoding without a desktop."""
import ctypes as C
import unittest
from vimm8bdm.rawinput import Header, Mouse, Keyboard, decode_packet


def packet(kind, data):
    return bytes(Header(kind=kind, size=C.sizeof(Header) + C.sizeof(data))) + bytes(data)


class RawInputTests(unittest.TestCase):
    def test_mouse_layout_and_signed_motion(self):
        self.assertEqual(C.sizeof(Mouse), 24)
        self.assertEqual(Mouse.dx.offset, 12)
        mouse = Mouse(dx=-17, dy=4)
        mouse.buttons.bits.flags = 0x402
        mouse.buttons.bits.data = 0xFF88
        result = decode_packet(packet(0, mouse))
        self.assertEqual((result['dx'], result['dy']), (-17, 4))
        self.assertEqual(result['wheel_delta'], -120)
        self.assertEqual(result['button_flags'], 0x402)
        self.assertFalse(result['absolute'])

    def test_absolute_mouse_and_extended_key_release(self):
        self.assertTrue(decode_packet(packet(0, Mouse(flags=1)))['absolute'])
        self.assertEqual(C.sizeof(Keyboard), 16)
        key = decode_packet(packet(1, Keyboard(scan=0x1D, flags=3, vk=0x11)))
        self.assertFalse(key['down'])
        self.assertEqual(key['extended'], 2)

    def test_malformed_packets_and_invalid_key(self):
        for raw in (b'', bytes(Header(kind=0, size=999)), bytes(Header(kind=1, size=1))):
            with self.assertRaises(ValueError):
                decode_packet(raw)
        self.assertIsNone(decode_packet(packet(1, Keyboard(vk=255))))
        with self.assertRaisesRegex(ValueError, 'overrun'):
            decode_packet(packet(1, Keyboard(scan=0xFF)))
