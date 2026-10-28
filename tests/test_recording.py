"""Check recording boundaries, event retention, and archive integrity."""
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
from vimm8bdm.control import Frame
from vimm8bdm.recording import RecordingSession, RecordingWriter, MAGIC, alignment_phase


class MemoryWriter:
    def __init__(self):
        self.items = []
        self.fail = False

    def put(self, kind, data):
        if self.fail:
            return False
        self.items.append((kind, data))
        return True


class RecordingTests(unittest.TestCase):
    def setUp(self):
        self.writer = MemoryWriter()
        self.focused = True
        self.session = RecordingSession(self.writer, lambda: self.focused)

    def key(self, vk, down=True, at=10):
        self.session.on_input({'type': 'key', 'vk': vk, 'scan': vk, 'extended': 0, 'down': down}, at)

    def toggle(self):
        self.key(0x77)
        self.key(0x77, False)

    def test_character_source_distinguishes_configured_identity_from_operator_report(self):
        from vimm8bdm.recording import main
        base = ['recording', '--window-id', '0x123', '--run-dir', 'unused-session', '--engine-version', 'example']
        for extra, expected in (([], 'operator_report_at_start'),
                                (['--character-source', 'launch_configuration', '--launch-record', 'launch.json'],
                                 'launch_configuration')):
            with patch('sys.argv', base + extra), patch('sys.platform', 'win32'), patch('vimm8bdm.recording.run') as run:
                main()
                self.assertEqual(run.call_args.args[0].character_source, expected)

    def test_configured_character_requires_a_launch_record(self):
        from vimm8bdm.recording import main
        args = ['recording', '--window-id', '0x123', '--run-dir', 'unused-session', '--engine-version', 'example',
                '--character-source', 'launch_configuration']
        with patch('sys.argv', args), patch('sys.platform', 'win32'), patch('sys.stderr', io.StringIO()), self.assertRaises(SystemExit):
            main()

    def test_alignment_phase_cues_and_end_are_bounded(self):
        self.assertEqual(alignment_phase(0), ('ready', None, 0))
        self.assertEqual(alignment_phase(2), ('horizontal', 'x', 1))
        self.assertEqual(alignment_phase(19.99), ('horizontal', 'x', 1))
        self.assertEqual(alignment_phase(20), ('vertical', 'y', 2))
        self.assertEqual(alignment_phase(38), ('complete', None, 3))

    def test_press_release_between_frames_are_both_retained(self):
        self.toggle()
        self.key(0x57, at=20)
        self.key(0x57, at=21)  # Hardware repeat does not add another press.
        self.key(0x57, False, at=22)
        keys = [e for kind, e in self.writer.items if kind == 'event' and e['type'] == 'key']
        self.assertEqual([(e['time_ns'], e['down']) for e in keys], [(20, True), (22, False)])

    def test_two_physical_modifiers_keep_independent_transitions(self):
        self.toggle()
        left = {'type': 'key', 'vk': 16, 'scan': 42, 'extended': 0, 'down': True}
        right = {**left, 'scan': 54}
        self.session.on_input(left, 20)
        self.session.on_input(right, 21)
        self.session.on_input({**left, 'down': False}, 22)
        self.assertIn(16, self.session.keys)
        self.session.on_input({**right, 'down': False}, 23)
        self.assertNotIn(16, self.session.keys)
        transitions = [e for _, e in self.writer.items if e['type'] == 'key']
        self.assertEqual([(e['scan'], e['down']) for e in transitions],
                         [(42, True), (54, True), (42, False), (54, False)])

    def test_background_keys_are_not_written_and_return_does_not_resume(self):
        self.toggle()
        self.key(0x57)
        self.focused = False
        self.key(0x41)
        self.assertFalse(self.session.armed)
        self.focused = True
        self.key(0x44)
        recorded = [e for kind, e in self.writer.items if kind == 'event' and e['type'] == 'key']
        self.assertEqual([e['vk'] for e in recorded], [0x57])
        self.assertFalse(self.session.armed)
        self.toggle()
        self.assertTrue(self.session.armed)
        self.assertEqual(self.session.segment, 2)

    def test_capture_does_not_cross_pause_or_segment_boundaries(self):
        self.toggle()
        state = self.session.capture_state()
        self.toggle()
        self.toggle()
        frame = Frame(2, 1, bytes((0, 0, 255, 255)) * 2, 100)
        self.assertFalse(self.session.accept_frame(frame, 110, state))
        self.assertFalse(any(kind == 'frame' for kind, _ in self.writer.items))

    def test_mouse_buttons_wheel_and_absolute_rejection(self):
        self.toggle()
        event = {'type': 'mouse', 'absolute': False, 'dx': -9, 'dy': 2,
                 'button_flags': 0x401, 'wheel_delta': 120}
        self.session.on_input(event, 20)
        self.assertEqual(self.session.buttons, {'left'})
        events = [e for kind, e in self.writer.items if kind == 'event']
        self.assertEqual([e['type'] for e in events], ['segment_start', 'button', 'motion', 'wheel'])
        self.session.on_input({**event, 'absolute': True}, 21)
        self.assertFalse(self.session.armed)
        self.assertEqual(self.session.last_reason, 'unsupported_absolute_mouse')

    def test_esc_and_f9_do_not_become_training_keys(self):
        self.toggle()
        self.key(0x1B)
        self.assertFalse(self.session.armed)
        self.key(0x78)
        self.assertTrue(self.session.stopped)
        self.assertFalse(any(e.get('type') == 'key' for _, e in self.writer.items))

    def test_writer_failure_stops_recording(self):
        self.toggle()
        self.writer.fail = True
        self.key(0x57)
        self.assertTrue(self.session.stopped)
        self.assertFalse(self.session.armed)

    def test_initial_held_inputs_are_preserved(self):
        self.session.snapshot = lambda: {'keys': [0x57], 'buttons': ['left']}
        self.toggle()
        self.assertEqual(self.session.keys, {0x57})
        self.assertEqual(self.writer.items[0][1]['held_buttons'], ['left'])

    def test_writer_constructor_waits_for_initialization(self):
        try:
            import PIL
        except ImportError:
            self.skipTest('Pillow is required for the writer test')
        import threading
        entered, release = threading.Event(), threading.Event()
        result = []
        class SlowWriter(RecordingWriter):
            def _run(self):
                entered.set()
                release.wait(3)
                super()._run()
        with tempfile.TemporaryDirectory() as directory:
            def create():
                result.append(SlowWriter(Path(directory)))
            creator = threading.Thread(target=create)
            creator.start()
            try:
                self.assertTrue(entered.wait(3))
                self.assertEqual(result, [])
            finally:
                release.set()
                creator.join(timeout=5)
            self.assertEqual(len(result), 1)
            self.assertTrue(result[0].ready.is_set())
            result[0].close()

    def test_archive_round_trip_and_exclusive_creation(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest('Pillow is required for the archive test')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            writer = RecordingWriter(path)
            frame = Frame(2, 1, bytes((0, 0, 255, 255, 0, 255, 0, 255)), 100)
            writer.put('event', {'type': 'recording_end', 'time_ns': 151, 'sequence': 1, 'segment': 1})
            writer.put('frame', (frame, {'frame': 0, 'capture_started_ns': 100,
                                         'capture_finished_ns': 150, 'segment': 1}))
            writer.close()
            self.assertIsNone(writer.error)
            index = json.loads((path / 'frames.jsonl').read_text())
            with (path / 'frames.vrec').open('rb') as archive:
                self.assertEqual(archive.read(len(MAGIC)), MAGIC)
                self.assertEqual(archive.tell(), index['offset'])
                length = struct.unpack('<I', archive.read(4))[0]
                self.assertEqual(length, index['png_bytes'])
                image = Image.open(io.BytesIO(archive.read(length)))
                self.assertEqual([image.getpixel((x, 0)) for x in range(2)], [(255, 0, 0), (0, 255, 0)])
            original = (path / 'frames.vrec').read_bytes()
            with self.assertRaises(RuntimeError):
                RecordingWriter(path)
            self.assertEqual((path / 'frames.vrec').read_bytes(), original)
            (path / 'session.json').write_text(json.dumps({'input_sent': False, 'training_ready': False}))
            (path / 'summary.json').write_text(json.dumps({'complete': True, 'dropped_items': 0,
                                                          'frames': 1, 'events': 1, 'segments': 1}))
            from vimm8bdm.recording_review import review
            result = review(path, path / 'preview.html')
            self.assertEqual(result['archive_integrity'], 'pass')
            self.assertIn('data:image/png', (path / 'preview.html').read_text())
            with (path / 'frames.vrec').open('ab') as archive:
                archive.write(b'unindexed')
            with self.assertRaisesRegex(ValueError, 'unindexed'):
                review(path)
