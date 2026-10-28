"""Record human input and foreground game images without input injection."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import io
import json
from pathlib import Path
import queue
import struct
import sys
import threading
import time

MAGIC = b'VIMMREC1\n'
BUTTONS = ('left', 'right', 'middle', 'x1', 'x2')
CONTROL_KEYS = {0x77, 0x78, 0x1B}  # F8, F9, Esc


def alignment_phase(elapsed):
    if elapsed < 2:
        return 'ready', None, 0
    if elapsed < 20:
        return 'horizontal', 'x', 1
    if elapsed < 38:
        return 'vertical', 'y', 2
    return 'complete', None, 3


def sound_cue(count, errors):
    try:
        import winsound
        for _ in range(count):
            winsound.Beep(880, 100)
            time.sleep(.1)
    except Exception as exc:
        errors.append(str(exc))


class RecordingWriter:
    def __init__(self, directory):
        self.directory = directory
        self.queue = queue.Queue(maxsize=8192)
        self.frame_slots = threading.BoundedSemaphore(8)
        self.error = None
        self.dropped = 0
        self.frames = 0
        self.events = 0
        self.ready = threading.Event()
        self.max_frame_queue_wait_ms = 0
        self.max_frame_write_ms = 0
        self.thread = threading.Thread(target=self._run, name='recording-writer', daemon=True)
        self.thread.start()
        if not self.ready.wait(15):
            self.error = 'The recording writer did not become ready'
        if self.error:
            self.close()
            raise RuntimeError(self.error)

    def put(self, kind, value):
        if self.error:
            return False
        slot = kind == 'frame'
        if slot and not self.frame_slots.acquire(blocking=False):
            self.error = 'The frame writer queue is full'
            self.dropped += 1
            return False
        try:
            self.queue.put_nowait((kind, value))
            return True
        except queue.Full:
            if slot:
                self.frame_slots.release()
            self.dropped += 1
            self.error = 'The input event writer queue is full'
            return False

    def _run(self):
        try:
            from PIL import Image
            Image.new('RGB', (2, 2)).save(io.BytesIO(), format='PNG', compress_level=1)
            with ((self.directory / 'frames.vrec').open('xb') as archive,
                  (self.directory / 'frames.jsonl').open('x', encoding='utf-8') as index,
                  (self.directory / 'events.jsonl').open('x', encoding='utf-8') as events):
                archive.write(MAGIC)
                archive.flush()
                self.ready.set()
                while True:
                    item = self.queue.get()
                    if item is None:
                        break
                    kind, value = item
                    if kind == 'event':
                        events.write(json.dumps(value) + '\n')
                        self.events += 1
                    else:
                        frame, metadata = value
                        work_started = time.perf_counter_ns()
                        self.max_frame_queue_wait_ms = max(self.max_frame_queue_wait_ms,
                                                          (work_started - metadata['capture_finished_ns']) / 1e6)
                        try:
                            output = io.BytesIO()
                            image = Image.frombytes('RGB', (frame.width, frame.height), frame.bgra, 'raw', 'BGRX')
                            image.save(output, format='PNG', compress_level=1)
                            png = output.getvalue()
                            offset = archive.tell()
                            archive.write(struct.pack('<I', len(png)))
                            archive.write(png)
                            index.write(json.dumps({**metadata, 'offset': offset, 'png_bytes': len(png),
                                                    'width': frame.width, 'height': frame.height}) + '\n')
                            self.frames += 1
                            self.max_frame_write_ms = max(self.max_frame_write_ms,
                                                          (time.perf_counter_ns() - work_started) / 1e6)
                        finally:
                            self.frame_slots.release()
        except BaseException as exc:
            self.error = str(exc)
        finally:
            self.ready.set()

    def close(self):
        try:
            self.queue.put(None, timeout=5)
        except queue.Full:
            self.error = 'The recording writer did not accept shutdown'
        self.thread.join(timeout=10)
        if self.thread.is_alive():
            self.error = 'The recording writer did not finish'


class RecordingSession:
    def __init__(self, writer, focused, snapshot=lambda: {'keys': [], 'buttons': []}):
        self.writer, self.focused, self.snapshot = writer, focused, snapshot
        self.lock = threading.RLock()
        self.armed = False
        self.stopped = False
        self.segment = 0
        self.sequence = 0
        self.frames = 0
        self.keys = set()
        self.key_identities = set()
        self.snapshot_keys = set()
        self.buttons = set()
        self.controls_down = set()
        self.first_enable_ns = None
        self.current_enable_ns = None
        self.last_reason = 'initial_pause'

    def emit(self, event, at=None, **values):
        self.sequence += 1
        record = {'type': event, 'time_ns': time.perf_counter_ns() if at is None else at,
                  'sequence': self.sequence, 'segment': self.segment, **values}
        if not self.writer.put('event', record):
            self.armed = False
            self.stopped = True
            self.last_reason = 'writer_error'

    def pause(self, reason):
        # Clear only dataset state. This class never sends key or button releases.
        with self.lock:
            if self.armed:
                self.emit('segment_end', reason=reason, held_keys=sorted(self.keys),
                          held_buttons=sorted(self.buttons))
            self.armed = False
            self.keys.clear()
            self.key_identities.clear()
            self.snapshot_keys.clear()
            self.buttons.clear()
            self.last_reason = reason

    def poll(self):
        with self.lock:
            if self.armed and not self.focused():
                self.pause('focus_lost')

    def on_input(self, event, received_ns):
        with self.lock:
            if self.stopped:
                return
            if event['type'] == 'key' and event['vk'] in CONTROL_KEYS:
                key = event['vk']
                if not event['down']:
                    self.controls_down.discard(key)
                    return
                if key in self.controls_down:
                    return
                self.controls_down.add(key)
                if key == 0x78:
                    self.pause('quit_key')
                    self.stopped = True
                elif key == 0x1B:
                    self.pause('escape_pause')
                elif self.armed:
                    self.pause('user_pause')
                elif self.focused():
                    state = self.snapshot()
                    self.keys = set(state['keys'])
                    self.snapshot_keys = self.keys.copy()
                    self.key_identities.clear()
                    self.buttons = set(state['buttons'])
                    self.segment += 1
                    self.armed = True
                    self.current_enable_ns = received_ns
                    if self.first_enable_ns is None:
                        self.first_enable_ns = received_ns
                    self.emit('segment_start', received_ns, snapshot_ns=time.perf_counter_ns(),
                              held_keys=sorted(self.keys), held_buttons=sorted(self.buttons))
                    self.last_reason = 'recording'
                print('Recorder:', self.last_reason, flush=True)
                return
            if not self.armed:
                return
            if not self.focused():
                self.pause('focus_lost')
                return
            if event['type'] == 'key':
                vk = event['vk']
                identity = (vk, event['scan'], event['extended'])
                repeated = event['down'] and identity in self.key_identities
                (self.key_identities.add if event['down'] else self.key_identities.discard)(identity)
                self.snapshot_keys.discard(vk)
                self.keys = self.snapshot_keys | {key[0] for key in self.key_identities}
                if not repeated:
                    self.emit('key', received_ns, **{k: v for k, v in event.items() if k != 'type'})
            elif event['type'] == 'mouse':
                if event['absolute']:
                    self.pause('unsupported_absolute_mouse')
                    return
                flags = event['button_flags']
                for number, button in enumerate(BUTTONS):
                    if flags & (1 << (number * 2)):
                        self.buttons.add(button)
                        self.emit('button', received_ns, button=button, down=True)
                    if flags & (2 << (number * 2)):
                        self.buttons.discard(button)
                        self.emit('button', received_ns, button=button, down=False)
                if event['dx'] or event['dy']:
                    self.emit('motion', received_ns, dx=event['dx'], dy=event['dy'])
                for mask, axis in ((0x400, 'vertical'), (0x800, 'horizontal')):
                    if flags & mask:
                        self.emit('wheel', received_ns, axis=axis, delta=event['wheel_delta'])

    def capture_state(self):
        with self.lock:
            self.poll()
            if not self.armed:
                return None
            return {'segment': self.segment, 'sequence_at_capture_start': self.sequence,
                    'held_keys_at_capture_start': sorted(self.keys),
                    'held_buttons_at_capture_start': sorted(self.buttons)}

    def accept_frame(self, frame, finished_ns, state):
        with self.lock:
            self.poll()
            if not self.armed or state['segment'] != self.segment:
                return False
            metadata = {**state, 'frame': self.frames, 'capture_started_ns': frame.captured_ns,
                        'capture_finished_ns': finished_ns, 'sequence_at_capture_end': self.sequence}
            if not self.writer.put('frame', (frame, metadata)):
                self.pause('writer_error')
                self.stopped = True
                return False
            self.frames += 1
            return True


def windows_snapshot():
    user = ctypes.WinDLL('user32', use_last_error=True)
    user.GetAsyncKeyState.argtypes = [ctypes.c_int]
    user.GetAsyncKeyState.restype = ctypes.c_int16
    mouse = {1: 'left', 2: 'right', 4: 'middle', 5: 'x1', 6: 'x2'}
    pressed = {vk for vk in range(1, 255) if user.GetAsyncKeyState(vk) & 0x8000}
    # Raw keyboard events use generic modifier VK values with scan/extended flags.
    keys = pressed - set(mouse) - CONTROL_KEYS - set(range(0xA0, 0xA6))
    return {'keys': sorted(keys), 'buttons': sorted(mouse[vk] for vk in pressed if vk in mouse)}


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def run(args):
    from .control import blank_frame
    from .desktop import Focus, WindowsWindowCapture
    from .rawinput import RawInput
    directory = args.run_dir
    directory.mkdir(parents=True, exist_ok=False)
    manifest = {'format': 'vimm8bdm-human-v1', 'clock': 'perf_counter_ns',
                'input_time_meaning': 'Raw Input callback receipt, not hardware event time',
                'target_hz': args.hz, 'engine_version': args.engine_version,
                'character': args.character, 'character_source': getattr(args, 'character_source', 'operator_report_at_start'),
                'class_selection': args.class_selection,
                'character_stable': args.class_selection == 'fixed', 'map': args.map,
                'purpose': args.purpose, 'task': args.task, 'training_ready': False, 'protocol': args.protocol,
                'mouse_units': 'Windows Raw Input relative counts; actuator equivalence unverified',
                'keyboard_identity': ['vk', 'scan', 'extended'],
                'initial_snapshot_has_scan_codes': False,
                'input_sent': False, 'menu_detection': 'Esc pauses; other menus require manual pause',
                'assets': [], 'settings_verified': False}
    from .winmouse import query_mouse_settings
    try:
        manifest['windows_pointer_settings'] = query_mouse_settings()
    except OSError as exc:
        manifest['windows_pointer_settings_error'] = str(exc)
    if args.launch_record:
        record = json.loads(args.launch_record.read_text())
        command = record['arguments']
        iw = command.index('-iwad')
        files = command.index('-file')
        for path in [command[0], command[iw + 1], *command[files + 1:files + 4]]:
            manifest['assets'].append({'name': Path(path).name, 'sha256': file_digest(path)})
        # The active ini can contain private paths. Keep its copy only with this local dataset.
        for flag, target in (('-config', 'game-settings.ini'), ('+exec', 'input-settings.cfg')):
            if flag in command:
                source = Path(command[command.index(flag) + 1])
                if source.is_file():
                    with (directory / target).open('xb') as out:
                        out.write(source.read_bytes())
    with (directory / 'session.json').open('x', encoding='utf-8') as output:
        json.dump(manifest, output, indent=2)
    focus = Focus(args.window_id)
    capture = raw = writer = session = None
    reason = 'error'
    protocol_segment = protocol_phase_name = None
    cue_errors = []
    cue_threads = []
    try:
        capture = WindowsWindowCapture(args.window_id)
        writer = RecordingWriter(directory)
        session = RecordingSession(writer, focus, windows_snapshot)
        raw = RawInput(session.on_input)
        print('Recorder ready. F8: record/pause. Esc: pause. F9: finish.', flush=True)
        print('Initial waiting has no timeout. The recorder sends no game input.', flush=True)
        while not session.stopped:
            begin = time.perf_counter_ns()
            if raw.error:
                raise RuntimeError('The Raw Input observer failed') from raw.error
            if writer.error:
                raise RuntimeError(writer.error)
            session.poll()
            if session.first_enable_ns is not None and (begin - session.first_enable_ns) / 1e9 >= args.seconds:
                reason = 'duration_limit'
                break
            protocol_done = False
            if args.protocol == 'mouse_alignment':
                with session.lock:
                    if session.armed:
                        phase, axis, cue = alignment_phase((begin - session.current_enable_ns) / 1e9)
                        if protocol_segment != session.segment:
                            protocol_segment = session.segment
                            protocol_phase_name = None
                        if phase != protocol_phase_name:
                            protocol_phase_name = phase
                            session.emit('protocol_phase', phase=phase, axis=axis, cue_count=cue)
                            print('Mouse alignment:', phase, flush=True)
                            if cue:
                                thread = threading.Thread(target=sound_cue, args=(cue, cue_errors), daemon=True)
                                cue_threads.append(thread)
                                thread.start()
                        if phase == 'complete':
                            session.pause('protocol_complete')
                            reason = 'protocol_complete'
                            protocol_done = True
            if protocol_done:
                break
            state = session.capture_state()
            if state is not None:
                frame = capture.grab()
                finished = time.perf_counter_ns()
                if blank_frame(frame):
                    session.pause('blank_frame')
                else:
                    session.accept_frame(frame, finished, state)
            time.sleep(max(0, 1 / args.hz - (time.perf_counter_ns() - begin) / 1e9))
        else:
            reason = session.last_reason
    except KeyboardInterrupt:
        reason = 'keyboard_interrupt'
    finally:
        cleanup_errors = []
        if session:
            with session.lock:
                session.pause('shutdown')
                session.stopped = True
        for resource in (raw, capture, focus):
            if resource:
                try:
                    resource.close()
                except Exception as exc:
                    cleanup_errors.append(str(exc))
        if session:
            session.emit('recording_end', reason=reason)
        if writer:
            writer.close()
        for thread in cue_threads:
            thread.join(timeout=1)
        summary = {'exit_reason': reason, 'frames': writer.frames if writer else 0,
                   'events': writer.events if writer else 0,
                   'dropped_items': writer.dropped if writer else 0,
                   'writer_error': writer.error if writer else 'writer_not_started',
                   'max_frame_queue_wait_ms': writer.max_frame_queue_wait_ms if writer else None,
                   'max_frame_write_ms': writer.max_frame_write_ms if writer else None,
                   'complete': reason != 'error' and writer is not None and not writer.error and not cleanup_errors,
                   'cleanup_errors': cleanup_errors, 'cue_errors': cue_errors,
                   'segments': session.segment if session else 0, 'input_sent': False}
        with (directory / 'summary.json').open('x', encoding='utf-8') as output:
            json.dump(summary, output, indent=2)
        print(json.dumps(summary, indent=2), flush=True)
        print('Recorder stopped. F8 is inactive.', flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description='Record human gameplay images and Raw Input')
    parser.add_argument('--window-id', type=lambda value: int(value, 0), required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=30)
    parser.add_argument('--hz', type=float, default=20)
    parser.add_argument('--purpose', choices=('interface_check', 'demonstration'), default='interface_check')
    parser.add_argument('--task', choices=('navigation', 'aim', 'combat', 'class_skill'))
    parser.add_argument('--protocol', choices=('free', 'mouse_alignment'), default='free')
    parser.add_argument('--character', default='unknown')
    parser.add_argument('--character-source', choices=('operator_report_at_start', 'launch_configuration', 'unknown'),
                        default='operator_report_at_start')
    parser.add_argument('--class-selection', choices=('random', 'fixed', 'unknown'), default='unknown')
    parser.add_argument('--map', default='unknown')
    parser.add_argument('--engine-version', required=True)
    parser.add_argument('--launch-record', type=Path)
    args = parser.parse_args()
    import math
    if sys.platform != 'win32':
        parser.error('Human recording requires Windows Python')
    if not math.isfinite(args.seconds) or not 0 < args.seconds <= 3600:
        parser.error('The duration must be finite and between zero and 3600 seconds')
    if not math.isfinite(args.hz) or not 1 <= args.hz <= 60:
        parser.error('The capture frequency must be between 1 and 60 Hz')
    if args.protocol == 'mouse_alignment' and args.seconds < 38:
        parser.error('The mouse alignment protocol requires a duration limit of at least 38 seconds')
    if args.character_source == 'launch_configuration' and args.launch_record is None:
        parser.error('A configured character requires a launch record')
    if args.purpose == 'demonstration' and (args.protocol != 'free' or args.class_selection != 'fixed'
            or args.character.strip().lower() in ('', 'unknown') or args.map.strip().lower() in ('', 'unknown') or args.task is None):
        parser.error('A demonstration requires a free protocol, fixed character, known map, and task')
    run(args)


if __name__ == '__main__':
    main()
