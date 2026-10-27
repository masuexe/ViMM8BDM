"""Run an attended timed-input check on the selected Windows game window."""
import argparse
import json
from pathlib import Path
import sys
import time

from .control import Action, blank_frame
from .timed_actions import TimedPacket, TimedWindow


def check_windows():
    idle = TimedWindow((TimedPacket(0, Action()), TimedPacket(50_000_000, Action())))
    result = [('ready', idle)] * 10
    for name, key, dx, dy in [('right_tap', 'd', 0, 0), ('left_tap', 'a', 0, 0),
                              ('yaw_right', None, 48, 0), ('yaw_left', None, -48, 0),
                              ('pitch_down', None, 0, 32), ('pitch_up', None, 0, -32)]:
        keys = frozenset({key}) if key else frozenset()
        # Two mouse packets exercise motion beyond the old per-call limit.
        window = TimedWindow((TimedPacket(0, Action()),
                              TimedPacket(5_000_000, Action(keys)),
                              TimedPacket(10_000_000, Action(keys, dx / 2, dy / 2)),
                              TimedPacket(15_000_000, Action(keys, dx / 2, dy / 2)),
                              TimedPacket(45_000_000, Action()),
                              TimedPacket(50_000_000, Action())))
        result.append((name, window.validate()))
        result.extend([('rest', idle)] * 6)
    return result


def collect(window_id, directory):
    from .desktop import WindowsWindowCapture
    from .input_worker import InputWorker

    directory.mkdir(parents=True, exist_ok=False)
    sequence = check_windows()
    index = frames = accepted = 0
    started = generation = None
    exit_reason = 'complete'
    capture = WindowsWindowCapture(window_id)
    try:
        with InputWorker(window_id, trace_path=directory / 'worker.local.jsonl') as worker:
            print('Timed input ready. Select the game. F8: enable. Esc: pause. F9: exit.', flush=True)
            print('Keep the real mouse still. The check uses short strafes and turns. It sends no attack.', flush=True)
            with (directory / 'capture.local.jsonl').open('x', encoding='utf-8') as log:
                while index < len(sequence) or worker.snapshot.get('timed_pending'):
                    begin = time.perf_counter()
                    worker.refresh()
                    if worker.stopped:
                        exit_reason = worker.reason
                        break
                    if started is not None and not worker.armed:
                        exit_reason = worker.reason
                        break
                    if not worker.armed:
                        time.sleep(.02)
                        continue
                    if started is None:
                        started, generation = begin, worker.generation
                    if begin - started > 30:
                        exit_reason = 'duration_limit'
                        break
                    frame = capture.grab()
                    frames += 1
                    if blank_frame(frame):
                        worker.pause('blank_frame')
                        exit_reason = worker.reason
                        break
                    applied = False
                    if not worker.snapshot.get('timed_pending') and index < len(sequence):
                        name, window = sequence[index]
                        applied = worker.submit_timed(window, frame.captured_ns, generation)
                        if applied:
                            accepted += 1
                            index += 1
                            if name not in ('ready', 'rest'):
                                print('Phase:', name, flush=True)
                    log.write(json.dumps({'captured_ns': frame.captured_ns, 'next_window': index,
                                          'accepted': applied}) + '\n')
                    time.sleep(max(0, .02 - (time.perf_counter() - begin)))
            worker.pause(exit_reason)
            worker.refresh()
            final_state = worker.snapshot.copy()
    finally:
        capture.close()
    summary = {'exit_reason': exit_reason, 'requested_windows': len(sequence),
               'accepted_windows': accepted, 'frames': frames,
               'elapsed_after_enable_s': None if started is None else time.perf_counter() - started,
               'final_state': final_state, 'gameplay_success_verified': False,
               'visible_response_latency_measured': False}
    with (directory / 'summary.local.json').open('x', encoding='utf-8') as output:
        json.dump(summary, output, indent=2)
    print(json.dumps(summary, indent=2), flush=True)
    print('Timed input stopped. F8 is inactive. Start a new run for another check.', flush=True)


def main():
    parser = argparse.ArgumentParser(description='Check short taps and bounded mouse packets')
    parser.add_argument('--window-id', type=lambda value: int(value, 0), required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != 'win32':
        parser.error('The live check requires Windows Python')
    collect(args.window_id, args.run_dir)


if __name__ == '__main__':
    main()
