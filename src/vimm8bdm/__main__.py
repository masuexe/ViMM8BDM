from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import platform
import statistics
import sys
import threading
import time
from pathlib import Path

from .control import InputLease, PixelProbe, Frame, synthetic_frame, ControlKeys, InputCheckPolicy, blank_frame, Action


class DrySink:
    def __init__(self):
        self.last = None

    def apply(self, action):
        self.last = action

    def release(self):
        self.last = None


def doctor():
    result = {"platform": platform.platform(), "python": platform.python_version(),
              "timing_clock": "perf_counter", "timing_resolution_ns": time.get_clock_info("perf_counter").resolution * 1e9,
              "display": os.getenv("DISPLAY"), "wayland_display": os.getenv("WAYLAND_DISPLAY"),
              "dependencies": {name: bool(importlib.util.find_spec(name))
                               for name in ("mss", "pynput", "torch", "vizdoom")}}
    try:
        from .desktop import Focus
        focus = Focus(0)
        try:
            result["foreground_window_id"] = hex(focus.current())
            result["desktop_access"] = True
        finally:
            focus.close()
    except Exception as exc:
        result["desktop_access"] = False
        result["desktop_error"] = str(exc)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def list_windows(args):
    from .windows import open_windows, candidates
    windows = open_windows()
    try:
        result = candidates(windows.enumerate(), args.all_windows)
        print(json.dumps({"input_sent": False,
                          "windows": [w.diagnostic_view(args.include_window_titles) for w in result]}, indent=2))
    finally:
        windows.close()


def check_window(args):
    from .desktop import Focus
    if args.window_id is None:
        raise ValueError("--check-window requires --window-id")
    focus = Focus(args.window_id)
    matched = samples = 0
    observed = set()
    try:
        started = time.perf_counter()
        for _ in range(args.frames):
            if time.perf_counter() - started >= args.seconds:
                break
            matched += focus()
            observed.add(hex(focus.current()))
            samples += 1
            time.sleep(1 / args.hz)
        print(json.dumps({"input_sent": False, "samples": samples, "matched_samples": matched,
                          "observed_focus_ids": sorted(observed),
                          "selected_window": focus.identity.diagnostic_view(args.include_window_titles)}, indent=2))
    finally:
        focus.close()


def summarize(rows, mode, policy_name, exit_reason):
    summary = {"mode": mode, "frames": len(rows), "units": "milliseconds",
               "timing_clock": "perf_counter",
               "timing_resolution_ns": time.get_clock_info("perf_counter").resolution * 1e9,
               "policy": policy_name,
               "exit_reason": exit_reason,
               "neural_inference_measured": policy_name == "continuousdoom",
               "accepted_actions": sum(row["applied"] for row in rows),
               "inactive_or_rejected_frames": sum(not row["applied"] for row in rows),
               "input_stage": "lease_and_sink" if any(row["applied"] for row in rows)
                              else "lease_gate_without_input_dispatch"}
    for key in ("capture_ms", "policy_ms", "input_ms", "loop_ms"):
        vals = sorted(row[key] for row in rows)
        if vals:
            summary[key] = {"p50": statistics.median(vals),
                            "p95": vals[max(0, math.ceil(len(vals) * .95) - 1)],
                            "max": vals[-1]}
    accepted = [row for row in rows if row["applied"]]
    summary["accepted_action_latency"] = {}
    for key in ("capture_ms", "policy_ms", "input_ms", "loop_ms"):
        vals = sorted(row[key] for row in accepted)
        if vals:
            summary["accepted_action_latency"][key] = {
                "p50": statistics.median(vals), "p95": vals[math.ceil(len(vals) * .95) - 1], "max": vals[-1]}
    print(json.dumps(summary, indent=2))


def run(args):
    if args.live and (args.window_id is None or (args.capture == "desktop" and args.region is None)):
        raise ValueError("--live requires --window-id. Desktop capture also requires --region")
    focus = sink = listener = capture = None
    stop = threading.Event()
    rows = []
    output = None
    lease = None
    watchdog = None
    exit_reason = "error"
    try:
        if args.output:
            output = Path(args.output).open("x", encoding="utf-8")
        if args.live:
            from .desktop import Focus, DesktopSink
            focus = Focus(args.window_id)
            sink = DesktopSink()
            import mss
            from pynput import keyboard
            if args.capture in ("x11-window", "window"):
                from .desktop import X11WindowCapture, WindowsWindowCapture
                capture = (WindowsWindowCapture(args.window_id) if args.capture == "window" and sys.platform == "win32"
                           else X11WindowCapture(args.window_id))
            else:
                capture = mss.MSS()
                left, top, width, height = args.region
                if width <= 0 or height <= 0:
                    raise ValueError("Capture width and height must be positive")
                region = {"left": left, "top": top, "width": width, "height": height}
            lease = InputLease(sink, focus)
            controls = ControlKeys(lease, stop)

            def on_press(key):
                state = controls.press(getattr(key, "name", None))
                if state is not None:
                    print("Control:", state, flush=True)

            def on_release(key):
                controls.release(getattr(key, "name", None))

            listener = keyboard.Listener(on_press=on_press, on_release=on_release)
            listener.start()
            listener.wait()
        else:
            sink = DrySink()
            lease = InputLease(sink, lambda: True)
            lease.toggle()

        def watch():
            while not stop.wait(.02):
                try:
                    lease.poll()
                    if listener is not None and not listener.is_alive():
                        lease.pause("listener_stopped")
                        stop.set()
                except Exception as exc:
                    print("Input watchdog failed:", exc, file=sys.stderr)
                    stop.set()
                    return

        watchdog = threading.Thread(target=watch, daemon=True)
        watchdog.start()
        if args.policy == "continuousdoom":
            if not args.continuous_source or not args.checkpoint or args.yaw_scale is None:
                raise ValueError("ContinuousDoom requires --continuous-source, --checkpoint, and --yaw-scale")
            from .continuous import ContinuousPolicy
            policy = ContinuousPolicy(args.continuous_source, args.checkpoint, args.yaw_scale)
        elif args.policy == "input_check":
            policy = InputCheckPolicy()
        else:
            policy = PixelProbe()
        # Model construction can exceed the lease. Start each policy paused.
        lease.pause("policy_ready")
        if not args.live:
            lease.toggle()
        else:
            print(f"Live probe paused. F8: enable/pause. Esc: pause. F9: exit. Run limit: {args.seconds:g} seconds.", flush=True)
        started = time.perf_counter()
        previously_armed = False
        previous_phase = None
        exit_reason = "frame_limit"
        for index in range(args.frames):
            if stop.is_set():
                exit_reason = lease.reason
                break
            if time.perf_counter() - started >= args.seconds:
                exit_reason = "duration_limit"
                break
            begin = time.perf_counter_ns()
            if args.live:
                if args.capture in ("x11-window", "window"):
                    frame = capture.grab()
                else:
                    shot = capture.grab(region)
                    frame = Frame(shot.width, shot.height, bytes(shot.bgra), begin)
            else:
                frame = synthetic_frame(index)
            captured = time.perf_counter_ns()
            armed = lease.armed
            if armed and not previously_armed and hasattr(policy, "reset"):
                policy.reset()
            previously_armed = armed
            invalid_frame = blank_frame(frame)
            if invalid_frame:
                lease.pause("blank_frame")
                action = Action()
            elif args.policy == "input_check" and not armed:
                action = Action()
            else:
                action = policy.predict(frame).bounded()
            phase = getattr(policy, "phase", None)
            if args.live and args.policy == "input_check" and armed and phase != previous_phase:
                print("Input check:", phase, flush=True)
                previous_phase = phase
            if args.policy == "input_check" and policy.complete:
                lease.pause("probe_complete")
            predicted = time.perf_counter_ns()
            applied = lease.submit(action, frame.captured_ns)
            end = time.perf_counter_ns()
            row = {"frame": index, "capture_started_ns": begin,
                   "capture_ms": (captured - begin) / 1e6,
                   "policy_ms": (predicted - captured) / 1e6,
                   "input_ms": (end - predicted) / 1e6, "loop_ms": (end - begin) / 1e6,
                   "applied": applied, "control_state": lease.reason,
                   "policy_phase": phase, "blank_frame": invalid_frame,
                   "action": {"keys": sorted(action.keys), "dx": action.mouse_dx,
                              "dy": action.mouse_dy, "attack": action.attack}}
            rows.append(row)
            if output:
                output.write(json.dumps(row) + "\n")
            if args.live:
                time.sleep(max(0, 1 / args.hz - (end - begin) / 1e9))
    except KeyboardInterrupt:
        exit_reason = "keyboard_interrupt"
    finally:
        stop.set()
        # Stop callbacks/watchdog before closing the display they use.
        if listener:
            listener.stop()
            listener.join(timeout=1)
        if watchdog:
            watchdog.join(timeout=1)
        try:
            if lease:
                lease.pause("shutdown")
            elif sink:
                sink.release()
        finally:
            if capture:
                capture.close()
            if focus:
                focus.close()
            if output:
                output.close()
    summarize(rows, "desktop" if args.live else "synthetic_dry_run", args.policy, exit_reason)
    if args.live:
        print("Control stopped:", exit_reason, flush=True)
        print("F8 no longer controls this process. Start a new run to enable AI input.", flush=True)


def main():
    parser = argparse.ArgumentParser(description="ViMM8BDM visual control interface probe")
    diagnostics = parser.add_mutually_exclusive_group()
    diagnostics.add_argument("--doctor", action="store_true")
    diagnostics.add_argument("--list-windows", action="store_true", help="List window metadata without sending input")
    diagnostics.add_argument("--check-window", action="store_true", help="Measure window focus without sending input")
    parser.add_argument("--all-windows", action="store_true", help="Include other visible windows in the local list")
    parser.add_argument("--include-window-titles", action="store_true", help="Include titles in local diagnostic output")
    parser.add_argument("--policy", choices=("red_marker_probe", "continuousdoom", "input_check"), default="red_marker_probe")
    parser.add_argument("--continuous-source", help="Trusted external upstream Python source directory")
    parser.add_argument("--checkpoint", help="External ContinuousDoom state_dict path")
    parser.add_argument("--yaw-scale", type=float, help="Mouse counts per model yaw unit, requires calibration")
    parser.add_argument("--live", action="store_true", help="Enable actual desktop capture and inputs")
    parser.add_argument("--capture", choices=("desktop", "x11-window", "window"), default="desktop")
    parser.add_argument("--window-id", type=lambda x: int(x, 0))
    parser.add_argument("--region", type=int, nargs=4, metavar=("LEFT", "TOP", "WIDTH", "HEIGHT"))
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--hz", type=float, default=20)
    parser.add_argument("--output", help="New JSONL path; existing files are never overwritten")
    args = parser.parse_args()
    if args.frames <= 0 or not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("frames and seconds must be positive and finite")
    if not math.isfinite(args.hz) or not 1 <= args.hz <= 60:
        parser.error("hz must be finite and between 1 and 60")
    if args.yaw_scale is not None and (not math.isfinite(args.yaw_scale) or args.yaw_scale <= 0):
        parser.error("yaw-scale must be positive and finite")
    try:
        if args.live and (args.doctor or args.list_windows or args.check_window):
            raise ValueError("Diagnostics cannot be combined with --live")
        if args.doctor:
            doctor()
        elif args.list_windows:
            list_windows(args)
        elif args.check_window:
            check_window(args)
        else:
            run(args)
    except (RuntimeError, ValueError, ImportError, OSError) as exc:
        parser.exit(2, f"ViMM8BDM: {exc}\n")


if __name__ == "__main__":
    main()
