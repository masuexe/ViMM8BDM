from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import platform
import statistics
import sys
import time
from pathlib import Path

from .control import (InputLease, PixelProbe, Frame, synthetic_frame, InputCheckPolicy,
                      WatchdogCheckPolicy, FocusCheckPolicy, blank_frame, Action)


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
               "neural_inference_measured": policy_name == "continuousdoom" or
                    (policy_name == "temporal_shadow" and any((row.get("model_output") or {}).get("inference_performed") for row in rows)),
               "accepted_actions": sum(row["applied"] for row in rows),
               "inactive_or_rejected_frames": sum(not row["applied"] for row in rows),
               "input_stage": ("input_process_roundtrip" if mode == "desktop" else "lease_and_sink")
                              if any(row["applied"] for row in rows) else "lease_gate_without_input_dispatch"}
    for key in ("capture_ms", "policy_ms", "input_ms", "loop_ms"):
        vals = sorted(row[key] for row in rows)
        if vals:
            summary[key] = {"p50": statistics.median(vals),
                            "p95": vals[max(0, math.ceil(len(vals) * .95) - 1)],
                            "max": vals[-1]}
    accepted = [row for row in rows if row["applied"]]
    summary["policy_actions_suppressed"] = any(row.get("observe_only", False) for row in rows)
    summary["dispatched_policy_actions"] = sum(row["applied"] and not row.get("observe_only", False) for row in rows)
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
    if args.policy in ("temporal_shadow", "body_shadow") and (not args.live or not getattr(args, "observe_only", False)
                                            or not getattr(args, "checkpoint", None)):
        raise ValueError("A shadow policy requires --live, --observe-only, and --checkpoint")
    sink = capture = worker = None
    rows = []
    output = None
    lease = None
    exit_reason = "error"
    try:
        if args.output:
            output = Path(args.output).open("x", encoding="utf-8")
        if args.live:
            import mss
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
        else:
            sink = DrySink()
            lease = InputLease(sink, lambda: True)
            lease.toggle()

        if args.policy == "continuousdoom":
            if not args.continuous_source or not args.checkpoint or args.yaw_scale is None:
                raise ValueError("ContinuousDoom requires --continuous-source, --checkpoint, and --yaw-scale")
            from .continuous import ContinuousPolicy
            policy = ContinuousPolicy(args.continuous_source, args.checkpoint, args.yaw_scale)
            print("Model loaded with strict parameter checks. CPU warm-up completed.", flush=True)
        elif args.policy == "input_check":
            policy = InputCheckPolicy()
        elif args.policy == "focus_check":
            policy = FocusCheckPolicy()
        elif args.policy == "watchdog_check":
            policy = WatchdogCheckPolicy()
        elif args.policy == "visual_navigation":
            from .visual_navigation import VisualNavigationPolicy
            policy = VisualNavigationPolicy()
        elif args.policy == "body_shadow":
            from .body_shadow import BodyShadowPolicy
            policy = BodyShadowPolicy(args.checkpoint)
            print("Body model loaded. CPU warm-up completed. This policy returns empty actions.", flush=True)
        elif args.policy == "temporal_shadow":
            from .temporal_shadow import TemporalShadowPolicy
            policy = TemporalShadowPolicy(args.checkpoint)
            print("Shadow model loaded. CPU warm-up completed. This policy returns empty actions.", flush=True)
        else:
            policy = PixelProbe()
        # Initialize desktop ownership only after the policy is ready.
        if args.live:
            from .input_worker import InputWorker
            worker = InputWorker(args.window_id, trace_path=args.input_trace)
            lease = worker
        lease.pause("policy_ready")
        if not args.live:
            lease.toggle()
        else:
            waiting_label = f"{args.arm_timeout:g} seconds" if args.arm_timeout else "none"
            print(f"Input worker ready. F8: enable/pause. Esc: pause. F9: exit. Waiting limit: {waiting_label}.", flush=True)
            print(f"The {args.seconds:g}-second run limit starts after the first enable command.", flush=True)
            if getattr(args, "observe_only", False):
                print("Observation mode: policy actions are suppressed.", flush=True)
        started = time.perf_counter()
        previously_armed = False
        previous_generation = None
        previous_phase = None
        fault_injected = False
        sequence_policy = args.policy in ("input_check", "watchdog_check", "focus_check")
        stateful_policy = sequence_policy or args.policy in ("visual_navigation", "temporal_shadow", "body_shadow")
        exit_reason = "frame_limit"
        frame_budget = args.frames
        if worker:
            waiting_until = time.perf_counter() + args.arm_timeout if args.arm_timeout else None
            while True:
                worker.refresh()
                if worker.stopped:
                    exit_reason = worker.reason
                    frame_budget = 0
                    break
                if worker.armed:
                    break
                if waiting_until is not None and time.perf_counter() >= waiting_until:
                    exit_reason = "arm_timeout"
                    frame_budget = 0
                    break
                time.sleep(.02)
            started = time.perf_counter()
        for index in range(frame_budget):
            if worker:
                worker.refresh()
                if worker.stopped:
                    exit_reason = worker.reason
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
            generation = worker.generation if worker else None
            if armed and (not previously_armed or generation != previous_generation) and hasattr(policy, "reset"):
                policy.reset()
            previously_armed = armed
            previous_generation = generation
            model_output = None
            invalid_frame = blank_frame(frame)
            if invalid_frame:
                if worker and args.policy == "focus_check":
                    worker.refresh()
                else:
                    lease.pause("blank_frame")
                action = Action()
            elif stateful_policy and not armed:
                action = Action()
            else:
                action = policy.predict(frame).bounded()
                model_output = getattr(policy, "last_prediction", None)
            phase = getattr(policy, "phase", None)
            if args.live and sequence_policy and armed and phase != previous_phase:
                print("Input check:", phase, flush=True)
                previous_phase = phase
            if sequence_policy and armed and policy.complete:
                lease.pause("probe_complete")
            predicted = time.perf_counter_ns()
            observe_only = getattr(args, "observe_only", False)
            dispatched = Action() if observe_only else action
            applied = (worker.submit(dispatched, frame.captured_ns, generation) if worker
                       else lease.submit(dispatched, frame.captured_ns))
            end = time.perf_counter_ns()
            row = {"frame": index, "capture_started_ns": begin,
                   "capture_ms": (captured - begin) / 1e6,
                   "policy_ms": (predicted - captured) / 1e6,
                   "input_ms": (end - predicted) / 1e6, "loop_ms": (end - begin) / 1e6,
                   "applied": applied, "control_state": lease.reason,
                   "policy_phase": phase, "blank_frame": invalid_frame,
                   "input_generation": generation, "observe_only": observe_only,
                   "model_output": model_output,
                   "dispatched_mouse_counts": [round(dispatched.mouse_dx), round(dispatched.mouse_dy)],
                   "action": {"keys": sorted(action.keys), "dx": action.mouse_dx,
                              "dy": action.mouse_dy, "attack": action.attack}}
            inject_fault = args.watchdog_fault and applied and phase == "hold" and not fault_injected
            if inject_fault:
                row["fault"] = args.watchdog_fault
                row["fault_started_ns"] = time.perf_counter_ns()
            rows.append(row)
            if output:
                output.write(json.dumps(row) + "\n")
            if inject_fault:
                fault_injected = True
                output.flush()
                print("Watchdog test fault:", args.watchdog_fault, flush=True)
                if args.watchdog_fault == "exit":
                    os._exit(86)
                time.sleep(.75)
            if args.live:
                time.sleep(max(0, 1 / args.hz - (end - begin) / 1e9))
    except KeyboardInterrupt:
        exit_reason = "keyboard_interrupt"
    finally:
        try:
            if worker:
                worker.close()
            elif lease:
                lease.pause("shutdown")
            elif sink:
                sink.release()
        finally:
            if capture:
                capture.close()
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
    parser.add_argument("--policy", choices=("red_marker_probe", "continuousdoom", "input_check", "watchdog_check", "focus_check", "visual_navigation", "temporal_shadow", "body_shadow"), default="red_marker_probe")
    parser.add_argument("--watchdog-fault", choices=("stall", "exit"), help="Explicit local fault test for the watchdog policy")
    parser.add_argument("--continuous-source", help="Trusted external upstream Python source directory")
    parser.add_argument("--checkpoint", help="Trusted local model checkpoint path")
    parser.add_argument("--yaw-scale", type=float, help="Mouse counts per model yaw unit, requires calibration")
    parser.add_argument("--observe-only", action="store_true", help="Infer from live images and suppress policy actions")
    parser.add_argument("--live", action="store_true", help="Enable actual desktop capture and inputs")
    parser.add_argument("--capture", choices=("desktop", "x11-window", "window"), default="desktop")
    parser.add_argument("--window-id", type=lambda x: int(x, 0))
    parser.add_argument("--region", type=int, nargs=4, metavar=("LEFT", "TOP", "WIDTH", "HEIGHT"))
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--arm-timeout", type=float, default=180, help="Initial waiting limit in seconds. Zero waits until F8 or F9.")
    parser.add_argument("--hz", type=float, default=20)
    parser.add_argument("--output", help="New JSONL path; existing files are never overwritten")
    parser.add_argument("--input-trace", help="New local JSONL file for input-worker audit events")
    args = parser.parse_args()
    if args.observe_only and (not args.live or args.watchdog_fault or args.policy == "watchdog_check"):
        parser.error("observe-only requires --live and cannot run a watchdog fault test")
    if args.watchdog_fault or args.policy == "watchdog_check":
        if not (args.live and args.policy == "watchdog_check" and args.watchdog_fault and args.output and args.input_trace):
            parser.error("A watchdog fault test requires --live, --policy watchdog_check, --watchdog-fault, --output, and --input-trace")
    if args.frames <= 0 or not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("frames and seconds must be positive and finite")
    if not math.isfinite(args.arm_timeout) or args.arm_timeout < 0:
        parser.error("arm-timeout must be nonnegative and finite")
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
