"""Keep the initial user wait outside the active run budget."""
import contextlib
import io
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from vimm8bdm.__main__ import run


class Clock:
    def __init__(self):
        self.now = 0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class Capture:
    def __init__(self):
        self.frames = 0

    def grab(self, unused):
        self.frames += 1
        return SimpleNamespace(width=160, height=120, bgra=bytes((35, 35, 35, 255)) * 160 * 120)

    def close(self):
        pass


class Worker:
    def __init__(self, clock, enable_at):
        self.clock, self.enable_at = clock, enable_at
        self.armed = self.stopped = self.closed = False
        self.generation = 0
        self.reason = "manual"
        self.actions = []

    def refresh(self):
        if self.clock() >= self.enable_at:
            self.armed = True
            self.generation = 1
            self.reason = "armed"

    def pause(self, reason):
        self.reason = reason
        self.armed = False

    def submit(self, action, captured_ns, generation):
        if self.armed:
            self.actions.append(action)
        return self.armed

    def close(self):
        self.closed = True


class RunTimingTests(unittest.TestCase):
    def execute(self, enable_at, waiting_limit):
        clock = Clock()
        worker = Worker(clock, enable_at)
        capture = Capture()
        args = SimpleNamespace(live=True, window_id=100, capture="desktop", region=(0, 0, 160, 120),
                               output=None, policy="red_marker_probe", input_trace=None, frames=3,
                               seconds=.5, arm_timeout=waiting_limit, hz=20, watchdog_fault=None)
        with (patch.dict("sys.modules", {"mss": SimpleNamespace(MSS=lambda: capture)}),
              patch("vimm8bdm.input_worker.InputWorker", return_value=worker),
              patch("vimm8bdm.__main__.time.perf_counter", clock),
              patch("vimm8bdm.__main__.time.sleep", clock.sleep),
              contextlib.redirect_stdout(io.StringIO())):
            run(args)
        return worker, capture

    def test_user_wait_does_not_consume_frames_or_run_duration(self):
        worker, capture = self.execute(enable_at=2, waiting_limit=5)
        self.assertEqual(len(worker.actions), 3)
        self.assertEqual(capture.frames, 3)
        self.assertTrue(worker.closed)

    def test_wait_timeout_closes_without_capture_or_input(self):
        worker, capture = self.execute(enable_at=10, waiting_limit=.1)
        self.assertEqual(worker.actions, [])
        self.assertEqual(capture.frames, 0)
        self.assertTrue(worker.closed)


if __name__ == "__main__":
    unittest.main()
