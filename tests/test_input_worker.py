"""Exercise real process failure without desktop input."""
import json
import multiprocessing as mp
from pathlib import Path
import tempfile
import time
import unittest

from vimm8bdm.control import Action
from vimm8bdm.input_worker import InputWorker


class AuditFocus:
    def __call__(self):
        return True


class AuditSink:
    def __init__(self, path):
        self.path = path
        self.held = Action()

    def write(self, event, **values):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"event": event, "time_ns": time.perf_counter_ns(), **values}) + "\n")

    def apply(self, action):
        self.held = action
        self.write("apply", keys=sorted(action.keys), attack=action.attack)

    def release(self):
        if self.held.keys or self.held.attack:
            self.held = Action()
            self.write("release", keys=[], attack=False)

    def close(self):
        self.write("closed")

    def owned_state(self):
        return {"held_keys": sorted(self.held.keys), "attack_held": self.held.attack}


class AuditFactory:
    def __init__(self, path):
        self.path = path

    def __call__(self):
        return AuditSink(self.path), AuditFocus()


def producer(path, ready, heartbeat):
    worker = InputWorker(backend_factory=AuditFactory(path))
    worker._request("test_key", key="f8")
    action = Action(frozenset({"w"}), attack=True)
    worker.submit(action, time.perf_counter_ns(), worker.generation)
    Path(ready).write_text("ready")
    # The test kills this process. Deliberately omit normal shutdown.
    while True:
        time.sleep(.02)
        if heartbeat:
            worker.submit(action, time.perf_counter_ns(), worker.generation)


class InputWorkerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="vimm8bdm-worker-")
        self.path = str(Path(self.directory.name) / "events.jsonl")

    def tearDown(self):
        self.directory.cleanup()

    def events(self):
        path = Path(self.path)
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line]

    def wait_event(self, kind, timeout=5):
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            try:
                matches = [e for e in self.events() if e["event"] == kind]
            except json.JSONDecodeError:
                matches = []
            if matches:
                return matches[-1]
            time.sleep(.01)
        self.fail(f"The audit did not record {kind}")

    def test_status_messages_do_not_extend_the_input_lease(self):
        with InputWorker(backend_factory=AuditFactory(self.path)) as worker:
            worker._request("test_key", key="f8")
            self.assertTrue(worker.submit(Action(frozenset({"w"}), attack=True), time.perf_counter_ns(), worker.generation))
            deadline = time.perf_counter() + .4
            while time.perf_counter() < deadline:
                worker.refresh()
                time.sleep(.02)
            self.assertFalse(worker.armed)
            self.assertEqual(worker.reason, "watchdog_or_focus")
            self.wait_event("release")

    def test_old_generation_cannot_apply_after_pause_and_reenable(self):
        with InputWorker(backend_factory=AuditFactory(self.path)) as worker:
            worker._request("test_key", key="f8")
            old = worker.generation
            worker._request("test_key", key="esc")
            worker._request("test_key", key="f8")
            self.assertFalse(worker.submit(Action(attack=True), time.perf_counter_ns(), old))
            self.assertTrue(worker.submit(Action(frozenset({"d"})), time.perf_counter_ns(), worker.generation))

    def test_f9_stops_the_worker_and_releases_input(self):
        with InputWorker(backend_factory=AuditFactory(self.path)) as worker:
            worker._request("test_key", key="f8")
            worker.submit(Action(frozenset({"w"}), attack=True), time.perf_counter_ns(), worker.generation)
            worker._request("test_key", key="f9")
            self.assertTrue(worker.stopped)
            self.assertEqual(worker.reason, "quit_key")
            self.wait_event("release")

    def test_trace_records_release_after_independent_expiry(self):
        trace = str(Path(self.directory.name) / "worker.jsonl")
        with InputWorker(backend_factory=AuditFactory(self.path), trace_path=trace) as worker:
            worker._request("test_key", key="f8")
            worker.submit(Action(frozenset({"w"}), attack=True), time.perf_counter_ns(), worker.generation)
            time.sleep(.3)
            worker.refresh()
            self.assertFalse(worker.armed)
        events = [json.loads(line) for line in Path(trace).read_text().splitlines()]
        released = next(e for e in events if e["event"] == "released")
        self.assertEqual(released["held_keys"], [])
        self.assertFalse(released["attack_held"])
        delay = (released["time_ns"] - released["lease_deadline_ns"] + 200_000_000) / 1e6
        self.assertLess(delay, 500)
        self.assertEqual(events[-1], {"event": "trace_end", "dropped_events": 0})

    def process_failure(self, heartbeat):
        ready = str(Path(self.directory.name) / "ready")
        process = mp.get_context("spawn").Process(target=producer, args=(self.path, ready, heartbeat))
        process.start()
        try:
            deadline = time.perf_counter() + 15
            while not Path(ready).exists() and time.perf_counter() < deadline:
                self.assertTrue(process.is_alive(), "The test producer ended before input")
                time.sleep(.01)
            self.assertTrue(Path(ready).exists(), "The test producer did not become ready")
            if not heartbeat:
                released = self.wait_event("release")
                applied = [e for e in self.events() if e["event"] == "apply"][-1]
                self.assertLess((released["time_ns"] - applied["time_ns"]) / 1e9, .5)
            process.terminate()
            process.join(timeout=3)
            self.wait_event("closed")
            self.assertFalse(process.is_alive())
            self.assertTrue(any(e["event"] == "release" for e in self.events()))
        finally:
            if process.is_alive():
                process.terminate()
                process.join(timeout=3)

    def test_inference_hang_does_not_hold_keys(self):
        self.process_failure(heartbeat=False)

    def test_killed_inference_process_releases_keys_and_worker_exits(self):
        self.process_failure(heartbeat=True)


if __name__ == "__main__":
    unittest.main()
