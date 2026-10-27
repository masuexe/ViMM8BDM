"""Own desktop input in a process independent from capture and inference."""
from __future__ import annotations

import atexit
import multiprocessing as mp
import json
import queue
import threading
import time

from .control import ControlKeys, InputLease


class WorkerTrace:
    """Write optional audit events without blocking the input lease."""

    def __init__(self, path):
        self.file = open(path, "x", encoding="utf-8")
        self.queue = queue.Queue(maxsize=256)
        self.dropped = 0
        self.thread = threading.Thread(target=self._write, daemon=True)
        self.thread.start()

    def __call__(self, event):
        try:
            self.queue.put_nowait(event)
        except queue.Full:
            self.dropped += 1

    def _write(self):
        try:
            while True:
                event = self.queue.get()
                if event is None:
                    break
                self.file.write(json.dumps(event) + "\n")
                self.file.flush()
            self.file.write(json.dumps({"event": "trace_end", "dropped_events": self.dropped}) + "\n")
        finally:
            self.file.close()

    def close(self):
        try:
            self.queue.put_nowait(None)
        except queue.Full:
            self.dropped += 1
        self.thread.join(timeout=1)


class InputOwner:
    """Serialize user commands and actions against a local input lease."""

    def __init__(self, sink, focused, trace=None):
        self.sink = sink
        self.lease = InputLease(sink, focused)
        self.stop = threading.Event()
        self.keys = ControlKeys(self.lease, self.stop)
        self.generation = 0
        self.trace = trace
        self.last_action = None

    def note(self, event, deadline=None):
        if self.trace:
            self.trace({"event": event, "time_ns": time.perf_counter_ns(),
                        "lease_deadline_ns": deadline, **self.state()})

    def state(self):
        with self.lease.lock:
            owned = self.sink.owned_state() if hasattr(self.sink, "owned_state") else {}
            return {"armed": self.lease.armed, "reason": self.lease.reason,
                    "generation": self.generation, "stopped": self.stop.is_set(), **owned}

    def press(self, name):
        with self.lease.lock:
            deadline = self.lease.deadline
            before = self.lease.armed
            reason = self.keys.press(name)
            if reason is not None:
                self.generation += 1
                self.note("released" if before and not self.lease.armed else "control", deadline)
            return reason

    def release_key(self, name):
        with self.lease.lock:
            self.keys.release(name)

    def pause(self, reason):
        with self.lease.lock:
            if self.lease.armed or self.lease.reason != reason:
                deadline = self.lease.deadline
                before = self.lease.armed
                self.lease.pause(reason)
                self.generation += 1
                self.note("released" if before else "control", deadline)

    def poll(self):
        with self.lease.lock:
            before = self.lease.armed
            deadline = self.lease.deadline
            self.lease.poll()
            if before and not self.lease.armed:
                self.generation += 1
                self.note("released", deadline)

    def submit(self, action, captured_ns, generation):
        with self.lease.lock:
            if self.stop.is_set() or generation != self.generation:
                return False
            before = self.lease.armed
            deadline = self.lease.deadline
            accepted = self.lease.submit(action, captured_ns)
            if before and not self.lease.armed:
                self.generation += 1
                self.note("released", deadline)
            elif accepted and action != self.last_action:
                self.last_action = action
                self.note("action", self.lease.deadline)
            return accepted


def _worker_main(connection, window_id, backend_factory, trace_path):
    owner = listener = focus = sink = trace = None
    try:
        if trace_path:
            trace = WorkerTrace(trace_path)
        if backend_factory is None:
            from .desktop import DesktopSink, Focus
            from pynput import keyboard
            focus = Focus(window_id)
            sink = DesktopSink()
        else:
            sink, focus = backend_factory()
        owner = InputOwner(sink, focus, trace)
        if backend_factory is None:
            def pressed(key):
                reason = owner.press(getattr(key, "name", None))
                if reason is not None:
                    print("Control:", reason, flush=True)

            def released(key):
                owner.release_key(getattr(key, "name", None))

            listener = keyboard.Listener(on_press=pressed, on_release=released)
            listener.start()
            listener.wait()
        connection.send({"kind": "ready", "state": owner.state()})
        while not owner.stop.is_set():
            owner.poll()
            if listener is not None and not listener.is_alive():
                owner.pause("listener_stopped")
                break
            parent = mp.parent_process()
            if parent is not None and not parent.is_alive():
                owner.pause("parent_lost")
                break
            if not connection.poll(.01):
                continue
            try:
                request = connection.recv()
            except EOFError:
                owner.pause("parent_lost")
                break
            accepted = False
            operation = request.get("op")
            if operation == "status":
                pass
            elif operation == "action":
                accepted = owner.submit(request["action"], request["captured_ns"], request["generation"])
            elif operation == "pause":
                owner.pause(request["reason"])
            elif operation == "shutdown":
                owner.pause("shutdown")
                owner.stop.set()
            elif operation == "test_key" and backend_factory is not None:
                owner.press(request["key"])
                owner.release_key(request["key"])
            else:
                owner.pause("protocol_error")
                raise ValueError("The input worker received an unsupported command")
            connection.send({"kind": "reply", "id": request["id"],
                             "applied": accepted, "state": owner.state()})
    except (EOFError, BrokenPipeError, ConnectionResetError):
        if owner:
            owner.pause("parent_lost")
    except Exception as exc:
        if owner:
            owner.pause("worker_error")
        try:
            connection.send({"kind": "error", "message": str(exc)})
        except (OSError, EOFError):
            pass
    finally:
        if listener:
            listener.stop()
            listener.join(timeout=1)
        try:
            if owner:
                owner.lease.pause(owner.lease.reason)
                owner.stop.set()
                try:
                    connection.send({"kind": "stopped", "state": owner.state()})
                except (OSError, EOFError):
                    pass
            elif sink:
                sink.release()
        finally:
            if focus is not None and hasattr(focus, "close"):
                focus.close()
            if sink is not None and hasattr(sink, "close"):
                sink.close()
            if owner:
                owner.note("worker_stopped")
            if trace:
                trace.close()
            connection.close()


class InputWorker:
    """Submit bounded actions. Only the worker's user hotkeys can enable input."""

    def __init__(self, window_id=None, *, backend_factory=None, startup_timeout=15, trace_path=None):
        context = mp.get_context("spawn")
        self.connection, child = context.Pipe()
        self.process = context.Process(target=_worker_main, args=(child, window_id, backend_factory, trace_path))
        self.closed = False
        self.sequence = 0
        self.snapshot = {"armed": False, "reason": "starting", "generation": 0, "stopped": False}
        try:
            self.process.start()
            child.close()
            if not self.connection.poll(startup_timeout):
                raise RuntimeError("The input worker did not become ready")
            message = self.connection.recv()
            if message.get("kind") != "ready":
                raise RuntimeError(message.get("message", "The input worker failed during startup"))
            self.snapshot = message["state"]
        except BaseException:
            child.close()
            self.connection.close()
            if self.process.pid is not None:
                self.process.join(timeout=1)
                if self.process.is_alive():
                    self.process.terminate()
                    self.process.join(timeout=1)
            raise
        atexit.register(self.close)

    @property
    def armed(self):
        return self.snapshot["armed"]

    @property
    def reason(self):
        return self.snapshot["reason"]

    @property
    def generation(self):
        return self.snapshot["generation"]

    @property
    def stopped(self):
        return self.snapshot["stopped"]

    def _request(self, operation, **fields):
        if self.closed:
            raise RuntimeError("The input worker connection is closed")
        if self.stopped:
            return False
        self.sequence += 1
        try:
            if self.connection.poll():
                event = self.connection.recv()
                if event.get("kind") == "stopped":
                    self.snapshot = event["state"]
                    return False
                if event.get("kind") == "error":
                    raise RuntimeError(event.get("message", "The input worker failed"))
                raise RuntimeError("The input worker sent an unexpected event")
            self.connection.send({"op": operation, "id": self.sequence, **fields})
            if not self.connection.poll(.75):
                raise RuntimeError("The input worker did not reply")
            reply = self.connection.recv()
        except (EOFError, OSError):
            try:
                if self.connection.poll(.05):
                    event = self.connection.recv()
                    if event.get("kind") == "stopped":
                        self.snapshot = event["state"]
                        return False
            except (EOFError, OSError):
                pass
            self.snapshot.update(armed=False, stopped=True, reason="worker_lost")
            return False
        if reply.get("kind") == "error":
            raise RuntimeError(reply.get("message", "The input worker failed"))
        if reply.get("kind") == "stopped":
            self.snapshot = reply["state"]
            return False
        if reply.get("id") != self.sequence:
            raise RuntimeError("The input worker returned an unexpected reply")
        self.snapshot = reply["state"]
        return reply["applied"]

    def refresh(self):
        self._request("status")
        return self.snapshot.copy()

    def submit(self, action, captured_ns, generation):
        return self._request("action", action=action.bounded(), captured_ns=captured_ns, generation=generation)

    def pause(self, reason="manual"):
        self._request("pause", reason=reason)

    def close(self):
        if self.closed:
            return
        try:
            if self.process.is_alive() and not self.stopped:
                self._request("shutdown")
        except (RuntimeError, OSError):
            pass
        finally:
            self.closed = True
            self.connection.close()
            self.process.join(timeout=2)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=1)
            atexit.unregister(self.close)

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.close()
