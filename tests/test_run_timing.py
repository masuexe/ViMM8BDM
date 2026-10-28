"""Keep the initial user wait outside the active run budget."""
import contextlib
import io
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

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
    def execute(self, enable_at, waiting_limit, observe_only=False,
                policy_name='red_marker_probe', worker_class=Worker):
        clock = Clock()
        worker = worker_class(clock, enable_at)
        capture = Capture()
        args = SimpleNamespace(live=True, window_id=100, capture="desktop", region=(0, 0, 160, 120),
                               output=None, policy=policy_name, input_trace=None, frames=3, checkpoint="fixture.pt",
                               observe_only=observe_only,
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

    def test_zero_waiting_limit_permits_late_user_enable(self):
        worker, capture = self.execute(enable_at=181, waiting_limit=0)
        self.assertEqual(len(worker.actions), 3)
        self.assertEqual(capture.frames, 3)
        self.assertTrue(worker.closed)

    def test_observation_mode_suppresses_movement_and_mouse_actions(self):
        from vimm8bdm.control import Action
        policy = SimpleNamespace(predict=lambda frame: Action(frozenset({"w"}), 3, 2, True))
        with patch("vimm8bdm.__main__.PixelProbe", return_value=policy):
            worker, capture = self.execute(enable_at=0, waiting_limit=5, observe_only=True)
        self.assertEqual(len(worker.actions), 3)
        self.assertEqual(capture.frames, 3)
        self.assertTrue(all(not a.keys and not a.attack and not a.mouse_dx and not a.mouse_dy
                            for a in worker.actions))

    def test_wait_timeout_closes_without_capture_or_input(self):
        worker, capture = self.execute(enable_at=10, waiting_limit=.1)
        self.assertEqual(worker.actions, [])
        self.assertEqual(capture.frames, 0)
        self.assertTrue(worker.closed)

    def test_temporal_shadow_requires_observation_mode(self):
        with self.assertRaisesRegex(ValueError, 'observe-only'):
            self.execute(0, 5, policy_name='temporal_shadow')

    def test_temporal_shadow_resets_and_suppresses_even_nonempty_actions(self):
        from vimm8bdm.control import Action
        policy = SimpleNamespace(reset=Mock(), predict=Mock(return_value=Action(frozenset({'w'}), 4, 2, True)))
        module = SimpleNamespace(TemporalShadowPolicy=lambda checkpoint: policy)
        with patch.dict('sys.modules', {'vimm8bdm.temporal_shadow': module}):
            worker, _ = self.execute(0, 5, policy_name='temporal_shadow', observe_only=True)
        policy.reset.assert_called_once()
        self.assertEqual(policy.predict.call_count, 3)
        self.assertTrue(all(a == Action() for a in worker.actions))

    def test_temporal_shadow_does_not_advance_while_paused(self):
        class PausedWorker(Worker):
            def refresh(self):
                self.calls = getattr(self, 'calls', 0) + 1
                self.armed = self.calls == 1
                self.generation = 1
        policy = SimpleNamespace(reset=Mock(), predict=Mock())
        module = SimpleNamespace(TemporalShadowPolicy=lambda checkpoint: policy)
        with patch.dict('sys.modules', {'vimm8bdm.temporal_shadow': module}):
            worker, _ = self.execute(0, 5, policy_name='temporal_shadow', observe_only=True,
                                     worker_class=PausedWorker)
        policy.predict.assert_not_called()
        self.assertEqual(worker.actions, [])

    def test_visual_navigation_resets_before_the_first_active_prediction(self):
        from vimm8bdm.control import Action
        policy = SimpleNamespace(reset=Mock(), predict=Mock(return_value=Action(frozenset({'w'}))))
        with patch('vimm8bdm.visual_navigation.VisualNavigationPolicy', return_value=policy):
            worker, _ = self.execute(0, 5, policy_name='visual_navigation')
        policy.reset.assert_called_once()
        self.assertEqual(policy.predict.call_count, 3)
        self.assertTrue(all(a.keys == frozenset({'w'}) for a in worker.actions))

    def test_visual_navigation_does_not_advance_while_paused(self):
        class PausedWorker(Worker):
            def refresh(self):
                self.calls = getattr(self, 'calls', 0) + 1
                self.armed = self.calls == 1  # Initial enable, followed by pause before capture.
                self.generation = 1
                self.reason = 'armed' if self.armed else 'manual'
        policy = SimpleNamespace(reset=Mock(), predict=Mock())
        with patch('vimm8bdm.visual_navigation.VisualNavigationPolicy', return_value=policy):
            worker, _ = self.execute(0, 5, policy_name='visual_navigation', worker_class=PausedWorker)
        policy.predict.assert_not_called()
        self.assertEqual(worker.actions, [])

    def test_idle_warmup_runs_during_the_user_wait_without_consuming_frames(self):
        from vimm8bdm.control import Action
        warm = Mock()
        policy = SimpleNamespace(predict=lambda frame: Action(), keep_ready=warm)
        with patch("vimm8bdm.__main__.PixelProbe", return_value=policy):
            worker, capture = self.execute(enable_at=2, waiting_limit=5)
        self.assertGreater(warm.call_count, 0)
        self.assertEqual(capture.frames, 3)
        self.assertEqual(len(worker.actions), 3)
        self.assertTrue(all(action == Action() for action in worker.actions))


if __name__ == "__main__":
    unittest.main()
