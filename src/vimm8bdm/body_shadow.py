"""Observe body candidates without producing game input."""
from pathlib import Path
import time

from PIL import Image
import torch
from torchvision.transforms.functional import pil_to_tensor

from .control import Action
from .local_zoom import refine_with_local_zoom
from .recording import file_digest
from .sprite_transfer import load_checkpoint


class BodyShadowPolicy:
    """Measure current-image inference and discard late candidates."""

    def __init__(self, checkpoint, clock=time.perf_counter_ns):
        torch.set_num_threads(4)
        self.model, _ = load_checkpoint(Path(checkpoint))
        self.checkpoint_sha256 = file_digest(Path(checkpoint))
        self.clock = clock
        with torch.inference_mode():
            for _ in range(5):
                self.model([torch.zeros(3, 480, 640)])
        self.reset()

    def reset(self):
        self.last_capture_ns = None
        self.last_prediction = None
        self.phase = 'observe'

    def predict(self, frame):
        started = self.clock()
        age = started - frame.captured_ns
        reason = None
        if not 0 <= age <= 75_000_000:
            reason = 'frame_age_invalid'
        elif self.last_capture_ns is not None and frame.captured_ns <= self.last_capture_ns:
            reason = 'capture_timestamp_invalid'
        self.last_prediction = dict(checkpoint_sha256=self.checkpoint_sha256,
                                    input_sent=False, candidates=[], stale=True,
                                    inference_performed=False, reset_reason=reason,
                                    capture_age_at_entry_ms=age / 1e6,
                                    enemy_or_life_state_verified=False)
        if reason:
            return Action()
        self.last_capture_ns = frame.captured_ns
        image = Image.frombytes('RGB', (frame.width, frame.height), frame.bgra, 'raw', 'BGRX')
        with torch.inference_mode():
            prediction = self.model([pil_to_tensor(image).float() / 255])[0]
            boxes, window = refine_with_local_zoom(self.model, image, prediction)
        finished = self.clock()
        stale = not 0 <= finished - frame.captured_ns <= 150_000_000
        for box in boxes:
            box['category'] = 'cutman_body_candidate'
        self.last_prediction.update(inference_performed=True, stale=stale,
                                    candidates=[] if stale else boxes,
                                    discarded_candidates=len(boxes) if stale else 0,
                                    local_zoom_crop=window,
                                    capture_age_at_result_ms=(finished-frame.captured_ns) / 1e6,
                                    processing_ms=(finished-started) / 1e6,
                                    reset_reason='result_age_invalid' if stale else None)
        return Action()
