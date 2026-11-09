"""Observe body candidates without producing game input."""
from pathlib import Path
import time

from PIL import Image
import torch
from torchvision.transforms.functional import pil_to_tensor

from .aim_tracking import measure_damage_cues
from .control import Action
from .local_zoom import refine_with_local_zoom
from .pretrained_detection import bind_single_class_postprocess as _bind_single_class_postprocess
from .recording import file_digest
from .sprite_transfer import load_checkpoint


class BodyShadowPolicy:
    """Measure current-image inference and discard late candidates."""

    def __init__(self, checkpoint, clock=time.perf_counter_ns, device=None):
        torch.set_num_threads(4)
        self.model, _ = load_checkpoint(Path(checkpoint))
        self.checkpoint_sha256 = file_digest(Path(checkpoint))
        self.clock = clock
        if device is None or device == 'auto':
            device = 'cuda' if hasattr(self.model, 'to') and torch.cuda.is_available() else 'cpu'
        self.device = str(torch.device(device))
        if hasattr(self.model, 'to'):
            self.model.to(self.device)
        if self.device.startswith('cuda'):
            _bind_single_class_postprocess(self.model)
            transform = getattr(self.model, 'transform', None)
            if transform is not None and hasattr(transform, 'image_mean') and hasattr(transform, 'image_std'):
                transform.image_mean = torch.as_tensor(transform.image_mean, dtype=torch.float32, device=self.device)
                transform.image_std = torch.as_tensor(transform.image_std, dtype=torch.float32, device=self.device)
        with torch.inference_mode():
            for h, w in [(480, 640)] * 4 + [(320, 320)]:
                self.model([torch.zeros(3, h, w, device=self.device)])
            if self.device.startswith('cuda'):
                torch.cuda.synchronize()
        self.idle_warmup_calls = 0
        self.last_ready_ns = self.clock()
        self.reset()

    def keep_ready(self):
        """Warm idle inference without input so CUDA stays in P0 clock state."""
        limit_ns = 120_000_000 if self.device.startswith('cuda') else 2_000_000_000
        if self.clock() - self.last_ready_ns < limit_ns:
            return False
        with torch.inference_mode():
            if self.device.startswith('cuda'):
                t = torch.zeros((3, 480, 640), dtype=torch.uint8).to(self.device).float().div_(255)
                self.model([t])
                self.model([t[:, 80:400, 160:480]])
                torch.cuda.synchronize()
            else:
                self.model([torch.zeros(3, 480, 640, device=self.device)])
        self.last_ready_ns = self.clock()
        self.idle_warmup_calls += 1
        return True

    def reset(self):
        self.last_capture_ns = None
        self.last_prediction = None
        self.last_image = None
        self.zoom_hint_box = None
        self.zoom_accept_score = .5
        self.phase = 'observe'

    def predict(self, frame):
        started = self.clock()
        self.last_image = None
        age = started - frame.captured_ns
        reason = None
        if not 0 <= age <= 75_000_000:
            reason = 'frame_age_invalid'
        elif self.last_capture_ns is not None and frame.captured_ns <= self.last_capture_ns:
            reason = 'capture_timestamp_invalid'
        self.last_prediction = dict(checkpoint_sha256=self.checkpoint_sha256,
                                    device=self.device,
                                    input_sent=False, candidates=[], stale=True,
                                    idle_warmup_calls=self.idle_warmup_calls,
                                    inference_performed=False, reset_reason=reason,
                                    capture_age_at_entry_ms=age / 1e6,
                                    enemy_or_life_state_verified=False)
        if reason:
            return Action()
        self.last_capture_ns = frame.captured_ns
        image = Image.frombytes('RGB', (frame.width, frame.height), frame.bgra, 'raw', 'BGRX')
        self.last_image = image
        flash_k, hp_pixels = measure_damage_cues(image)
        with torch.inference_mode():
            raw_tensor = pil_to_tensor(image)
            tensor = (raw_tensor.to(self.device).float().div_(255)
                      if self.device.startswith('cuda') else (raw_tensor.float() / 255).to(self.device))
            if flash_k >= 12:
                alpha = flash_k / 255.
                inv = 1. / (1. - alpha)
                tensor = tensor.clone()
                tensor[0].sub_(alpha).mul_(inv).clamp_(0., 1.)
                tensor[1].sub_(alpha).mul_(inv).clamp_(0., 1.)
                tensor[2].mul_(inv).clamp_(0., 1.)
            prediction = self.model([tensor])[0]
            boxes, window = refine_with_local_zoom(
                self.model, image, prediction, tensor,
                hint_box=getattr(self, 'zoom_hint_box', None),
                accept_score=getattr(self, 'zoom_accept_score', .5))
            if self.device.startswith('cuda'):
                torch.cuda.synchronize()
        finished = self.clock()
        self.last_ready_ns = finished
        stale = not 0 <= finished - frame.captured_ns <= 150_000_000
        for box in boxes:
            box['category'] = 'cutman_body_candidate'
        self.last_prediction.update(inference_performed=True, stale=stale,
                                    candidates=[] if stale else boxes,
                                    discarded_candidates=len(boxes) if stale else 0,
                                    local_zoom_crop=window,
                                    damage_flash_k=flash_k,
                                    hud_hp_pixels=hp_pixels,
                                    capture_age_at_result_ms=(finished-frame.captured_ns) / 1e6,
                                    processing_ms=(finished-started) / 1e6,
                                    reset_reason='result_age_invalid' if stale else None)
        return Action()
