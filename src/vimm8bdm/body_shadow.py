"""Observe body candidates without producing game input."""
from pathlib import Path
import time

from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.ops import boxes as box_ops
from torchvision.transforms.functional import pil_to_tensor

from .control import Action
from .local_zoom import refine_with_local_zoom
from .recording import file_digest
from .sprite_transfer import load_checkpoint


def _bind_single_class_postprocess(model):
    """Avoid 90-class host-device syncs in SSD.postprocess_detections when only class 1 is used."""
    if not (hasattr(model, 'postprocess_detections') and hasattr(model, 'box_coder')):
        return
    def postprocess(head_outputs, image_anchors, image_shapes):
        bbox_regression = head_outputs['bbox_regression']
        pred_scores = F.softmax(head_outputs['cls_logits'], dim=-1)
        detections = []
        for boxes, scores, anchors, image_shape in zip(bbox_regression, pred_scores, image_anchors, image_shapes):
            boxes = model.box_coder.decode_single(boxes, anchors)
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)
            s = scores[:, 1]
            keep = s > model.score_thresh
            s, b = s[keep], boxes[keep]
            keep = box_ops.remove_small_boxes(b, min_size=1e-2)
            s, b = s[keep], b[keep]
            if s.numel() > model.topk_candidates:
                s, topk = s.topk(model.topk_candidates)
                b = b[topk]
            labels = torch.ones_like(s, dtype=torch.int64)
            keep_nms = box_ops.batched_nms(b, s, labels, model.nms_thresh)[:model.detections_per_img]
            detections.append(dict(boxes=b[keep_nms], scores=s[keep_nms], labels=labels[keep_nms]))
        return detections
    model.postprocess_detections = postprocess


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
        with torch.inference_mode():
            for _ in range(5):
                self.model([torch.zeros(3, 480, 640, device=self.device)])
            if self.device.startswith('cuda'):
                torch.cuda.synchronize()
        self.idle_warmup_calls = 0
        self.last_ready_ns = self.clock()
        self.reset()

    def keep_ready(self):
        """Warm idle inference at most once per two seconds without input."""
        if self.clock() - self.last_ready_ns < 2_000_000_000:
            return False
        with torch.inference_mode():
            self.model([torch.zeros(3, 480, 640, device=self.device)])
            if self.device.startswith('cuda'):
                torch.cuda.synchronize()
        self.last_ready_ns = self.clock()
        self.idle_warmup_calls += 1
        return True

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
        with torch.inference_mode():
            tensor = (pil_to_tensor(image).float() / 255).to(self.device)
            prediction = self.model([tensor])[0]
            boxes, window = refine_with_local_zoom(self.model, image, prediction)
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
                                    capture_age_at_result_ms=(finished-frame.captured_ns) / 1e6,
                                    processing_ms=(finished-started) / 1e6,
                                    reset_reason='result_age_invalid' if stale else None)
        return Action()
