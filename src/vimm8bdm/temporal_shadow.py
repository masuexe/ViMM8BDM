"""Run visual inference and return empty actions for every frame."""
from collections import deque
import time

import numpy as np
from PIL import Image
import torch

from .control import Action
from .dataset import select_history
from .behavior_cloning import IMAGE_SIZE, image_features
from .steering import CLASSES
from .recording import file_digest
from .temporal_steering import load_checkpoint

STEP_NS = 50_000_000


class TemporalShadowPolicy:
    """Observe a bounded image history without producing game input."""

    def __init__(self, checkpoint, clock=time.perf_counter_ns):
        torch.set_num_threads(4)
        self.encoder, self.model, self.mean, self.scale, self.steps = load_checkpoint(checkpoint)
        self.checkpoint_sha256 = file_digest(checkpoint)
        self.clock = clock
        with torch.inference_mode():
            self.encoder(torch.zeros(1, 4, IMAGE_SIZE[1], IMAGE_SIZE[0]))
            self.model(torch.zeros(1, self.steps, 128), torch.tensor([self.steps]))
        self.reset()

    def reset(self):
        self.images = deque(maxlen=12)
        self.features = deque(maxlen=self.steps)
        self.feature_times = deque(maxlen=self.steps)
        self.shape = None
        self.last_capture_ns = None
        self.last_arrival_ns = None
        self.first_arrival_ns = None
        self.last_anchor_ns = None
        self.next_anchor_ns = None
        self.last_prediction = None
        self.phase = 'collect_history'

    def predict(self, frame):
        arrived = self.clock()  # The image is complete no later than policy entry.
        shape = (frame.width, frame.height)
        reason = None
        if self.shape is not None and shape != self.shape:
            reason = 'capture_size_changed'
        elif self.last_capture_ns is not None and frame.captured_ns <= self.last_capture_ns:
            reason = 'capture_timestamp_invalid'
        elif self.last_arrival_ns is not None and not 0 < arrived - self.last_arrival_ns <= 100_000_000:
            reason = 'capture_interval_invalid'
        if reason:
            self.reset()
        self.shape = shape
        self.last_capture_ns, self.last_arrival_ns = frame.captured_ns, arrived
        self.phase = 'collect_history'
        self.last_prediction = {'type': 'temporal_steering_shadow', 'phase': 'collect_history',
                                'inference_performed': False, 'reset_reason': reason,
                                'context_steps': len(self.features), 'input_sent': False,
                                'image_complete_time': 'policy_entry_upper_bound',
                                'checkpoint_sha256': self.checkpoint_sha256}
        if not 0 <= arrived - frame.captured_ns <= 75_000_000:
            self.reset()
            self.last_prediction = {'type': 'temporal_steering_shadow', 'phase': 'collect_history',
                                    'inference_performed': False, 'reset_reason': 'frame_age_invalid',
                                    'context_steps': 0, 'input_sent': False}
            return Action()
        preprocess_started = time.perf_counter_ns()
        image = Image.frombytes('RGB', shape, frame.bgra, 'raw', 'BGRX')
        gray = image.convert('L').resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
        self.images.append((arrived, torch.from_numpy(np.asarray(gray).copy())))
        if self.first_arrival_ns is None:
            self.first_arrival_ns = arrived
            self.next_anchor_ns = arrived + 3 * STEP_NS
        if arrived < self.next_anchor_ns:
            if self.features:
                self.phase = 'wait_for_next_anchor'
                self.last_prediction['phase'] = self.phase
            return Action()
        latest_anchor = self.first_arrival_ns + (arrived - self.first_arrival_ns) // STEP_NS * STEP_NS
        times = [row[0] for row in self.images]
        preprocess_ns = time.perf_counter_ns() - preprocess_started
        encoder_ns = 0
        new_histories = []
        # Reconstruct missed feature anchors from images already complete at each anchor.
        # No old action or prediction enters the input worker.
        while self.next_anchor_ns <= latest_anchor:
            preprocessing = time.perf_counter_ns()
            anchor = self.next_anchor_ns
            self.next_anchor_ns += STEP_NS
            selected = select_history(times, anchor, 4, STEP_NS)
            if (selected is None or any(abs(anchor - distance * STEP_NS - times[index]) > 75_000_000
                                       for distance, index in zip((3, 2, 1, 0), selected))):
                self.features.clear()
                self.feature_times.clear()
                self.last_anchor_ns = None
                self.last_prediction['reset_reason'] = 'image_history_invalid'
                preprocess_ns += time.perf_counter_ns() - preprocessing
                continue
            inputs = torch.stack([self.images[index][1] for index in selected]).unsqueeze(0).float() / 255
            encoding = time.perf_counter_ns()
            preprocess_ns += encoding - preprocessing
            with torch.inference_mode():
                feature = self.encoder(image_features(inputs, 'motion'))[0]
            encoder_ns += time.perf_counter_ns() - encoding
            self.features.append(feature)
            self.feature_times.append(anchor)
            self.last_anchor_ns = anchor
            new_histories.append({'anchor_ns': anchor, 'image_complete_ns': [times[i] for i in selected]})
        if self.last_anchor_ns != latest_anchor:
            self.last_prediction['context_steps'] = len(self.features)
            return Action()
        anchor = latest_anchor
        temporal_started = time.perf_counter_ns()
        with torch.inference_mode():
            context = (torch.stack(list(self.features)) - self.mean) / self.scale
            logits = self.model(context.unsqueeze(0), torch.tensor([len(self.features)]))[0]
            if not torch.isfinite(logits).all():
                raise ValueError('The shadow model produced nonfinite logits')
            probabilities = logits.softmax(0).tolist()
        completed = time.perf_counter_ns()
        self.phase = 'infer'
        self.last_prediction.update({'phase': self.phase, 'inference_performed': True,
                                     'class': CLASSES[int(logits.argmax())],
                                     'probabilities': probabilities, 'classes': CLASSES,
                                     'context_steps': len(self.features), 'maximum_context_steps': self.steps,
                                     'anchor_ns': anchor, 'latest_image_age_ms': (anchor - times[selected[-1]]) / 1e6,
                                     'history_image_complete_ns': [times[i] for i in selected],
                                     'feature_anchor_ns': list(self.feature_times),
                                     'new_feature_histories': new_histories,
                                     'backfilled_feature_steps': max(0, len(new_histories) - 1),
                                     'preprocess_ms': preprocess_ns / 1e6,
                                     'encoder_ms': encoder_ns / 1e6,
                                     'temporal_ms': (completed - temporal_started) / 1e6})
        # The model has no action-magnitude or movement decoder.
        return Action()
