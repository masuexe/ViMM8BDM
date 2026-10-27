"""Experimental adapter for separately supplied ContinuousDoom research files.

The repository does not include the upstream source or checkpoint.
The caller must trust the supplied Python files before this module imports them.
"""
from collections import deque
import importlib
import math
import time
from pathlib import Path
import sys

from .control import Action


def map_prediction(choice, yaw, yaw_scale):
    """Match the pinned defend_the_center.cfg button order."""
    if choice not in range(6):
        raise ValueError("The model action index must be between zero and five")
    if not math.isfinite(yaw) or not math.isfinite(yaw_scale) or yaw_scale <= 0:
        raise ValueError("Yaw must be finite. The yaw scale must be finite and positive")
    key = {2: "d", 3: "a", 4: "w", 5: "s"}.get(choice)
    return Action(frozenset({key}) if key else frozenset(), yaw * yaw_scale, 0, choice == 1)


class ContinuousPolicy:
    def __init__(self, source, checkpoint, yaw_scale):
        import numpy as np
        from PIL import Image
        import torch
        self.np, self.Image, self.torch = np, Image, torch
        self.yaw_scale = yaw_scale
        self.frames = deque(maxlen=4)
        source = Path(source).resolve(strict=True)
        if not all((source / name).is_file() for name in ("actor_critic.py", "modules.py")):
            raise ValueError("The source directory must contain actor_critic.py and modules.py")
        # Isolate this research import from unrelated modules with common names.
        if "modules" in sys.modules or "actor_critic" in sys.modules:
            raise RuntimeError("ContinuousDoom module names already exist in this process")
        sys.path.insert(0, str(source))
        try:
            upstream = importlib.import_module("actor_critic")
            architecture = importlib.import_module("modules").ConvFeatureExtractor
        finally:
            sys.path.remove(str(source))
        torch.set_num_threads(4)
        # Strict loading replaces every parameter. Random initialization is unnecessary.
        initialize = upstream.weights_init
        upstream.weights_init = lambda *args, **kwargs: None
        try:
            self.model = upstream.ActorCritic(4, 512, 6, 1, feature_extractor=architecture)
        finally:
            upstream.weights_init = initialize
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        self.last_prediction = None
        self.warmup_ms = []
        with torch.inference_mode():
            for _ in range(3):
                started = time.perf_counter_ns()
                self.model.predict(torch.zeros((1, 4, 240, 320)))
                self.warmup_ms.append((time.perf_counter_ns() - started) / 1e6)

    def reset(self):
        self.frames.clear()
        self.last_prediction = None

    def predict(self, frame):
        image = self.Image.frombytes("RGB", (frame.width, frame.height), frame.bgra,
                                     "raw", "BGRX")
        gray = image.resize((320, 240), self.Image.Resampling.BILINEAR).convert("L")
        self.frames.append(self.np.asarray(gray, dtype=self.np.float32) / 255.0)
        while len(self.frames) < 4:
            self.frames.append(self.frames[-1].copy())
        observation = self.torch.from_numpy(self.np.stack(self.frames)[None])
        with self.torch.inference_mode():
            categorical, yaw = self.model.predict(observation)
        choice = int(categorical.item())
        raw_yaw = float(yaw.item())
        action = map_prediction(choice, raw_yaw, self.yaw_scale)
        self.last_prediction = {"choice": choice, "yaw": raw_yaw, "yaw_scale": self.yaw_scale}
        return action
