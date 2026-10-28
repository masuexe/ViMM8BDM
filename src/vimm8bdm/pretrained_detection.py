"""Run an official COCO detector without game control or implicit downloads."""
import math
from pathlib import Path

from .recording import file_digest

CHECKPOINT_SHA256 = 'a79551df90c79834bcd3bb3845ef9d966b5449a3a9b2833ae8404778ca5d65d2'


def person_boxes(prediction, width, height, threshold=.5):
    """Retain finite COCO person boxes in source-image coordinates."""
    if not 0 <= threshold <= 1 or min(width, height) <= 0:
        raise ValueError('Supply a valid score threshold and image dimensions')
    output = []
    rows = zip(prediction['boxes'].tolist(), prediction['labels'].tolist(),
               prediction['scores'].tolist())
    for box, label, score in rows:
        if label != 1 or not math.isfinite(score) or score < threshold:
            continue
        if len(box) != 4 or not all(math.isfinite(value) for value in box):
            continue
        left, top, right, bottom = box
        left, right = max(0., left), min(float(width), right)
        top, bottom = max(0., top), min(float(height), bottom)
        if right <= left or bottom <= top:
            continue
        output.append(dict(box=[left, top, right - left, bottom - top], score=score,
                           category='coco_person', enemy_identification_verified=False))
    return output


def load_model(checkpoint):
    if file_digest(Path(checkpoint)) != CHECKPOINT_SHA256:
        raise ValueError('Supply the official SSDLite320 COCO V1 checkpoint with its expected hash')
    import torch
    from torchvision.models.detection import ssdlite320_mobilenet_v3_large
    model = ssdlite320_mobilenet_v3_large(weights=None, weights_backbone=None)
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
    model.eval()
    return model
