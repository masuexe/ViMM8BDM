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


def check_transfer_evaluation(metadata, checkpoint, validation=False):
    """Reject training recordings and require complete provenance for reserved validation."""
    expected = 'validation' if validation else 'calibration'
    if metadata.get('format') != 'vimm8bdm-visible-coco-v1' or metadata.get('split') != expected:
        raise ValueError('Select the intended calibration or reserved validation split explicitly')
    primary = checkpoint.get('training_source_session_id')
    sources = set(checkpoint.get('training_source_session_ids', [primary]))
    archives = checkpoint.get('training_source_archives', {})
    if (metadata.get('source_session_id') in sources
            or metadata.get('source_hashes', {}).get('frames.vrec') in archives.values()):
        raise ValueError('The evaluation recording or image archive supplied training data')
    if validation and (metadata.get('reserved_for_validation') is not True
                       or not primary or primary not in sources or not sources
                       or set(archives) != sources or not all(archives.values())):
        raise ValueError('Reserved validation requires complete project recording provenance')


def complete_synthetic_provenance(checkpoint, dataset, dataset_sha256):
    """Recover legacy recording provenance only from the checkpoint-bound synthetic dataset."""
    if (dataset_sha256 != checkpoint.get('synthetic_dataset_sha256')
            or dataset.get('format') != 'vimm8bdm-synthetic-body-coco-v1'
            or dataset.get('source_session_id') != checkpoint.get('training_source_session_id')
            or not dataset.get('source_hashes', {}).get('frames.vrec')):
        raise ValueError('The provenance dataset does not match the legacy synthetic checkpoint')
    source = dataset['source_session_id']
    return {**checkpoint, 'training_source_session_ids': [source],
            'training_source_archives': {source: dataset['source_hashes']['frames.vrec']},
            'provenance_companion_sha256': dataset_sha256}
