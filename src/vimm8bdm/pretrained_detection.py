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


def bind_single_class_postprocess(model):
    """Avoid 90-class host-device syncs in SSD.postprocess_detections when only class 1 is used."""
    if not (hasattr(model, 'postprocess_detections') and hasattr(model, 'box_coder')):
        return
    import torch
    import torch.nn.functional as F
    from torchvision.ops import boxes as box_ops

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
