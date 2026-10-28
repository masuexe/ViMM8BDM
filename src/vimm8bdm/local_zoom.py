"""Inspect one enlarged image region after a weak body proposal."""
from .pretrained_detection import person_boxes


def crop_window(prediction, width, height):
    """Return one bounded square only when no full-image candidate passes."""
    if person_boxes(prediction, width, height, .5):
        return None
    proposals = person_boxes(prediction, width, height, .2)
    if not proposals:
        return None
    best = max(proposals, key=lambda row: row['score'])
    x, y, w, h = best['box']
    size = min(320, width, height)
    left = max(0, min(width - size, round(x + w / 2 - size / 2)))
    top = max(0, min(height - size, round(y + h / 2 - size / 2)))
    return left, top, left + size, top + size


def restore_boxes(boxes, window):
    """Translate bounded crop boxes without changing their original values."""
    left, top, _, _ = window
    return [{**row, 'box': [row['box'][0] + left, row['box'][1] + top,
                           row['box'][2], row['box'][3]]} for row in boxes]


def refine_with_local_zoom(model, image, full_prediction):
    """Use at most one extra model call with the unchanged score threshold."""
    from torchvision.transforms.functional import pil_to_tensor
    window = crop_window(full_prediction, image.width, image.height)
    if window is None:
        return person_boxes(full_prediction, image.width, image.height, .5), None
    crop = image.crop(window)
    prediction = model([pil_to_tensor(crop).float() / 255])[0]
    return restore_boxes(person_boxes(prediction, crop.width, crop.height, .5), window), window
