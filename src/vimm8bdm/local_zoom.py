"""Inspect one enlarged image region after a weak body proposal."""
from .pretrained_detection import person_boxes


def crop_window(prediction, width, height, accepted=None):
    """Return one bounded square only when no full-image candidate passes."""
    if (person_boxes(prediction, width, height, .5) if accepted is None else accepted):
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


def refine_with_local_zoom(model, image, full_prediction, full_tensor=None):
    """Use at most one extra model call with the unchanged score threshold."""
    from torchvision.transforms.functional import pil_to_tensor
    boxes = person_boxes(full_prediction, image.width, image.height, .5)
    if boxes:
        return boxes, None
    window = crop_window(full_prediction, image.width, image.height, boxes)
    if window is None:
        return boxes, None
    left, top, right, bottom = window
    if full_tensor is not None and tuple(full_tensor.shape[-2:]) == (image.height, image.width):
        tensor = full_tensor[:, top:bottom, left:right]
    else:
        crop = image.crop(window)
        raw = pil_to_tensor(crop)
        parameters = getattr(model, 'parameters', None)
        device = next(parameters(), raw).device if callable(parameters) else raw.device
        tensor = (raw.to(device).float().div_(255)
                  if str(device).startswith('cuda') else (raw.float() / 255).to(device))
    prediction = model([tensor])[0]
    return restore_boxes(person_boxes(prediction, right - left, bottom - top, .5), window), window
