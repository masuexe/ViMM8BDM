"""Inspect one enlarged image region after a weak body proposal."""
from .pretrained_detection import person_boxes


def crop_window(prediction, width, height, accepted=None, hint_box=None, accept_score=.5):
    """Return one bounded square only when no full-image candidate passes."""
    if (person_boxes(prediction, width, height, accept_score) if accepted is None else accepted):
        return None
    proposals = person_boxes(prediction, width, height, .2)
    if proposals:
        x, y, w, h = max(proposals, key=lambda row: row['score'])['box']
    elif hint_box is not None:
        x, y, w, h = hint_box
    else:
        return None
    size = min(320, width, height)
    left = max(0, min(width - size, round(x + w / 2 - size / 2)))
    top = max(0, min(height - size, round(y + h / 2 - size / 2)))
    return left, top, left + size, top + size


def restore_boxes(boxes, window):
    """Translate bounded crop boxes without changing their original values."""
    left, top, _, _ = window
    return [{**row, 'box': [row['box'][0] + left, row['box'][1] + top,
                           row['box'][2], row['box'][3]]} for row in boxes]


def refine_with_local_zoom(model, image, full_prediction, full_tensor=None, hint_box=None, accept_score=.5):
    """Use at most one extra model call with the unchanged score threshold."""
    from torchvision.transforms.functional import pil_to_tensor
    boxes = person_boxes(full_prediction, image.width, image.height, accept_score)
    if boxes:
        return boxes, None
    window = crop_window(full_prediction, image.width, image.height, boxes,
                         hint_box=hint_box, accept_score=accept_score)
    if window is None:
        return person_boxes(full_prediction, image.width, image.height, .5), None
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
    zoom_boxes = restore_boxes(person_boxes(prediction, right - left, bottom - top, .5), window)
    return zoom_boxes or person_boxes(full_prediction, image.width, image.height, .5), window
