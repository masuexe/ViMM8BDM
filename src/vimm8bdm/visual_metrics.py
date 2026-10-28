"""Measure overlap between visible image boxes."""


def box_iou(a, b):
    """Return intersection over union for two positive-area xywh boxes."""
    if len(a) != 4 or len(b) != 4 or min(a[2], a[3], b[2], b[3]) <= 0:
        raise ValueError('Supply two boxes with positive width and height')
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    return intersection / (a[2] * a[3] + b[2] * b[3] - intersection)
