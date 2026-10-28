"""Test fixed Cutman color proposals on private calibration images only."""
import argparse
import json
from pathlib import Path
import statistics
import time

import cv2
import numpy as np
from PIL import Image

from evaluate_sprite_perception import overlap
from vimm8bdm.recording import file_digest


def proposals(image):
    pixels = np.asarray(image.resize((320, 240), Image.Resampling.BILINEAR), dtype=np.float32)
    red, green, blue = pixels[:, :, 0], pixels[:, :, 1], pixels[:, :, 2]
    mask = ((red >= 90) & (red > 1.7 * green) & (blue > .15 * red) & (blue < .85 * red))
    # The bottom 40 pixels contain the HUD. The remaining weapon can still cause proposals.
    mask[200:] = False
    merged = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(merged, connectivity=8)
    boxes = []
    for stat in stats[1:count]:
        x, y, width, height, area = map(int, stat)
        if area < 6 or width < 2 or height < 3:
            continue
        left, top = max(0, x - round(width * .3)), max(0, y - round(height * .2))
        right, bottom = min(320, x + width + round(width * .3)), min(200, y + height + round(height * .2))
        boxes.append([left * image.width / 320, top * image.height / 240,
                      (right - left) * image.width / 320, (bottom - top) * image.height / 240])
    return boxes


def evaluate(dataset, output):
    if output.exists():
        raise FileExistsError('The evaluation output already exists')
    metadata = json.loads((dataset / 'dataset.json').read_text())
    if metadata['format'] != 'vimm8bdm-visible-coco-v1' or metadata['split'] != 'calibration':
        raise ValueError('Supply a visible-body calibration export')
    coco = json.loads((dataset / 'annotations.coco.json').read_text())
    expected = {}
    for annotation in coco['annotations']:
        expected.setdefault(annotation['image_id'], []).append(annotation['bbox'])
    cv2.setNumThreads(4)
    cases = []
    for row in coco['images']:
        path = dataset / row['file_name']
        if file_digest(path) != metadata['image_sha256'][str(row['id'])]:
            raise ValueError('A calibration image changed after export')
        with Image.open(path) as image:
            rgb = image.convert('RGB')
        started = time.perf_counter()
        boxes = proposals(rgb)
        elapsed = (time.perf_counter() - started) * 1000
        targets = expected.get(row['id'], [])
        iou = max((overlap(a, b) for a in boxes for b in targets), default=0)
        cases.append(dict(frame=row['id'], positive=bool(targets), proposals=boxes,
                          best_iou=iou, processing_ms=elapsed))
    positives = [row for row in cases if row['positive']]
    negatives = [row for row in cases if not row['positive']]
    result = dict(kind='fixed_color_calibration_diagnostic', independent_test=False,
                  human_box_reviewed=metadata['human_box_reviewed'], input_sent=False,
                  parameters=dict(red_minimum=90,red_green_ratio_minimum=1.7,
                                  blue_red_ratio_range=[.15,.85],closing_kernel=5,minimum_area=6,
                                  minimum_width=2,minimum_height=3,scene_bottom=200,
                                  horizontal_expansion=.3,vertical_expansion=.2),
                  summary=dict(positive_images=len(positives),negative_images=len(negatives),
                               localized_positives=sum(row['best_iou'] >= .5 for row in positives),
                               negative_images_with_proposals=sum(bool(row['proposals']) for row in negatives),
                               total_proposals=sum(len(row['proposals']) for row in cases),
                               median_processing_ms=statistics.median(row['processing_ms'] for row in cases)),
                  cases=cases)
    with output.open('x') as stream:
        json.dump(result,stream,indent=2)
    print(json.dumps(result['summary'],indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    evaluate(args.dataset,args.output)
