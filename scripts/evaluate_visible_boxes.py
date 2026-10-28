"""Measure a rejected sprite baseline against private calibration boxes."""
import argparse
import json
from pathlib import Path
import statistics
import time

import cv2
from PIL import Image

from evaluate_sprite_perception import overlap
from vimm8bdm.recording import file_digest
from vimm8bdm.sprite_perception import SpriteMatcher


def evaluate(dataset, bank, output):
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
    result = dict(kind='real_image_calibration_diagnostic', independent_test=False,
                  human_box_reviewed=metadata['human_box_reviewed'], input_sent=False,
                  threshold=.97, minimum_iou=.5, variants={})
    for mode in ('neutral', 'opaque'):
        matcher = SpriteMatcher(bank, mask=mode, threshold=.97)
        cases = []
        for image in coco['images']:
            path = dataset / image['file_name']
            if file_digest(path) != metadata['image_sha256'][str(image['id'])]:
                raise ValueError('A calibration image changed after export')
            with Image.open(path) as pixels:
                rgb = pixels.convert('RGB')
            started = time.perf_counter()
            prediction = matcher.best(rgb)
            elapsed = (time.perf_counter() - started) * 1000
            boxes = expected.get(image['id'], [])
            iou = max((overlap(box, prediction['box']) for box in boxes), default=0)
            cases.append(dict(frame=image['id'], positive=bool(boxes), iou=iou,
                              processing_ms=elapsed, **prediction))
        positives = [row for row in cases if row['positive']]
        negatives = [row for row in cases if not row['positive']]
        result['variants'][mode] = dict(
            positive_images=len(positives), negative_images=len(negatives),
            localized_positives=sum(r['candidate'] and r['iou'] >= .5 for r in positives),
            negative_candidates=sum(r['candidate'] for r in negatives),
            median_processing_ms=statistics.median(r['processing_ms'] for r in cases), cases=cases)
    with output.open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({k: {n: v for n, v in row.items() if n != 'cases'}
                      for k, row in result['variants'].items()}, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--bank', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.dataset, args.bank, args.output)
