"""Evaluate SSDLite body candidates on private calibration images."""
import argparse
import json
import math
from pathlib import Path
import statistics
import time

from PIL import Image
import torch
import torchvision
from torchvision.transforms.functional import pil_to_tensor

from vimm8bdm.visual_metrics import box_iou as overlap
from vimm8bdm.pretrained_detection import load_model, person_boxes, check_transfer_evaluation
from vimm8bdm.recording import file_digest


def evaluate(dataset, checkpoint, output, transfer=False, validation=False, provenance_dataset=None, validation_reuse=False):
    if output.exists():
        raise FileExistsError('The evaluation directory already exists')
    metadata = json.loads((dataset / 'dataset.json').read_text())
    if validation_reuse and not validation:
        raise ValueError('Declare validation reuse only with an explicit reserved validation run')
    expected_split = 'validation' if validation else 'calibration'
    if metadata['format'] != 'vimm8bdm-visible-coco-v1' or metadata['split'] != expected_split:
        raise ValueError('Select the intended calibration or reserved validation export')
    if validation and not transfer:
        raise ValueError('Reserved validation requires a transfer checkpoint with project provenance')
    coco = json.loads((dataset / 'annotations.coco.json').read_text())
    expected = {}
    for annotation in coco['annotations']:
        expected.setdefault(annotation['image_id'], []).append(annotation['bbox'])
    torch.set_num_threads(4)
    torch.manual_seed(0)
    if transfer:
        from vimm8bdm.sprite_transfer import load_checkpoint
        model, training = load_checkpoint(checkpoint, provenance_dataset=provenance_dataset)
        check_transfer_evaluation(metadata, training, validation=validation)
    else:
        model = load_model(checkpoint)
    threshold = .5
    cases = []
    with torch.inference_mode():
        # Exclude initialization and five warmup calls from measured processing.
        for _ in range(5):
            model([torch.zeros(3, 480, 640)])
        for row in coco['images']:
            path = dataset / row['file_name']
            if file_digest(path) != metadata['image_sha256'][str(row['id'])]:
                raise ValueError('A calibration image changed after export')
            with Image.open(path) as image:
                rgb = image.convert('RGB')
            started = time.perf_counter()
            tensor = pil_to_tensor(rgb).to(dtype=torch.float32) / 255
            prediction = model([tensor])[0]
            boxes = person_boxes(prediction, rgb.width, rgb.height, threshold)
            if transfer:
                for box in boxes:
                    box['category'] = 'cutman_body_candidate'
            elapsed = (time.perf_counter() - started) * 1000
            targets = expected.get(row['id'], [])
            iou = max((overlap(box['box'], target) for box in boxes for target in targets), default=0)
            all_person_scores = prediction['scores'][prediction['labels'] == 1].tolist()
            cases.append(dict(frame=row['id'],positive=bool(targets),predictions=boxes,
                              best_iou=iou,maximum_person_score=max(all_person_scores,default=0),
                              processing_ms=elapsed))
    positives = [row for row in cases if row['positive']]
    negatives = [row for row in cases if not row['positive']]
    times = sorted(row['processing_ms'] for row in cases)
    located = sum(row['best_iou'] >= .5 for row in positives)
    false_positive = sum(bool(row['predictions']) for row in negatives)
    conditions = dict(recall_at_least_0_8=located / len(positives) >= .8,
                      zero_negative_candidates=false_positive == 0,
                      median_processing_ms_at_most_50=statistics.median(times) <= 50)
    kind = ('ssdlite_separate_session_visual_validation' if validation else
            'ssdlite_sprite_transfer_calibration_diagnostic' if transfer else 'pretrained_ssdlite_calibration_diagnostic')
    result = dict(kind=kind,independent_test=validation and not validation_reuse,independent_session_validation=validation,
                  validation_reused_after_prior_result=validation_reuse,
                  unseen_map_test=False,evaluation_split=expected_split,
                  annotations_sha256=file_digest(dataset/'annotations.coco.json'),
                  export_metadata_sha256=file_digest(dataset/'dataset.json'),
                  torch=torch.__version__,torchvision=torchvision.__version__,device='cpu',threads=4,
                  checkpoint_sha256=file_digest(checkpoint),checkpoint_bytes=checkpoint.stat().st_size,
                  model='ssdlite320_mobilenet_v3_large',weights='local_sprite_transfer' if transfer else 'COCO_V1',class_id=1,
                  score_threshold=threshold,minimum_iou=.5,warmup_calls=5,
                  human_box_reviewed=metadata['human_box_reviewed'],input_sent=False,
                  training_performed=transfer,source_session_id=metadata['source_session_id'],
                  summary=dict(positive_images=len(positives),negative_images=len(negatives),
                               localized_positives=located,negative_images_with_predictions=false_positive,
                               total_predictions=sum(len(row['predictions']) for row in cases),
                               processing_ms=dict(p50=statistics.median(times),
                                                  p95=times[math.ceil(.95*len(times))-1],maximum=times[-1])),
                  conditions=conditions,conditions_passed=all(conditions.values()),cases=cases)
    output.mkdir(parents=True,exist_ok=False)
    with (output/'report.local.json').open('x') as stream:
        json.dump(result,stream,indent=2)
    print(json.dumps({k:result[k] for k in ('summary','conditions','conditions_passed')},indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--transfer',action='store_true')
    parser.add_argument('--validation',action='store_true')
    parser.add_argument('--provenance-dataset',type=Path)
    parser.add_argument('--validation-reuse',action='store_true')
    args = parser.parse_args()
    evaluate(args.dataset,args.checkpoint,args.output,transfer=args.transfer,validation=args.validation,
             provenance_dataset=args.provenance_dataset,validation_reuse=args.validation_reuse)
