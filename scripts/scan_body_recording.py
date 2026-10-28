"""Scan a private recording without game input or new ground-truth claims."""
import argparse
import io
import json
import math
from pathlib import Path
import statistics
import struct
import time

from PIL import Image
import torch
from torchvision.transforms.functional import pil_to_tensor

from vimm8bdm.local_zoom import refine_with_local_zoom
from vimm8bdm.pretrained_detection import check_transfer_evaluation, person_boxes
from vimm8bdm.recording import file_digest
from vimm8bdm.recording_review import review
from vimm8bdm.sprite_transfer import load_checkpoint
from vimm8bdm.visual_labels import source_identity
from vimm8bdm.visual_metrics import box_iou


def distribution(values):
    ordered = sorted(values)
    return dict(p50=statistics.median(ordered), p95=ordered[math.ceil(.95*len(ordered))-1],
                maximum=ordered[-1])


def scan(recording, checkpoint, output, local_zoom=False):
    if output.exists():
        raise FileExistsError('The scan directory already exists')
    integrity = review(recording)
    source, hashes = source_identity(recording)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    model, training = load_checkpoint(checkpoint)
    check_transfer_evaluation(dict(format='vimm8bdm-visible-coco-v1', split='calibration',
                                   source_session_id=source, source_hashes=hashes), training)
    frames = [json.loads(line) for line in (recording/'frames.jsonl').read_text().splitlines()]
    output.mkdir(parents=True, exist_ok=False)
    protocol = dict(format='vimm8bdm-continuous-body-scan-v1', source_session_id=source,
                    source_hashes=hashes, checkpoint_sha256=file_digest(checkpoint),
                    local_zoom=local_zoom, score_threshold=.5, threads=4, warmup_calls=5,
                    independent_test=False, ground_truth_available=False, input_sent=False)
    (output/'protocol.local.json').write_text(json.dumps(protocol, indent=2))
    cases = []
    with torch.inference_mode(), (recording/'frames.vrec').open('rb') as archive, (output/'frames.local.jsonl').open('x') as stream:
        for _ in range(5):
            model([torch.zeros(3, 480, 640)])
        started_scan = time.perf_counter()
        for row in frames:
            started = time.perf_counter()
            archive.seek(row['offset'])
            length = struct.unpack('<I', archive.read(4))[0]
            if length != row['png_bytes']:
                raise ValueError('A recorded image has an invalid archive length')
            raw = archive.read(length)
            if len(raw) != length:
                raise ValueError('A recorded image is incomplete')
            with Image.open(io.BytesIO(raw)) as image:
                rgb = image.convert('RGB')
            decoded = time.perf_counter()
            prediction = model([pil_to_tensor(rgb).float()/255])[0]
            boxes = person_boxes(prediction, rgb.width, rgb.height, .5)
            window = None
            if local_zoom:
                boxes, window = refine_with_local_zoom(model, rgb, prediction)
            for box in boxes:
                box['category'] = 'cutman_body_candidate'
            finished = time.perf_counter()
            capture_ms = (row['capture_finished_ns']-row['capture_started_ns'])/1e6
            case = dict(frame=row['frame'], segment=row['segment'],
                        recorded_time_ns=row['capture_started_ns'], predictions=boxes,
                        local_zoom_crop=window, decode_ms=(decoded-started)*1000,
                        processing_ms=(finished-decoded)*1000,
                        archive_to_result_ms=(finished-started)*1000,
                        recorded_capture_ms=capture_ms,
                        estimated_capture_plus_processing_ms=capture_ms+(finished-decoded)*1000)
            cases.append(case)
            stream.write(json.dumps(case)+'\n')
            if len(cases)%200 == 0:
                stream.flush()
                print(json.dumps(dict(scanned=len(cases), total=len(frames))), flush=True)
        elapsed = time.perf_counter()-started_scan
    unchanged = source_identity(recording) == (source, hashes)
    if not unchanged:
        raise ValueError('The recording changed during the scan')
    adjacent = [(a,b) for a,b in zip(cases,cases[1:]) if a['segment']==b['segment']]
    both = [(a,b) for a,b in adjacent if a['predictions'] and b['predictions']]
    def best(case):
        return max(case['predictions'], key=lambda box: box['score'])['box']
    summary = dict(frames=len(cases), frames_with_candidates=sum(bool(c['predictions']) for c in cases),
                   total_candidates=sum(len(c['predictions']) for c in cases),
                   frames_with_multiple_candidates=sum(len(c['predictions'])>1 for c in cases),
                   extra_model_calls=sum(c['local_zoom_crop'] is not None for c in cases),
                   candidate_presence_changes=sum(bool(a['predictions'])!=bool(b['predictions']) for a,b in adjacent),
                   adjacent_candidate_pairs=len(both),
                   adjacent_top_boxes_iou_below_point_one=sum(box_iou(best(a),best(b))<.1 for a,b in both),
                   processing_ms=distribution([c['processing_ms'] for c in cases]),
                   archive_to_result_ms=distribution([c['archive_to_result_ms'] for c in cases]),
                   estimated_capture_plus_processing_ms=distribution([c['estimated_capture_plus_processing_ms'] for c in cases]),
                   measured_processing_over_50_ms=sum(c['processing_ms']>50 for c in cases),
                   measured_processing_over_100_ms=sum(c['processing_ms']>100 for c in cases),
                   scan_seconds=elapsed)
    report = dict(protocol=protocol, integrity=integrity, summary=summary,
                  source_unchanged=unchanged, input_sent=False, recall_measured=False,
                  false_positive_rate_measured=False, live_capture_measured=False,
                  estimates_include_queuing=False, training_performed=False)
    (output/'report.local.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--local-zoom', action='store_true')
    args = parser.parse_args()
    scan(args.recording, args.checkpoint, args.output, args.local_zoom)
