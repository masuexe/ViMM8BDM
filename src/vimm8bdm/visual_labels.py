"""Export reviewed visible boxes without changing the original recording."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct

from .recording import file_digest
from .recording_review import review

FORMAT = 'vimm8bdm-visual-box-review-v1'
SOURCE_FILES = ('session.json', 'summary.json', 'events.jsonl', 'frames.jsonl', 'frames.vrec')


def source_identity(directory):
    hashes = {name: file_digest(directory / name) for name in SOURCE_FILES}
    session = hashlib.sha256(json.dumps({name: hashes[name] for name in
                                        ('events.jsonl', 'frames.jsonl', 'frames.vrec')},
                                       sort_keys=True).encode()).hexdigest()
    return session, hashes


def validate_labels(labels, frames):
    if labels.get('format') != FORMAT or labels.get('split') not in ('calibration', 'validation') or not labels.get('annotator'):
        raise ValueError('Supply an identified calibration or validation review in the supported format')
    if type(labels.get('human_box_reviewed')) is not bool:
        raise ValueError('State whether a human reviewed the actual boxes')
    previous = -1
    for row in labels['frames']:
        index = row['frame']
        if type(index) is not int or not previous < index < len(frames):
            raise ValueError('Reviewed frame identifiers must be ordered, unique, and present')
        previous = index
        boxes, presence = row['boxes'], row['visible_presence']
        if row['status'] == 'exclude':
            if not row.get('reason') or boxes:
                raise ValueError('An excluded frame requires a reason and no accepted boxes')
            continue
        if row['status'] != 'usable' or presence not in ('present', 'absent'):
            raise ValueError('Only reviewed present or absent frames can enter the image export')
        if (presence == 'present') != bool(boxes):
            raise ValueError('Visible-presence labels must agree with the accepted boxes')
        for box in boxes:
            rect = box['xywh']
            if box['class'] != 'cutman_body' or len(rect) != 4 or any(type(v) is not int for v in rect):
                raise ValueError('Each Cutman body box requires four integer coordinates')
            x, y, width, height = rect
            if (min(x, y) < 0 or min(width, height) <= 0
                    or x + width > frames[index]['width'] or y + height > frames[index]['height']):
                raise ValueError('A visible box extends outside its source image')
            if any(type(box.get(k)) is not bool for k in ('occluded', 'truncated')):
                raise ValueError('State the box occlusion and truncation flags')
    if previous < 0:
        raise ValueError('The visual review contains no frames')


def export(recording, label_file, output):
    recording, output = recording.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError('The image export directory already exists')
    if recording in output.parents:
        raise ValueError('Export labels outside the original recording directory')
    integrity = review(recording)
    session, hashes = source_identity(recording)
    labels = json.loads(label_file.read_text())
    if labels.get('source_session_id') != session:
        raise ValueError('The visual review belongs to a different recording')
    frames = [json.loads(line) for line in (recording / 'frames.jsonl').read_text().splitlines()]
    validate_labels(labels, frames)
    usable = [row for row in labels['frames'] if row['status'] == 'usable']
    if not usable:
        raise ValueError('The visual review has no accepted images')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'images').mkdir()
    images, annotations, image_hashes = [], [], {}
    with (recording / 'frames.vrec').open('rb') as archive:
        for row in usable:
            source = frames[row['frame']]
            archive.seek(source['offset'])
            length = struct.unpack('<I', archive.read(4))[0]
            if length != source['png_bytes']:
                raise ValueError('An exported image has an invalid archive length')
            raw = archive.read(length)
            filename = f'images/{row["frame"]:06d}.png'
            with (output / filename).open('xb') as stream:
                stream.write(raw)
            image_hashes[str(row['frame'])] = hashlib.sha256(raw).hexdigest()
            images.append({'id': row['frame'], 'file_name': filename, 'width': source['width'],
                           'height': source['height'], 'view_state': row['view_state']})
            for box in row['boxes']:
                annotations.append({'id': len(annotations) + 1, 'image_id': row['frame'],
                                    'category_id': 1, 'bbox': box['xywh'],
                                    'area': box['xywh'][2] * box['xywh'][3], 'iscrowd': 0,
                                    'occluded': box['occluded'], 'truncated': box['truncated']})
    if source_identity(recording) != (session, hashes):
        raise ValueError('The original recording changed during export')
    coco = {'info': {'description': 'Candidate visible Cutman body labels', 'split': labels['split'],
                      'source_session_id': session, 'training_ready': False},
            'images': images, 'annotations': annotations,
            'categories': [{'id': 1, 'name': 'cutman_body'}]}
    metadata = {'format': 'vimm8bdm-visible-coco-v1', 'source_session_id': session,
                'source_directory': Path(os.path.relpath(recording, output)).as_posix(),
                'source_hashes': hashes, 'source_integrity': integrity, 'split': labels['split'],
                'reserved_for_validation': labels['split'] == 'validation',
                'review_sha256': file_digest(label_file), 'annotator': labels['annotator'],
                'human_box_reviewed': labels['human_box_reviewed'], 'source_files_changed': False,
                'images': len(images), 'positive_images': sum(bool(row['boxes']) for row in usable),
                'negative_images': sum(not row['boxes'] for row in usable), 'boxes': len(annotations),
                'excluded_reviewed_frames': len(labels['frames']) - len(usable),
                'image_sha256': image_hashes, 'training_ready': False,
                'action_training_eligible': False, 'enemy_or_life_state_verified': False}
    with (output / 'annotations.coco.json').open('x') as stream:
        json.dump(coco, stream, indent=2)
    with (output / 'dataset.json').open('x') as stream:
        json.dump(metadata, stream, indent=2)
    return metadata


def main():
    parser = argparse.ArgumentParser(description='Export private reviewed body boxes as COCO')
    parser.add_argument('--recording', type=Path, required=True)
    parser.add_argument('--review', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = export(args.recording, args.review, args.output)
    print(json.dumps({k: result[k] for k in ('images', 'boxes', 'excluded_reviewed_frames', 'training_ready')}, indent=2))


if __name__ == '__main__':
    main()
