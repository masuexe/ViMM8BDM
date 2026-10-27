"""Evaluate a private sprite bank on recorded backgrounds and artificial positives."""
import argparse
import io
import json
import math
from pathlib import Path
import random
import statistics
import struct
import time
import zipfile

import cv2
import numpy as np
from PIL import Image, ImageDraw

from vimm8bdm.recording import file_digest
from vimm8bdm.recording_review import review
from vimm8bdm.sprite_perception import SpriteMatcher


def overlap(a, b):
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    return intersection / (a[2] * a[3] + b[2] * b[3] - intersection)


def evaluate(args):
    if args.output.exists():
        raise FileExistsError('The evaluation directory already exists')
    cv2.setNumThreads(4)
    integrity = review(args.recording)
    manifest = json.loads((args.bank / 'bank.json').read_text())
    if file_digest(args.base_game) != manifest['sources'][0]['sha256']:
        raise ValueError('The base-game palette must match the sprite bank source')
    index = [json.loads(line) for line in (args.recording / 'frames.jsonl').read_text().splitlines()]
    selected = np.linspace(0, len(index) - 1, args.negative_frames, dtype=int).tolist()
    backgrounds = []
    with (args.recording / 'frames.vrec').open('rb') as archive:
        for number in selected:
            archive.seek(index[number]['offset'])
            length = struct.unpack('<I', archive.read(4))[0]
            with Image.open(io.BytesIO(archive.read(length))) as image:
                backgrounds.append(image.convert('RGB').resize((320, 240), Image.Resampling.BILINEAR))
    with zipfile.ZipFile(args.base_game) as archive:
        palette = np.frombuffer(archive.read('PLAYPAL.pal'), dtype=np.uint8).reshape(256, 3)
    rng = random.Random(7)
    synthetic = []
    for i in range(args.synthetic_cases):
        exact = i < args.synthetic_cases // 2
        height = ((16, 24, 32) if exact else (20, 28, 40))[i % 3]
        rotation = i % 8 + 1
        with Image.open(args.bank / f'rotation-{rotation}.png') as image:
            pixels = np.array(image.convert('RGBA'))
        recolored = bool(rng.randrange(2))
        if recolored:
            original = pixels[:, :, :3].copy()
            # These three simple index replacements come from the supplied CBM color source.
            for source, destination in ((192, 4), (198, 41), (202, 96)):
                pixels[:, :, :3][np.all(original == palette[source], axis=2)] = palette[destination]
        brightness = rng.choice((.75, 1.))
        pixels[:, :, :3] = np.rint(pixels[:, :, :3] * brightness).astype(np.uint8)
        sprite = Image.fromarray(pixels)
        width = round(sprite.width * height / sprite.height)
        resample = rng.choice((Image.Resampling.NEAREST, Image.Resampling.BILINEAR))
        sprite = sprite.resize((width, height), resample)
        left, top = rng.randrange(80, 210), rng.randrange(35, 80)
        scene = backgrounds[i % len(backgrounds)].copy()
        scene.paste(sprite, (left, top), sprite)
        synthetic.append((scene, [left, top, width, height],
                          {'rotation': rotation, 'height': height, 'exact_bank_scale': exact,
                           'brightness': brightness, 'recolored': recolored, 'resample': resample.name}))
    args.output.mkdir(parents=True, exist_ok=False)
    sheet = Image.new('RGB', (1280, 140 * math.ceil(len(backgrounds) / 8)))
    draw = ImageDraw.Draw(sheet)
    for i, image in enumerate(backgrounds):
        x, y = i % 8 * 160, i // 8 * 140
        sheet.paste(image.resize((160, 120)), (x, y))
        draw.text((x + 3, y + 121), str(selected[i]), fill='white')
    sheet.save(args.output / 'negative-contact-sheet.local.png')
    result = {'kind': 'offline_sprite_perception', 'opencv': cv2.__version__,
              'source_integrity': integrity, 'source_hashes': manifest['sources'],
              'negative_frame_indices': selected, 'threshold': .97, 'variants': {},
              'negative_frames_visually_reviewed': False, 'synthetic_positive_game_render_verified': False,
              'actual_enemy_detection_verified': False, 'input_sent': False}
    for mode in ('neutral', 'opaque'):
        matcher = SpriteMatcher(args.bank, mask=mode, threshold=.97)
        negative, positive, times = [], [], []
        for image, number in zip(backgrounds, selected):
            started = time.perf_counter()
            match = matcher.best(image)
            times.append((time.perf_counter() - started) * 1000)
            negative.append({'frame': number, **match})
        for image, box, context in synthetic:
            started = time.perf_counter()
            match = matcher.best(image)
            times.append((time.perf_counter() - started) * 1000)
            positive.append({**context, **match, 'expected_box': box, 'iou': overlap(box, match['box'])})
        exact = [r for r in positive if r['exact_bank_scale']]
        heldout = [r for r in positive if not r['exact_bank_scale']]
        recall = lambda rows: sum(r['candidate'] and r['iou'] >= .5 for r in rows) / len(rows)
        times.sort()
        summary = {'negative_candidates': sum(r['candidate'] for r in negative),
                   'negative_samples': len(negative), 'exact_scale_recall_at_iou_0_5': recall(exact),
                   'heldout_scale_recall_at_iou_0_5': recall(heldout),
                   'processing_ms': {'samples': len(times), 'p50': statistics.median(times),
                                     'p95': times[math.ceil(.95 * len(times)) - 1], 'maximum': times[-1]}}
        conditions = {'no_negative_candidates': summary['negative_candidates'] == 0,
                      'exact_scale_recall_at_least_0_9': recall(exact) >= .9,
                      'heldout_scale_recall_at_least_0_8': recall(heldout) >= .8,
                      'median_ms_at_most_50': summary['processing_ms']['p50'] <= 50}
        result['variants'][mode] = {'summary': summary, 'conditions': conditions,
                                    'conditions_passed': all(conditions.values()),
                                    'negative_cases': negative, 'synthetic_cases': positive}
        print(json.dumps({'mask': mode, 'summary': summary, 'conditions': conditions}), flush=True)
    with (args.output / 'report.local.json').open('x') as stream:
        json.dump(result, stream, indent=2)
    return result


def main():
    parser = argparse.ArgumentParser(description='Check an offline sprite baseline without desktop input')
    parser.add_argument('--recording', type=Path, required=True)
    parser.add_argument('--bank', type=Path, required=True)
    parser.add_argument('--base-game', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--negative-frames', type=int, default=32)
    parser.add_argument('--synthetic-cases', type=int, default=48)
    args = parser.parse_args()
    if not 8 <= args.negative_frames <= 256 or not 12 <= args.synthetic_cases <= 256 or args.synthetic_cases % 2:
        parser.error('Select 8 through 256 backgrounds and an even count of 12 through 256 artificial cases')
    evaluate(args)


if __name__ == '__main__':
    main()
