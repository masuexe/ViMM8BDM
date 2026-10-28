"""Create private Cutman image composites from original assets and reviewed backgrounds."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import random
import re
import struct
import zipfile

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

from .recording import file_digest
from .recording_review import review
from .visual_labels import source_identity

POSES = 'ABCDEFGJKLMQRSTVW'
BANK_FORMAT = 'vimm8bdm-cutman-upright-bank-v1'
DATA_FORMAT = 'vimm8bdm-synthetic-body-coco-v1'


def build_bank(packages, output):
    if output.exists():
        raise FileExistsError('The sprite training bank already exists')
    sources, resolved = [], {}
    palette, actor_hash = None, None
    for package in packages:
        sources.append(dict(name=package.name, sha256=file_digest(package)))
        with zipfile.ZipFile(package) as archive:
            if palette is None and 'PLAYPAL.pal' in archive.namelist():
                palette = np.frombuffer(archive.read('PLAYPAL.pal')[:768], dtype=np.uint8).reshape(256, 3)
            for member in archive.namelist():
                normalized = member.replace('\\', '/')
                if normalized.lower() == 'actors/megaman/cutman.txt':
                    actor_hash = hashlib.sha256(archive.read(member)).hexdigest()
                if not normalized.lower().startswith('sprites/'):
                    continue
                name = normalized.rsplit('/', 1)[-1].upper()
                match = re.fullmatch(r'CUTM([A-Z])([0-8])(?:([A-Z])([0-8]))?\.PNG', name)
                if match:
                    resolved[name[:-4]] = (archive.read(member), package.name, member, match.groups())
    if palette is None:
        raise ValueError('The original base package must supply its PLAYPAL palette')
    views = {}
    for raw, package, member, pairs in resolved.values():
        with Image.open(io.BytesIO(raw)) as source:
            image = source.convert('RGBA')
        for frame, turn, mirrored in ((pairs[0], pairs[1], False), (pairs[2], pairs[3], True)):
            if frame is None or frame not in POSES:
                continue
            selected = ImageOps.mirror(image) if mirrored else image.copy()
            # The three simple replacements follow the supplied Cutman translation source.
            pixels = np.array(selected)
            original = pixels[:, :, :3].copy()
            for start, end in ((192, 4), (198, 41), (202, 96)):
                pixels[:, :, :3][np.all(original == palette[start], axis=2)] = palette[end]
            selected = Image.fromarray(pixels)
            bounds = selected.getchannel('A').getbbox()
            if bounds is None:
                raise ValueError('An upright reference has no visible pixels')
            for rotation in (range(1, 9) if turn == '0' else (int(turn),)):
                views[frame, rotation] = (selected.crop(bounds), package, member, mirrored)
    if set(views) != {(pose, rotation) for pose in POSES for rotation in range(1, 9)}:
        raise ValueError('The original packages must supply all reviewed upright views')
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for (pose, rotation), (image, package, member, mirrored) in sorted(views.items()):
        filename = f'{pose}-{rotation}.png'
        image.save(output / filename)
        rows.append(dict(pose=pose, rotation=rotation, image=filename,
                         sha256=file_digest(output / filename), source_package=package,
                         source_member=member, mirrored=mirrored))
    for package, expected in zip(packages, sources):
        if file_digest(package) != expected['sha256']:
            raise ValueError('An original package changed during reference extraction')
    metadata = dict(format=BANK_FORMAT, sources=sources, templates=rows, upright_frames=list(POSES),
                    actor_source_sha256=actor_hash, translation_pairs=[[192, 4], [198, 41], [202, 96]],
                    source_files_changed=False, actual_game_render_verified=False, input_sent=False)
    (output / 'bank.json').write_text(json.dumps(metadata, indent=2))
    return metadata


def composite(background, sprite, rng):
    """Return one RGB image and the visible xywh box after clipping and optional occlusion."""
    image = background.convert('RGB').resize((320, 240), Image.Resampling.BILINEAR)
    height = round(2 ** rng.uniform(3.6, 7.65))
    width = max(2, round(sprite.width * height / sprite.height * rng.uniform(.9, 1.1)))
    sprite = sprite.resize((width, height), rng.choice((Image.Resampling.NEAREST, Image.Resampling.BILINEAR)))
    sprite = ImageEnhance.Brightness(sprite).enhance(rng.uniform(.65, 1.1))
    left = rng.randint(0, max(0, 320 - width))
    top = rng.randint(8, max(8, 198 - height))
    if rng.random() < .2:
        left = rng.choice((-round(width * .25), 320 - round(width * .75)))
    if rng.random() < .15:
        top = -round(height * .25)
    layer = Image.new('RGBA', image.size)
    layer.paste(sprite, (left, top))
    alpha = np.array(layer.getchannel('A'))
    alpha[200:] = 0
    if rng.random() < .25:
        x = max(0, left + width // 2)
        y = max(0, top + height // 2)
        alpha[y:min(200, y + max(2, height // 3)), x:min(320, x + max(2, width // 2))] = 0
    mask = Image.fromarray(alpha)
    bounds = mask.getbbox()
    if bounds is None:
        raise ValueError('The composite has no visible body pixels')
    image.paste(layer.convert('RGB'), (0, 0), mask)
    x0, y0, x1, y1 = bounds
    return image, [x0, y0, x1 - x0, y1 - y0]


def build_dataset(bank, recording, reviewed_frames, output, seed=7, train_count=512, validation_count=128):
    if output.exists():
        raise FileExistsError('The synthetic image directory already exists')
    if recording.resolve() in output.resolve().parents:
        raise ValueError('Create synthetic images outside the original recording directory')
    if min(train_count, validation_count) < 1:
        raise ValueError('Both synthetic splits require images')
    metadata = json.loads((bank / 'bank.json').read_text())
    if metadata['format'] != BANK_FORMAT:
        raise ValueError('Supply a reviewed upright sprite bank')
    sprites = []
    for row in metadata['templates']:
        path = bank / row['image']
        if path.parent != bank or file_digest(path) != row['sha256']:
            raise ValueError('An upright sprite does not match its bank hash')
        with Image.open(path) as image:
            sprites.append(image.convert('RGBA'))
    integrity = review(recording)
    session, source_hashes = source_identity(recording)
    index = [json.loads(line) for line in (recording / 'frames.jsonl').read_text().splitlines()]
    if (len(reviewed_frames) < 4 or reviewed_frames != sorted(set(reviewed_frames))
            or min(reviewed_frames) < 0 or max(reviewed_frames) >= len(index)):
        raise ValueError('Supply ordered unique background frame identifiers from a visual review')
    backgrounds = []
    with (recording / 'frames.vrec').open('rb') as archive:
        for number in reviewed_frames:
            archive.seek(index[number]['offset'])
            length = struct.unpack('<I', archive.read(4))[0]
            if length != index[number]['png_bytes']:
                raise ValueError('A background image has an invalid archive length')
            with Image.open(io.BytesIO(archive.read(length))) as image:
                backgrounds.append(image.convert('RGB'))
    boundary = len(backgrounds) * 3 // 4
    output.mkdir(parents=True, exist_ok=False)
    image_hashes, plans, counts = {}, [], {}
    rng = random.Random(seed)
    for split, count, available in (('train', train_count, list(range(boundary))),
                                     ('validation', validation_count, list(range(boundary, len(backgrounds))))):
        (output / split).mkdir()
        images, annotations = [], []
        for number in range(count):
            background, sprite = rng.choice(available), rng.randrange(len(sprites))
            image, box = composite(backgrounds[background], sprites[sprite], rng)
            filename = f'{split}/{number:06d}.png'
            image.save(output / filename)
            image_hashes[filename] = file_digest(output / filename)
            images.append(dict(id=number, file_name=filename, width=320, height=240))
            annotations.append(dict(id=number + 1, image_id=number, category_id=1, bbox=box,
                                    area=box[2] * box[3], iscrowd=0))
            plans.append(dict(split=split, image=number, background_frame=reviewed_frames[background],
                              sprite_index=sprite))
        coco = dict(images=images, annotations=annotations, categories=[dict(id=1, name='cutman_body_candidate')])
        (output / f'{split}.coco.json').write_text(json.dumps(coco, indent=2))
        counts[split] = count
    if source_identity(recording) != (session, source_hashes):
        raise ValueError('The background recording changed during image generation')
    result = dict(format=DATA_FORMAT, seed=seed, counts=counts, image_sha256=image_hashes,
                  annotation_sha256={split:file_digest(output / f'{split}.coco.json') for split in counts},
                  bank_sha256=file_digest(bank / 'bank.json'), source_session_id=session,
                  source_hashes=source_hashes, source_integrity=integrity, background_frames=reviewed_frames,
                  train_background_frames=reviewed_frames[:boundary],validation_background_frames=reviewed_frames[boundary:],
                  source_files_changed=False, actual_game_render_verified=False, independent_validation=False,
                  input_sent=False, action_training_eligible=False, plans=plans)
    (output / 'dataset.json').write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    bank = sub.add_parser('bank')
    bank.add_argument('--packages', type=Path, nargs='+', required=True)
    bank.add_argument('--output', type=Path, required=True)
    dataset = sub.add_parser('dataset')
    dataset.add_argument('--bank', type=Path, required=True)
    dataset.add_argument('--recording', type=Path, required=True)
    dataset.add_argument('--frames', type=int, nargs='+', required=True)
    dataset.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'bank':
        result = build_bank(args.packages, args.output)
        print(json.dumps(dict(views=len(result['templates']), poses=len(result['upright_frames']))))
    else:
        result = build_dataset(args.bank, args.recording, args.frames, args.output)
        print(json.dumps(result['counts']))


if __name__ == '__main__':
    main()
