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
OCCLUDER_FORMAT = 'vimm8bdm-foreground-occluder-bank-v1'
DATA_FORMAT = 'vimm8bdm-synthetic-body-coco-v1'
OCCLUDER_PREFIXES = (
    'MXP1', 'MXP2', 'MXP3', 'MXP4', 'MMFX', 'MMDX', 'NEDR',
    'BUST', 'BUS2', 'W_0B', 'MBU2', 'PCOI', 'CBUS', 'RCUT', 'RCTC',
)


def build_bank(packages, output, allow_partial_poses=False):
    if output.exists():
        raise FileExistsError('The sprite training bank already exists')
    sources, resolved = [], {}
    palette, actor_hash = None, None
    skin_prefixes = {'CUTM', '$CUT'}
    for package in packages:
        sources.append(dict(name=package.name, sha256=file_digest(package)))
        with zipfile.ZipFile(package) as archive:
            if palette is None and 'PLAYPAL.pal' in archive.namelist():
                palette = np.frombuffer(archive.read('PLAYPAL.pal')[:768], dtype=np.uint8).reshape(256, 3)
            for member in archive.namelist():
                normalized = member.replace('\\', '/')
                lower = normalized.lower()
                if lower in ('skininfo', 'skininfo.txt'):
                    text = archive.read(member).decode('latin-1', errors='ignore')
                    for block in re.findall(r'\{[^}]*\}', text, re.S):
                        if re.search(r'name\s*=\s*"?cutman"?', block, re.I):
                            m_sp = re.search(r'sprite\s*=\s*([^\s\}]+)', block, re.I)
                            if m_sp and len(m_sp.group(1).strip('"')) == 4:
                                skin_prefixes.add(m_sp.group(1).strip('"').upper())
                if lower == 'actors/megaman/cutman.txt':
                    actor_hash = hashlib.sha256(archive.read(member)).hexdigest()
                if not lower.startswith('sprites/'):
                    continue
                name = normalized.rsplit('/', 1)[-1].upper()
                if len(name) >= 8 and name[:4] in skin_prefixes and name.endswith('.PNG'):
                    match = re.fullmatch(r'([A-Z])([0-8])(?:([A-Z])([0-8]))?\.PNG', name[4:])
                    if match:
                        resolved['CUTM' + name[4:-4]] = (archive.read(member), package.name, member, match.groups())
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
    complete_poses = sorted(pose for pose in POSES if all((pose, r) in views for r in range(1, 9)))
    expected_poses = complete_poses if (allow_partial_poses and complete_poses) else list(POSES)
    if set(views) != {(pose, rotation) for pose in expected_poses for rotation in range(1, 9)}:
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
    metadata = dict(format=BANK_FORMAT, sources=sources, templates=rows, upright_frames=expected_poses,
                    actor_source_sha256=actor_hash, translation_pairs=[[192, 4], [198, 41], [202, 96]],
                    source_files_changed=False, actual_game_render_verified=False, input_sent=False)
    (output / 'bank.json').write_text(json.dumps(metadata, indent=2))
    return metadata


def build_occluder_bank(packages, output):
    """Extract foreground projectile and hit-smoke sprites in PK3 load order."""
    if output.exists():
        raise FileExistsError('The foreground occluder bank already exists')
    sources, resolved = [], {}
    for package in packages:
        sources.append(dict(name=package.name, sha256=file_digest(package)))
        with zipfile.ZipFile(package) as archive:
            for member in archive.namelist():
                normalized = member.replace('\\', '/')
                lower = normalized.lower()
                if not (lower.startswith('sprites/') and lower.endswith('.png')):
                    continue
                stem = normalized.rsplit('/', 1)[-1][:-4].upper()
                if len(stem) >= 6 and (stem[:4] in OCCLUDER_PREFIXES or '/explosions/' in lower):
                    resolved[stem] = (archive.read(member), package.name, member)
    if not resolved:
        raise ValueError('The supplied packages contain no recognized foreground occluder sprites')
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for index, (stem, (raw, package, member)) in enumerate(sorted(resolved.items())):
        with Image.open(io.BytesIO(raw)) as source:
            image = source.convert('RGBA')
        bounds = image.getchannel('A').getbbox()
        if bounds is None:
            continue
        cropped = image.crop(bounds)
        if min(cropped.size) < 2:
            continue
        safe_stem = re.sub(r'[^A-Z0-9_]', '_', stem)
        filename = f'occluder-{index:04d}-{safe_stem}.png'
        cropped.save(output / filename)
        rows.append(dict(lump=stem, prefix=stem[:4], image=filename,
                         width=cropped.width, height=cropped.height,
                         sha256=file_digest(output / filename),
                         source_package=package, source_member=member))
    if not rows:
        raise ValueError('No visible foreground occluder sprites were extracted')
    for package, expected in zip(packages, sources):
        if file_digest(package) != expected['sha256']:
            raise ValueError('An original package changed during occluder extraction')
    metadata = dict(format=OCCLUDER_FORMAT, sources=sources, templates=rows,
                    prefixes=sorted({row['prefix'] for row in rows}),
                    source_files_changed=False, input_sent=False)
    (output / 'occluders.json').write_text(json.dumps(metadata, indent=2))
    return metadata


def composite(background, sprite, rng, profile="baseline", occluders=None):
    """Return one RGB image and the visible xywh box after clipping and optional occlusion."""
    if profile not in ('baseline', 'combat', 'occlusion'):
        raise ValueError('The synthetic generation profile is unsupported')
    image = background.convert('RGB').resize((320, 240), Image.Resampling.BILINEAR)
    if profile in ('combat', 'occlusion'):
        lower, upper = (4.32, 6.) if rng.random() < .7 else (6., 7.65)
        height = round(2 ** rng.uniform(lower, upper))
    else:
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
    fg_layer = None
    if profile == 'occlusion' and occluders and rng.random() < .7:
        occ = rng.choice(occluders)
        occ_h = max(6, round(height * rng.uniform(.35, .85)))
        occ_w = max(6, round(occ.width * occ_h / max(1, occ.height)))
        occ_img = occ.resize((occ_w, occ_h), Image.Resampling.NEAREST)
        if rng.random() < .5:
            # Partial foreground occlusion near the body edge
            ox = left + ( -occ_w // 2 if rng.random() < .5 else max(1, width - occ_w // 2))
            oy = top + rng.randint(0, max(0, height - occ_h // 2))
        else:
            # Separate foreground projectile/smoke distractor elsewhere in the scene
            ox = rng.randint(0, max(0, 320 - occ_w))
            oy = rng.randint(8, max(8, 198 - occ_h))
        fg_layer = Image.new('RGBA', image.size)
        fg_layer.paste(occ_img, (ox, oy))
        fg_alpha = np.array(fg_layer.getchannel('A'))
        fg_alpha[200:] = 0
        candidate_alpha = alpha.copy()
        candidate_alpha[fg_alpha >= 128] = 0
        if np.count_nonzero(candidate_alpha) >= max(8, int(np.count_nonzero(alpha) * .35)):
            alpha = candidate_alpha
            fg_layer.putalpha(Image.fromarray(fg_alpha))
        else:
            fg_layer = None
    elif rng.random() < .25:
        x = max(0, left + width // 2)
        y = max(0, top + height // 2)
        alpha[y:min(200, y + max(2, height // 3)), x:min(320, x + max(2, width // 2))] = 0
    mask = Image.fromarray(alpha)
    bounds = mask.getbbox()
    if bounds is None:
        raise ValueError('The composite has no visible body pixels')
    image.paste(layer.convert('RGB'), (0, 0), mask)
    if fg_layer is not None:
        image.paste(fg_layer.convert('RGB'), (0, 0), fg_layer.getchannel('A'))
    x0, y0, x1, y1 = bounds
    return image, [x0, y0, x1 - x0, y1 - y0]


def augment_scene(image, rng):
    """Change RGB appearance without changing image coordinates."""
    image = ImageEnhance.Color(image).enhance(rng.uniform(.6, 1.3))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(.75, 1.2))
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(.75, 1.15))
    if rng.random() < .5:
        color = rng.choice(((255, 0, 0), (255, 220, 0), (255, 255, 255)))
        image = Image.blend(image, Image.new('RGB', image.size, color), rng.uniform(.08, .35))
    return image


def build_dataset(bank, recording, reviewed_frames, output, seed=7, train_count=512, validation_count=128,
                  profile="baseline", occluder_bank=None):
    if output.exists():
        raise FileExistsError('The synthetic image directory already exists')
    if recording.resolve() in output.resolve().parents:
        raise ValueError('Create synthetic images outside the original recording directory')
    if profile not in ('baseline', 'combat', 'occlusion'):
        raise ValueError('The synthetic generation profile is unsupported')
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
    occluders, occluder_hash = None, None
    if occluder_bank is not None:
        occ_meta = json.loads((occluder_bank / 'occluders.json').read_text())
        if occ_meta.get('format') != OCCLUDER_FORMAT:
            raise ValueError('Supply a verified foreground occluder bank')
        occluders = []
        for row in occ_meta['templates']:
            path = occluder_bank / row['image']
            if path.parent != occluder_bank or file_digest(path) != row['sha256']:
                raise ValueError('A foreground occluder does not match its bank hash')
            with Image.open(path) as image:
                occluders.append(image.convert('RGBA'))
        occluder_hash = file_digest(occluder_bank / 'occluders.json')
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
            image, box = composite(backgrounds[background], sprites[sprite], rng, profile=profile, occluders=occluders)
            if profile in ('combat', 'occlusion'):
                image = augment_scene(image, rng)
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
    result = dict(format=DATA_FORMAT, seed=seed, generation_profile=profile, counts=counts, image_sha256=image_hashes,
                  annotation_sha256={split:file_digest(output / f'{split}.coco.json') for split in counts},
                  bank_sha256=file_digest(bank / 'bank.json'),
                  occluder_bank_sha256=occluder_hash,
                  source_session_id=session,
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
    bank.add_argument('--allow-partial-poses', action='store_true')
    occ = sub.add_parser('occluders')
    occ.add_argument('--packages', type=Path, nargs='+', required=True)
    occ.add_argument('--output', type=Path, required=True)
    dataset = sub.add_parser('dataset')
    dataset.add_argument('--bank', type=Path, required=True)
    dataset.add_argument('--occluders', type=Path)
    dataset.add_argument('--recording', type=Path, required=True)
    dataset.add_argument('--frames', type=int, nargs='+', required=True)
    dataset.add_argument('--output', type=Path, required=True)
    dataset.add_argument('--profile', choices=('baseline', 'combat', 'occlusion'), default='baseline')
    dataset.add_argument('--train-count', type=int, default=512)
    dataset.add_argument('--validation-count', type=int, default=128)
    args = parser.parse_args()
    if args.command == 'bank':
        result = build_bank(args.packages, args.output, allow_partial_poses=args.allow_partial_poses)
        print(json.dumps(dict(views=len(result['templates']), poses=len(result['upright_frames']))))
    elif args.command == 'occluders':
        result = build_occluder_bank(args.packages, args.output)
        print(json.dumps(dict(occluders=len(result['templates']), prefixes=result['prefixes'])))
    else:
        result = build_dataset(args.bank, args.recording, args.frames, args.output,
                               profile=args.profile, train_count=args.train_count,
                               validation_count=args.validation_count, occluder_bank=args.occluders)
        print(json.dumps(result['counts']))


if __name__ == '__main__':
    main()
