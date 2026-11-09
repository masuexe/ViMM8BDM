"""Export version-agnostic PK3 sprite/occluder bundles and run GPU detector training."""
import argparse
import json
from pathlib import Path
import random
import shutil
import tarfile
import tempfile

import numpy as np
from PIL import Image

from .pretrained_detection import load_model
from .recording import file_digest
from .sprite_synthesis import BANK_FORMAT, OCCLUDER_FORMAT, build_bank, build_dataset, build_occluder_bank
from .sprite_transfer import SyntheticImages, load_checkpoint, train
from .visual_metrics import box_iou

BUNDLE_FORMAT = 'vimm8bdm-colab-training-bundle-v1'

COLAB_RUNNER_SOURCE = '''"""Standalone Google Colab GPU training entry point for a ViMM8BDM bundle."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if (ROOT / "src").is_dir():
    sys.path.insert(0, str(ROOT / "src"))

from vimm8bdm.colab_bundle import train_from_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    result = train_from_bundle(
        args.bundle,
        args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
    )
    print(json.dumps({
        "device": result["protocol"]["device"],
        "seconds": result["seconds"],
        "checkpoint_sha256": result["checkpoint_sha256"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
'''


def _overlay_occluder_on_annotated_image(image, bbox, occluders, rng):
    """Paste a foreground occluder (left/right/bottom or distractor) and optional yellow hit-flash."""
    out = image.copy().convert('RGB')
    x, y, w, h = [float(v) for v in bbox]
    occ = rng.choice(occluders)
    occ_h = max(6, round(h * rng.uniform(.35, .75)))
    occ_w = max(6, round(occ.width * occ_h / max(1, occ.height)))
    occ_img = occ.resize((occ_w, occ_h), Image.Resampling.NEAREST)
    new_box = [x, y, w, h]
    placed = False
    bottom_clip = False
    if rng.random() < .6 and w >= 16 and h >= 16:
        edge = rng.choice(('left', 'right', 'bottom'))
        if edge == 'bottom':
            clip_h = max(2, min(occ_h // 2, int(h * .3)))
            ox = max(0, min(out.width - occ_w, round(x + rng.uniform(0, max(0., w - occ_w)))))
            oy = round(y + h - clip_h)
            new_box = [x, y, w, h - clip_h]
            bottom_clip = True
        else:
            clip_w = max(2, min(occ_w // 2, int(w * .25)))
            oy = max(0, min(200 - occ_h, round(y + rng.uniform(0, max(0., h - occ_h)))))
            if edge == 'left':
                ox = round(x + clip_w - occ_w)
                new_box = [x + clip_w, y, w - clip_w, h]
            else:
                ox = round(x + w - clip_w)
                new_box = [x, y, w - clip_w, h]
        placed = True
    if not placed:
        for _ in range(32):
            ox = rng.randint(0, max(0, out.width - occ_w))
            oy = rng.randint(8, max(8, min(198, out.height) - occ_h))
            if box_iou([ox, oy, occ_w, occ_h], [x, y, w, h]) == 0:
                placed = True
                break
        if not placed:
            ox = 0 if x > out.width / 2 else max(0, out.width - occ_w)
            oy = 8
    fg = Image.new('RGBA', out.size)
    fg.paste(occ_img, (ox, oy))
    fg_alpha = np.array(fg.getchannel('A'))
    if not bottom_clip:
        fg_alpha[200:] = 0
    out.paste(fg.convert('RGB'), (0, 0), Image.fromarray(fg_alpha))
    if rng.random() < .35:
        alpha = rng.uniform(.05, .30)
        out = Image.blend(out, Image.new('RGB', out.size, (255, 255, 0)), alpha)
    return out, new_box


def _copy_verified_dataset(source_dir, target_dir, occluder_dir=None, seed=7, augment_occluders=None):
    train_split = SyntheticImages(source_dir, 'train')
    val_split = SyntheticImages(source_dir, 'validation')
    for split_ds in (train_split, val_split):
        for i in range(len(split_ds)):
            _ = split_ds[i]
    shutil.copytree(source_dir, target_dir)
    if occluder_dir is not None:
        occ_meta = json.loads((occluder_dir / 'occluders.json').read_text())
        occluders = []
        for row in occ_meta['templates']:
            with Image.open(occluder_dir / row['image']) as img:
                occluders.append(img.convert('RGBA'))
        count = max(1, len(train_split) // 8) if augment_occluders is None else int(augment_occluders)
        if count > 0 and occluders:
            rng = random.Random(seed)
            metadata = json.loads((target_dir / 'dataset.json').read_text())
            coco_path = target_dir / 'train.coco.json'
            coco = json.loads(coco_path.read_text())
            by_id = {row['image_id']: row['bbox'] for row in coco['annotations']}
            base_rows = list(coco['images'])
            for idx in range(count):
                src_row = base_rows[idx % len(base_rows)]
                with Image.open(target_dir / src_row['file_name']) as src_img:
                    aug_img, aug_box = _overlay_occluder_on_annotated_image(
                        src_img, by_id[src_row['id']], occluders, rng
                    )
                new_id = len(coco['images'])
                rel_name = f'train/{new_id:06d}.png'
                aug_img.save(target_dir / rel_name)
                metadata['image_sha256'][rel_name] = file_digest(target_dir / rel_name)
                coco['images'].append(dict(id=new_id, file_name=rel_name,
                                           width=aug_img.width, height=aug_img.height))
                coco['annotations'].append(dict(id=len(coco['annotations']) + 1, image_id=new_id,
                                                category_id=1, bbox=aug_box,
                                                area=aug_box[2] * aug_box[3], iscrowd=0))
            coco_path.write_text(json.dumps(coco, indent=2))
            metadata['counts']['train'] = len(coco['images'])
            metadata['annotation_sha256']['train'] = file_digest(coco_path)
            metadata['occluder_bank_sha256'] = file_digest(occluder_dir / 'occluders.json')
            (target_dir / 'dataset.json').write_text(json.dumps(metadata, indent=2))
            train_split = SyntheticImages(target_dir, 'train')
    return len(train_split), len(val_split)


def export_bundle(packages, initial, output, *, dataset=None, recording=None, reviewed_frames=None,
                  train_count=512, validation_count=128, seed=7, initial_transfer=False,
                  allow_partial_poses=False, augment_occluders=None):
    """Build a self-contained directory or .tar.gz training bundle from any PK3 stack."""
    if output.exists():
        raise FileExistsError('The Colab bundle output already exists')
    if (dataset is None) == (recording is None):
        raise ValueError('Supply either an existing --dataset or --recording with --frames')
    if recording is not None and not reviewed_frames:
        raise ValueError('Building a dataset from a recording requires --frames')
    previous_ckpt = None
    if initial_transfer:
        _, previous_ckpt = load_checkpoint(initial)
    else:
        load_model(initial)
    initial_sha256 = file_digest(initial)

    is_archive = output.name.lower().endswith(('.tar.gz', '.tgz'))
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / 'bundle' if is_archive else output
        stage.mkdir(parents=True, exist_ok=False)
        upright_dir = stage / 'upright_bank'
        occluder_dir = stage / 'occluder_bank'
        dataset_dir = stage / 'dataset'
        initial_dir = stage / 'initial'
        initial_dir.mkdir()

        upright_meta = build_bank(packages, upright_dir, allow_partial_poses=allow_partial_poses)
        occluder_meta = build_occluder_bank(packages, occluder_dir)

        if dataset is not None:
            train_n, val_n = _copy_verified_dataset(
                dataset, dataset_dir, occluder_dir=occluder_dir, seed=seed, augment_occluders=augment_occluders
            )
        else:
            build_dataset(
                upright_dir,
                recording,
                reviewed_frames,
                dataset_dir,
                seed=seed,
                train_count=train_count,
                validation_count=validation_count,
                profile='occlusion',
                occluder_bank=occluder_dir,
            )
            train_n, val_n = train_count, validation_count

        if previous_ckpt is not None:
            ds_meta = json.loads((dataset_dir / 'dataset.json').read_text())
            source_ids = ds_meta.get('training_source_session_ids', [ds_meta['source_session_id']])
            inherited = previous_ckpt.get('training_source_session_ids', [previous_ckpt['training_source_session_id']])
            if not set(inherited).issubset(source_ids):
                raise ValueError('Dataset provenance must include every inherited training recording')

        target_ckpt = initial_dir / 'checkpoint.pt'
        shutil.copy2(initial, target_ckpt)
        if file_digest(target_ckpt) != initial_sha256:
            raise ValueError('The initial checkpoint changed during bundle export')

        src_root = Path(__file__).resolve().parent
        pkg_target = stage / 'src' / 'vimm8bdm'
        pkg_target.mkdir(parents=True)
        for py_file in sorted(src_root.glob('*.py')):
            shutil.copy2(py_file, pkg_target / py_file.name)

        (stage / 'train_colab.py').write_text(COLAB_RUNNER_SOURCE, encoding='utf-8')
        manifest = dict(
            format=BUNDLE_FORMAT,
            pk3_sources=upright_meta['sources'],
            upright_poses=upright_meta['upright_frames'],
            upright_views=len(upright_meta['templates']),
            upright_bank_sha256=file_digest(upright_dir / 'bank.json'),
            occluder_count=len(occluder_meta['templates']),
            occluder_prefixes=occluder_meta['prefixes'],
            occluder_bank_sha256=file_digest(occluder_dir / 'occluders.json'),
            dataset_sha256=file_digest(dataset_dir / 'dataset.json'),
            initial_checkpoint_sha256=initial_sha256,
            initial_transfer=bool(initial_transfer),
            train_images=train_n,
            validation_images=val_n,
            colab_command='python train_colab.py --output trained_run --device cuda --epochs 8 --batch-size 16',
            source_files_changed=False,
            input_sent=False,
        )
        (stage / 'colab_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        if is_archive:
            output.parent.mkdir(parents=True, exist_ok=True)
            with tarfile.open(output, 'w:gz') as tar:
                for item in sorted(stage.rglob('*')):
                    rel = item.relative_to(stage).as_posix()
                    tar.add(item, arcname=rel, recursive=False)
            manifest['archive_sha256'] = file_digest(output)
        return manifest


def _safe_extract_tar(archive_path, target_dir):
    root = target_dir.resolve()
    with tarfile.open(archive_path, 'r:*') as tar:
        for member in tar.getmembers():
            if member.issym() or member.islnk():
                raise ValueError('Symbolic or hard links are not allowed in Colab bundles')
            dest = (root / member.name).resolve()
            if root != dest and root not in dest.parents:
                raise ValueError('Unsafe relative path in Colab bundle archive')
        tar.extractall(root, filter='data')


def train_from_bundle(bundle, output, *, epochs=8, batch_size=16, learning_rate=.0005, seed=0, device='auto'):
    """Verify a Colab bundle directory or .tar.gz archive and run detector training."""
    if output.exists():
        raise FileExistsError('The training output already exists')
    bundle = Path(bundle)
    with tempfile.TemporaryDirectory() as tmp:
        if bundle.is_file():
            stage = Path(tmp) / 'unpacked'
            stage.mkdir()
            _safe_extract_tar(bundle, stage)
        elif bundle.is_dir():
            stage = bundle
        else:
            raise ValueError('Supply an existing bundle directory or .tar.gz archive')

        manifest_path = stage / 'colab_manifest.json'
        if not manifest_path.is_file():
            raise ValueError('The Colab bundle is missing colab_manifest.json')
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest.get('format') != BUNDLE_FORMAT:
            raise ValueError('The Colab bundle format is unsupported')

        upright_json = stage / 'upright_bank' / 'bank.json'
        occluder_json = stage / 'occluder_bank' / 'occluders.json'
        dataset_json = stage / 'dataset' / 'dataset.json'
        initial_ckpt = stage / 'initial' / 'checkpoint.pt'
        if not all(p.is_file() for p in (upright_json, occluder_json, dataset_json, initial_ckpt)):
            raise ValueError('The Colab bundle is missing a required component file')
        if (file_digest(upright_json) != manifest['upright_bank_sha256']
                or file_digest(occluder_json) != manifest['occluder_bank_sha256']
                or file_digest(dataset_json) != manifest['dataset_sha256']
                or file_digest(initial_ckpt) != manifest['initial_checkpoint_sha256']):
            raise ValueError('A Colab bundle component does not match its manifest SHA256')
        if (json.loads(upright_json.read_text()).get('format') != BANK_FORMAT
                or json.loads(occluder_json.read_text()).get('format') != OCCLUDER_FORMAT):
            raise ValueError('A sprite or occluder bank inside the bundle has an invalid format')

        result = train(
            stage / 'dataset',
            initial_ckpt,
            output,
            epochs=epochs,
            batch_size=batch_size,
            seed=seed,
            initial_transfer=bool(manifest['initial_transfer']),
            learning_rate=learning_rate,
            device=device,
        )
        provenance = dict(
            format=BUNDLE_FORMAT,
            colab_manifest_sha256=file_digest(manifest_path),
            pk3_sources=manifest['pk3_sources'],
            upright_views=manifest['upright_views'],
            occluder_count=manifest['occluder_count'],
            occluder_prefixes=manifest['occluder_prefixes'],
            checkpoint_sha256=result['checkpoint_sha256'],
            device=result['protocol']['device'],
            epochs=epochs,
            batch_size=batch_size,
            seconds=result['seconds'],
            input_sent=False,
        )
        (output / 'colab_bundle_provenance.local.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)

    exp = sub.add_parser('export')
    exp.add_argument('--packages', type=Path, nargs='+', required=True)
    exp.add_argument('--initial', type=Path, required=True)
    exp.add_argument('--output', type=Path, required=True)
    exp.add_argument('--dataset', type=Path)
    exp.add_argument('--recording', type=Path)
    exp.add_argument('--frames', type=int, nargs='+')
    exp.add_argument('--train-count', type=int, default=512)
    exp.add_argument('--validation-count', type=int, default=128)
    exp.add_argument('--seed', type=int, default=7)
    exp.add_argument('--initial-transfer', action='store_true')
    exp.add_argument('--allow-partial-poses', action='store_true')
    exp.add_argument('--augment-occluders', type=int)

    trn = sub.add_parser('train')
    trn.add_argument('--bundle', type=Path, required=True)
    trn.add_argument('--output', type=Path, required=True)
    trn.add_argument('--epochs', type=int, default=8)
    trn.add_argument('--batch-size', type=int, default=16)
    trn.add_argument('--learning-rate', type=float, default=.0005)
    trn.add_argument('--seed', type=int, default=0)
    trn.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')

    args = parser.parse_args()
    if args.command == 'export':
        manifest = export_bundle(
            args.packages,
            args.initial,
            args.output,
            dataset=args.dataset,
            recording=args.recording,
            reviewed_frames=args.frames,
            train_count=args.train_count,
            validation_count=args.validation_count,
            seed=args.seed,
            initial_transfer=args.initial_transfer,
            allow_partial_poses=args.allow_partial_poses,
            augment_occluders=args.augment_occluders,
        )
        print(json.dumps(dict(
            upright_views=manifest['upright_views'],
            occluder_count=manifest['occluder_count'],
            train_images=manifest['train_images'],
            validation_images=manifest['validation_images'],
            archive_sha256=manifest.get('archive_sha256'),
        ), indent=2), flush=True)
    else:
        result = train_from_bundle(
            args.bundle,
            args.output,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
            device=args.device,
        )
        print(json.dumps(dict(
            device=result['protocol']['device'],
            seconds=result['seconds'],
            checkpoint_sha256=result['checkpoint_sha256'],
        ), indent=2), flush=True)


if __name__ == '__main__':
    main()
