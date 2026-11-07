"""Export version-agnostic PK3 sprite/occluder bundles and run GPU detector training."""
import argparse
import json
from pathlib import Path
import shutil
import tarfile
import tempfile

from .pretrained_detection import load_model
from .recording import file_digest
from .sprite_synthesis import BANK_FORMAT, OCCLUDER_FORMAT, build_bank, build_dataset, build_occluder_bank
from .sprite_transfer import SyntheticImages, load_checkpoint, train

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


def _copy_verified_dataset(source_dir, target_dir):
    train_split = SyntheticImages(source_dir, 'train')
    val_split = SyntheticImages(source_dir, 'validation')
    for split_ds in (train_split, val_split):
        for i in range(len(split_ds)):
            _ = split_ds[i]
    shutil.copytree(source_dir, target_dir)
    return len(train_split), len(val_split)


def export_bundle(packages, initial, output, *, dataset=None, recording=None, reviewed_frames=None,
                  train_count=512, validation_count=128, seed=7, initial_transfer=False,
                  allow_partial_poses=False):
    """Build a self-contained directory or .tar.gz training bundle from any PK3 stack."""
    if output.exists():
        raise FileExistsError('The Colab bundle output already exists')
    if (dataset is None) == (recording is None):
        raise ValueError('Supply either an existing --dataset or --recording with --frames')
    if recording is not None and not reviewed_frames:
        raise ValueError('Building a dataset from a recording requires --frames')
    if initial_transfer:
        load_checkpoint(initial)
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
            train_n, val_n = _copy_verified_dataset(dataset, dataset_dir)
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
