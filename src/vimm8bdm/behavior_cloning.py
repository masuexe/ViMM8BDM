"""Train an experimental visual navigation model without desktop input."""
from __future__ import annotations

import argparse
import copy
import io
import json
import math
from pathlib import Path
import random
import struct
import time

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

from .dataset import check_splits, filter_reviewed_samples, make_samples, verify_source
from .recording import file_digest

KEYS = ('w', 'a', 's', 'd')
VKS = (87, 65, 83, 68)
IMAGE_SIZE = (112, 84)
FEATURE_MODES = ('raw', 'centered', 'motion')


def image_features(images, mode):
    """Transform past images without using labels or changing the input tensor."""
    if images.ndim != 4 or images.shape[1] != 4:
        raise ValueError('Image features require four chronological channels')
    if mode == 'raw':
        return images
    if mode == 'centered':
        return images * 2 - 1
    if mode == 'motion':
        return torch.cat((images[:, -1:] * 2 - 1,
                          (images[:, 1:] - images[:, :-1]) * 2), dim=1)
    raise ValueError('The image feature mode is unsupported')


def label_target(sample):
    if sample['action_end_ns'] - sample['anchor_ns'] != 50_000_000:
        raise ValueError('Navigation labels require a 50-millisecond action window')
    label = sample['action']
    keys = set(label['keys_at_start']) | set(label['keys_at_end'])
    keys.update(map(int, label['key_held_fraction']))
    keys.update(e['vk'] for e in label['events'] if e['type'] == 'key')
    if not keys <= set(VKS):
        raise ValueError('The navigation target supports only W, A, S, and D')
    if (label['buttons_at_start'] or label['buttons_at_end'] or label['button_held_fraction']
            or label['wheel_vertical'] or label['wheel_horizontal']
            or any(e['type'] not in ('key', 'motion') for e in label['events'])):
        raise ValueError('The navigation target does not support buttons or wheel input')
    fractions = [label['key_held_fraction'].get(str(vk), 0) for vk in VKS]
    values = [*fractions, label['mouse_dx_raw'], label['mouse_dy_raw']]
    if not all(math.isfinite(v) for v in values) or any(not 0 <= v <= 1 for v in fractions):
        raise ValueError('Navigation targets must contain finite counts and valid held fractions')
    return values


def load_inputs(directory, expected_split):
    """Check source labels and load only the four referenced past images."""
    directory = Path(directory)
    meta = json.loads((directory / 'dataset.json').read_text())
    if (meta['split'] != expected_split or meta['task'] != 'navigation'
            or meta['source_purpose'] != 'demonstration' or meta['source_protocol'] != 'free'):
        raise ValueError('The dataset must contain navigation demonstrations in the requested split')
    verify_source(directory)
    source = directory / meta['source_directory']
    frames = [json.loads(line) for line in (source / 'frames.jsonl').read_text().splitlines()]
    events = [json.loads(line) for line in (source / 'events.jsonl').read_text().splitlines()]
    samples = [json.loads(line) for line in (directory / 'samples.jsonl').read_text().splitlines()]
    expected, _ = make_samples(frames, events)
    expected, _ = filter_reviewed_samples(expected, frames, meta.get('reviewed_frame_exclusions'),
                                         meta['source_session_id'])
    if samples != expected or len(samples) != meta['samples'] or not samples:
        raise ValueError('Candidate samples do not match the causal source reconstruction')
    targets = torch.tensor([label_target(sample) for sample in samples], dtype=torch.float32)
    cache = {}
    needed = {h['frame'] for sample in samples for h in sample['history']}
    with (source / 'frames.vrec').open('rb') as archive:
        for row in frames:
            if row['frame'] not in needed:
                continue
            archive.seek(row['offset'])
            length = struct.unpack('<I', archive.read(4))[0]
            if length != row['png_bytes']:
                raise ValueError('The image length does not match its source index')
            with Image.open(io.BytesIO(archive.read(length))) as image:
                gray = image.convert('L').resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
                cache[row['frame']] = np.asarray(gray).copy()
    images = torch.from_numpy(np.stack([np.stack([cache[h['frame']] for h in sample['history']])
                                      for sample in samples]))
    if images.shape[1:] != (4, IMAGE_SIZE[1], IMAGE_SIZE[0]):
        raise ValueError('The model requires four image channels')
    identity = {key: meta[key] for key in ('source_session_id', 'split', 'character', 'map', 'samples')}
    identity['samples_sha256'] = file_digest(directory / 'samples.jsonl')
    return images, targets, identity


class NavigationModel(nn.Module):
    """Predict four held fractions and two raw mouse totals from past images."""

    def __init__(self, features='raw'):
        super().__init__()
        if features not in FEATURE_MODES:
            raise ValueError('The image feature mode is unsupported')
        self.features = features
        self.layers = nn.Sequential(
            nn.Conv2d(4, 16, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 6)), nn.Flatten(),
            nn.Linear(32 * 4 * 6, 128), nn.ReLU(), nn.Linear(128, 6))

    def forward(self, images):
        return self.layers(image_features(images, self.features))


def load_checkpoint(path):
    """Load old raw-image checkpoints or versioned feature configurations."""
    checkpoint = torch.load(path, weights_only=True, map_location='cpu')
    version = checkpoint.get('format')
    if version not in ('vimm8bdm-navigation-bc-v1', 'vimm8bdm-navigation-bc-v2'):
        raise ValueError('The navigation checkpoint format is unsupported')
    features = checkpoint.get('features', 'raw') if version.endswith('v1') else checkpoint.get('features')
    if version.endswith('v1') and features != 'raw':
        raise ValueError('Version 1 checkpoints require raw image features')
    if checkpoint.get('image_size') != list(IMAGE_SIZE) or checkpoint.get('keys') != list(KEYS):
        raise ValueError('The checkpoint image size or key order is unsupported')
    mean, scale = checkpoint['mouse_mean'], checkpoint['mouse_scale']
    if (not isinstance(mean, torch.Tensor) or not isinstance(scale, torch.Tensor)
            or mean.shape != (2,) or scale.shape != (2,) or not torch.isfinite(mean).all()
            or not torch.isfinite(scale).all() or not (scale > 0).all()):
        raise ValueError('The checkpoint requires finite mouse normalization and positive scales')
    model = NavigationModel(features)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.eval()
    return model, mean, scale


def normalization(targets):
    # Validation targets must never enter this calculation.
    return targets[:, 4:].mean(0), targets[:, 4:].std(0, correction=0).clamp_min(1)


def metrics(predictions, targets, scale):
    probabilities = predictions[:, :4].clamp(1e-6, 1 - 1e-6)
    difference = predictions[:, 4:] - targets[:, 4:]
    mouse_mae = difference.abs().mean(0)
    bce = F.binary_cross_entropy(probabilities, targets[:, :4]).item()
    result = {
        'key_bce': bce,
        'key_held_fraction_mae': dict(zip(KEYS, (probabilities - targets[:, :4]).abs().mean(0).tolist())),
        'key_target_mean': dict(zip(KEYS, targets[:, :4].mean(0).tolist())),
        'key_prediction_mean': dict(zip(KEYS, probabilities.mean(0).tolist())),
        'mouse_mae_raw_counts': dict(zip(('x', 'y'), mouse_mae.tolist())),
        'mouse_rmse_raw_counts': dict(zip(('x', 'y'), difference.square().mean(0).sqrt().tolist())),
        'selection_score': bce + (mouse_mae / scale).mean().item(),
    }
    sign = {}
    for axis, name in enumerate(('x', 'y')):
        active = targets[:, axis + 4].abs() >= 1
        sign[name] = {'nonzero_targets': active.sum().item(),
                      'accuracy': ((predictions[active, axis + 4].sign() == targets[active, axis + 4].sign())
                                   .float().mean().item()) if active.any() else None}
    result['mouse_sign'] = sign
    return result


def predict(model, images, mean, scale, batch_size, zero_images=False, repeat_latest=False):
    outputs = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            batch = images[start:start + batch_size].float() / 255
            if zero_images:
                batch.zero_()
            if repeat_latest:
                batch = batch[:, -1:].expand(-1, 4, -1, -1).contiguous()
            output = model(batch)
            outputs.append(torch.cat((output[:, :4].sigmoid(), output[:, 4:] * scale + mean), dim=1))
    return torch.cat(outputs)


def read_review(path):
    review = json.loads(Path(path).read_text())
    if not review.get('operator_report') or review.get('source_files_changed') is not False:
        raise ValueError('Supply a separate operator review that leaves source files unchanged')
    return file_digest(path)


def train(args):
    if not args.experimental:
        raise ValueError('This candidate-data experiment requires --experimental')
    if args.output.exists():
        raise FileExistsError('The output directory already exists. Select a new directory')
    reviews = {'train': read_review(args.train_review), 'validation': read_review(args.validation_review)}
    split_result = check_splits([args.train, args.validation])
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    started = time.perf_counter()
    train_x, train_y, train_id = load_inputs(args.train, 'train')
    validation_x, validation_y, validation_id = load_inputs(args.validation, 'validation')
    if train_id['character'] != validation_id['character'] or train_id['map'] != validation_id['map']:
        raise ValueError('This first experiment requires the same character and map in both splits')
    args.output.mkdir(parents=True, exist_ok=False)
    mean, scale = normalization(train_y)
    constants = torch.cat((train_y[:, :4].mean(0), train_y[:, 4:].median(0).values))
    baselines = {'train_median_mouse': metrics(constants.expand(len(validation_y), -1), validation_y, scale)}
    mean_constants = torch.cat((train_y[:, :4].mean(0), mean))
    baselines['train_mean_mouse'] = metrics(mean_constants.expand(len(validation_y), -1), validation_y, scale)
    features = getattr(args, 'features', 'raw')
    model = NavigationModel(features)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    generator = torch.Generator().manual_seed(args.seed)
    history, best_state, best_epoch, best_score = [], None, None, math.inf
    print('Inputs checked. Begin offline training.', flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(train_y), generator=generator)
        losses = []
        for indices in order.split(args.batch_size):
            output = model(train_x[indices].float() / 255)
            labels = train_y[indices]
            loss = (F.binary_cross_entropy_with_logits(output[:, :4], labels[:, :4])
                    + F.smooth_l1_loss(output[:, 4:], (labels[:, 4:] - mean) / scale))
            if not torch.isfinite(loss):
                raise ValueError('The training loss is not finite')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append((loss.item(), len(indices)))
        evaluation = metrics(predict(model, validation_x, mean, scale, args.batch_size), validation_y, scale)
        row = {'epoch': epoch, 'train_loss': sum(v * n for v, n in losses) / len(train_y),
               'validation': evaluation}
        history.append(row)
        print(json.dumps({'epoch': epoch, 'train_loss': row['train_loss'],
                          'validation_score': evaluation['selection_score']}), flush=True)
        if evaluation['selection_score'] < best_score:
            best_epoch, best_score = epoch, evaluation['selection_score']
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state, strict=True)
    validation_predictions = predict(model, validation_x, mean, scale, args.batch_size)
    permutation = torch.randperm(len(validation_y), generator=torch.Generator().manual_seed(args.seed + 1))
    report = {
        'format': 'vimm8bdm-navigation-bc-v2', 'experimental': True, 'features': features,
        'train_source': train_id, 'validation_source': validation_id, 'operator_review_hashes': reviews,
        'source_split_check': split_result, 'seed': args.seed, 'epochs': args.epochs,
        'batch_size': args.batch_size, 'learning_rate': args.learning_rate, 'cpu_threads': args.threads,
        'torch_version': torch.__version__, 'device': 'cpu', 'image_size': list(IMAGE_SIZE),
        'history_frames': 4, 'keys': list(KEYS), 'parameters': sum(p.numel() for p in model.parameters()),
        'mouse_mean_train_only': mean.tolist(), 'mouse_scale_train_only': scale.tolist(),
        'selection_rule': 'Minimum validation key BCE plus mean mouse MAE divided by training mouse scales',
        'best_epoch': best_epoch, 'history': history, 'baselines': baselines,
        'train_metrics': metrics(predict(model, train_x, mean, scale, args.batch_size), train_y, scale),
        'validation_metrics': metrics(validation_predictions, validation_y, scale),
        'shuffled_validation_images': metrics(validation_predictions[permutation], validation_y, scale),
        'zero_validation_images': metrics(predict(model, validation_x, mean, scale, args.batch_size, True),
                                          validation_y, scale),
        'repeated_latest_validation_images': metrics(
            predict(model, validation_x, mean, scale, args.batch_size, repeat_latest=True), validation_y, scale),
        'elapsed_seconds': time.perf_counter() - started,
        'input_sent': False, 'training_ready': False, 'deployment_ready': False,
        'mouse_unit_conversion_verified': False, 'unseen_map_evaluation': False,
    }
    with (args.output / 'model.pt').open('xb') as output:
        torch.save({'format': report['format'], 'state_dict': best_state,
                    'features': features,
                    'mouse_mean': mean, 'mouse_scale': scale, 'image_size': list(IMAGE_SIZE),
                    'keys': list(KEYS), 'seed': args.seed}, output)
    # Check the saved artifact without importing executable checkpoint objects.
    loaded, restored_mean, restored_scale = load_checkpoint(args.output / 'model.pt')
    restored_predictions = predict(loaded, validation_x[:8], restored_mean, restored_scale, 8)
    report['checkpoint_reload_matches'] = torch.allclose(restored_predictions, validation_predictions[:8], atol=1e-6)
    if not report['checkpoint_reload_matches']:
        raise ValueError('The saved checkpoint does not reproduce the model predictions')
    with (args.output / 'report.local.json').open('x') as output:
        json.dump(report, output, indent=2)
    print(json.dumps({key: report[key] for key in ('best_epoch', 'validation_metrics', 'baselines',
                                                  'elapsed_seconds', 'checkpoint_reload_matches')}, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description='Train an offline experimental navigation model')
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--validation', type=Path, required=True)
    parser.add_argument('--train-review', type=Path, required=True)
    parser.add_argument('--validation-review', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--experimental', action='store_true')
    parser.add_argument('--features', choices=FEATURE_MODES, default='raw')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=.001)
    args = parser.parse_args()
    if not (1 <= args.epochs <= 1000 and 1 <= args.batch_size <= 4096 and 1 <= args.threads <= 32):
        parser.error('Epochs, batch size, and thread count must remain within their supported ranges')
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0 or not 0 <= args.seed < 2**32:
        parser.error('Use a positive finite learning rate and an unsigned 32-bit seed')
    train(args)


if __name__ == '__main__':
    main()
