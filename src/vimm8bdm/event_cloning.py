"""Fit explicit navigation events with image history and owned-key context."""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import time

import torch
from torch import nn
from torch.nn import functional as F

from .action_events import binary_events, contiguous, evaluate_explicit_events
from .behavior_cloning import IMAGE_SIZE, NavigationModel, image_features, load_inputs, read_review
from .dataset import check_splits
from .event_targets import EventTarget, KEYS, VKS, WINDOW_NS, encode_label, target_label, compile_target

FORMAT = 'vimm8bdm-event-policy-v1'


def load_data(directory, split):
    images, old_targets, identity = load_inputs(directory, split)
    samples = [json.loads(line) for line in (directory / 'samples.jsonl').read_text().splitlines()]
    states, fields = [], []
    for sample in samples:
        owned, target = encode_label(sample['action'], sample['action_end_ns'] - sample['anchor_ns'])
        states.append([int(key in owned) for key in KEYS])
        fields.append(target.training_fields())
    data = {'states': torch.tensor(states, dtype=torch.float32),
            'counts': torch.tensor([f['key_counts'] for f in fields]),
            'times': torch.tensor([f['key_offset_fraction'] for f in fields], dtype=torch.float32),
            'time_mask': torch.tensor([f['key_offset_mask'] for f in fields]),
            'mouse': old_targets[:, 4:]}
    data['mouse_class'] = data['mouse'].sign().long() + 1
    return images, data, samples, identity


def training_statistics(data):
    def weights(targets):
        counts = torch.stack([torch.bincount(targets[:, k], minlength=3) for k in range(targets.shape[1])])
        raw = torch.where(counts > 0, counts.clamp_min(1).float().rsqrt(), 0.)
        factor = (counts * raw).sum(1, keepdim=True) / len(targets)
        return raw / factor, counts
    count_weights, key_counts = weights(data['counts'])
    mouse_weights, mouse_counts = weights(data['mouse_class'])
    scales = []
    for axis in range(2):
        active = data['mouse'][:, axis] != 0
        scales.append(data['mouse'][active, axis].abs().mean().clamp_min(1) if active.any() else torch.tensor(1.))
    return {'key_weights': count_weights, 'mouse_weights': mouse_weights,
            'key_class_counts': key_counts, 'mouse_class_counts': mouse_counts,
            'mouse_scale': torch.stack(scales)}


class EventModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = NavigationModel('motion').layers[:-1]
        self.head = nn.Linear(132, 28)

    def forward(self, images, states):
        if (states.shape != (len(images), 4) or not torch.isfinite(states).all()
                or not ((states == 0) | (states == 1)).all()):
            raise ValueError('Event inference requires four owned-key context values per image history')
        features = self.encoder(image_features(images, 'motion'))
        return self.head(torch.cat((features, states.to(features.dtype)), dim=1))


def unpack(raw):
    counts = raw[:, :12].reshape(-1, 4, 3)
    time_logits = raw[:, 12:20].reshape(-1, 4, 2)
    first = time_logits[:, :, 0].sigmoid()
    second = first + (1 - first) * time_logits[:, :, 1].sigmoid()
    times = torch.stack((first, second), dim=2)
    return counts, times, raw[:, 20:26].reshape(-1, 2, 3), F.softplus(raw[:, 26:28])


def event_loss(raw, data, stats):
    counts, times, classes, magnitudes = unpack(raw)
    key_loss = sum(F.cross_entropy(counts[:, k], data['counts'][:, k], weight=stats['key_weights'][k],
                                  reduction='none').mean() for k in range(4)) / 4
    mouse_loss = sum(F.cross_entropy(classes[:, k], data['mouse_class'][:, k], weight=stats['mouse_weights'][k],
                                    reduction='none').mean() for k in range(2)) / 2
    time_loss = (F.smooth_l1_loss(times, data['times'], reduction='none') * data['time_mask']).sum()
    time_loss = time_loss / data['time_mask'].sum().clamp_min(1)
    active = data['mouse'] != 0
    magnitude_loss = (F.smooth_l1_loss(magnitudes, data['mouse'].abs() / stats['mouse_scale'],
                                      reduction='none') * active).sum() / active.sum().clamp_min(1)
    return key_loss + mouse_loss + time_loss + magnitude_loss


def infer_raw(model, images, states, batch_size=64):
    model.eval()
    with torch.inference_mode():
        return torch.cat([model(images[i:i + batch_size].float() / 255, states[i:i + batch_size])
                          for i in range(0, len(images), batch_size)])


def activity_score(raw, data):
    counts, _, mouse_class, _ = unpack(raw)
    reports = []
    for prediction, truth in [(counts.argmax(2) != 0, data['counts'] != 0),
                              (mouse_class.argmax(2) != 1, data['mouse_class'] != 1)]:
        group = [binary_events(truth[:, k].tolist(), prediction[:, k].tolist()) for k in range(truth.shape[1])]
        defined = [r['f1'] for r in group if r['f1'] is not None]
        reports.append({'per_channel': group, 'macro_f1': sum(defined) / len(defined) if defined else 0.})
    return {'key_change': reports[0], 'mouse_activity': reports[1],
            'selection_score': (reports[0]['macro_f1'] + reports[1]['macro_f1']) / 2}


def decode_raw(raw, mouse_scale):
    if raw.shape != (1, 28) or not torch.isfinite(raw).all():
        raise ValueError('Each event prediction requires 28 finite outputs')
    if (not isinstance(mouse_scale, torch.Tensor) or mouse_scale.shape != (2,)
            or not torch.isfinite(mouse_scale).all() or not (mouse_scale > 0).all()):
        raise ValueError('Mouse magnitude scales must be finite and positive')
    counts, times, classes, magnitudes = unpack(raw)
    counts = counts.argmax(2)[0].tolist()
    offsets = tuple(tuple(int(float(t) * WINDOW_NS) for t in times[0, k, :count])
                    for k, count in enumerate(counts))
    signs = classes.argmax(2)[0] - 1
    mouse = (signs * magnitudes[0] * mouse_scale).tolist()
    return EventTarget(offsets, *mouse).validate()


def decode_batch(raw, states, scale):
    labels = []
    for index in range(len(raw)):
        owned = frozenset(key for key, value in zip(KEYS, states[index].tolist()) if value == 1)
        labels.append(target_label(decode_raw(raw[index:index + 1], scale), owned))
    return labels


def feedback_labels(model, images, samples, scale):
    """Use only earlier predicted keys; reset to released keys at every source gap."""
    owned, previous, labels, resets = frozenset(), None, [], 0
    model.eval()
    with torch.inference_mode():
        for index, sample in enumerate(samples):
            if not contiguous(previous, sample):
                owned = frozenset()
                resets += 1
            state = torch.tensor([[int(key in owned) for key in KEYS]], dtype=torch.float32)
            raw = model(images[index:index + 1].float() / 255, state)
            label = target_label(decode_raw(raw, scale), owned)
            owned = frozenset(key for key, vk in zip(KEYS, VKS) if vk in label['keys_at_end'])
            labels.append(label)
            previous = sample
    return labels, resets


def decoded_errors(samples, labels):
    return {'key_fraction_mae': {name: sum(abs(s['action']['key_held_fraction'].get(str(vk), 0)
                                               - p['key_held_fraction'].get(str(vk), 0))
                                            for s, p in zip(samples, labels)) / len(samples)
                                 for name, vk in zip(KEYS, VKS)},
            'mouse_mae_raw_counts': {axis: sum(abs(s['action'][field] - p[field])
                                               for s, p in zip(samples, labels)) / len(samples)
                                     for axis, field in [('x', 'mouse_dx_raw'), ('y', 'mouse_dy_raw')]}}


def save_checkpoint(path, model, scale):
    with path.open('xb') as output:
        torch.save({'format': FORMAT, 'state_dict': model.state_dict(), 'mouse_scale': scale,
                    'keys': list(KEYS), 'image_size': list(IMAGE_SIZE), 'window_ns': WINDOW_NS,
                    'features': 'motion', 'maximum_key_events': 2}, output)


def load_checkpoint(path):
    value = torch.load(path, weights_only=True, map_location='cpu')
    if (value.get('format') != FORMAT or value.get('keys') != list(KEYS)
            or value.get('image_size') != list(IMAGE_SIZE) or value.get('window_ns') != WINDOW_NS
            or value.get('features') != 'motion' or value.get('maximum_key_events') != 2):
        raise ValueError('The event checkpoint configuration is unsupported')
    scale = value['mouse_scale']
    if not isinstance(scale, torch.Tensor) or scale.shape != (2,) or not torch.isfinite(scale).all() or not (scale > 0).all():
        raise ValueError('Event checkpoint mouse scales must be finite and positive')
    model = EventModel()
    model.load_state_dict(value['state_dict'], strict=True)
    model.eval()
    return model, scale


def train(args):
    if not args.experimental:
        raise ValueError('Event training with candidate data requires --experimental')
    if args.output.exists():
        raise FileExistsError('The output directory already exists. Select a new directory')
    reviews = {'train': read_review(args.train_review), 'validation': read_review(args.validation_review)}
    split_check = check_splits([args.train, args.validation])
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.seed)
    started = time.perf_counter()
    train_x, train_data, train_samples, train_id = load_data(args.train, 'train')
    val_x, val_data, val_samples, val_id = load_data(args.validation, 'validation')
    if (train_id['character'], train_id['map']) != (val_id['character'], val_id['map']):
        raise ValueError('This event experiment requires the same map and character in both splits')
    args.output.mkdir(parents=True, exist_ok=False)
    stats = training_statistics(train_data)
    model = EventModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    generator = torch.Generator().manual_seed(args.seed)
    history, best, best_epoch, best_score = [], None, None, -1.
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for indices in torch.randperm(len(train_x), generator=generator).split(args.batch_size):
            raw = model(train_x[indices].float() / 255, train_data['states'][indices])
            data = {key: value[indices] for key, value in train_data.items()}
            loss = event_loss(raw, data, stats)
            if not torch.isfinite(loss):
                raise ValueError('The event training loss is not finite')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append((len(indices), loss.item()))
        evaluation = activity_score(infer_raw(model, val_x, val_data['states'], args.batch_size), val_data)
        row = {'epoch': epoch, 'train_loss': sum(n * loss for n, loss in losses) / len(train_x),
               'validation_activity': evaluation}
        history.append(row)
        print(json.dumps(row), flush=True)
        if evaluation['selection_score'] > best_score:
            best, best_epoch, best_score = copy.deepcopy(model.state_dict()), epoch, evaluation['selection_score']
    model.load_state_dict(best, strict=True)
    raw = infer_raw(model, val_x, val_data['states'], args.batch_size)
    labels = decode_batch(raw, val_data['states'], stats['mouse_scale'])
    feedback, resets = feedback_labels(model, val_x, val_samples, stats['mouse_scale'])
    permutation = torch.randperm(len(val_x), generator=torch.Generator().manual_seed(args.seed + 1))
    shuffled = decode_batch(infer_raw(model, val_x[permutation], val_data['states'], args.batch_size),
                            val_data['states'], stats['mouse_scale'])
    idle = EventTarget(((), (), (), ()))
    hold_reference = [target_label(idle, frozenset(key for key, v in zip(KEYS, row.tolist()) if v == 1))
                      for row in val_data['states']]
    released_reference = [target_label(idle, ()) for _ in val_samples]
    compile_counts = {}
    for name, decoded in [('recorded_state_reference', labels), ('predicted_state_feedback', feedback)]:
        packets = 0
        for label in decoded:
            owned, target = encode_label(label)
            packets += len(compile_target(target, owned, 1.06, .92).packets)
        compile_counts[name] = packets
    report = {'format': FORMAT, 'train_source': train_id, 'validation_source': val_id,
              'operator_review_hashes': reviews, 'source_split_check': split_check,
              'seed': args.seed, 'epochs': args.epochs, 'batch_size': args.batch_size,
              'learning_rate': args.learning_rate, 'cpu_threads': args.threads,
              'parameters': sum(p.numel() for p in model.parameters()), 'best_epoch': best_epoch,
              'training_statistics': {key: value.tolist() for key, value in stats.items()},
              'selection_rule': 'Maximum mean of key-change macro F1 and mouse-activity macro F1 with recorded starting states',
              'history': history, 'recorded_state_reference': evaluate_explicit_events(val_samples, labels),
              'predicted_state_feedback': evaluate_explicit_events(val_samples, feedback),
              'shuffled_images_recorded_state': evaluate_explicit_events(val_samples, shuffled),
              'recorded_state_errors': decoded_errors(val_samples, labels),
              'feedback_errors': decoded_errors(val_samples, feedback),
              'no_change_zero_mouse_recorded_state_reference': evaluate_explicit_events(val_samples, hold_reference),
              'always_released_zero_mouse_reference': evaluate_explicit_events(val_samples, released_reference),
              'no_change_zero_mouse_errors': decoded_errors(val_samples, hold_reference),
              'feedback_resets_to_released': resets, 'compiled_packet_counts': compile_counts,
              'elapsed_seconds': time.perf_counter() - started, 'input_sent': False,
              'independent_policy_rollout': False, 'deployment_ready': False, 'training_ready': False}
    save_checkpoint(args.output / 'model.pt', model, stats['mouse_scale'])
    restored, restored_scale = load_checkpoint(args.output / 'model.pt')
    report['checkpoint_reload_matches'] = torch.allclose(
        infer_raw(restored, val_x[:8], val_data['states'][:8]), raw[:8], atol=1e-6)
    if not report['checkpoint_reload_matches'] or not torch.equal(restored_scale, stats['mouse_scale']):
        raise ValueError('The event checkpoint failed its reload check')
    with (args.output / 'report.local.json').open('x') as output:
        json.dump(report, output, indent=2)
    print(json.dumps({'best_epoch': best_epoch, 'event_selection_score': best_score,
                      'recorded_state_errors': report['recorded_state_errors'],
                      'feedback_errors': report['feedback_errors'], 'input_sent': False}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description='Train explicit navigation events offline')
    for name in ('train', 'validation', 'train-review', 'validation-review', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--experimental', action='store_true')
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=.001)
    args = parser.parse_args()
    if not (1 <= args.epochs <= 1000 and 1 <= args.batch_size <= 4096 and 1 <= args.threads <= 32
            and 0 <= args.seed < 2**32 and math.isfinite(args.learning_rate) and args.learning_rate > 0):
        parser.error('Training arguments must remain within their supported ranges')
    train(args)


if __name__ == '__main__':
    main()
