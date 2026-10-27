"""Compare bounded causal visual memory without desktop input."""
import argparse
import copy
import json
from pathlib import Path
import time

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence

from .action_events import contiguous
from .behavior_cloning import (IMAGE_SIZE, image_features, load_inputs,
                               load_training_inputs, read_review)
from .dataset import check_splits
from .recording import file_digest
from .steering import (CLASSES, SteeringModel, candidate_conditions, direction_metrics,
                       direction_targets, load_checkpoint as load_encoder_checkpoint,
                       tolerant_transitions, transition_metrics)

FORMAT = 'vimm8bdm-temporal-steering-v1'


def causal_windows(groups, steps):
    """Use only current and earlier contiguous anchors within one source group."""
    if type(steps) is not int or not 1 <= steps <= 64:
        raise ValueError('The visual context must contain one through 64 steps')
    windows, offset = [], 0
    for samples in groups:
        begin = 0
        for i, sample in enumerate(samples):
            if i and not contiguous(samples[i - 1], sample):
                begin = i
            windows.append(list(range(offset + max(begin, i - steps + 1), offset + i + 1)))
        offset += len(samples)
    return windows


def encode(encoder, images, batch_size=64):
    encoder.eval()
    encoder.requires_grad_(False)
    with torch.no_grad():
        return torch.cat([encoder(image_features(batch.float() / 255, 'motion'))
                          for batch in images.split(batch_size)])


def normalization(features):
    return features.mean(0), features.std(0, correction=0).clamp_min(.001)


class TemporalSteering(nn.Module):
    def __init__(self):
        super().__init__()
        self.memory = nn.GRU(128, 64, batch_first=True)
        self.head = nn.Linear(64, 3)

    def forward(self, padded, lengths):
        packed = pack_padded_sequence(padded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.memory(packed)
        return self.head(hidden[-1])


def batch_windows(features, windows, indices, mode='normal'):
    rows = []
    for index in indices:
        positions = windows[int(index)]
        if mode == 'reverse':
            positions = list(reversed(positions))
        elif mode == 'repeat_latest':
            positions = [positions[-1]] * len(positions)
        elif mode != 'normal':
            raise ValueError('The temporal comparison mode is unsupported')
        rows.append(features[positions])
    return pad_sequence(rows, batch_first=True), torch.tensor([len(row) for row in rows])


def predict(model, features, windows, batch_size=64, mode='normal'):
    outputs = []
    model.eval()
    with torch.inference_mode():
        for indices in torch.arange(len(windows)).split(batch_size):
            outputs.append(model(*batch_windows(features, windows, indices, mode)))
    return torch.cat(outputs)


def load_checkpoint(path):
    value = torch.load(path, weights_only=True, map_location='cpu')
    steps = value.get('context_steps')
    if (value.get('format') != FORMAT or value.get('classes') != CLASSES
            or value.get('image_size') != list(IMAGE_SIZE) or value.get('features') != 'motion'
            or value.get('action_window_ms') != 50 or value.get('history_spacing_ms') != 50
            or type(steps) is not int or not 1 <= steps <= 64):
        raise ValueError('The temporal checkpoint configuration is unsupported')
    mean, scale = value['feature_mean'], value['feature_scale']
    if (mean.shape != (128,) or scale.shape != (128,) or not torch.isfinite(mean).all()
            or not torch.isfinite(scale).all() or not (scale > 0).all()):
        raise ValueError('The temporal checkpoint normalization is invalid')
    encoder, model = SteeringModel().layers[:-1], TemporalSteering()
    encoder.load_state_dict(value['encoder_state_dict'], strict=True)
    model.load_state_dict(value['temporal_state_dict'], strict=True)
    if any(not torch.isfinite(p).all() for module in (encoder, model) for p in module.parameters()):
        raise ValueError('The temporal checkpoint contains nonfinite parameters')
    encoder.eval().requires_grad_(False)
    model.eval()
    return encoder, model, mean, scale, steps


def fit(train_features, labels, windows, val_features, val_labels, val_windows, args):
    """Update temporal parameters from training features and targets only."""
    torch.manual_seed(args.seed)
    mean, scale = normalization(train_features)
    train_features, val_features = (train_features - mean) / scale, (val_features - mean) / scale
    model = TemporalSteering()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    generator = torch.Generator().manual_seed(args.seed)
    best, epoch_at_best, score_at_best, history = None, None, -1., []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.
        for indices in torch.randperm(len(labels), generator=generator).split(args.batch_size):
            output = model(*batch_windows(train_features, windows, indices))
            loss = F.cross_entropy(output, labels[indices])
            if not torch.isfinite(loss):
                raise ValueError('The temporal classification loss is not finite')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(indices)
        metrics = direction_metrics(val_labels, predict(model, val_features, val_windows, args.batch_size).argmax(1))
        history.append({'epoch': epoch, 'loss': total_loss / len(labels), 'validation': metrics})
        print(json.dumps({'epoch': epoch, 'macro_f1': metrics['macro_f1']}), flush=True)
        if metrics['macro_f1'] > score_at_best:
            best, epoch_at_best, score_at_best = copy.deepcopy(model.state_dict()), epoch, metrics['macro_f1']
    model.load_state_dict(best, strict=True)
    return model, mean, scale, history, epoch_at_best


def train(args):
    if not args.experimental:
        raise ValueError('This candidate-data experiment requires --experimental')
    if args.output.exists():
        raise FileExistsError('The output directory already exists')
    if len(args.train) != len(args.train_review):
        raise ValueError('Supply one review for each training source')
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    reviews = {'train': [read_review(p) for p in args.train_review],
               'validation': read_review(args.validation_review)}
    splits = check_splits([*args.train, args.validation])
    started = time.perf_counter()
    train_x, targets, identities = load_training_inputs(args.train)
    val_x, val_targets, val_id = load_inputs(args.validation, 'validation')
    if any(row[k] != val_id[k] for row in identities for k in ('map', 'character')):
        raise ValueError('The temporal diagnostic requires the same map and character')
    encoder_report_path = args.encoder.with_name('report.local.json')
    encoder_report = json.loads(encoder_report_path.read_text())
    provenance = lambda rows: {(row['source_session_id'], row['derived_samples_sha256']) for row in rows}
    if (provenance(encoder_report['train_sources']) != provenance(identities)
            or encoder_report['validation_source']['source_session_id'] != val_id['source_session_id']):
        raise ValueError('The encoder source report does not match this training experiment')
    groups = [[json.loads(line) for line in (p / 'samples.jsonl').read_text().splitlines()] for p in args.train]
    val_samples = [json.loads(line) for line in (args.validation / 'samples.jsonl').read_text().splitlines()]
    windows, val_windows = causal_windows(groups, args.context_steps), causal_windows([val_samples], args.context_steps)
    encoder = load_encoder_checkpoint(args.encoder).layers[:-1]
    encoder_before = {k: v.clone() for k, v in encoder.state_dict().items()}
    features, val_features = encode(encoder, train_x), encode(encoder, val_x)
    labels, val_labels = direction_targets(targets), direction_targets(val_targets)
    model, mean, scale, history, best_epoch = fit(features, labels, windows, val_features, val_labels, val_windows, args)
    normalized = (val_features - mean) / scale
    logits = predict(model, normalized, val_windows, args.batch_size)
    predictions = logits.argmax(1)
    result = direction_metrics(val_labels, predictions)
    permutation = torch.randperm(len(val_labels), generator=torch.Generator().manual_seed(args.seed + 1))
    shuffled = direction_metrics(val_labels, logits[permutation].argmax(1))
    tolerant = [tolerant_transitions(val_samples, val_labels, predictions, ms * 1_000_000) for ms in (0, 50, 100, 200)]
    conditions = candidate_conditions(result, shuffled)
    for kind in ('start', 'stop'):
        metrics = tolerant[2]['transitions'][kind]
        conditions[kind + '_precision_and_recall_at_100ms_at_least_0_4'] = min(metrics['precision'], metrics['recall']) >= .4
    frozen = all(torch.equal(v, encoder_before[k]) for k, v in encoder.state_dict().items())
    if not frozen:
        raise ValueError('The frozen encoder changed during temporal training')
    report = {'format': FORMAT, 'context_steps': args.context_steps,
              'nominal_visual_history_ms': 150 + 50 * (args.context_steps - 1),
              'epochs': args.epochs, 'seed': args.seed, 'batch_size': args.batch_size, 'threads': args.threads,
              'learning_rate': .001, 'class_weights': 'uniform', 'train_sources': identities,
              'validation_source': val_id, 'source_split_check': splits, 'review_hashes': reviews,
              'encoder_checkpoint_sha256': file_digest(args.encoder), 'encoder_unchanged': frozen,
              'encoder_report_sha256': file_digest(encoder_report_path),
              'encoder_sources_checked': True, 'encoder_selected_with_same_validation': True,
              'frozen_encoder_parameters': sum(p.numel() for p in encoder.parameters()),
              'temporal_parameters': sum(p.numel() for p in model.parameters()), 'history': history,
              'selection_rule': 'Maximum validation macro F1, earliest epoch on ties', 'best_epoch': best_epoch,
              'validation': result, 'shuffled_windows': shuffled,
              'reversed_past_features': direction_metrics(val_labels, predict(model, normalized, val_windows, args.batch_size, 'reverse').argmax(1)),
              'repeated_latest_features': direction_metrics(val_labels, predict(model, normalized, val_windows, args.batch_size, 'repeat_latest').argmax(1)),
              'activity_transitions': transition_metrics(val_samples, val_labels, predictions),
              'tolerant_transitions': tolerant, 'candidate_conditions': conditions,
              'candidate_conditions_passed': all(conditions.values()),
              'source_boundary_resets': len(groups), 'train_context_resets': sum(i == 0 or not contiguous(rows[i - 1], row)
                                          for rows in groups for i, row in enumerate(rows)),
              'validation_context_resets': sum(i == 0 or not contiguous(val_samples[i - 1], row)
                                               for i, row in enumerate(val_samples)),
              'feature_mean_train_only': mean.tolist(), 'feature_scale_train_only': scale.tolist(),
              'elapsed_seconds': time.perf_counter() - started, 'torch_version': torch.__version__,
              'input_sent': False, 'deployment_ready': False, 'recorded_input_context_used': False}
    args.output.mkdir(parents=True, exist_ok=False)
    torch.save({'format': FORMAT, 'context_steps': args.context_steps, 'classes': CLASSES,
                'image_size': list(IMAGE_SIZE), 'features': 'motion', 'action_window_ms': 50,
                'history_spacing_ms': 50, 'feature_mean': mean, 'feature_scale': scale,
                'encoder_state_dict': encoder.state_dict(), 'temporal_state_dict': model.state_dict()}, args.output / 'model.pt')
    restored_encoder, restored_model, restored_mean, restored_scale, restored_steps = load_checkpoint(args.output / 'model.pt')
    restored_features = encode(restored_encoder, val_x)
    restored_logits = predict(restored_model, (restored_features - restored_mean) / restored_scale,
                              causal_windows([val_samples], restored_steps), args.batch_size)
    report['checkpoint_reload_matches'] = torch.allclose(restored_logits, logits, atol=1e-6)
    report['checkpoint_reload_checked_samples'] = len(val_x)
    if not report['checkpoint_reload_matches']:
        raise ValueError('The temporal checkpoint does not reproduce its predictions')
    with (args.output / 'report.local.json').open('x') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({'best_epoch': best_epoch, 'validation': result, 'conditions': conditions}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description='Train an offline visual memory diagnostic')
    parser.add_argument('--train', nargs='+', type=Path, required=True)
    parser.add_argument('--train-review', nargs='+', type=Path, required=True)
    parser.add_argument('--validation', type=Path, required=True)
    parser.add_argument('--validation-review', type=Path, required=True)
    parser.add_argument('--encoder', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--context-steps', type=int, choices=(1, 32), required=True)
    parser.add_argument('--experimental', action='store_true')
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if not (1 <= args.epochs <= 100 and 0 <= args.seed < 2**32
            and 1 <= args.batch_size <= 4096 and 1 <= args.threads <= 32):
        parser.error('The training options exceed the supported ranges')
    train(args)


if __name__ == '__main__':
    main()
