"""Compare horizontal input classification without desktop input."""
import argparse
import copy
from collections import defaultdict
import json
from pathlib import Path
import time

import torch
from torch import nn
from torch.nn import functional as F

from .action_events import binary_events, contiguous
from .behavior_cloning import (IMAGE_SIZE, NavigationModel, load_inputs,
                               load_training_inputs, read_review)
from .dataset import check_splits

FORMAT = 'vimm8bdm-steering-classifier-v1'
CLASSES = ['negative', 'zero', 'positive']


class SteeringModel(NavigationModel):
    def __init__(self):
        super().__init__('motion')
        self.layers[-1] = nn.Linear(128, 3)


def direction_targets(targets):
    if targets.ndim != 2 or targets.shape[1] != 6 or not torch.isfinite(targets).all():
        raise ValueError('Steering labels require six finite navigation targets')
    return targets[:, 4].sign().long() + 1


def class_statistics(labels, weighted):
    counts = torch.bincount(labels, minlength=3)
    if weighted:
        weights = torch.where(counts > 0, counts.clamp_min(1).float().rsqrt(), 0.)
        weights /= (counts * weights).sum() / len(labels)
    else:
        weights = torch.ones(3)
    return counts, weights


def direction_metrics(labels, predicted):
    if (labels.ndim != 1 or predicted.shape != labels.shape or not len(labels)
            or labels.dtype not in (torch.int32, torch.int64)
            or predicted.dtype not in (torch.int32, torch.int64)
            or any(not ((x >= 0) & (x <= 2)).all() for x in (labels, predicted))):
        raise ValueError('Supply equal nonempty class vectors with values zero through two')
    confusion = torch.bincount(labels.long() * 3 + predicted.long(), minlength=9).reshape(3, 3)
    per_class = {}
    for k, name in enumerate(CLASSES):
        tp = int(confusion[k, k])
        support, selected = int(confusion[k].sum()), int(confusion[:, k].sum())
        per_class[name] = {'support': support, 'predicted': selected,
                           'precision': tp / selected if selected else 0.,
                           'recall': tp / support if support else 0.,
                           'f1': 2 * tp / (support + selected) if support + selected else 0.}
    zero = per_class['zero']['support']
    return {'confusion_rows_actual_columns_predicted': confusion.tolist(),
            'classes': CLASSES, 'per_class': per_class,
            'accuracy': float(confusion.diag().sum() / len(labels)),
            'macro_f1': sum(v['f1'] for v in per_class.values()) / 3,
            'zero_window_false_turn_rate': 1 - per_class['zero']['recall'] if zero else None}


def predict(model, images, batch_size=64, repeat_latest=False):
    outputs = []
    model.eval()
    with torch.inference_mode():
        for batch in images.split(batch_size):
            batch = batch.float() / 255
            if repeat_latest:
                batch = batch[:, -1:].expand(-1, 4, -1, -1).contiguous()
            outputs.append(model(batch))
    return torch.cat(outputs)


def load_checkpoint(path):
    value = torch.load(path, weights_only=True, map_location='cpu')
    if (value.get('format') != FORMAT or value.get('classes') != CLASSES
            or value.get('image_size') != list(IMAGE_SIZE)
            or value.get('features') != 'motion' or value.get('action_window_ms') != 50):
        raise ValueError('The steering checkpoint configuration is unsupported')
    model = SteeringModel()
    model.load_state_dict(value['state_dict'], strict=True)
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError('The steering checkpoint contains nonfinite parameters')
    model.eval()
    return model


def candidate_conditions(result, shuffled):
    return {'macro_f1_at_least_half': result['macro_f1'] >= .5,
            'all_class_recalls_at_least_0_4': all(v['recall'] >= .4 for v in result['per_class'].values()),
            'zero_false_turn_rate_at_most_0_1': result['zero_window_false_turn_rate'] is not None
                and result['zero_window_false_turn_rate'] <= .1,
            'shuffle_macro_f1_drop_at_least_0_05': result['macro_f1'] - shuffled['macro_f1'] >= .05}


def transition_metrics(samples, labels, predicted):
    """Compare input activity changes only across contiguous source windows."""
    if len(samples) != len(labels) or predicted.shape != labels.shape:
        raise ValueError('Supply aligned source windows and direction classes')
    direction_metrics(labels, predicted)
    truth, estimates = {k: [] for k in ('start', 'stop', 'reversal')}, {k: [] for k in ('start', 'stop', 'reversal')}
    excluded = 1
    for i in range(1, len(samples)):
        if not contiguous(samples[i - 1], samples[i]):
            excluded += 1
            continue
        for values, output in ((labels, truth), (predicted, estimates)):
            previous, current = int(values[i - 1]), int(values[i])
            output['start'].append(previous == 1 and current != 1)
            output['stop'].append(previous != 1 and current == 1)
            output['reversal'].append(previous != 1 and current != 1 and previous != current)
    return {'transitions': {k: binary_events(truth[k], estimates[k]) for k in truth},
            'boundary_windows_excluded': excluded}


def tolerant_transitions(samples, labels, predicted, tolerance_ns):
    """Match each same-direction event at most once within a symmetric time tolerance."""
    transition_metrics(samples, labels, predicted)
    if type(tolerance_ns) is not int or tolerance_ns < 0:
        raise ValueError('The event tolerance must not be negative')

    def events(values):
        result, block = defaultdict(list), 0
        for i in range(1, len(samples)):
            if not contiguous(samples[i - 1], samples[i]):
                block += 1
                continue
            previous, current = int(values[i - 1]), int(values[i])
            kind = ('start' if previous == 1 and current != 1 else
                    'stop' if previous != 1 and current == 1 else
                    'reversal' if previous != 1 and current != 1 and previous != current else None)
            if kind:
                direction = previous if kind == 'stop' else current
                result[(kind, block, direction)].append(samples[i]['anchor_ns'])
        return result

    truth, estimates = events(labels), events(predicted)
    result = {}
    for kind in ('start', 'stop', 'reversal'):
        tp = actual = selected = 0
        for group in set(truth) | set(estimates):
            if group[0] != kind:
                continue
            a, b = truth.get(group, []), estimates.get(group, [])
            actual += len(a)
            selected += len(b)
            i = j = 0
            while i < len(a) and j < len(b):
                if b[j] < a[i] - tolerance_ns:
                    j += 1
                elif b[j] > a[i] + tolerance_ns:
                    i += 1
                else:
                    tp += 1
                    i += 1
                    j += 1
        result[kind] = {'true_events': actual, 'predicted_events': selected, 'tp': tp,
                        'precision': tp / selected if selected else 0.,
                        'recall': tp / actual if actual else 0.,
                        'f1': 2 * tp / (actual + selected) if actual + selected else 0.}
    return {'symmetric_tolerance_ns': tolerance_ns, 'requires_same_input_direction': True,
            'transitions': result}


def train(args):
    if not args.experimental:
        raise ValueError('This candidate-data experiment requires --experimental')
    if args.output.exists():
        raise FileExistsError('The output directory already exists')
    if len(args.train) != len(args.train_review):
        raise ValueError('Supply one review for each training source')
    reviews = {'train': [read_review(p) for p in args.train_review],
               'validation': read_review(args.validation_review)}
    splits = check_splits([*args.train, args.validation])
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.seed)
    started = time.perf_counter()
    train_x, targets, ids = load_training_inputs(args.train)
    val_x, val_targets, val_id = load_inputs(args.validation, 'validation')
    if any(i[k] != val_id[k] for i in ids for k in ('character', 'map')):
        raise ValueError('This diagnostic requires the same map and character in both splits')
    labels, val_labels = direction_targets(targets), direction_targets(val_targets)
    counts, weights = class_statistics(labels, args.weighted)
    model = SteeringModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    generator = torch.Generator().manual_seed(args.seed)
    history, best, best_epoch, best_score = [], None, None, -1.
    args.output.mkdir(parents=True, exist_ok=False)
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for ix in torch.randperm(len(labels), generator=generator).split(args.batch_size):
            loss = F.cross_entropy(model(train_x[ix].float() / 255), labels[ix],
                                   weight=weights, reduction='none').mean()
            if not torch.isfinite(loss):
                raise ValueError('The classification loss is not finite')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append((float(loss.detach()), len(ix)))
        result = direction_metrics(val_labels, predict(model, val_x, args.batch_size).argmax(1))
        row = {'epoch': epoch, 'loss': sum(a * n for a, n in losses) / len(labels),
               'validation': result}
        history.append(row)
        print(json.dumps({'epoch': epoch, 'macro_f1': result['macro_f1']}), flush=True)
        if result['macro_f1'] > best_score:
            best, best_epoch, best_score = copy.deepcopy(model.state_dict()), epoch, result['macro_f1']
    model.load_state_dict(best, strict=True)
    logits = predict(model, val_x, args.batch_size)
    result = direction_metrics(val_labels, logits.argmax(1))
    permutation = torch.randperm(len(val_labels), generator=torch.Generator().manual_seed(args.seed + 1))
    shuffled = direction_metrics(val_labels, logits[permutation].argmax(1))
    conditions = candidate_conditions(result, shuffled)
    report = {'format': FORMAT, 'weighted': args.weighted, 'seed': args.seed,
              'epochs': args.epochs, 'batch_size': args.batch_size, 'threads': args.threads,
              'learning_rate': .001, 'train_sources': ids, 'validation_source': val_id,
              'source_split_check': splits, 'review_hashes': reviews,
              'train_class_counts': counts.tolist(), 'train_class_weights': weights.tolist(),
              'selection_rule': 'Maximum validation three-class macro F1, with the earliest epoch on ties',
              'best_epoch': best_epoch, 'history': history, 'validation': result,
              'shuffled_images': shuffled, 'repeated_latest_images': direction_metrics(
                  val_labels, predict(model, val_x, args.batch_size, True).argmax(1)),
              'train_majority_baseline': direction_metrics(val_labels, counts.argmax().expand_as(val_labels)),
              'zero_baseline': direction_metrics(val_labels, torch.ones_like(val_labels)),
              'candidate_conditions': conditions, 'candidate_conditions_passed': all(conditions.values()),
              'elapsed_seconds': time.perf_counter() - started,
              'parameters': sum(p.numel() for p in model.parameters()),
              'torch_version': torch.__version__, 'input_sent': False, 'deployment_ready': False,
              'predicts_magnitude': False, 'predicts_movement_keys': False}
    samples = [json.loads(line) for line in (args.validation / 'samples.jsonl').read_text().splitlines()]
    report['activity_transitions'] = transition_metrics(samples, val_labels, logits.argmax(1))
    report['tolerant_transitions'] = [tolerant_transitions(samples, val_labels, logits.argmax(1), ms * 1_000_000)
                                     for ms in (0, 50, 100, 200)]
    torch.save({'format': FORMAT, 'classes': CLASSES, 'image_size': list(IMAGE_SIZE),
                'features': 'motion', 'action_window_ms': 50, 'state_dict': best}, args.output / 'model.pt')
    restored = predict(load_checkpoint(args.output / 'model.pt'), val_x[:8], 8)
    report['checkpoint_reload_matches'] = torch.allclose(restored, logits[:8], atol=1e-6)
    if not report['checkpoint_reload_matches']:
        raise ValueError('The saved classifier does not reproduce its outputs')
    with (args.output / 'report.local.json').open('x') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({'best_epoch': best_epoch, 'validation': result, 'conditions': conditions}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description='Train an offline horizontal input classifier')
    parser.add_argument('--train', nargs='+', type=Path, required=True)
    parser.add_argument('--train-review', nargs='+', type=Path, required=True)
    parser.add_argument('--validation', type=Path, required=True)
    parser.add_argument('--validation-review', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--weighted', action='store_true')
    parser.add_argument('--experimental', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if not (1 <= args.epochs <= 100 and 1 <= args.batch_size <= 4096
            and 1 <= args.threads <= 32 and 0 <= args.seed < 2**32):
        parser.error('Training options exceed the supported ranges')
    train(args)


if __name__ == '__main__':
    main()
