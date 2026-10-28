"""Adapt the existing SSDLite detector with reviewed body images."""
import argparse
import json
from pathlib import Path
import statistics
import time

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.transforms.functional import pil_to_tensor

from .pretrained_detection import CHECKPOINT_SHA256, load_model, person_boxes, complete_synthetic_provenance
from .recording import file_digest
from .sprite_synthesis import DATA_FORMAT
from .real_body_data import FORMAT as MIXED_FORMAT
from .visual_metrics import box_iou

FORMAT = 'vimm8bdm-ssdlite-sprite-transfer-v1'


class SyntheticImages(Dataset):
    def __init__(self, directory, split):
        self.directory = directory
        metadata = json.loads((directory / 'dataset.json').read_text())
        if (metadata['format'] not in (DATA_FORMAT, MIXED_FORMAT) or split not in ('train', 'validation')
                or metadata.get('reserved_for_validation', False)
                or (metadata['format'] == MIXED_FORMAT and metadata.get('image_gradient_training_allowed') is not True)):
            raise ValueError('Supply an eligible body training dataset and a supported split')
        annotation = directory / f'{split}.coco.json'
        if file_digest(annotation) != metadata['annotation_sha256'][split]:
            raise ValueError('The body annotations changed after generation')
        coco = json.loads(annotation.read_text())
        self.rows = coco['images']
        self.targets = {}
        self.hashes = metadata['image_sha256']
        for row in coco['annotations']:
            if row['category_id'] != 1:
                raise ValueError('Body transfer supports one body category')
            x, y, width, height = row['bbox']
            self.targets.setdefault(row['image_id'], []).append([x, y, x + width, y + height])
        if len(self.rows) != metadata['counts'][split]:
            raise ValueError('The split count does not match its saved images')
        if not self.rows or any(len(self.targets.get(row['id'], [])) != 1 for row in self.rows):
            raise ValueError('Each training or monitoring image requires exactly one body')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = (self.directory / row['file_name']).resolve()
        if self.directory.resolve() not in path.parents or file_digest(path) != self.hashes[row['file_name']]:
            raise ValueError('A synthetic image does not match its dataset hash')
        with Image.open(path) as source:
            image = pil_to_tensor(source.convert('RGB')).to(torch.float32) / 255
        boxes = torch.tensor(self.targets[row['id']], dtype=torch.float32)
        if (torch.any(boxes[:, :2] < 0) or torch.any(boxes[:, 2:] <= boxes[:, :2])
                or torch.any(boxes[:, 2] > image.shape[2]) or torch.any(boxes[:, 3] > image.shape[1])):
            raise ValueError('A synthetic body box extends outside its image')
        return image, dict(boxes=boxes, labels=torch.ones(len(boxes), dtype=torch.int64))


def load_checkpoint(path, provenance_dataset=None):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if (checkpoint.get('format') != FORMAT or checkpoint.get('initial_checkpoint_sha256') != CHECKPOINT_SHA256
            or checkpoint.get('training_source_session_id') is None or checkpoint.get('class_id') != 1):
        raise ValueError('The synthetic transfer checkpoint has an unsupported identity')
    if provenance_dataset is not None:
        checkpoint = complete_synthetic_provenance(checkpoint, json.loads(provenance_dataset.read_text()),
                                                  file_digest(provenance_dataset))
    model = ssdlite320_mobilenet_v3_large(weights=None, weights_backbone=None)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.eval()
    return model, checkpoint


def validate(model, images):
    model.eval()
    matches, false_predictions = 0, 0
    with torch.inference_mode():
        for i in range(len(images)):
            image, target = images[i]
            prediction = person_boxes(model([image])[0], image.shape[2], image.shape[1], .5)
            a = target['boxes'][0].tolist()
            expected = [a[0], a[1], a[2]-a[0], a[3]-a[1]]
            hits = [box_iou(row['box'], expected) >= .5 for row in prediction]
            matches += any(hits)
            false_predictions += len(hits) - int(any(hits))
    return dict(images=len(images), localized=matches, recall=matches / len(images),
                unmatched_predictions=false_predictions, score_threshold=.5, minimum_iou=.5)


def train(dataset, initial, output, epochs=8, batch_size=4, seed=0, initial_transfer=False, learning_rate=.002):
    if output.exists():
        raise FileExistsError('The training output already exists')
    import math
    if not math.isfinite(learning_rate) or not 0 < learning_rate <= .1:
        raise ValueError('Supply a finite learning rate greater than zero and at most 0.1')
    if type(epochs) is not int or epochs < 1 or batch_size < 2:
        raise ValueError('Supply positive epochs and a batch size of at least two')
    training, validation = SyntheticImages(dataset, 'train'), SyntheticImages(dataset, 'validation')
    metadata = json.loads((dataset / 'dataset.json').read_text())
    if set(metadata['train_background_frames']) & set(metadata['validation_background_frames']):
        raise ValueError('Synthetic splits cannot reuse the same background frames')
    dataset_hash = file_digest(dataset / 'dataset.json')
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    source_ids = metadata.get('training_source_session_ids', [metadata['source_session_id']])
    source_archives = metadata.get('training_source_archives')
    if source_archives is None:
        source_archives = {metadata['source_session_id']: metadata['source_hashes']['frames.vrec']}
    if initial_transfer:
        model, previous = load_checkpoint(initial)
        inherited = previous.get('training_source_session_ids', [previous['training_source_session_id']])
        if not set(inherited).issubset(source_ids):
            raise ValueError('Dataset provenance must include every inherited training recording')
    else:
        model = load_model(initial)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=.9, weight_decay=.0005)
    loader = DataLoader(training, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True,
                        generator=torch.Generator().manual_seed(seed), collate_fn=lambda rows: tuple(zip(*rows)))
    if not len(loader):
        raise ValueError('The training split is smaller than one complete batch')
    output.mkdir(parents=True, exist_ok=False)
    protocol = dict(format=FORMAT, epochs=epochs, batch_size=batch_size, seed=seed, threads=4, device='cpu',
                    learning_rate=learning_rate, momentum=.9, weight_decay=.0005, gradient_clip=10,
                    selection='fixed_final_epoch', train_images=len(training), validation_images=len(validation),
                    initial_checkpoint_sha256=CHECKPOINT_SHA256, synthetic_dataset_sha256=dataset_hash,
                    starting_checkpoint_sha256=file_digest(initial), starting_from_transfer=initial_transfer,
                    training_source_session_ids=source_ids, training_source_archives=source_archives,
                    training_source_session_id=metadata['source_session_id'], input_sent=False,
                    real_calibration_images_used_for_gradients=metadata['format'] == MIXED_FORMAT, independent_validation=False)
    (output / 'protocol.local.json').write_text(json.dumps(protocol, indent=2))
    records, steps = [], 0
    started = time.perf_counter()
    with (output / 'progress.local.jsonl').open('x') as stream:
        for epoch in range(1, epochs + 1):
            model.train()
            losses = []
            for batch, (images, targets) in enumerate(loader, 1):
                optimizer.zero_grad(set_to_none=True)
                parts = model(list(images), list(targets))
                loss = sum(parts.values())
                if not torch.isfinite(loss):
                    raise ValueError('The body detector loss is not finite')
                loss.backward()
                gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
                if not torch.isfinite(gradient):
                    raise ValueError('The body detector gradient is not finite')
                optimizer.step()
                losses.append(float(loss.detach()))
                steps += 1
                if batch == 1 or batch % 32 == 0:
                    print(json.dumps(dict(epoch=epoch,batch=batch,batches=len(loader),loss=losses[-1],
                                          elapsed_seconds=time.perf_counter()-started)),flush=True)
            record = dict(epoch=epoch, mean_loss=statistics.mean(losses), steps=steps,
                          validation=validate(model, validation), elapsed_seconds=time.perf_counter()-started)
            records.append(record)
            stream.write(json.dumps(record) + '\n')
            stream.flush()
            print(json.dumps(record), flush=True)
    if file_digest(dataset / 'dataset.json') != dataset_hash:
        raise ValueError('The training dataset metadata changed during training')
    checkpoint = dict(format=FORMAT, state_dict=model.state_dict(), class_id=1,
                      initial_checkpoint_sha256=CHECKPOINT_SHA256,
                      training_source_session_id=metadata['source_session_id'],
                      synthetic_dataset_sha256=dataset_hash, epochs=epochs, steps=steps, seed=seed,
                      training_source_session_ids=source_ids, training_source_archives=source_archives,
                      starting_checkpoint_sha256=file_digest(initial),real_image_training=metadata['format'] == MIXED_FORMAT)
    with (output / 'model.pt').open('xb') as stream:
        torch.save(checkpoint, stream)
    result = dict(protocol=protocol, epochs=records, seconds=time.perf_counter()-started,
                  checkpoint_sha256=file_digest(output / 'model.pt'), input_sent=False)
    (output / 'training.local.json').write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--initial', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--initial-transfer', action='store_true')
    parser.add_argument('--learning-rate', type=float, default=.002)
    args = parser.parse_args()
    result = train(args.dataset, args.initial, args.output, epochs=args.epochs,
                   initial_transfer=args.initial_transfer, learning_rate=args.learning_rate)
    print(json.dumps(dict(seconds=result['seconds'], checkpoint_sha256=result['checkpoint_sha256'])), flush=True)


if __name__ == '__main__':
    main()
