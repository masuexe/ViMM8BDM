"""Build an experimental training mixture without reserved validation recordings."""
import argparse
import json
from pathlib import Path
import random

from PIL import Image, ImageEnhance

from .recording import file_digest
from .sprite_synthesis import BANK_FORMAT, DATA_FORMAT, composite
from .visual_labels import source_identity

FORMAT = 'vimm8bdm-mixed-body-training-v1'
REVIEW_FORMAT = 'vimm8bdm-body-image-training-review-v1'


def checked_image(directory, row, hashes):
    path = (directory / row['file_name']).resolve()
    if directory.resolve() not in path.parents or file_digest(path) != hashes[row['file_name']]:
        raise ValueError('A source image does not match its reviewed hash')
    with Image.open(path) as image:
        return image.convert('RGB')


def build(synthetic, calibrations, bank, review_file, output, positive_repeats=16, negative_repeats=8):
    if output.exists():
        raise FileExistsError('The mixed training directory already exists')
    if min(positive_repeats, negative_repeats) < 1:
        raise ValueError('Image repeat counts must be positive')
    for source in [synthetic, *calibrations, bank]:
        if source.resolve() in output.resolve().parents:
            raise ValueError('Create training images outside their source directories')
    permission = json.loads(review_file.read_text())
    if (permission.get('format') != REVIEW_FORMAT or permission.get('image_gradient_training_allowed') is not True
            or permission.get('action_training_allowed') is not False or not permission.get('reviewer')):
        raise ValueError('Supply an explicit experimental body-image training review')
    baseline = json.loads((synthetic / 'dataset.json').read_text())
    if baseline['format'] != DATA_FORMAT:
        raise ValueError('Supply the original synthetic training dataset')
    reference = json.loads((bank / 'bank.json').read_text())
    if reference['format'] != BANK_FORMAT or file_digest(bank / 'bank.json') != baseline['bank_sha256']:
        raise ValueError('The upright bank differs from the synthetic source')
    sprites = []
    for row in reference['templates']:
        path = bank / row['image']
        if path.parent != bank or file_digest(path) != row['sha256']:
            raise ValueError('An upright sprite changed after bank creation')
        with Image.open(path) as image:
            sprites.append(image.convert('RGBA'))
    positives, negatives, source_identities = [], [], {}
    source_archives = {baseline['source_session_id']: baseline['source_hashes']['frames.vrec']}
    for directory in calibrations:
        metadata = json.loads((directory / 'dataset.json').read_text())
        if (metadata.get('format') != 'vimm8bdm-visible-coco-v1' or metadata.get('split') != 'calibration'
                or metadata.get('reserved_for_validation', False)):
            raise ValueError('Reserved validation images cannot enter experimental training')
        if metadata['source_session_id'] != permission['source_session_id']:
            raise ValueError('The image training review names a different recording')
        expected = permission['exports'][directory.name]
        if (file_digest(directory / 'dataset.json') != expected['dataset_sha256']
                or file_digest(directory / 'annotations.coco.json') != expected['annotations_sha256']):
            raise ValueError('The image export differs from its explicit training review')
        source = (directory / metadata['source_directory']).resolve()
        if source_identity(source) != (metadata['source_session_id'], metadata['source_hashes']):
            raise ValueError('A real source recording changed after image review')
        source_identities[source] = (metadata['source_session_id'], metadata['source_hashes'])
        source_archives[metadata['source_session_id']] = metadata['source_hashes']['frames.vrec']
        coco = json.loads((directory / 'annotations.coco.json').read_text())
        boxes = {}
        for row in coco['annotations']:
            if row['category_id'] != 1:
                raise ValueError('Experimental transfer supports one visible-body category')
            boxes.setdefault(row['image_id'], []).append(row['bbox'])
        for row in coco['images']:
            path = (directory / row['file_name']).resolve()
            if directory.resolve() not in path.parents or file_digest(path) != metadata['image_sha256'][str(row['id'])]:
                raise ValueError('A real image changed after annotation review')
            record = dict(directory=directory, image=row, path=path, boxes=boxes.get(row['id'], []))
            if len(record['boxes']) > 1:
                raise ValueError('This first real-image transfer requires at most one body per image')
            (positives if record['boxes'] else negatives).append(record)
    keys = [(record['directory'].name, record['image']['id']) for record in positives + negatives]
    frame_keys = [record['image']['id'] for record in positives + negatives]
    if len(keys) != len(set(keys)) or len(frame_keys) != len(set(frame_keys)):
        raise ValueError('Do not repeat the same reviewed source frame across training exports')
    if not positives or not negatives:
        raise ValueError('The image training review requires both body and empty examples')
    output.mkdir(parents=True, exist_ok=False)
    hashes, origins, counts = {}, [], {}
    rng = random.Random(19)
    for split in ('train', 'validation'):
        if file_digest(synthetic / f'{split}.coco.json') != baseline['annotation_sha256'][split]:
            raise ValueError('The original synthetic annotations changed')
        coco = json.loads((synthetic / f'{split}.coco.json').read_text())
        by_id = {row['image_id']: row['bbox'] for row in coco['annotations']}
        images, annotations = [], []
        (output / split).mkdir()

        def emit(image, box, origin):
            number = len(images)
            filename = f'{split}/{number:06d}.png'
            image.save(output / filename)
            hashes[filename] = file_digest(output / filename)
            images.append(dict(id=number, file_name=filename, width=image.width, height=image.height))
            annotations.append(dict(id=number+1,image_id=number,category_id=1,bbox=box,
                                    area=box[2]*box[3],iscrowd=0))
            origins.append(dict(split=split,image=number,**origin))

        for row in coco['images']:
            image = checked_image(synthetic, row, baseline['image_sha256'])
            emit(image, by_id[row['id']], dict(kind='existing_synthetic', source_image=row['id']))
        if split == 'train':
            for record in positives:
                with Image.open(record['path']) as source:
                    original = source.convert('RGB').resize((320,240),Image.Resampling.BILINEAR)
                x,y,width,height = record['boxes'][0]
                box = [x*320/record['image']['width'],y*240/record['image']['height'],
                       width*320/record['image']['width'],height*240/record['image']['height']]
                for repeat in range(positive_repeats):
                    image = original.copy()
                    if repeat:
                        image = ImageEnhance.Color(image).enhance(rng.uniform(.9,1.1))
                        image = ImageEnhance.Contrast(image).enhance(rng.uniform(.9,1.1))
                        image = ImageEnhance.Brightness(image).enhance(rng.uniform(.9,1.1))
                    emit(image,box,dict(kind='reviewed_real_positive',source_frame=record['image']['id'],repeat=repeat))
            for record in negatives:
                with Image.open(record['path']) as source:
                    background = source.convert('RGB')
                for repeat in range(negative_repeats):
                    for attempt in range(64):
                        image,box = composite(background,rng.choice(sprites),rng)
                        # Preserve the original upper-left HUD as background for hard-negative selection.
                        if box[0] >= 45 and box[1] >= 10:
                            break
                    else:
                        raise ValueError('No visible body placement preserved the HUD region')
                    emit(image,box,dict(kind='sprite_on_reviewed_empty_real_image',source_frame=record['image']['id'],repeat=repeat))
        result = dict(images=images,annotations=annotations,categories=[dict(id=1,name='cutman_body_candidate')])
        (output / f'{split}.coco.json').write_text(json.dumps(result,indent=2))
        counts[split] = len(images)
    for source,identity in source_identities.items():
        if source_identity(source) != identity:
            raise ValueError('A real recording changed during mixed image generation')
    metadata = dict(format=FORMAT,purpose='experimental_body_image_training',image_gradient_training_allowed=True,
                    reserved_for_validation=False,action_training_eligible=False,human_box_reviewed=False,
                    source_session_id=permission['source_session_id'],training_source_session_ids=sorted(source_archives),
                    training_source_archives=source_archives,counts=counts,image_sha256=hashes,
                    annotation_sha256={split:file_digest(output/f'{split}.coco.json') for split in counts},
                    train_background_frames=baseline['train_background_frames'],
                    validation_background_frames=baseline['validation_background_frames'],
                    synthetic_validation_only=True,independent_validation=False,
                    explicit_image_review_sha256=file_digest(review_file),seed=19,
                    real_positive_frames=len(positives),real_negative_frames=len(negatives),
                    positive_repeats=positive_repeats,negative_repeats=negative_repeats,
                    source_files_changed=False,input_sent=False,origins=origins)
    (output/'dataset.json').write_text(json.dumps(metadata,indent=2))
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--synthetic',type=Path,required=True)
    parser.add_argument('--calibrations',type=Path,nargs='+',required=True)
    parser.add_argument('--bank',type=Path,required=True)
    parser.add_argument('--review',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    result = build(args.synthetic,args.calibrations,args.bank,args.review,args.output)
    print(json.dumps(dict(counts=result['counts'],real_positive_frames=result['real_positive_frames'],real_negative_frames=result['real_negative_frames'])))
