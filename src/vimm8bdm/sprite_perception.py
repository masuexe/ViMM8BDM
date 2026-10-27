"""Build private sprite references and locate one visual candidate without input."""
import argparse
import io
import json
from pathlib import Path
import re
import zipfile

import cv2
import numpy as np
from PIL import Image, ImageOps

from .recording import file_digest

FORMAT = 'vimm8bdm-cutman-template-bank-v1'


def build_bank(packages, output):
    """Resolve Cut Man standing sprite lumps in the supplied original load order."""
    if output.exists():
        raise FileExistsError('The template directory already exists')
    resolved, sources = {}, []
    for package in packages:
        digest = file_digest(package)
        sources.append({'name': package.name, 'sha256': digest})
        with zipfile.ZipFile(package) as archive:
            for entry in archive.namelist():
                normalized = entry.replace('\\', '/')
                if not normalized.lower().startswith('sprites/'):
                    continue
                name = normalized.rsplit('/', 1)[-1].upper()
                match = re.fullmatch(r'CUTMA([1-8])(?:A([1-8]))?\.PNG', name)
                if match:
                    # Sprite names are global within their namespace, regardless of directory case.
                    resolved[name[:-4]] = (archive.read(entry), package.name, entry, match.groups())
    rotations = {}
    for lump, (raw, package, entry, turns) in resolved.items():
        with Image.open(io.BytesIO(raw)) as source:
            image = source.convert('RGBA')
        rotations[int(turns[0])] = (image.copy(), package, entry, False)
        if turns[1]:
            rotations[int(turns[1])] = (ImageOps.mirror(image), package, entry, True)
    if set(rotations) != set(range(1, 9)):
        raise ValueError('The original packages do not supply all eight Cut Man standing rotations')
    output.mkdir(parents=True, exist_ok=False)
    templates = []
    for rotation, (image, package, entry, mirrored) in sorted(rotations.items()):
        bounds = image.getchannel('A').getbbox()
        if bounds is None:
            raise ValueError('A sprite reference has no opaque pixels')
        image = image.crop(bounds)
        filename = f'rotation-{rotation}.png'
        image.save(output / filename)
        templates.append({'rotation': rotation, 'image': filename, 'sha256': file_digest(output / filename),
                          'source_package': package, 'source_member': entry, 'mirrored': mirrored,
                          'width': image.width, 'height': image.height})
    manifest = {'format': FORMAT, 'actor': 'cutman', 'sprite_prefix': 'CUTM', 'pose': 'A',
                'sources': sources, 'templates': templates, 'source_files_changed': False,
                'actual_game_render_verified': False, 'input_sent': False}
    with (output / 'bank.json').open('x') as stream:
        json.dump(manifest, stream, indent=2)
    return manifest


def template_mask(rgba, mode):
    opaque = rgba[:, :, 3] >= 128
    if mode == 'neutral':
        rgb = rgba[:, :, :3].astype(np.int16)
        opaque &= rgb.max(2) - rgb.min(2) <= 8
    elif mode != 'opaque':
        raise ValueError('The sprite mask mode is unsupported')
    return np.repeat((opaque.astype(np.uint8) * 255)[:, :, None], 3, axis=2)


class SpriteMatcher:
    """Return a single sprite-like candidate. Do not identify an enemy or send actions."""

    def __init__(self, bank, *, heights=(12, 16, 24, 32, 48), mask='neutral', threshold=.97):
        self.bank = Path(bank)
        metadata = json.loads((self.bank / 'bank.json').read_text())
        if metadata.get('format') != FORMAT:
            raise ValueError('The sprite bank format is unsupported')
        if not heights or any(type(h) is not int or not 8 <= h <= 96 for h in heights):
            raise ValueError('Sprite heights must be integers from eight through 96')
        if not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError('The correlation threshold must be finite and between zero and one')
        self.threshold, self.templates = threshold, []
        for reference in metadata['templates']:
            path = self.bank / reference['image']
            if path.parent != self.bank or file_digest(path) != reference['sha256']:
                raise ValueError('The sprite image does not match its bank record')
            with Image.open(path) as image:
                rgba = image.convert('RGBA')
            for height in heights:
                width = max(1, round(rgba.width * height / rgba.height))
                pixels = np.asarray(rgba.resize((width, height), Image.Resampling.NEAREST))
                selected = template_mask(pixels, mask)
                if np.count_nonzero(selected[:, :, 0]) < 12:
                    continue
                self.templates.append((np.ascontiguousarray(pixels[:, :, :3]), selected,
                                       reference['rotation'], height))
        if not self.templates:
            raise ValueError('The sprite bank supplies no usable template masks')

    def best(self, image):
        """Search the upper central scene and return an uncalibrated correlation score."""
        original_width, original_height = image.size
        frame = np.asarray(image.convert('RGB').resize((320, 240), Image.Resampling.BILINEAR))
        x0, y0, x1, y1 = 38, 12, 282, 144
        scene = np.ascontiguousarray(frame[y0:y1, x0:x1])
        winner = None
        for template, mask, rotation, height in self.templates:
            response = cv2.matchTemplate(scene, template, cv2.TM_CCORR_NORMED, mask=mask)
            np.nan_to_num(response, copy=False, nan=-1, posinf=-1, neginf=-1)
            _, score, _, location = cv2.minMaxLoc(response)
            score = min(score, 1.)
            if winner is None or score > winner['score']:
                left, top = location[0] + x0, location[1] + y0
                winner = {'score': score, 'rotation': rotation, 'template_height': height,
                          'box': [round(left * original_width / 320), round(top * original_height / 240),
                                  round(template.shape[1] * original_width / 320), round(height * original_height / 240)]}
        winner['candidate'] = winner['score'] >= self.threshold
        winner['kind'] = 'cutman_standing_sprite_candidate'
        winner['enemy_identification_verified'] = False
        winner['input_sent'] = False
        return winner


def main():
    parser = argparse.ArgumentParser(description='Build private Cut Man standing sprite references')
    parser.add_argument('--packages', type=Path, nargs=3, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = build_bank(args.packages, args.output)
    print(json.dumps({'templates': len(result['templates']), 'source_files_changed': False,
                      'input_sent': False}, indent=2))


if __name__ == '__main__':
    main()
