"""Check sprite precedence, mirrored rotations, and finite matching scores."""
import io
from pathlib import Path
import tempfile
import unittest
import zipfile

try:
    import numpy as np
    from PIL import Image
    from vimm8bdm.sprite_perception import SpriteMatcher, build_bank, template_mask
except ImportError:
    np = None


def png(marker):
    pixels = np.zeros((8, 8, 4), dtype=np.uint8)
    pixels[1:7, 1:7] = [0, 0, 0, 255]
    pixels[2:6, 2:6] = [255, 255, 255, 255]
    pixels[2, 2] = [marker, 30, 90, 255]
    stream = io.BytesIO()
    Image.fromarray(pixels).save(stream, format='PNG')
    return stream.getvalue()


@unittest.skipIf(np is None, 'Install the optional vision packages')
class SpritePerceptionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.base, self.patch = self.root / 'base.pk3', self.root / 'patch.pk3'
        with zipfile.ZipFile(self.base, 'w') as archive:
            for name in ('CUTMA1', 'CUTMA5', 'CUTMA6A4', 'CUTMA7A3', 'CUTMA8A2'):
                archive.writestr('sprites/body/' + name + '.png', png(120))
        with zipfile.ZipFile(self.patch, 'w') as archive:
            archive.writestr('Sprites/Other/CUTMA1.PNG', png(230))
        self.bank = self.root / 'bank'

    def test_later_sprite_wins_without_changing_packages(self):
        original = self.base.read_bytes(), self.patch.read_bytes()
        manifest = build_bank([self.base, self.patch], self.bank)
        self.assertEqual(len(manifest['templates']), 8)
        self.assertEqual(manifest['templates'][0]['source_package'], 'patch.pk3')
        with Image.open(self.bank / 'rotation-1.png') as image:
            self.assertEqual(image.getpixel((1, 1)), (230, 30, 90, 255))
        self.assertEqual(original, (self.base.read_bytes(), self.patch.read_bytes()))
        with self.assertRaises(FileExistsError):
            build_bank([self.base, self.patch], self.bank)

    def test_second_rotation_uses_horizontal_mirror(self):
        build_bank([self.base], self.bank)
        with Image.open(self.bank / 'rotation-6.png') as a, Image.open(self.bank / 'rotation-4.png') as b:
            self.assertTrue(np.array_equal(np.asarray(a)[:, ::-1], np.asarray(b)))

    def test_neutral_mask_ignores_colored_and_transparent_pixels(self):
        rgba = np.array([[[0, 0, 0, 255], [255, 255, 255, 255], [0, 120, 252, 255],
                          [0, 248, 252, 255], [255, 255, 255, 0]]], dtype=np.uint8)
        self.assertEqual(template_mask(rgba, 'neutral')[0, :, 0].tolist(), [255, 255, 0, 0, 0])

    def test_matching_finds_a_known_box_and_rejects_zero_energy_background(self):
        build_bank([self.base], self.bank)
        matcher = SpriteMatcher(self.bank, heights=(24,), mask='opaque', threshold=.99)
        scene = Image.new('RGB', (320, 240), (70, 90, 110))
        with Image.open(self.bank / 'rotation-1.png') as image:
            image = image.resize((24, 24), Image.Resampling.NEAREST)
            scene.paste(image, (140, 55), image)
        match = matcher.best(scene)
        self.assertTrue(match['candidate'])
        self.assertEqual(match['box'], [140, 55, 24, 24])
        empty = matcher.best(Image.new('RGB', (320, 240)))
        self.assertTrue(np.isfinite(empty['score']))
        self.assertFalse(empty['candidate'])
        self.assertFalse(match['enemy_identification_verified'])
        self.assertFalse(match['input_sent'])

    def test_modified_template_fails_its_bank_hash(self):
        build_bank([self.base], self.bank)
        (self.bank / 'rotation-1.png').write_bytes(png(10))
        with self.assertRaisesRegex(ValueError, 'bank record'):
            SpriteMatcher(self.bank)


if __name__ == '__main__':
    unittest.main()
