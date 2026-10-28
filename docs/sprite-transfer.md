# Synthetic sprite transfer

This experiment adapts the existing SSDLite320 detector with private synthetic Cutman images.
It reuses TorchVision matching, loss, optimization inputs, and detection postprocessing.
The experiment does not change the engine or game packages.
It sends no game input.

## Training source

A read-only review checked the supplied Gondola Cutman actor and the three target packages in their normal load order.
The actor uses seventeen normal-action frames: A, B, C, D, E, F, G, J, K, L, M, Q, R, S, T, V, and W.
They cover standing, movement, jumping, attacks, and missing-cutter states.
The bank excludes H and U, which the actor uses in both pain and death entries.
The extraction resolves 85 sprite lumps into 136 directional references.
Mirrored rotation pairs supply their second view through horizontal reflection.
The bank records original package and image hashes.
A second package-hash check found no source changes.

The bank applies three simple palette replacements from the inspected Cutman color source.
It preserves transparency and excludes transparent image borders.
These replacements do not reproduce all engine lighting, team colors, or rendering behavior.
The extracted images and synthetic images remain private.

The backgrounds are the same 32 sampled images reviewed for the earlier sprite-matching experiment.
The development agent identified no other player in those sampled images.
This review does not establish every image of that source recording.
Twenty-four background images supply training composites.
The other eight supply synthetic validation composites.
The two groups do not reuse the same background frames.
Both groups come from the same recording, so this is not independent session validation.

## Image generation

The generator creates 512 training images and 128 synthetic validation images at 320 by 240 pixels.
Each image contains one body with a known visible bounding box.
The generator varies direction, pose, size, brightness, interpolation, position, and minor width distortion.
It also varies image-edge clipping and rectangular occlusion.
The bottom forty pixels remain outside the body placement area.
The generator derives each box from the remaining visible alpha mask.
A sampled image review checked the pasted bodies and box overlays before training.

These composites can place bodies at unrealistic map positions or in front of the visible weapon.
They do not reproduce game geometry, motion, projectile effects, or complete depth occlusion.
The known box labels describe image composition only.
They do not establish enemy identity or a living actor.

The generator records image hashes, annotation hashes, sprite-bank identity, background source hashes, and generation plans.
It checks the original recording before generation and its hashes after generation.
Seed 7 determines image generation.
The dataset loader checks annotation and image hashes before use.
It rejects invalid body boxes and unsupported dataset formats.
The recorded real-body calibration dataset cannot enter this training loader.

## Training protocol

The model starts from the verified official SSDLite320 COCO V1 checkpoint.
All model parameters remain trainable.
The existing category 1 output serves as the experimental body-candidate output after adaptation.
It no longer establishes general COCO person semantics.

Training uses eight fixed epochs, batches of four images, four CPU threads, and random seed zero.
SGD uses a learning rate of 0.002, momentum of 0.9, and weight decay of 0.0005.
Gradient clipping uses a maximum norm of ten.
The program rejects nonfinite loss or gradient values.
The final epoch supplies the checkpoint.
Neither real calibration scores nor synthetic validation scores select an earlier checkpoint.

The pinned [SSD loss implementation](https://github.com/pytorch/vision/blob/v0.29.1/torchvision/models/detection/ssd.py) selects background examples in proportion to matched foreground anchors.
An entirely empty image therefore produces no classification or regression signal in this path.
Each training composite contains a body so the existing loss can select background examples from that image.
A unit test checks this behavior of the reused library.
Unmodified negative real images remain necessary for false-candidate evaluation.

The real combat calibration recording supplies no training image, background, or gradient target.
The evaluator rejects a checkpoint whose training-background recording identity matches the real calibration recording.
The twenty reviewed real images remain a calibration diagnostic because their content informed the experiment design.
The operator has not reviewed their candidate box coordinates.

## Results

The eight-epoch run completed 1,024 optimizer steps in 298.98 seconds on the CPU.
Mean training loss decreased from 2.4481 in the first epoch to 0.5503 in the final epoch.
The final checkpoint located 94 of 128 synthetic validation bodies at the fixed thresholds.
It produced no additional unmatched predictions in those positive synthetic images.
This result does not measure false candidates in entirely empty images.

The same final checkpoint then ran on the twenty real calibration images.
The score threshold remained 0.5, and the minimum box intersection-over-union remained 0.5.

| Measurement | Linux CPU | Native Windows CPU |
| --- | --- | --- |
| Located positive images | 6 of 9 | 6 of 9 |
| Negative images with candidates | 0 of 11 | 0 of 11 |
| Median processing | 30.05 ms | 32.56 ms |
| 95th percentile processing | 41.90 ms | 36.13 ms |
| Maximum processing | 42.82 ms | 43.66 ms |

The unchanged COCO model located zero of nine positive images in the earlier check.
Synthetic transfer improved location quality on these calibration images.
However, recall of 0.6667 remains below the required 0.8.
The model passed the preliminary negative-image and median-processing conditions but failed the complete set.
The experiment does not establish an independent real-session result.
These runs also do not isolate operating-system effects on processing time.

An overlay review checked all nine positive predictions or misses.
The three missed images contain smaller bodies, with scene tint, effects, or combat text in some views.
This review does not isolate the cause of each miss.
The model assigns higher scores to some negative wall details than to some missed bodies.
One negative image reaches approximately 0.453, while missed bodies reach approximately 0.274 to 0.393.
Lowering the score threshold alone therefore cannot retain every missed body while rejecting those negative candidates.
The fixed threshold remained unchanged.

The [aggregate evidence](evidence/sprite-transfer.json) records the protocol, epoch results, checkpoint hash, and real-image results.
Five additional tests check pose coverage, mirrored views, visible mask boxes, reproducible image generation, source integrity, changed data, checkpoint identity, and empty-image loss.
Linux and Windows each passed all 208 tests without skips.
The original recordings and all three original package hashes remain unchanged.

The model remains an offline body-candidate detector.
Automatic aim and attack remain disabled.
The next experiment must improve the smaller-body and background separation before an independent live check.

## Reproduce

Use the tested optional detector dependencies from the [pretrained check](pretrained-detection.md).
Build a new private upright bank:

```bash
python -m vimm8bdm.sprite_synthesis bank \
  --packages assets/MM8BDM-v6a.pk3 assets/classes-v9fh.pk3 assets/gondolapatchv2.7h2.pk3 \
  --output .local/perception/new-cutman-upright
```

The asset directory is a local placeholder.
For image generation, supply the exact frame identifiers from a completed visual background review.
The generator does not automatically establish that a background contains no opponent.

```bash
python -m vimm8bdm.sprite_synthesis dataset \
  --bank .local/perception/new-cutman-upright \
  --recording data/wall-recovery-demo-1 \
  --frames 0 76 153 230 307 384 461 538 \
  --output data/new-cutman-synthetic
python -m vimm8bdm.sprite_transfer \
  --dataset data/new-cutman-synthetic \
  --initial .local/perception/torchvision/ssdlite320_mobilenet_v3_large_coco-a79551df.pth \
  --output runs/new-sprite-transfer
python scripts/evaluate_pretrained_detection.py \
  --dataset data/cutman-body-calibration-1 \
  --checkpoint runs/new-sprite-transfer/model.pt --transfer \
  --output runs/new-transfer-calibration
```

The short frame list illustrates the interface and does not reproduce the 32-background experiment.
All output directories must be new.
Keep model weights, sprite art, screenshots, complete labels, and local generation plans outside the public repository.
