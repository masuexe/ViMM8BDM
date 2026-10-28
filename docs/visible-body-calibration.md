# Visible Cutman body calibration

The first real combat recording contains visible Cutman bodies, projectiles, deaths, and respawns.
The operator reported Cutman and interruptions.
The report does not identify all interruption times or establish a constant player class throughout the recording.
The source remains outside action training.

## Recording checks

The recording contains 1,193 images and 5,116 input events in one segment.
The archive check passed with zero dropped items.
Median image capture took 7.15 milliseconds.
The largest interval between image starts was 51.47 milliseconds.
The recorder sent no game input.

A development agent inspected 24 sampled images and candidate box overlays.
The review accepted 20 images for calibration: nine positive images and eleven negative images.
Each positive image has one box around the visible Cutman body.
The review excluded four images with unclear body extent, death fragments, or a horizontal death sprite.
The operator has not reviewed these box coordinates.
The boxes are candidate annotations, not an independent test standard.

A negative label means that the reviewer identified no clear Cutman body in that image.
It does not establish that no opponent exists elsewhere in the scene.
One image contains an upright opponent while the local player is dead.
Such an image can support appearance analysis but cannot establish a valid player action.

The HUD can contain the Cutman name without a clearly visible body.
Do not use this text alone as a body location or attack command.
The initial labels also exclude death sprites and projectile shapes.
They do not establish team membership or whether a visible actor is alive.

## Export

The exporter copies original PNG bytes into a private COCO dataset.
It records source hashes, image hashes, review identity, occlusion, and image-edge clipping.
It checks source integrity before export and source hashes after export.
It rejects an existing output directory, a different source identity, duplicate image identifiers, and boxes outside the image.
The export always records `training_ready: false` and `action_training_eligible: false`.

```bash
python -m vimm8bdm.visual_labels \
  --recording data/cutman-visual-demo-1 \
  --review runs/cutman-visual-demo-1/body-review.local.json \
  --output data/new-cutman-calibration
```

Keep the review, PNG images, and complete dataset private.
The public repository contains code and aggregate evidence only.

## Existing matcher results

The existing matcher retained its correlation threshold of 0.97.
A successful location also requires box intersection-over-union of at least 0.5.
Both mask variants failed to locate all nine positive images.
Neither variant produced a candidate in the eleven negative images.

| Mask | Located positive images | Negative candidates | Linux median processing ms |
| --- | --- | --- | --- |
| Neutral | 0 of 9 | 0 of 11 | 144.84 |
| Opaque | 0 of 9 | 0 of 11 | 135.37 |

This result is a calibration diagnostic from one short recording.
It does not measure general detection quality or establish performance on an independent session.
Close bodies, image-edge clipping, effects, scale changes, and different poses exceed the initial standing bank coverage.
The result does not isolate the contribution of each cause.
The matcher remains outside the live controller.

```bash
python scripts/evaluate_visible_boxes.py \
  --dataset data/new-cutman-calibration \
  --bank .local/perception/cutman-standing-1 \
  --output runs/new-visible-box-evaluation.local.json
```

The [aggregate evidence](evidence/visible-body-calibration.json) records the result.
Five additional tests check export provenance, immutable source files, annotation bounds, source identity, and exclusion rules.
Linux and Windows each passed all 199 tests without skips.

## Fixed color experiment

A second offline experiment tested fixed color proposals on the same reviewed images.
It used RGB ratios, a five-pixel closing kernel, connected components, and fixed box expansion.
The parameters remained unchanged after the first result.
The method covered seven of nine annotated bodies at box intersection-over-union of at least 0.5.
However, all eleven negative images produced proposals.
The twenty images produced 892 regions in total.
Median processing took 2.71 milliseconds on Linux.
Fast processing did not establish useful body identification.
The method remains an offline diagnostic.

```bash
python scripts/evaluate_color_proposals.py \
  --dataset data/new-cutman-calibration \
  --output runs/new-color-evaluation.local.json
```

An initial report write failed because a NumPy integer did not support JSON serialization.
The script now converts component coordinates to Python integers.
A new output file contains the complete result.

## Next experiment

Evaluate a detector that uses body shape and scene context before another live experiment.
Treat a proposal as a possible body region only.
A color match cannot establish an enemy, a living actor, or permission to attack.
Use a separate recording for any later independent quality test.
Do not split nearby frames from this recording across training and test sets.

The [official SSDLite transfer check](pretrained-detection.md) completed the next pretrained-model experiment.
The unchanged COCO model did not identify the reviewed bodies.
