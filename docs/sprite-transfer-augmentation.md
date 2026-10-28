# Sprite transfer with appearance changes

This experiment tests a second synthetic training profile after the first model missed three reviewed real bodies.
The first model also assigned appreciable scores to some negative wall details.
The experiment retains the existing SSDLite320 architecture and detection thresholds.
It sends no game input.

## Fixed protocol

The new `combat` generation profile creates 1,024 training images and 256 synthetic validation images.
It uses the same upright sprite bank and reviewed background-frame split as the first experiment.
Generation seed seven determines image content.
Seventy percent of the requested sprite heights use a logarithmic distribution from approximately 20 to 64 pixels.
The remaining heights use a logarithmic distribution from 64 to approximately 201 pixels.
Image-edge clipping can reduce the final visible height.
The first profile used approximately 12 to 201 pixels without this mixture.

The new profile changes complete-image color saturation, contrast, and brightness.
Half the generated images also receive a red, yellow, or white overlay with varying strength.
These operations change RGB appearance without changing image coordinates.
They do not emulate the exact engine palette or damage effects.
The original placement, partial occlusion, clipping, and interpolation variations remain available.
The `baseline` profile remains the default for existing commands.

The first training set used 133 of the 136 bank views.
The second set uses all 136 views.
Median visible body height is 46 pixels in the first set and 44 pixels in the second set.
The second set reduces the proportion of extremely small and extremely large bodies.

Both runs start from the same verified official COCO checkpoint.
The optimizer, learning rate, batch size, CPU thread count, random seed, and eight-epoch duration remain unchanged.
The second run therefore makes 2,048 optimizer steps, twice the first run's count.
The final epoch supplies the checkpoint.
The real calibration images supply no background or gradient target.

The experiment changes sample count, size distribution, and appearance together.
Its results cannot isolate the contribution of each change.
The score threshold remains 0.5, and the minimum box intersection-over-union remains 0.5.
The existing recall, false-candidate, and processing conditions remain unchanged.

## Additional real-image review

During training, the development agent reviewed 23 additional samples from the same real combat recording.
The new sample indices lie between the original sampled indices.
The review preceded model predictions on these additional images.
Four images contain one clear body each.
Thirteen images contain no clear body.
The review excluded six images with ambiguous body extent, projectile occlusion, or death fragments.
The agent checked box overlays for all four positive images.
The operator has not reviewed these box coordinates.

The new private COCO export contains seventeen images and four candidate body boxes.
The original real recording and initial calibration export remain unchanged.
These are additional calibration samples from the same recording.
They do not establish independent session validation.

The first model located three of four additional bodies and produced a candidate in one of thirteen additional negative images.
The false candidate overlaps the local health meter during a view dominated by an explosion and death text.
The extended review therefore identified a false candidate absent from the original eleven negative samples.
The original and additional sample groups must both remain in the comparison.
The first-model additional check ran while training was active.
Its timing does not serve as an uncontended processing benchmark.

## Results and decision

The second run completed 2,048 optimizer steps in 518.16 seconds on the Linux CPU.
The final epoch located 174 of 256 synthetic validation bodies and produced one additional unmatched prediction.
The first and second synthetic validation sets differ in content and count.
Their recall values do not form a controlled comparison.

| Model | Original group locations | Additional group locations | Combined locations | Combined negative images with candidates |
| --- | --- | --- | --- | --- |
| First profile | 6 of 9 | 3 of 4 | 9 of 13 | 1 of 24 |
| Appearance profile | 8 of 9 | 1 of 4 | 9 of 13 | 3 of 24 |

The second model passed the preliminary conditions on the original twenty images.
It failed on the additional seventeen images.
The combined recall is 0.6923 for both models.
The second model increases negative-image candidates.
The experiment therefore rejects it as a replacement for the first offline checkpoint.
Neither model passed the expanded calibration conditions.
Automatic aim and attack remain disabled.

Native Windows reproduced both location counts and negative-image candidate counts for the second model.
Median processing took 34.47 milliseconds on the original group and 36.86 milliseconds on the additional group.
Linux medians were 25.60 and 27.45 milliseconds respectively.
These measurements include conversion, inference, postprocessing, and filtering after five warmup calls.
They exclude capture, disk decoding, model loading, and game input.
They do not isolate operating-system effects.

Linux and Windows each passed all 210 tests without skips.
The added tests check deterministic appearance changes, unchanged object geometry, and unsupported profiles.
The original recordings and all first-profile synthetic images retain their saved hashes.
The [aggregate evidence](evidence/sprite-transfer-augmentation.json) records both groups and the complete decision.

## Next data requirement

Synthetic adaptation supplies a useful starting point but has not established reliable real-image recognition.
Use the first combat recording for candidate real-image training after an explicit image-label review.
Keep action training separate because the recording includes death and respawn transitions.
Collect a new complete recording for validation before adapting the detector to real images.
Do not divide nearby frames from the first recording between training and independent validation.
Keep the new validation recording outside gradient training.
A new recording of the same map checks a new session, not an unseen map.

## Reproduce

Use the same reviewed frame list and private bank as the first [sprite transfer](sprite-transfer.md).
Select the new profile and image counts:

```bash
python -m vimm8bdm.sprite_synthesis dataset \
  --bank .local/perception/cutman-upright-1 \
  --recording data/wall-recovery-demo-1 \
  --frames 0 76 153 230 307 384 461 538 \
  --profile combat --train-count 1024 --validation-count 256 \
  --output data/new-combat-synthetic
python -m vimm8bdm.sprite_transfer \
  --dataset data/new-combat-synthetic \
  --initial .local/perception/torchvision/ssdlite320_mobilenet_v3_large_coco-a79551df.pth \
  --output runs/new-combat-transfer
```

The short frame list illustrates the interface and does not reproduce the complete background selection.
All output directories must be new.
Keep game art, model weights, screenshots, labels, and complete local reports private.

The [separate-session intake](visual-validation-session.md) completed the new recording requirement.
The recording remains outside gradient training.
The later [real-image transfer](real-body-transfer.md) completed the first fixed-checkpoint validation on its reviewed images.
