# Conditional local image enlargement

This experiment retains the expanded-image checkpoint and all model weights.
It changes inference only.
The detector remains outside automatic aim and attack.

## Method

The first model call processes the complete image.
An accepted body requires a score of at least 0.5.
If a body passes, the detector retains the complete-image result and makes no additional call.
Otherwise, it selects the highest body proposal with a score of at least 0.2.
If no proposal passes this lower trigger threshold, processing ends without a candidate.

A weak proposal starts one additional model call on a square image region.
The square has sides of 320 source pixels, or the smaller image dimension for smaller images.
Its center follows the proposal center.
The square stays within the source image.
The detector does not mask the HUD or remove any region from the first call.

The second call requires the same acceptance score of 0.5.
The lower trigger threshold does not accept a body by itself.
The detector translates accepted crop coordinates into complete-image coordinates.
It does not change box widths or heights during this translation.
The model already returns boxes in its input-image coordinates.
The procedure permits at most one additional call per image.

The [implementation](../src/vimm8bdm/local_zoom.py) includes separate crop and coordinate functions.
Unit tests check image boundaries, trigger thresholds, smaller images, and coordinate translation.

## Evaluation scope

The private initial protocol precedes the repeated validation call.
The first check uses all 107 accepted images from the training recording.
These images supplied training and cannot establish generalization.
The result retains the same 42 of 44 locations and zero candidates in 63 negative images.
Six images require an additional call.

The second check uses the same seventeen reviewed images from the reserved recording.
These images remain outside gradient training.
Earlier results already informed development, so this check uses `--validation-reuse`.
The report records `independent_test: false`.
It does not constitute a new blind test or an unseen-map test.
The score threshold remains 0.5, and the minimum box overlap remains 0.5.

The enlarged view recovers the HUD-adjacent body in frame 259.
Its box overlap reaches 0.7692.
The small partly hidden body in frame 362 remains undetected.
Two negative images also start an additional call but produce no accepted body.

## Result

| Measurement | Linux CPU | Native Windows CPU |
| --- | --- | --- |
| Located positive images | 8 of 9 | 8 of 9 |
| Negative images with candidates | 0 of 8 | 0 of 8 |
| Additional model calls | 4 | 4 |
| Median processing | 23.66 ms | 38.02 ms |
| Maximum processing | 57.53 ms | 106.58 ms |

Each run produces eight accepted boxes in total.
Both runs use four CPU threads and five warmup calls.
Processing includes tensor conversion, conditional inference, filtering, and coordinate translation.
It excludes image capture, disk decoding, model loading, and input execution.
The small sample makes its 95th percentile equal to its maximum.

The three existing sample conditions pass: recall of at least 0.8, no candidates in negative images, and median processing below 50 milliseconds.
These conditions do not limit the slowest frame.
The Windows maximum exceeds 100 milliseconds and requires attention before a control-loop test.
A passing repeated sample does not prove general detection quality or acceptable reaction time.

The next step is an observation-only test on continuous images.
That test must retain normal human control and send no game input.
It must measure capture and inference together, candidate continuity, and stale-frame behavior.
Automatic aim and attack remain disabled.
A later independent recording remains necessary for a broader quality claim.
The [aggregate evidence](evidence/local-zoom.json) records both platforms and the initial training-image check.

Linux and native Windows each passed all 226 tests without skips.
The public-file check excludes private images, weights, and recording archives.
