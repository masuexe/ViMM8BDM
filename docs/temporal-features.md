# Temporal image features

The first two offline experiments produced almost constant actions.
This comparison tests whether an explicit representation of image changes helps the same network.
It uses the existing directed recordings without further collection or game input.

## Fixed protocol

All variants use 1879 filtered training candidates and 1796 validation candidates.
The source identities, sample hashes, mouse normalization, and parameter counts match.
The network contains 114710 parameters in every variant.
The targets, loss, optimizer, seed, batch size, and checkpoint-selection rule remain unchanged.
Each new variant uses 12 epochs, seed 0, batches of 64, and Adam at a learning rate of 0.001.
The previous raw-image checkpoint supplies the control result.

The four input images remain chronological and causal.
Each pixel initially has a value from zero through one.

| Feature mode | Four input channels |
| --- | --- |
| `raw` | The four original grayscale images |
| `centered` | Each original image multiplied by two, minus one |
| `motion` | The centered latest image, followed by three signed adjacent differences multiplied by two |

The signed differences follow the original chronological order.
They add no future information or engine observations.
They preserve the information needed to reconstruct the four images, before floating-point rounding.
The experiment changes the input representation rather than adding a larger network.

## Results

Lower values indicate smaller errors.
Mouse errors use raw counts per 50-millisecond action window.

| Predictor | Key cross-entropy | Horizontal mouse MAE | Vertical mouse MAE | Combined score |
| --- | --- | --- | --- | --- |
| Constant training median mouse | 0.3092 | 7.7233 | 0.3630 | 0.6041 |
| Raw images | 0.3085 | 8.0486 | 0.3808 | 0.6166 |
| Centered images | 0.3047 | 8.2635 | 0.3807 | 0.6177 |
| Explicit image motion | 0.2603 | 9.5630 | 0.4239 | 0.6172 |

The centered variant selected epoch 2, and the motion variant selected epoch 10.
Motion features reduced key cross-entropy by approximately 15.8 percent against the constant baseline.
Their mouse errors increased, and the combined score still failed to beat that baseline.
Do not interpret the key improvement as complete navigation ability.

At a diagnostic key threshold of 0.5, raw and centered models selected no movement keys.
The motion model selected W in 44.77 percent of validation samples.
Its forward predictions contained 514 true positives, 290 false positives, and 147 false negatives.
Forward F1 was 0.7017.
F1 combines precision and recall for the thresholded forward target.
The motion model still selected none of A, S, or D at this threshold.
These threshold checks do not define a live action decoder.

## Remove image correspondence and history

The comparison includes two distinct checks:

1. Shuffle complete validation image histories relative to the labels.
2. Repeat the latest image four times, preserving the current view but removing historical changes.

The second check occurs before the feature transformation.
For the motion variant, this produces zero difference channels.

| Motion-model input | Key cross-entropy | Combined score |
| --- | --- | --- |
| Original history | 0.2603 | 0.6172 |
| Shuffled history correspondence | 0.4109 | 0.7696 |
| Latest image repeated | 0.3641 | 0.7129 |

These changes support a dependency on corresponding image history for key prediction.
They do not establish route planning or when the AI should start moving.
The experiments reuse the development validation session for model comparisons.
They provide no independent final test or unseen-map result.
Full-session operator confirmation for the validation recording remains pending.

## Past-input reference and steering errors

A separate reference uses the human's recorded key state immediately before each decision anchor.
It estimates each future key fraction conditional on that prior state using only training labels.
It uses the training median for mouse output.
This reference achieved key cross-entropy of approximately 0.0346 and a combined score of 0.3295.
The visual models receive no such key-state input.

This reference uses actual human history throughout evaluation.
It does not use future inputs or hidden engine state.
It also does not represent an autonomous policy that must live with its own previous actions.
Its result highlights action persistence and motivates separate transition checks.
Recovering ongoing movement from pixels does not establish when the AI should start, stop, or change direction.

Horizontal mouse targets were zero in 1369 of 1796 validation windows.
The motion model produced mean absolute horizontal output of 2.0461 counts in those zero-target windows.
The raw model produced 0.4464 counts.
For nonzero horizontal targets, motion-model error was also higher: 33.6630 versus 32.4220 counts.
The new representation therefore did not solve steering in either group.

## Run and load

The offline training command now accepts `--features raw`, `--features centered`, or `--features motion`.
Use the [existing command](behavior-cloning.md#run-the-experiment) with a new output directory for each run.
Do not change the source recordings or filtered sample files.

New checkpoints use format `vimm8bdm-navigation-bc-v2` and record their feature mode.
Use `load_checkpoint` to restore the model and normalization together.
It also accepts the original version 1 raw-image checkpoints.
It rejects unsupported formats, image sizes, key orders, and invalid mouse normalization.
Strict parameter loading remains active.
Do not load motion weights into a model that assumes raw inputs.

Linux and Windows each passed 123 automated tests.
Tests check signed differences, input preservation, repeated-image behavior, and feature-aware checkpoint loading.
Both new saved checkpoints reproduced their predictions after loading.
Raw data and all weights remain private.
The [evidence file](evidence/temporal-features.json) contains the metrics and error groups.

## Next step

Keep motion features as a candidate for movement prediction.
Use the current data to evaluate key press and release transitions separately from long held intervals.
Inspect mouse turn initiation, continuation, and stopping errors separately.
Define the role of the agent's own previous actions before a state-conditioned experiment.
Do not treat a reference with actual human history as a deployment result.
Keep all current checkpoints offline.
