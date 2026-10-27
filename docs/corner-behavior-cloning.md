# Second offline navigation experiment

This run uses the directed corner demonstrations.
The filtered training dataset contains 1879 candidates after the reviewed notification exclusion.
The new validation dataset contains 1796 candidates from a separate recording.
The original model and all source recordings remain unchanged.

## Validation recording

The recorder saved 1789 images and 2249 events in one segment, with zero dropped items.
The archive check passed.
The active segment lasted 90.04 seconds, and the image span was 89.97 seconds.
Median capture duration was 7.35 milliseconds.
The largest capture interval was 51.17 milliseconds.

The recorded input includes 25 W press/release pairs, nine A pairs, four S pairs, and eleven D pairs.
The initial and final held-key states were empty.
The candidate labels retain all 2246 source action events.
Two incomplete final windows produced no samples.
All candidates passed causal reconstruction and timed compilation, producing 5928 packets.

Five sampled images showed gameplay without a visible overlay.
A color check for the previous notification icon found no candidates.
This heuristic is not a general notification detector.
The operator reported completion of the recording.
Full-session confirmation of the requested actions and absence of interruptions remains pending.
The experiment therefore retains candidate-data status.

The source asset digests and pointer-setting snapshot match the directed training recording.
Source-hash checks passed.
The split check found four distinct source sessions without session or identical-archive overlap across splits.
No validation labels entered training updates or normalization.
The [validation evidence](evidence/navigation-corners-validation.json) contains the aggregate checks.

## Fixed comparison

The run retained the first experiment's architecture and hyperparameters.
It used 12 epochs, seed 0, batches of 64, and a learning rate of 0.001.
Data loading, training, and final metrics took 23.55 seconds on four CPU threads.
Epoch 10 supplied the selected checkpoint.
The saved checkpoint reproduced its predictions after loading.

The table evaluates every predictor on the new validation session.
Both constant baselines use key fractions from the new training dataset.
Scores use the new training mouse scales for every row, including the previous model.
Lower values indicate smaller errors.

| Predictor | Key cross-entropy | Horizontal mouse MAE | Vertical mouse MAE | Common score |
| --- | --- | --- | --- | --- |
| Constant training median mouse | 0.3092 | 7.7233 | 0.3630 | 0.6041 |
| Constant training mean mouse | 0.3092 | 9.0560 | 0.4151 | 0.6516 |
| Previous model on new validation | 0.6222 | 8.5709 | 0.5228 | 0.9881 |
| Selected second model | 0.3085 | 8.0486 | 0.3808 | 0.6166 |
| Second model with shuffled image correspondence | 0.3110 | 8.0676 | 0.3809 | 0.6196 |

Mouse errors use raw counts per 50-millisecond action window.
The second model improves on the previous model for this validation session.
It still fails to beat the median-mouse constant on the combined score or either mouse error.
Shuffling image correspondence changes the combined score by approximately 0.48 percent.
These results do not establish useful visual navigation.

At a diagnostic threshold of 0.5, the model selects no movement key in any validation sample.
The earlier model selected forward movement in every sample of its original validation session.
Changing action frequencies alone did not resolve the tendency toward constant behavior.
The weights remain local and offline.

## Training-only diagnosis

A separate small-subset check used 32 examples from the filtered training dataset.
Sixteen examples had forward held fractions of at least 0.95.
Sixteen had zero forward held fraction.
The selection spread examples across the training recording.
The check used the same architecture, seed 123, and 200 full-batch Adam updates at a learning rate of 0.001.
It used no validation examples.

Loss decreased from 0.7537 to 0.0706.
Forward classification at threshold 0.5 improved from 50 percent to 90.625 percent on those same 32 examples.
This result establishes partial memorization of a small training subset.
It does not establish independent validation performance or a complete diagnosis of the failure.
The check took approximately 1.70 seconds and sent no game input.

Keep the current recordings before requesting more data.
Inspect temporal image features, target timing, and the held-fraction decoder next.
Compare every later change with the same constant baselines and image-correspondence check.
Do not enable live control with the current checkpoint.
The [experiment evidence](evidence/corner-behavior-cloning.json) contains the metrics and epoch history.
