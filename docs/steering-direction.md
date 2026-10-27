# Horizontal input direction experiment

This experiment separates horizontal input direction from key timing and mouse magnitude.
It uses the same checked human recordings as the combined-source regression experiment.
It sends no desktop input and does not change the visual navigation rule.
The classifier alone cannot control a player.

## Target audit

Each target describes the sign of the total horizontal raw mouse input in the next 50 milliseconds.
The classes are negative, zero, and positive.
These names specify recorded input counts, not verified camera directions.
The four causal input images still cover 150 milliseconds with 50-millisecond spacing.

| Source | Negative | Zero | Positive | Total |
| --- | --- | --- | --- | --- |
| Directed training | 132 | 1578 | 169 | 1879 |
| Wall recovery training | 455 | 1568 | 372 | 2395 |
| Separate validation | 279 | 1369 | 148 | 1796 |

Zero targets dominate all three sources.
The directed source contains one window with opposite horizontal events.
The recovery source contains four such windows, including two with exactly zero net horizontal input.
The validation source contains none.
A zero net total therefore does not always mean an absence of mouse movement inside a window.
The audit leaves these source events unchanged.

The audit also classifies earlier regression outputs with a one-count deadband.
Predictions at or below minus one count enter the negative class.
Predictions at or above one count enter the positive class.
Other predictions enter the zero class.
This fixed deadband matches the earlier mouse-activity audit.

| Regression model | Turn predictions in 1369 zero-target windows | Horizontal MAE in zero-target windows | Horizontal MAE in active-target windows |
| --- | --- | --- | --- |
| Earlier motion model | 892 | 2.0461 | 33.6630 |
| Combined-data model | 967 | 2.5990 | 33.1155 |

These false predictions can introduce steering drift if a controller executes them.
The audit does not measure a live drift trajectory.
It also does not identify which turns would avoid a wall.

## Fixed comparison

Both classifiers reuse the existing motion-image CNN and replace its final layer with three class logits.
The classifiers predict horizontal input direction only.
They predict no movement keys, mouse magnitude, vertical aim, attack, or class skill.
They read no recorded action history or hidden engine state during prediction.

Both variants use the same 4274 training samples and 1796 validation samples.
They use seed 0, 12 epochs, batches of 64, four CPU threads, and Adam with a learning rate of 0.001.
One variant uses uniform class weights.
The other uses inverse square roots of training class counts.
Weighted mean sample weight equals one, and missing classes receive zero target weight.
Validation labels supply no weights or parameter updates.

The training loss is three-class cross-entropy.
Checkpoint selection maximizes validation macro F1 across all three classes.
The earliest epoch wins a tie.
Constant training-majority and zero-output predictors provide references.
Shuffled histories and repeated latest images test image dependence.
Validation remains a reused selection set rather than an independent final test.
Its full-session operator confirmation also remains pending.

Before training, the experiment set these candidate conditions:

1. Macro F1 must be at least 0.5.
2. Recall for every class must be at least 0.4.
3. The false-turn rate on zero-target windows must not exceed 0.1.
4. Shuffling image histories must reduce macro F1 by at least 0.05.

These values are engineering screening targets, not gameplay success thresholds.
Checkpoint loading must also reproduce the logits.
Passing these conditions would still require action-magnitude, timing, and attended game checks.

## Evidence status

All three runs completed without game input.
The classifiers contain 114323 parameters.
Linux and Windows each passed 176 automated tests.
The saved checkpoints reproduced their logits after loading.
New tests cover class counts, missing classes, confusion matrices, training isolation, source boundaries, and event matching.
Model outputs do not enter the live controller.
Recordings, review files, predictions, and model weights remain private.


## Results

The zero-output baseline has macro F1 of 0.288362.
All learned variants improved this diagnostic score.
None passed every candidate condition.

| Variant | Selected epoch | Macro F1 | Negative recall | Positive recall | False-turn rate on zero targets |
| --- | --- | --- | --- | --- | --- |
| Uniform weights, 12 epochs | 12 | 0.465660 | 0.175627 | 0.256757 | 0.063550 |
| Weighted classes, 12 epochs | 10 | 0.514164 | 0.354839 | 0.344595 | 0.140248 |
| Uniform weights, 36 epochs | 14 | 0.512038 | 0.240143 | 0.331081 | 0.078890 |

The first uniform run selected its final epoch.
A separate plan therefore extended the budget to 36 epochs without changing data, thresholds, or other training options.
The extended run reproduced the first 12 epochs exactly.
Its selected epoch was 14, and later training did not produce a better macro F1.
The longer run reduced false turns from the regression model's 967 to 108 zero-target windows.
It still missed most nonzero directions.

The longer run's shuffled-history macro F1 was 0.333404.
Repeating only the latest image reduced macro F1 to the zero-output reference of 0.288362.
The image differences contain useful signal, but that signal did not establish adequate direction recall.
Class weighting increased recall and also increased false turns.
This tradeoff does not justify live deployment.

## Input activity changes

Validation contains 49 horizontal activity starts, 49 stops, and three direct direction reversals across contiguous windows.
The longer run predicted 96 starts and 96 stops.
It matched no starts and ten stops in their original windows.
These activity counts alone do not require the same direction.
The separate tolerant audit requires the same input direction and matches each event at most once.
It also prevents matching across source gaps.

| Symmetric time tolerance | Matched same-direction starts, out of 49 | Start F1 |
| --- | --- | --- |
| 0 ms | 0 | 0 |
| 50 ms | 4 | 0.0552 |
| 100 ms | 11 | 0.1517 |
| 200 ms | 15 | 0.2069 |

The wider comparisons still show low start recall and many false predictions.
The failure cannot be described only as a one-window delay.
These comparisons use 50-millisecond window anchors.
They do not measure hardware input time, visible response delay, or network latency.
They do not shift source labels or train on future images.

A reference that copies the preceding recorded human direction achieves macro F1 of 0.905850.
Its exact start recall is zero because it starts one window late by construction.
This reference reads past human input and is not an autonomous prediction sequence.
It must not substitute for the model's own previous commands during deployment.
It demonstrates temporal persistence, not a solution to turn initiation.

## Reproduce the comparison

Use a new output directory:

```bash
python -m vimm8bdm.steering \
  --train data/navigation-corners-train-clean-slices-1 data/wall-recovery-slices-1 \
  --train-review runs/navigation-corners-train-1/operator-review.local.json \
    runs/wall-recovery-demo-1/operator-review.local.json \
  --validation data/navigation-corners-validation-slices-1 \
  --validation-review runs/navigation-corners-validation-1/collection-review.local.json \
  --output runs/steering-direction-1/new-unweighted --experimental \
  --epochs 12 --seed 0 --batch-size 64 --threads 4
```

Add `--weighted` for training-class weights.
Use `--epochs 36` for the longer uniform run.
The [evidence file](evidence/steering-direction.json) records class errors, checkpoint results, and the event audits.

## Next decision

Retain the existing rule and keep all three classifiers offline.
Do not request another recording only because these runs failed.
Compare longer causal visual context with the current short-history classifier next.
The comparison must clear its memory at source gaps and manual resets.
It must use its own inference state rather than recorded human action state.
Keep movement, turn magnitude, and gameplay acceptance as separate unresolved tasks.
