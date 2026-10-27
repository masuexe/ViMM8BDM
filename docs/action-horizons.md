# Action window comparison

This offline experiment compares future action windows of 50 and 200 milliseconds.
It uses human recordings only.
It does not change the current visual rule or send desktop input.

## Protocol

Both variants use four causal images with 50-millisecond spacing.
The images cover the same 150-millisecond history.
The longer variant changes the future action duration and the decision spacing.
Its selected future windows do not overlap.
It preserves raw mouse totals and key-held fractions across the complete window.
It applies the reviewed notification exclusion across both the history and the future action interval.

The loader first checks the original dataset and its source hashes.
It then derives the longer windows from the source events without changing those events.
A hash identifies the derived sample list.
Source sessions remain separate between training and validation.
The longer windows reduce the counts from 1879 to 467 training samples and from 1796 to 449 validation samples.

Both runs use motion features, 12 epochs, seed 0, batches of 64, and four CPU threads.
Both use Adam with a learning rate of 0.001.
The network still contains 114710 parameters.
Mouse normalization uses training labels only.
Checkpoint selection minimizes key cross-entropy plus normalized mouse error on validation data.
These settings match the previous motion experiment.

The pre-run decision conditions require improvement over the constant baseline for the combined score, key cross-entropy, and both mouse errors.
Shuffled images must worsen the combined score.
Checkpoint loading must reproduce the predictions.
Input timing still needs separate checks before any deployment.
These conditions screen candidates and do not prove autonomous navigation.

The same epoch count gives the longer-window run fewer optimizer updates.
This comparison does not isolate action duration from training exposure or sample count.
Repeated selection on this validation recording also prevents an independent final-test claim.
Full-session operator confirmation for that recording remains pending.

## Results

Lower scores indicate smaller errors.
Each constant baseline uses its own training-window statistics.
The combined scores across different window durations are not directly equivalent task measurements.

| Window | Selected epoch | Model combined score | Constant combined score | Key cross-entropy | Forward held-state F1 |
| --- | --- | --- | --- | --- | --- |
| 50 ms | 10 | 0.617206 | 0.604115 | 0.260341 | 0.701706 |
| 200 ms | 8 | 0.693274 | 0.683569 | 0.305676 | 0 |

The 50-millisecond run reproduced the previous selected weights exactly.
Its validation score also matched exactly.
The 200-millisecond run predicted all four movement keys as released at the 0.5 held-fraction threshold.
Neither checkpoint passed the decision conditions.
Neither checkpoint enters the live controller.

| Window | Model mouse X MAE | Constant mouse X MAE | Model mouse Y MAE | Constant mouse Y MAE |
| --- | --- | --- | --- | --- |
| 50 ms | 9.5630 | 7.7233 | 0.4239 | 0.3630 |
| 200 ms | 31.2377 | 30.5635 | 1.3347 | 1.2650 |

Mouse errors use raw counts per complete action window.
A 200-millisecond total does not preserve the exact trajectory inside that window.
Opposite mouse movements can cancel in the total.
Held-state F1 measures thresholded occupancy, not exact input transitions.
The earlier [transition audit](action-events.md) explains this distinction.

Shuffling image histories increased the 50-millisecond combined score to 0.769591.
It increased the 200-millisecond score to 0.696681.
The latter change is small and does not establish useful steering.
These results do not show that longer actions can never work.
They show that this fixed experiment did not produce a usable candidate.

## Verification and reproduction

Linux and Windows each passed 166 tests after the extension.
New tests cover future-window boundaries, unchanged causal images, reviewed exclusions, and model duration metadata.
The training-isolation test changes validation labels and checks identical training updates and normalization with 200-millisecond targets.
Both saved checkpoints reproduced their predictions after loading.
Version 3 checkpoints store the action duration explicitly.
The loader still supports version 1 and version 2 checkpoints as 50-millisecond models.

Use a new output directory for each run:

```bash
python -m vimm8bdm.behavior_cloning \
  --train data/navigation-corners-train-clean-slices-1 \
  --validation data/navigation-corners-validation-slices-1 \
  --train-review runs/navigation-corners-train-1/operator-review.local.json \
  --validation-review runs/navigation-corners-validation-1/collection-review.local.json \
  --output runs/navigation-horizon-bc-1/new-action-200 --experimental \
  --features motion --action-ms 200 --epochs 12 --seed 0 \
  --batch-size 64 --threads 4 --learning-rate 0.001
```

Use `--action-ms 50` for the control run.
The [evidence file](evidence/action-horizons.json) records the measured results and decision conditions.
Recordings, weights, and configuration files remain private.

## Next collection

Keep the existing visual rule unchanged as a diagnostic baseline.
The latest operator report still identifies unresolved behavior in that rule.
Do not mark wall navigation as accepted.

Collect a separate human session for wall approaches and recovery.
Use fixed Grenade Man on MM1CUT for this first addition.
Include normal forward movement before each approach, then a human turn away from the wall.
Include angled approaches that cause slow movement along a wall.
Use normal ground navigation without attacks, jumps, or ladder movement in this session.
Pause before menus, death, chat, or character changes.

This collection targets the reported failure cases.
It does not establish that missing data caused the model failures.
Keep the new source separate from validation and record the operator's review before training.
