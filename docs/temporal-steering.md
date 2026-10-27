# Bounded visual memory experiment

This experiment tests whether longer visual context improves horizontal input prediction.
It uses the same human recordings and direction labels as the [direction classifier](steering-direction.md).
It does not change the current rule or send game input.

## Model and controls

The experiment reuses the selected image encoder from the 36-epoch direction run.
The encoder contains 113936 parameters and remains fixed during both new runs.
Its source report must match the checked training samples and validation source.
The encoder already used the same validation recording for checkpoint selection.
The new results therefore do not constitute independent validation.

Each encoder output contains 128 numerical features from four causal images.
A gated recurrent unit, or GRU, reads a sequence of these features and updates a numerical memory.
The GRU has 64 memory values and a three-class output layer.
Its 37443 trainable parameters match between the control and longer-history models.
No recorded key state or mouse input enters the memory.

| Variant | Feature steps | Nominal complete image history |
| --- | --- | --- |
| Single-step control | 1 | 150 ms |
| Bounded memory | 32 | 1700 ms |

Each prediction starts its GRU state at zero and reads its complete bounded window.
Short windows at source starts contain only available past features.
Packed sequences prevent padding from entering the memory.
Context never crosses a source change, segment change, or missing anchor.
The training data contains four continuity blocks, and validation contains one.
A future observation cannot change an earlier prediction.

Both runs use 4274 training samples and 1796 validation samples.
Both use seed 0, 12 epochs, batches of 64, four CPU threads, and Adam with a learning rate of 0.001.
Both use uniform class weights.
Feature means and scales use training features only.
The earliest maximum validation macro F1 selects each checkpoint.
The longer run requires more feature processing per sample, so these runs do not have equal computation costs.

The original classification conditions remain unchanged.
Before training, this experiment also required start and stop precision and recall of at least 0.4 within a symmetric 100-millisecond tolerance.
The event comparisons require the same input direction and allow each event to match only once.
These candidate conditions do not establish complete gameplay ability.
The model still supplies no movement keys, turn magnitude, vertical aim, attack, or class skills.

## Results

| Variant | Selected epoch | Macro F1 | Negative recall | Positive recall | False turns on zero targets |
| --- | --- | --- | --- | --- | --- |
| Single step | 2 | 0.527928 | 0.347670 | 0.398649 | 0.139518 |
| 32 steps | 6 | 0.528339 | 0.329749 | 0.317568 | 0.102995 |

The macro scores are nearly equal.
This single-seed comparison does not establish a reliable improvement from longer context.
The longer model produced fewer false turns but missed more active directions.
Neither model passed all conditions.
Both checkpoints remain offline.

| Variant | Start precision at 100 ms | Start recall at 100 ms | Stop precision at 100 ms | Stop recall at 100 ms |
| --- | --- | --- | --- | --- |
| Single step | 0.138211 | 0.346939 | 0.170732 | 0.428571 |
| 32 steps | 0.135417 | 0.265306 | 0.177083 | 0.346939 |

The longer model matched 13 of 49 starts and 17 of 49 stops within that tolerance.
These comparisons use source-window anchors rather than hardware or visible-response timestamps.
They do not establish autonomous turn initiation.

Reversing the longer model's past feature order reduced macro F1 to 0.305868.
Repeating its latest feature across the window gave macro F1 of 0.512004.
Shuffling complete windows gave macro F1 of 0.336861.
The model depends on image content and order, but these tests do not establish adequate steering.

## Verification

Linux and Windows each passed 189 tests after the memory and shadow-runtime changes.
Tests cover source boundaries, capture gaps, future-feature isolation, and variable sequence lengths.
A training test changes validation labels and checks identical parameter updates and normalization.
An end-to-end test checks encoder preservation and complete validation replay after checkpoint loading.
Both actual runs preserved every encoder parameter.
Both saved checkpoints reproduced all 1796 validation logits.

The new runtime also checks image conversion against the recorded-image path.
It discards cached context after a capture gap, size change, invalid history, or manual reset.
Causal reconstruction fills missed feature anchors without replaying old actions.
These are offline runtime checks rather than attended game results.
The [evidence file](evidence/temporal-steering.json) records the actual runs and tests.

## Reproduce

Use a new output directory:

```bash
python -m vimm8bdm.temporal_steering \
  --train data/navigation-corners-train-clean-slices-1 data/wall-recovery-slices-1 \
  --train-review runs/navigation-corners-train-1/operator-review.local.json \
    runs/wall-recovery-demo-1/operator-review.local.json \
  --validation data/navigation-corners-validation-slices-1 \
  --validation-review runs/navigation-corners-validation-1/collection-review.local.json \
  --encoder runs/steering-direction-1/unweighted-36/model.pt \
  --output runs/temporal-steering-1/new-context-32 --context-steps 32 \
  --experimental --epochs 12 --seed 0 --batch-size 64 --threads 4
```

Use `--context-steps 1` for the control.
Keep the encoder checkpoint and its source report together.
Recordings, feature statistics, reviews, and weights remain private.
The validation recording still needs full-session operator confirmation.

## Next runtime check

The separate [shadow check](temporal-shadow.md) measures preprocessing and inference on the native Windows runtime.
It uses the longer model to exercise the largest tested context.
This selection does not approve that model for game control.
The shadow policy always returns an empty action, even when its predicted class is nonzero.
It cannot establish navigation quality.
