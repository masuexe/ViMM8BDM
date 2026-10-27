# First offline navigation experiment

This experiment trains a small visual model with PyTorch.
It uses the two reviewed Grenade Man recordings on MM1CUT.
The experiment sends no desktop input and does not connect to a game server.
It does not use the ContinuousDoom checkpoint or upstream source code.
That checkpoint lacks the required independent navigation outputs, and its reuse terms remain unresolved.

## Inputs and targets

Each input contains four grayscale images resized to 112 by 84 pixels.
The images come from the causal sample history.
All four images completed capture before the decision time.
The model receives no key labels, engine variables, map coordinates, or future images as input.

The output has six values:

- The held fraction for each of W, A, S, and D during the next 50 milliseconds.
- The horizontal raw mouse total during that interval.
- The vertical raw mouse total during that interval.

The four held fractions are independent.
The model can therefore predict simultaneous forward and sideways movement.
Raw mouse targets retain their original magnitude.
Training uses no actuator clipping or response-delay correction.
Mouse normalization uses only training labels.

Held fractions do not specify the exact time of a press or release.
Mouse totals also omit the sequence of movements inside the window.
The raw dataset retains that information, but this first prediction target does not encode it.
A later live adapter must define and test this approximation before executing model predictions.
The model has no attack, jump, ladder, weapon, or skill target.

## Fixed first protocol

The training split contains 2005 candidates from the first recording.
The validation split contains 1794 candidates from the second recording.
The tool checks source hashes and split identities before loading images.
It also reconstructs causal samples from the original sources and compares the result with the candidate files.
Changed candidate labels or image references cause an error.

The model uses three convolution layers and two linear layers.
Training uses Adam, a learning rate of 0.001, batches of 64, and seed 0.
The fixed run contains 12 epochs on four CPU threads.
The loss adds key binary cross-entropy and normalized mouse Smooth L1 loss.
Smooth L1 limits the effect of large errors without changing the stored targets.

After each epoch, the tool measures validation key cross-entropy and raw mouse mean absolute error.
The selection score adds key cross-entropy to mean mouse error divided by the training mouse scales.
The checkpoint comes from the epoch with the lowest selection score.
Validation selects the checkpoint but does not supply gradients or normalization parameters.
It is therefore not an independent final test set.

Compare the model with two constant baselines:

1. Training mean key fractions and training median mouse totals.
2. Training mean key fractions and training mean mouse totals.

Also evaluate shuffled validation image predictions against the original labels.
This check breaks the correspondence between image history and future action.
A separate check replaces every validation image with zeros.
These checks measure image dependence, not navigation success.
Report key errors and mouse errors separately, even when the selection score improves.
Do not claim successful visual learning from a lower combined score alone.

## Run the experiment

Install the optional training packages in the project environment:

```bash
python -m pip install -e '.[train]'
```

Use the reviewed private recordings and a new output directory:

```bash
python -m vimm8bdm.behavior_cloning \
  --train data/navigation-slices-1 \
  --validation data/navigation-validation-slices-1 \
  --train-review runs/navigation-demo-1/session-review.local.json \
  --validation-review runs/navigation-validation-1/operator-review.local.json \
  --output runs/navigation-bc-1/artifacts --experimental \
  --epochs 12 --seed 0 --batch-size 64 --threads 4 --learning-rate 0.001
```

The explicit experimental option permits this run with candidate data.
It does not remove dataset blockers or establish mouse-unit equivalence.
The tool reads separate operator records without changing the source manifests.
The review arguments must identify the records for the corresponding sessions.
The tool checks their required fields but does not interpret the operator's prose automatically.

The private output contains `model.pt` and `report.local.json`.
The tool loads the saved checkpoint with `weights_only=True` and checks its predictions.
The report contains learning curves, normalization parameters, baseline metrics, and image-dependence checks.
Model weights and raw recordings remain excluded from the public repository.
This module provides no live control command.

## First result

Data loading, training, and final metrics took 23.43 seconds on the CPU.
The run completed all 12 epochs.
The model contains 114710 parameters.
Epoch 3 produced the selected checkpoint.
The saved checkpoint reproduced its validation predictions after loading.
Linux and Windows each passed 118 automated tests, including synthetic training checks.
The full target-game training run used Linux.

Lower values indicate smaller errors in this table:

| Predictor | Key cross-entropy | Horizontal mouse MAE | Vertical mouse MAE | Selection score |
| --- | --- | --- | --- | --- |
| Constant training median mouse | 0.3463 | 18.5753 | 1.4660 | 0.7795 |
| Constant training mean mouse | 0.3463 | 21.0913 | 1.4728 | 0.8136 |
| Selected visual model | 0.3410 | 19.0810 | 1.5651 | 0.7935 |
| Model with shuffled image correspondence | 0.3424 | 19.1145 | 1.5646 | 0.7952 |

Both constant baselines use training mean key fractions.
Mouse errors use raw counts per 50-millisecond window.
The model slightly reduced key cross-entropy but increased mouse error against the median baseline.
Its key held-fraction absolute errors also exceeded that baseline for every key.
The combined score did not beat the median baseline.
Shuffling image correspondence changed that score by approximately 0.22 percent.

Thresholding held fractions at 0.5 selected W in every validation sample and no other movement key.
This threshold is a diagnostic rule, not an installed control adapter.
These results do not establish useful visual navigation.
Keep this checkpoint offline.

Forward input occupied 94.04 percent of training time and 96.48 percent of validation time within the candidate windows.
Rightward strafe occupied 15.72 percent and 32.50 percent, respectively.
These distributions favor constant predictions and differ between sessions.
They do not establish the sole cause of the model result.
The small dataset, ambiguous route choices, model representation, and training settings can also affect the result.

A separate CPU timing check measured 100 single-history forward passes after three warm-up passes.
Median inference time was 0.185 milliseconds, and the 95th percentile was 0.343 milliseconds.
This check excluded capture, image preparation, action decoding, and input execution.
Fast inference does not compensate for prediction errors.
The [evidence file](evidence/behavior-cloning.json) contains the metrics and epoch scores.

## Next experiment

Keep both current recordings and the first result as a reference.
Do not add epochs until the model beats appropriate constant baselines.
Collect short, task-specific navigation sessions before another model comparison.
Use the [corner collection plan](navigation-corners.md) for the next training recording.

1. Start at varied positions and headings near clear corridors and corners.
2. Demonstrate forward movement, deliberate stops, and left and right turns.
3. Include sideways movement in both directions and brief backward recovery.
4. Avoid arbitrary camera motion when the route does not require it.
5. Keep training and validation demonstrations in separate complete sessions.

Use the natural motion required by each situation.
Do not change recorded magnitudes to balance the dataset.
Report action frequencies and motion direction for each split before training.
Keep jumps, ladders, combat, and skills as separate later tasks.
Define and test a decoder for the predicted targets before any live model check.

## Limits

The recordings contain approximately three minutes of one character on one map.
Adjacent samples share past images and remain strongly correlated.
The validation session has no backward movement examples.
Neither session covers jumps, ladders, combat, or character skills.
The model cannot establish those abilities from these labels.
The candidate mouse scales remain unverified and are not applied during offline training.
Independent map tests and attended control checks remain necessary before deployment.
