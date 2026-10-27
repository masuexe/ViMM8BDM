# Explicit event model experiment

This experiment trains the [explicit navigation targets](event-targets.md) with image history and starting key context.
The context contains four binary values for W, A, S, and D.
During training and the recorded-state reference, these values come from recorded human input before the decision anchor.
During prediction feedback, they come only from earlier model predictions.
No engine state enters either calculation.

## Model and fixed protocol

The network reuses the motion-image encoder from the earlier experiment.
It combines 128 image features with four key-state values.
Its 28 outputs contain:

- Twelve logits for four key-event counts, with three classes per key.
- Eight values for two event times per key.
- Six logits for negative, zero, or positive mouse totals on two axes.
- Two nonnegative mouse magnitudes.

The second time follows the first time by construction.
Absent event times contribute no timing loss.
Zero mouse classes produce exactly zero output regardless of the magnitude head.
Active magnitude loss uses only windows with nonzero source totals.
All original raw counts remain unchanged.

The run used 1879 training candidates and 1796 separate validation candidates.
It used 12 epochs, seed 0, batches of 64, four CPU threads, and Adam at a learning rate of 0.001.
The model contains 117660 parameters.
Category weights use inverse square roots of training class counts, normalized to unit mean sample weight.
Missing classes have zero target weight.
Mouse magnitude scales use mean absolute nonzero training totals, with a minimum scale of one.
Validation data supplies no weights, normalization, or gradients.

The loss adds count classification, masked event timing, mouse class classification, and masked magnitude error.
Checkpoint selection maximizes the mean of key-change macro F1 and mouse-activity macro F1 using recorded starting states.
This diagnostic score differs from the older combined error score.
It does not constitute a deployment acceptance test.
Epoch 10 supplied the selected checkpoint, and the run took approximately 28.21 seconds.

## Results and state feedback

The selected checkpoint detected none of the recorded key changes in their original windows.
It produced two false A releases in the recorded-state reference.
The predicted-state feedback produced one false S press without a subsequent S release.
It never started forward movement in that feedback sequence.
There were no correctly matched key events for a meaningful timing-error estimate.

| Evaluation | Forward held-fraction MAE | Horizontal mouse MAE | Vertical mouse MAE |
| --- | --- | --- | --- |
| Hold recorded keys, zero mouse | 0.0131 | 7.7233 | 0.3630 |
| Event model with recorded starting keys | 0.0131 | 9.7416 | 0.4862 |
| Event model with earlier predicted keys | 0.3685 | 10.1908 | 0.5004 |

Mouse errors use raw counts per 50-millisecond action window.
The small forward error with recorded states mainly reflects persistence of the human's existing commands.
It does not establish movement initiation.
The prediction-feedback sequence starts with all keys released and resets to released keys after source gaps.
It reads no recorded key states during prediction.
The images still come from human play and do not respond to the model's commands.
This sequence is not an autonomous game rollout.

Mouse activity classification improved some detection counts, but steering error still exceeded the zero-mouse reference.
The checkpoint therefore remains offline.
Its low error under recorded-state evaluation must not be reported as playable navigation.
Full-session operator confirmation for the validation recording also remains pending.

## Verification and reproduction

The tool reconstructs causal source samples before creating event tensors.
Both recorded-state and prediction-feedback outputs passed bounded timed compilation with candidate scales 1.06 and 0.92.
The saved event checkpoint reproduced its outputs after loading.
The checkpoint format records the image mode, key order, window size, and event-count limit.
The loader rejects incompatible configuration or invalid magnitude scales.

Linux and Windows each passed 144 tests before the later visual baseline work.
New tests cover absent-target gradients, chronological event times, zero-mouse gating, and checkpoint loading.
A feedback test changes the stored human key states and verifies that earlier predictions alone determine context.
A training test changes validation labels and verifies identical parameter updates and training statistics.

Run the experiment with a new output directory:

```bash
python -m vimm8bdm.event_cloning \
  --train data/navigation-corners-train-clean-slices-1 \
  --validation data/navigation-corners-validation-slices-1 \
  --train-review runs/navigation-corners-train-1/operator-review.local.json \
  --validation-review runs/navigation-corners-validation-1/collection-review.local.json \
  --output runs/navigation-event-bc-1/new-artifacts --experimental \
  --epochs 12 --seed 0 --batch-size 64 --threads 4 --learning-rate 0.001
```

The [evidence file](evidence/event-cloning.json) contains epoch metrics, state-feedback results, and class statistics.
Model weights and recordings remain private.
The training command sends no desktop input.

## Next practical baseline

The model does not supply usable movement initiation or steering yet.
Use the separate [visual navigation rule](visual-navigation.md) to establish a measurable local-game baseline.
Its actions must remain labelled as rule output, not human demonstration or learned-policy output.
Keep the neural model and its failed event metrics as a reference for later changes.
