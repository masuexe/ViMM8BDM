# Action transition audit

This audit separates persistent actions from key changes and mouse motion phases.
It evaluates the existing checkpoints on the 1796 directed validation candidates.
Source verification and causal sample reconstruction occur before inference.
No desktop input occurs.

## Key definitions

The source labels retain individual key press and release events.
The current models predict the fraction of each future window during which a key remains down.
The diagnostic decoder thresholds that fraction at 0.5 and treats the result as a desired final state.
This rule is an approximation, not a deployed adapter.

The audit reports two distinct comparisons:

1. Compare the desired state with the recorded human key state at the window start.
2. Compare it with the previous predicted state across contiguous windows.

The first comparison uses recorded human history as a reference.
The second uses only earlier predictions for state changes.
Both still use recorded images rather than images from an autonomous game session.
Neither is an independent policy rollout.

A target event counts as detected only in its original action window.
The tool matches event presence per key and direction, rather than precise event times.
It does not measure hardware latency or match arbitrary valid navigation choices.
The tool excludes the first window of each contiguous interval from the second comparison.
It never joins gaps, exclusion intervals, or separate recording segments.

## Existing model result

The motion model's contiguous prediction sequence produced these forward-key results:

| Event | Recorded positive windows | Predicted positive windows | True positives | False positives | False negatives |
| --- | --- | --- | --- | --- | --- |
| W press | 25 | 84 | 1 | 83 | 24 |
| W release | 25 | 84 | 1 | 83 | 24 |

F1 was approximately 0.0183 for each direction.
This result differs from the earlier 0.7017 forward occupancy F1.
That earlier measure checks whether a key is generally active, rather than when it changes.
The motion model predicted no A, S, or D changes at the diagnostic threshold.
The raw model predicted no movement-key changes.

## Target and decoder mismatch

The recording contains 98 key events across all four movement keys.
The occupancy threshold disagrees with the recorded final state for 47 combinations of key and action window.
These disagreements do not mean that the recorder lost inputs.
They expose a limitation of the target-to-action approximation.

For example, a key can start released and become pressed after 40 milliseconds.
Its held fraction is 0.2 during a 50-millisecond window.
Thresholding at 0.5 incorrectly selects a released final state.
A decoder needs separate evidence of the state change and its timing.

Two diagnostic oracles isolate the target representation from model error:

- Supply the exact recorded future held fractions to the occupancy decoder.
- Supply the exact recorded future final key states to the same state-change comparison.

Both oracles use future labels and cannot serve as deployment policies.
The first oracle still misses transitions.
Its W press and release F1 values were 0.6486 and 0.6842 against recorded starting states.
The final-state oracle detected all key changes in this validation session.
This session contains no window with both a press and a release for the same key.
Final-state prediction alone therefore does not establish support for short taps in other recordings.
The evaluator preserves and reports those cases separately.
The training dataset contains one A release followed by another press inside the same window.
Its start and end states match, and its held fraction is approximately 0.9936.
A new target must preserve both events rather than infer them from the final state alone.
The recorded offsets identify callback receipt times, not precise hardware timing.

## Mouse definitions and result

The tool treats an absolute raw delta of at least one count as nonzero motion.
For each axis, it compares adjacent complete action windows:

| Phase | Meaning |
| --- | --- |
| Start | Zero motion followed by nonzero motion |
| Continue | Nonzero motion with the same sign |
| Reverse | Nonzero motion with the opposite sign |
| Stop | Nonzero motion followed by zero motion |
| Idle | Zero motion in both windows |
| Boundary | No contiguous earlier window exists |

These phases describe input totals, not engine camera angles or projectile behavior.
The model's phase sequence derives from its own predicted totals.
The tool reports raw-count errors by actual phase and separate phase-detection metrics.

The validation recording contained 49 horizontal starts and 49 horizontal stops.
The motion model matched five starts and one stop in their original windows.
It predicted 190 starts and 189 stops.
During 1319 actual idle windows, it predicted nonzero horizontal motion in 849 windows.
It predicted no vertical motion above the one-count threshold.
These results leave the current steering output below the conditions required for a live control check.

## Run the audit

Create a private output directory, then run:

```bash
python -m vimm8bdm.action_events \
  --dataset data/navigation-corners-validation-slices-1 \
  --checkpoint runs/navigation-bc-motion-1/artifacts/model.pt \
  --output runs/navigation-action-events-1/new-audit.local.json
```

Use a new output file.
The default split is `validation`.
Use `--split train` only for a training-data diagnosis.
The [public evidence](evidence/action-events.json) contains aggregate results and checkpoint digests.
Linux and Windows each passed 129 automated tests.
New tests cover short taps, state-history differences, gaps, segments, and motion phases.

## Next model contract

Keep the original recordings, fractional targets, and existing checkpoints unchanged.
Evaluate explicit key changes and their offsets before another live-control adapter.
The [explicit target encoder](event-targets.md) now preserves keyboard changes and compiles them through the bounded timed path.
The model may receive its own previous input state during deployment.
That state must describe commands owned by the agent, not hidden engine state or actual human commands during autonomous evaluation.
Separate teacher-forced tests from tests that feed earlier predictions into later decisions.

Report initiation, stopping, and false changes separately from persistent actions.
Retain short taps and repeated changes as explicit events.
Evaluate mouse activity separately from signed motion magnitude.
Do not use a lower overall loss as the sole acceptance condition.
