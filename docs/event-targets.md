# Explicit navigation targets

The [transition audit](action-events.md) identified a mismatch between occupancy thresholds and key changes.
The new target representation separates event count, event time, and mouse totals.
It compiles through the existing bounded timed-action path.
This change does not train a new model or send desktop input.

## Keyboard representation

Each 50-millisecond target contains four chronological lists of integer offsets.
The lists correspond to W, A, S, and D.
Each list supports zero, one, or two changes inside the window.
Offsets preserve nanosecond differences between source callback receipt and the decision anchor.
Zero is a valid offset.
The encoder rejects offsets at the window endpoint.

The encoder checks that source press and release events alternate from the recorded starting state.
It checks the final state and held fractions against the reconstructed result.
Unsupported keys, buttons, wheels, extra changes, or inconsistent labels cause an error.
The encoder does not discard or clip those inputs.

The decoder requires the caller's owned key state.
Each event toggles the corresponding key from that state.
The target contains no future final state for the decoder to copy.
During source reconstruction, the caller supplies the recorded starting state.
An autonomous caller must instead supply its own input state.
It must not substitute recorded human state during a claimed autonomous evaluation.

The representation preserves both kinds of two-event windows:

- A released key becomes pressed and then released.
- A held key becomes released and then pressed again.

For training, each key has a count target from zero through two.
Its two time slots contain normalized offsets and an explicit presence mask.
An absent slot has zero padding with a false mask.
An event at time zero has a true mask.
The mask therefore prevents confusion between missing events and events at the window start.

## Mouse representation

The target retains horizontal and vertical raw totals and whether each total is nonzero.
It does not retain the complete mouse trajectory inside the window.
The decoder places one aggregate motion event at the 25-millisecond midpoint.
The original source still contains every raw mouse event.

Opposing motions can cancel in the total.
The audit reports windows with such cancellation instead of claiming complete mouse reconstruction.
The directed training and validation datasets contain four and seven such windows, respectively.
This approximation requires a separate game-response check before live model control.
It cannot establish equivalent camera motion or combat timing from arithmetic checks alone.

Compilation requires explicit candidate mouse scales.
The existing compiler splits larger totals into packets without changing their sum.
Each packet remains within the existing per-axis bound.
Fractional remainder handling remains in the input worker.
The candidate scales 1.06 and 0.92 remain experimental.

## Dataset checks

The audit reconstructs causal candidates from verified source files before encoding.
It then checks compiled key events against the original offsets and directions for every key.
It also checks scaled mouse totals across the compiled packets.

| Dataset | Samples | Key events retained | Two-change cases retained |
| --- | --- | --- | --- |
| Directed training after exclusion | 1879 | 114 | 1 |
| Directed validation | 1796 | 98 | 0 |
| Original training | 2005 | 234 | 1 |
| Original validation | 1794 | 344 | 1 |

All 7474 candidates passed these checks.
The compiled output retained 790 key events and three two-change cases.
Mouse totals also matched in every candidate.
These results establish representation and compilation, not neural prediction accuracy.
They do not measure input execution time or visible game effects.
The worker's polling interval can prevent exact execution of very short recorded intervals.

Linux and Windows each passed 138 automated tests.
The new tests cover late changes, two-event windows, simultaneous keys, zero offsets, masks, and mouse bounds.
They also check rejection of inconsistent labels and packet overflow.
The [evidence file](evidence/event-targets.json) contains aggregate results for all four datasets.

## Run the audit

Create a private output directory, then use a new output file:

```bash
python -m vimm8bdm.event_targets \
  --dataset data/navigation-corners-train-clean-slices-1 --split train \
  --scale-x 1.06 --scale-y 0.92 \
  --output runs/navigation-event-targets-1/new-audit.local.json
```

Use `--split validation` for the validation dataset.
The command fails on unsupported or inconsistent labels.
It does not create a filtered copy or change source data.

## Next experiment

Train event counts and masked offsets as separate targets.
Retain explicit context for the agent's own starting key state.
Evaluate both recorded-state references and sequences that use earlier predicted states.
Report false changes, missed changes, and time-offset errors separately.
Keep mouse activity and displacement errors separate from keyboard results.
Do not use target reconstruction as evidence of learned navigation.
Keep the future model offline until its prediction and decoder checks support an attended trial.
