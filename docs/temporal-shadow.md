# Temporal model shadow check

The shadow policy observes game images and records model predictions.
It always returns an empty action.
The command also requires observation mode, which suppresses policy actions before the input worker.
The existing visual rule remains unchanged.

This check tests a native deployment interface for the project's own model.
It does not test autonomous movement or approve the model's steering quality.
The [offline memory experiment](temporal-steering.md) did not pass its candidate conditions.

## Image and memory path

The policy converts BGRA pixels to grayscale and resizes them to 112 by 84 pixels.
This path matches the recorded RGB PNG path in a replay test.
Each completed image receives a conservative completion-time bound at policy entry.
The bound can be later than the actual capture completion.

A fixed 50-millisecond decision grid starts from the first completion bound.
The policy selects four distinct images nearest the 150, 100, 50, and zero millisecond history targets.
Every selected image must be complete by the decision anchor.
An image that arrives after an anchor cannot enter that anchor's prediction.
The maximum distance from each image target is 75 milliseconds.
These limits match the offline sample builder.

The selected image history produces one encoder feature.
The policy keeps at most 32 consecutive features and recomputes the bounded GRU state for each prediction.
It reads no recorded human input or hidden engine state.
It discards context after capture gaps above 100 milliseconds, invalid timestamps, size changes, or invalid image histories.
When a capture arrives after a grid boundary, it reconstructs missed features from images already complete at each missed anchor.
It sends no prediction or action for those earlier anchors.
Manual re-enabling also clears the policy.
The policy does not advance while paused.

The log contains class probabilities, context length, anchor times, selected image times, and the checkpoint digest.
It also contains separate preprocessing, encoder, and memory-stage durations.
No screenshot archive enters this check's output.
Logs remain private.

## Run an attended check

Use the stock Windows alpha client and the current fixed Grenade Man setup on MM1CUT.
Keep the game window visible and close menus and desktop notifications.
Use the current window identifier and new output files:

```powershell
.local\windows-venv\Scripts\python -m vimm8bdm `
  --live --capture window --window-id 0x12345 --policy temporal_shadow `
  --checkpoint runs/temporal-steering-1/context-32/model.pt --observe-only `
  --seconds 30 --frames 1200 --hz 20 --arm-timeout 0 `
  --output runs/temporal-shadow-check-1/actions.local.jsonl `
  --input-trace runs/temporal-shadow-check-1/worker.local.jsonl
```

1. Wait for the ready message.
2. Select the game window.
3. Close its menu.
4. Press F8 to start observation.
5. Move and turn with the real keyboard and mouse for 30 seconds.
6. Check that the player remains under human control.

F8 or Esc pauses the process.
F9 exits it.
Focus loss pauses it, and returning does not resume it.
The initial wait has no timeout.
The duration limit starts after the first F8 command.
The program does not move, aim, or attack for the player.

## Evidence status

Linux and Windows each passed 189 tests after the runtime addition.
Tests check causal image selection, pixel-path equivalence, context resets, and empty actions.
A dispatcher test also suppresses deliberately nonempty mock actions in observation mode.
A native Windows synthetic benchmark processed 100 artificial 640-by-480 images without desktop capture.
It performed 97 inference steps, including 66 steps with full context.
Full-context prediction took approximately 2.69 milliseconds at the median and 3.57 milliseconds at the 95th percentile.
Every returned action was empty.
Its result does not measure capture delay, engine response, or gameplay performance.
The attended game check remains pending.
The [evidence file](evidence/temporal-shadow.json) separates these results.


## Recorded-image replay

A native Windows replay processed all 2385 frames from the recovery recording.
All 2395 labelled feature histories matched the offline dataset's selected images.
The runtime also processed one final causal feature without a complete future action label.
That feature needs no future input during inference.
The complete grayscale input tensor had the same hash on Linux and Windows.

One hundred probability comparisons used offline encoded images and the runtime's actual bounded history.
The maximum probability difference was approximately 0.00000018.
All returned actions remained empty.
These checks use recorded images and do not replace live capture checks.

The first sampling revision cleared memory 81 times at skipped grid anchors.
Causal feature reconstruction removed those unnecessary resets in the repeated replay.
The updated replay performed 2314 inference steps and 82 additional feature steps.
It reached full context for 2288 inference steps.
No old action entered the worker.
