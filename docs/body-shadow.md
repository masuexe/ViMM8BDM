# Body observation runtime

The `body_shadow` policy measures body detection on current game images.
It always returns an empty action.
The command also requires `--observe-only`, which suppresses policy actions before dispatch.
This mode does not control aim, movement, or attack.

The model still produces projectile, foreground, and duplicate-box errors.
This observation test does not establish combat quality.
The [projectile follow-up](projectile-transfer.md) records those limits.

## Processing rules

The policy loads a trusted local body-transfer checkpoint.
It uses four CPU threads and five warmup calls before input-worker setup.
It converts captured BGRA bytes into RGB pixels.
It uses the complete-image detector and the conditional local-zoom procedure.

The policy rejects a future image timestamp, a repeated timestamp, or an image older than 75 milliseconds at entry.
After inference, it discards candidates older than 150 milliseconds from capture start.
A discarded result retains its candidate count and timing for diagnosis.
It supplies no candidate for later use.
The policy never retains a previous candidate for a newer frame.
Pause and enable operations reset its timestamp and output state.
These thresholds diagnose result age and do not establish acceptable combat reaction time.

The existing runtime captures each image immediately before inference.
It does not place images in a capture queue.
Window focus checks, F8 pause, Esc pause, F9 exit, and the input watchdog remain active.
The runtime log records model output, capture time, processing time, and input-worker status.
The input trace provides separate evidence about dispatch.

## Local test

Use the native Windows Python environment with the detector dependencies installed.
Keep the game assets and checkpoint in private local storage.
Select the current official game-window identifier.
Use a new output file and a new input-trace file.

```text
python -m vimm8bdm --live --observe-only --policy body_shadow --checkpoint PRIVATE_CHECKPOINT --capture window --window-id WINDOW_ID --seconds 30 --frames 1200 --hz 10 --arm-timeout 0 --output runs/body-observation.local.jsonl --input-trace runs/body-input.local.jsonl
```

Wait for the model warmup and input-worker readiness messages.
The initial wait has no time limit.
The thirty-second duration starts after the first F8 enable command.

1. Select the game window.
2. Close the menu if necessary.
3. Press F8 to start observation.
4. Play normally for approximately thirty seconds.
5. Check that your controls remain normal.
6. Use Esc or F8 if you need to pause.
7. Use F9 if you need to end the test early.

The test does not save a new training recording.
It measures runtime behavior rather than new detector ground truth.
The first attended body-observation test is complete.

Linux and native Windows each passed all 231 tests without skips.
The native Windows policy also processed one real archived image and returned an empty action.
That check does not measure current capture.
The [preflight evidence](evidence/body-shadow-preflight.json) retains these results.


## First attended result

The native Windows client completed approximately thirty seconds of observation at a target rate of ten images per second.
The operator reported no problem.
The runtime processed 298 images and performed inference on every image.
All 298 action messages remained empty.
The input trace retained no owned key or attack state and dropped no events.
The input worker stopped normally at the duration limit.

Screenshot capture took a median of 7.40 milliseconds.
Policy processing took a median of 41.24 milliseconds and a 95th percentile of 82.04 milliseconds.
The complete loop took a median of 49.55 milliseconds and a maximum of 108.35 milliseconds.
The result age reached a maximum of 107.60 milliseconds.
No result exceeded the 150-millisecond age limit.
This session therefore does not exercise the stale-result rejection branch.
Unit tests separately check that branch.

Candidates appeared in 171 images, and 82 images required an extra crop call.
These counts do not measure detector accuracy.
The observation log does not supply new visual ground truth.
The earlier detection failures remain relevant, and automatic combat remains disabled.

The original console summary incorrectly reported that no neural inference occurred.
Its condition omitted the new policy name.
The per-frame records show all 298 inference calls.
The summary condition now includes the body observation policy.
A regression test checks both performed and skipped inference.
The original runtime records remain unchanged.
The [attended evidence](evidence/body-shadow-attended.json) retains the observed results and the summary error.

After the summary fix, Linux and native Windows each passed all 232 tests without skips.
