# Held-input focus check

This local test holds forward movement and attack while the user changes the active window.
The stock client must use the normal game assets and input bindings.
The test does not read engine state or game packets.

## Run the test

Use an attended loopback game without opponents.
Keep unrelated input controllers stopped.
Use the selected game window ID from the normal window selector.
Run this command from the repository root in PowerShell:

```powershell
.local\windows-venv\Scripts\python -m vimm8bdm --live --capture window `
  --window-id 0xYOUR_WINDOW_ID --policy focus_check --arm-timeout 0 `
  --seconds 120 --frames 2400 --hz 20 `
  --output runs/focus-frames.jsonl --input-trace runs/focus-worker.jsonl
```

Use new output files.
Wait for the input-worker ready message.
The initial waiting phase has no timeout with `--arm-timeout 0`.
The run limit starts after the first F8 enable command.

1. Select the game window.
2. Press F8.
3. Wait for automatic forward movement and attack.
4. Immediately use Alt+Tab to select another window.
5. Wait two seconds.
6. Use Alt+Tab to return to the game.
7. Observe the game for two seconds without F8.
8. Press F9 to finish the test.

The player must stop after focus loss.
Returning to the game must not enable input automatically.
A later F8 may enable input again.
Esc also pauses AI input and can open the game menu.
Closing the menu does not enable AI input.
Close the menu before using F8 again.
The test limits each hold to eight seconds if focus remains valid.
An expired hold does not supply focus-loss evidence.

## Worker records

The input worker checks focus independently from capture and policy execution.
A failed focus check while armed records a `focus_lost` event before release.
That event includes the owned inputs, the query interval, and the previous successful query start.
A later `released` event records the state after the release calls return.
The release event must contain no held keys and no held attack button.
The final `trace_end` event must report zero dropped events for a complete audit.

In this test, an unfocused blank image does not issue a parent pause command.
The worker still checks focus and image age before every action.
The worker can therefore release input through its own focus check.
Other policies retain the existing blank-image pause behavior.

## Interpret the timing

Measure from `focus_observed_ns` to the release timestamp for detection-to-release time.
This interval excludes the delay before the worker detects focus loss.
The trace does not timestamp the user's Alt+Tab event or an OS focus-change notification.

The previous successful query start provides an earlier reference point.
For one focus transition, the interval from that start to release bounds the transition-to-release delay.
This bound includes query and polling time.
Multiple focus transitions between queries can invalidate that interpretation.
Record both intervals instead of claiming an exact OS event latency.
Neither interval measures the visible game-stop tick.

Check the parent frame log after focus returns.
Visible nonblank frames must remain rejected until a new user enable command.
Record the operator's observation separately from worker release-call timing.
Keep raw traces and desktop metadata in the excluded `runs/` directory.

## Automated checks

Linux and Windows each passed 58 tests.
A real spawned worker holds movement and attack against a simulated input receiver.
A shared focus flag then changes to false.
The test checks release, trace ordering, and generation changes.
It also checks that restored focus and submitted actions cannot enable input.
A new test checks unlimited initial waiting beyond 180 seconds.
These tests do not replace the stock-client focus test.

## Stock Windows alpha result

The live test used the official `3.3-alpha-r260925-2352` client and a stock loopback server.
The loaded stack was MM8BDM v6a, CBM v9fh, and Gondola Patch v2.7h2.
The worker recorded two focus losses while it owned forward movement and attack.
Both release records contained no held keys and no held attack button.

| Focus loss | Detection to completed release calls | Previous successful query start to release |
| --- | --- | --- |
| 1 | 3.56 ms | 20.15 ms |
| 2 | 6.63 ms | 22.45 ms |

The largest observed detection-to-release interval was 6.63 milliseconds.

The larger query-based interval was 22.45 milliseconds.
The trace does not measure the exact OS event or visible game-stop latency.
The test does not establish a worst-case guarantee.

After focus returned, nonblank frames remained rejected before any later user enable command.
The operator reported that the player stopped and did not resume automatically.
The trace also recorded two Esc pauses that released held movement and attack.
The operator separately reported that Esc stopped control.

See the [sanitized focus record](evidence/focus-check.json).

The controller reached its 120-second run limit and closed normally.
The worker trace ended with zero dropped events and no held inputs.
The frame log contains 235 accepted actions and 2136 inactive or rejected frames.
Those inactive frames include the intended pause after focus loss.
