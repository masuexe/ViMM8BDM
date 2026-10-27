# Mouse calibration from game images

The calibration tool sends small relative mouse pulses to the stock Windows client.
It does not move the player, jump, fire, or select weapons.
Each cycle tests both axes with positive and negative pulses of 4 and 8 counts.
The input worker retains its focus checks, hotkeys, and 200 ms lease.

## Prepare the scene

Use the exact engine build, asset stack, resolution, and mouse settings intended for deployment.
Use an attended local game without opponents.
Keep the player stationary near a textured wall.
Keep moving objects, sky, menus, and HUD elements outside the measurement region.
Do not move the real mouse during image collection.

The tool measures an upper central image region.
It compares a stationary baseline pair before each pulse.
It waits after each pulse before collecting the result image.
All translation analysis and PNG writes occur after the input worker closes.

## Run a short test

List the game windows with the normal window selector.
Start this command from the repository root in PowerShell:

```powershell
.local\windows-venv\Scripts\python -m vimm8bdm.calibration `
  --window-id 0xYOUR_WINDOW_ID --run-dir runs/mouse-short --cycles 3
```

Use a new run directory for each test.
Wait for the ready message.
The initial waiting phase has no default timeout.
Use `--arm-timeout SECONDS` to set a positive waiting limit.
F9 remains available during that waiting phase.
Select the game window.
Press F8 to start.
Press F8 or Esc to pause.
Press F8 to enable input again.
Press F9 to stop the run.

A pause discards the current incomplete measurement.
An interrupted pulse can leave a small view offset.
Opposite pulse counts do not guarantee an exact return to the original view.
The program does not correct the view through engine state.

## Read the result

The local run directory contains these files:

- `frames.local.jsonl`: capture times, accepted actions, generations, and pulse counts.
- `worker.local.jsonl`: independent input events and release records.
- `NNN-baseline.png`: the first stationary image.
- `NNN-before.png`: the image immediately before the pulse.
- `NNN-after.png`: the image after the settling interval.
- `summary.local.json`: individual measurements and axis medians.

Keep these files outside Git.
Inspect the image pairs before accepting a calibration.
Check the worker trace for unintended pauses and dropped events.

A valid measurement requires a stationary baseline and a distinct translation match.
The tool rejects low texture, uncertain matches, search-boundary matches, and absent visible response.
The search range is 24 pixels in either direction.
A rejected trial does not prove that input failed.
A camera response outside that range also causes rejection.

The result uses image pixels per injected mouse count.
Positive image displacement means that scene features moved right or down.
The scene usually moves opposite to the viewing direction.
This value is a local approximation for the selected image region.
Perspective, scene depth, view settings, and OS mouse processing can change it.
The tool does not calculate engine degrees or measure visible response latency.
Do not use the result directly as the ContinuousDoom yaw scale.
That adapter requires a separate action-unit conversion.

## Extended capture test

After the short test produces useful image pairs, repeat the command with `--cycles 14`.
The sequence requires approximately 123 seconds without pauses.
This checks repeated capture, input dispatch, and the active control lease.
It does not test navigation, combat, or a held movement key.
Inspect the frame log for capture gaps and rejected actions.
Inspect the worker trace for release and normal shutdown.
Record manual takeover observations separately from timing measurements.

## Automated checks

Synthetic images test both axes, both signs, moving baselines, absent response, and search-boundary rejection.
A simulated desktop test checks that each pulse occurs once without movement or attack.
These tests do not check the stock client's response.
The Windows short test produced 24 accepted image measurements.
The operator observed small movements on both axes.
The initial waiting regression test also covers a delay longer than 180 seconds.

## Windows short-test result

The stock alpha client completed 24 pulses in 26.62 seconds.
All 529 captured frames produced accepted actions.
The maximum interval between capture completions was 57.09 ms.
The independent worker closed with no held inputs and zero dropped audit events.
The operator observed small horizontal and vertical view movements.
Inspection of a horizontal image pair also showed the expected scene displacement.

| Axis | Input count | Image displacement in each of three cycles |
| --- | --- | --- |
| Horizontal | +4 / -4 | -4 / +4 pixels |
| Horizontal | +8 / -8 | -8 / +8 pixels |
| Vertical | +4 / -4 | -3 / +3 pixels |
| Vertical | +8 / -8 | -5 / +5 pixels |

The signed median ratios were -1.0 for horizontal input and -0.6875 for vertical input.
The search uses integer pixel offsets.
These ratios do not establish subpixel precision or an angle calibration.
See the [sanitized measurement record](evidence/mouse-calibration.json).

The first attempt expired before F8 and sent no input.
Calibration now waits for F8 or F9 without a default initial timeout.
The extended repetition test remains pending.
