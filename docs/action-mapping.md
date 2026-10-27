# Dataset labels and desktop actions

The audit compares recorded labels with the current bounded `Action` interface.
It does not replay samples or send game input.
The existing live mouse test remains a separate attended experiment.
The [timed execution path](timed-actions.md) provides a separate interface for transitions and larger mouse totals.
This audit still checks the original single-action interface.

## Supported scope

The keyboard mapping follows the Windows [virtual-key definitions](https://learn.microsoft.com/en-us/windows/win32/inputdev/virtual-key-codes).
It maps W, A, S, D, Space, and number keys 1 through 9 to their existing executor names.
Shift exists in the action interface, but the audit requires a binding and side review before mapping it.
A virtual-key match alone does not prove a character-specific game effect.
Grenade Man's single-weapon recording did not test weapon selection.

The interface can hold movement keys and primary attack together.
Its mouse fields contain relative horizontal and vertical requests.
Each request has a limit of 24 counts per axis.
The desktop backend rounds each request to an integer.

The current interface cannot represent every recorded label:

- It has no secondary mouse button or wheel action.
- It cannot reproduce arbitrary keys outside its allowed set.
- It cannot schedule a press or release inside one 50-millisecond action step.
- A small fractional mouse request can disappear during integer rounding.

The audit reports these conditions instead of removing labels or clipping motion silently.
A short tap requires a timed decoder or an explicitly declared training approximation.
Keep the original transitions for that later decision.
The audit does not remove samples from the source dataset.

## Run the audit

Run this command from the repository root in PowerShell:

```powershell
.local\windows-venv\Scripts\python -m vimm8bdm.action_mapping `
  --dataset data/timing-slices --output runs/action-audit.json
```

Use a new output file.
Without a mouse scale, nonzero mouse labels remain blocked.
Idle labels and representable constant key/button states can still produce offline candidates.
The tool checks source hashes before reading labels.

Supply both candidate scales only for an explicit experiment:

```powershell
.local\windows-venv\Scripts\python -m vimm8bdm.action_mapping `
  --dataset data/timing-slices --candidate-scale-x 1.06 --candidate-scale-y 0.92 `
  --output runs/action-audit-candidate.json
```

These arguments do not install a runtime profile or enable AI control.
The output retains `mouse_unit_conversion_verified: false` and `training_ready: false`.
It also reports motion that integer rounding would reduce to zero.

## Mouse units and settings

Windows [Raw Input mouse data](https://learn.microsoft.com/en-us/windows/win32/api/winuser/ns-winuser-rawmouse) and [injected mouse input](https://learn.microsoft.com/en-us/windows/win32/api/winuser/ns-winuser-mouseinput) use different processing paths.
The SDK describes pointer settings for relative injected motion.
That description alone does not establish the exact input path used by this game build.
Use displayed game response to check the current setup.

The new recorder metadata includes read-only Windows pointer settings when the query succeeds.
The query does not change pointer speed, thresholds, or acceleration.
Older recordings do not contain this snapshot.
Do not assign the new snapshot to old recordings as a verified historical fact.

The test snapshot reported pointer speed 5, thresholds 6 and 10, and acceleration value 1.
Those values describe the inspected Windows settings.
They do not prove how the engine processes every injected event.

## Live pulse result and candidate scale

The stock client completed 24 injected pulses over 26.63 seconds.
The input worker accepted all 529 actions.
Both axes had twelve valid image measurements.
The operator observed small movements on both axes followed by an automatic stop.
The worker closed with no held inputs and zero dropped audit events.

The injected image response had a zero-intercept least-squares slope of -1.0 horizontally and -0.65 vertically.
The two earlier physical-input fits had horizontal slopes of approximately -1.014 and -1.101.
Their vertical slopes were approximately -0.608 and -0.587.
All slopes use displayed image pixels per input count for their respective input path.

Dividing the mean physical-input slope by the injected slope gives an experimental starting scale.
Rounded values are 1.06 horizontally and 0.92 vertically.
The source scenes and movement patterns differ.
Pixel quantization and the camera-translation approximation also affect the estimates.
These values do not establish a precise or universal raw-to-injected conversion.
No global Windows settings changed.

The 756-sample diagnostic dataset produced these audit results:

| Audit | Offline candidates | Remaining condition |
| --- | --- | --- |
| No scale | 511 | 245 nonzero-motion labels need a scale |
| Unit scales | 756 | Unit equivalence remains unverified |
| Candidate scales 1.06 / 0.92 | 756 | Experimental units and game bindings still need review |

The candidate-scale audit found no motion requests that rounded to zero in these samples.
This dataset contains mouse calibration actions and idle periods.
It does not test every keyboard, attack, or class-skill combination.
All outputs remain outside the training set.

Linux and Windows each passed 100 automated tests.
Tests cover combined keys and attack, unsupported labels, taps, clipping, scales, and read-only pointer queries.
See the [sanitized action record](evidence/action-mapping.json).

## Next collection step

Use fixed Grenade Man for a short navigation demonstration.
Keep the current client, bindings, mouse settings, FOV, and resolution.
Use movement and view changes to demonstrate routes and wall avoidance.
Leave class skills and weapon actions for separate recordings and execution support.
Preserve raw labels so later unit corrections do not require another recording.
The recorder sends no input during the human demonstration.
