# Mouse input and image response alignment

The first human recording uses one clock for images and input events.
That fact does not establish the delay between input receipt and visible game response.
This check estimates that delay from ordinary displayed images.
It does not read game angles, tick counters, memory, or packets.

## Prepare the test

Use the stock Windows alpha client and the original target assets.
Select Grenade Man through the normal game menu.
Use fixed selection for this check.
Enter the map after the selection takes effect.
Stand still near a textured wall with room for small camera movements.
Keep the current resolution and mouse settings.
Stop AI controllers.

Start the recorder from the repository root in PowerShell:

```powershell
.local\windows-venv\Scripts\python -m vimm8bdm.recording `
  --window-id 0xYOUR_WINDOW_ID --run-dir runs/mouse-timing/session `
  --protocol mouse_alignment --seconds 45 --hz 60 `
  --character grenademan --class-selection fixed --map MM1CUT `
  --engine-version 3.3-alpha-r260925-2352
```

The initial waiting phase has no timeout.
The target capture rate is 60 Hz.
The recorder stores actual capture timestamps rather than assuming that rate.
It sends no game input.

## Follow the sound cues

1. Wait for the ready message.
2. Select the game window.
3. Close the menu.
4. Release all gameplay keys and mouse buttons.
5. Press F8.
6. Keep the mouse still during the first two seconds.
7. After one beep, make small horizontal movements in both directions.
8. Pause briefly between movements.
9. After two beeps, make small vertical movements in both directions.
10. Pause briefly between movements.
11. After three beeps, stop moving the mouse.

The recorder ends the protocol approximately 38 seconds after enabling.
Do not move, jump, attack, or change weapons during this check.
Use short movements with stops instead of one constant pan.
Avoid large camera rotations and pitch limits.
F8 and Esc pause recording.
F9 ends recording.
A new recording segment restarts the cue sequence.
The overall 45-second limit still includes pauses.
An interrupted protocol can therefore end before all phases finish.

The first beep marks horizontal capture after a two-second preparation interval.
The two-beep cue marks vertical capture at approximately 20 seconds.
The final three-beep cue marks completion at approximately 38 seconds.
Cue failures appear in the private session summary.
Do not treat a failed or misunderstood cue sequence as a completed timing experiment.

## Analyze the result

After recording ends, run this command:

```powershell
.local\windows-venv\Scripts\python -m vimm8bdm.alignment `
  --run-dir runs/mouse-timing/session --output runs/mouse-timing/alignment.json
```

Use a new output filename.
The checker first verifies archive integrity.
It analyzes grayscale copies at 320 by 240 pixels.
It measures translations in an upper central image region.
It rejects uncertain matches, segment crossings, held keys, and intervals near key events.
Guided recordings also restrict each axis to its requested phase.
The phase filter excludes the first 600 milliseconds and the final 150 milliseconds of each phase.

The default image interval is at least 80 milliseconds.
Samples do not overlap within a segment.
This interval reduces the effect of repeated rendered images at a higher capture rate.
Actual image times use the midpoint of each capture interval.
The analysis does not claim that the midpoint is the exact display sampling time.
Use `--window-ms` to inspect interval sensitivity in a separate output file.
Do not select an interval merely because it gives a preferred answer.

The fitter integrates raw mouse events over candidate shifted intervals.
It searches delays from -100 through +150 milliseconds in five-millisecond steps.
A positive delay means that recorded input precedes the observed image response.
A negative estimate can reflect callback scheduling or a failed response model.
The fitter expects opposite scene and device movement under the current mouse settings.
Inverted settings require a separate direction check.

The first chronological portion selects the delay and image scale.
A later portion checks the result with the scale unchanged.
A 300-millisecond gap separates fitting and validation samples.
The validation data does not select the delay.
Both portions need positive and negative motion samples.
The report rejects insufficient samples, weak validation, broad delay ranges, and search-boundary results.
The near-best range is a search diagnostic rather than a statistical confidence interval.
The five-millisecond grid does not establish five-millisecond measurement precision.

## Keep label time separate from response time

The report estimates image response to input receipt.
It does not measure the physical hardware event, exact OS processing, or human reaction time.
Camera rotation is only approximately a translation in the selected image region.
Animation, scene geometry, renderer timing, and capture timing can affect the result.
A candidate needs repetition before use as a calibration.

The analysis does not change images or input timestamps.
Do not move an action label onto a later image that already contains its effect.
Behavior-cloning labels need a causal observation and a defined future action interval.
Retain image history and the original events for later dataset construction.
Physical mouse counts also require a separate comparison with injected actuator counts.

## First mixed recording

The earlier recording contains movement, jump, attack, and mouse motion.
After interference rejection, adjacent-image analysis retained 32 horizontal and 21 vertical active samples.
The 80-millisecond analysis retained 12 horizontal and eight vertical active samples.
Both analyses reported insufficient data.
Neither analysis selected a response delay or changed training labels.
The first guided mouse-only run later produced candidate estimates.
See the live result below.


## Automated checks and current limits

Linux and Windows each passed 81 automated tests.
Synthetic data checks known positive and negative delays, constant panning, unrelated image changes, and failed validation.
Archive tests cover phase guards and large capture gaps.
These checks do not establish the stock client's response delay.
The first guided run passed the integrity and fitting checks.
Repetition remains necessary before accepting a calibration.
See the [sanitized alignment record](evidence/input-alignment.json).


## First guided attempt

The first guided run stopped after eight frames because the writer queue filled during startup.
The run did not reach the horizontal phase or call a phase cue.
It supplied no response-delay measurement.
The recorder now waits for encoder readiness before accepting recording commands.
The updated writer passed isolated Windows throughput checks.
The live retry completed all phases with no dropped frames or cue errors.
The operator reported audible game sound and phase cues.
See the [startup correction record](evidence/recorder-startup.json).

The earlier local launcher disabled game sound and music.
The launcher now enables client audio by default.
Use `--mute` only when a quiet client test is necessary.
The audio restart preserved a separate copy of the saved configuration.
The stock client initialized FMOD after the restart.
Successful audio initialization or a cue API return does not prove audible output.
The operator must check the actual sound.


## First complete guided run

The retry completed all phases and saved 2238 frames and 1383 events.
It reported no queue drops, cleanup errors, or cue errors.
Maximum frame queue wait was 1.37 milliseconds.
Maximum frame write duration was 9.40 milliseconds.
The operator reported both audio paths working and both movement phases complete.

| Axis | Active image pairs | Candidate delay | Near-best training range | Validation R-squared |
| --- | --- | --- | --- | --- |
| Horizontal | 141 | 5 milliseconds | 5 to 10 milliseconds | 0.978 |
| Vertical | 135 | 10 milliseconds | 10 milliseconds on the tested grid | 0.908 |

The horizontal validation error was 1.46 pixels RMS.
The vertical validation error was 2.37 pixels RMS.
The first run passed the candidate criteria on both axes.
These estimates describe receipt-to-image response under this setup.
They do not establish exact hardware latency or a universal fixed offset.
The range is not a statistical confidence interval.
The analysis preserved the original image and event timestamps.
A second recording under the same conditions remains pending.
