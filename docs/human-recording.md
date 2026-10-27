# Human gameplay recording

The Windows recorder receives human input and captures the selected game window.
It sends no keyboard or mouse input.
Stop AI controllers before starting a human recording.

## Install and start

Use Windows Python on the same desktop as the official client.
Install the recording packages in the project environment:

```powershell
.local\windows-venv\Scripts\python -m pip install -e ".[record]"
```

Use the normal window selector to obtain the current game window ID.
Run this command from the repository root:

```powershell
.local\windows-venv\Scripts\python -m vimm8bdm.recording `
  --window-id 0xYOUR_WINDOW_ID --run-dir runs/human-check/session `
  --seconds 30 --hz 20 --character grenademan --class-selection random `
  --map MM1CUT --engine-version 3.3-alpha-r260925-2352
```

The session directory must not exist before startup.
The initial waiting phase has no timeout.
The duration limit starts at the first F8 command and includes later pauses.

Use `--launch-record` with a local startup record when available.
The recorder then hashes the executable, IWAD, and three MOD files.
It copies the selected configuration files into the private session directory.
Those files can contain private paths.
The copies do not prove that every setting matches the client's current in-memory value.

## Record a short check

1. Wait for the ready message.
2. Select the game window.
3. Close the menu.
4. Release gameplay keys and mouse buttons.
5. Press F8 to record.
6. Move forward and release the key.
7. Move sideways and jump.
8. Turn horizontally and vertically with the real mouse.
9. Press and release primary attack.
10. Change weapons if the selected character supports weapon selection.
11. Wait for the automatic recording end.

For a character with one weapon, omit weapon selection and mark that check as not exercised.

F8 pauses or starts a new recording segment.
Esc pauses recording and remains available to the game menu.
F9 ends the session.
The recorder never releases human-held inputs when recording stops.
The player remains under human control.

Focus loss pauses recording and ends the current segment.
Returning to the game does not resume recording.
Use F8 to start another segment after returning.
The recorder discards frames that cross a detected pause or segment boundary.
The recorder checks focus before and after image capture.
Close unrelated overlays before recording.

Pause before chat, menus, death, or a change of character.
Only Esc and focus loss currently provide automatic pause conditions.
The recorder does not detect death, chat, or every menu from images.

## Character identity

`character` names the actual character reported by the operator at session start.
`class_selection` records whether the game uses fixed or random selection.
These fields describe different facts.
A random selection label does not identify the active character.

The first check uses the operator's Grenade Man report and random selection.
The session does not guarantee that this character persists after respawn.
Use a fixed character for the first training dataset.
Record later character changes as new sessions until explicit character markers exist.

## Saved files

| File | Content |
| --- | --- |
| `session.json` | Version, map, character report, selection mode, capture target, and available asset digests |
| `events.jsonl` | Segment boundaries, key transitions, relative mouse motion, buttons, and wheel events |
| `frames.vrec` | Lossless RGB PNG images in a sequential binary archive |
| `frames.jsonl` | Image offsets, dimensions, capture intervals, event sequence boundaries, and starting held inputs |
| `summary.json` | Saved counts, queue losses, cleanup errors, and completion status |

The archive starts with `VIMMREC1` and a newline.
Each image has a four-byte little-endian length followed by PNG bytes.
The writer uses a limited queue.
A full queue stops recording instead of silently dropping training data.
An incomplete session must not enter a training dataset.

All current sessions have `training_ready: false`.
The first version collects interface-check data.
It does not implement a behavior-cloning dataset builder or training.
Keep recordings, previews, and configuration copies in the excluded `runs/` or `data/` directory.
The public file check rejects `.vrec` archives and staged files from those directories.

## Input and timing

The observer registers mouse and keyboard [Windows Raw Input](https://learn.microsoft.com/en-us/windows/win32/inputdev/about-raw-input).
A separate message window receives the input without replacing the game's registrations.
The observer does not request exclusive input or disable legacy game input.
The [registration API](https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-registerrawinputdevices) documents the registration scope within each process.

Mouse labels use [relative device counts](https://learn.microsoft.com/en-us/windows/win32/api/winuser/ns-winuser-rawmouse).
Absolute mouse devices pause the session because their labels require another conversion.
The observer does not estimate motion from cursor positions or game state.
The equivalence between physical Raw Input counts and injected actuator counts remains unverified.
Keyboard records include virtual keys, scan codes, and extended-key flags.
Repeated key-down messages for the same virtual key, scan code, and extended flag do not add another transition.
Left and right modifiers can retain separate transitions despite a shared virtual key.
Recording starts with an OS snapshot of held inputs.

Frames and events use `perf_counter_ns` in one process.
Input timestamps identify callback receipt, rather than the physical hardware event time.
Image records preserve capture start and completion times.
The first format retains raw events before selecting a training interval.
Do not pair an earlier action with an image captured after that action and claim a causal observation.
Do not join samples across pauses or character changes.

## Review a session

Run the checker after the recorder exits:

```powershell
.local\windows-venv\Scripts\python -m vimm8bdm.recording_review `
  --run-dir runs/human-check/session --preview runs/human-check/preview.html
```

Use a new preview filename.
The checker verifies record counts, event sequence continuity, image offsets, PNG integrity, and capture intervals.
The preview contains at most 120 sampled images and adjacent input events.
Open the local HTML file in a browser.
Move its slider to compare actions with the images.
The preview selects events after capture completion until the next frame in the same segment.
This interval is a review aid, not a verified training-label conversion.
The checker does not execute input or reconstruct hidden game state.


## First live result

The stock Windows alpha session saved 416 images and 564 events in one segment.
The image span was 20.88 seconds.
The operator ended recording with F9 before the 30-second duration limit.
The writer closed with no dropped items or cleanup errors.
The archive checker verified all images, offsets, record counts, and event sequence numbers.

The event stream contains 48 key transitions, 505 relative-motion events, and eight mouse-button transitions.
WASD and Space each have matching press and release counts.
Primary attack has three press/release pairs.
The right mouse button has one press/release pair.
The initial and final recorded held-input sets are empty.
Weapon selection was not exercised because the operator reported one available weapon for Grenade Man.

The median capture duration was 7.27 milliseconds.
The maximum interval between successive captures in the segment was 51.03 milliseconds.
The inspected first, middle, and final images contain game content and changed views.
Selected mouse-motion comparisons had consistent direction but different displacement magnitudes.
This check does not establish precise action alignment or device-to-actuator conversion.

The session used the reported character Grenade Man with random selection enabled.
It remains interface-check data with `training_ready: false`.
Active recording did not include a focus-loss or Esc-pause test.
Automated tests cover those boundaries.
Linux and Windows each passed 73 tests after the final modifier-state correction.
The live session exercised WASD and Space rather than paired modifier keys.
See the [sanitized recording record](evidence/human-recording.json).

Use the [mouse alignment protocol](input-alignment.md) before selecting a response delay.
It adds sound cues and phase markers to the existing recorder.


## Writer startup correction

The first 60 Hz timing attempt stopped before its first phase cue.
The writer queue filled after eight frames while the encoder was still initializing.
The session has `complete: false` and cannot supply alignment data.
A separate replay reproduced the same eight-frame overflow with the earlier writer.

The recorder now waits for Pillow, PNG encoder warm-up, and archive creation before reporting readiness.
The frame queue retains its existing limit.
A new regression test checks delayed writer initialization.
Writer-only Windows replay tests then saved 480 and 240 frames at a target rate of 60 Hz.
Both tests reported zero dropped items.
The cold-import test waited approximately 347 milliseconds before becoming ready.
These writer tests do not replace the live capture retry.

The summary now reports maximum frame queue wait and maximum frame write duration.
Linux and Windows each passed 82 automated tests.
See the [startup correction record](evidence/recorder-startup.json).

The subsequent live timing run saved 2238 frames with no dropped items or cue errors.
The operator reported that game audio and the cues were audible.
The response estimates remain candidates pending repetition.
