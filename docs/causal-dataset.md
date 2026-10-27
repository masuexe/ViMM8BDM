# Causal behavior-cloning samples

The dataset builder pairs image history with future human input.
It preserves original images, timestamps, and input events.
It does not train a model or execute game input.

## Sample contract

Each sample contains four distinct images and a future action window.
The decision anchors follow a contiguous 50-millisecond grid within each recording segment.
The history targets are 150, 100, 50, and zero milliseconds before the anchor.
The builder selects nearby images in chronological order.
Every selected image must finish capture at or before the decision anchor.
It never uses a future image as an observation.

Capture timing can differ slightly from the target grid.
The history retains actual timestamps and signed target offsets.
The newest image age also appears in the sample.
The builder permits at most 75 milliseconds of deviation from each history target.
It rejects capture gaps above 100 milliseconds within the history and label coverage.
It does not duplicate an image or insert a blank image to fill a missing history.

The action interval starts at the anchor and ends 50 milliseconds later.
The interval includes its start and excludes its end.
An event at the end belongs to the next window.
Future image timestamps provide a coverage check only.
The future image pixels never enter the observation.

| Action field | Meaning |
| --- | --- |
| `keys_at_start` / `buttons_at_start` | Inputs held before events at the anchor |
| `key_held_fraction` | Fraction of the window that each virtual key remains held |
| `button_held_fraction` | Fraction of the window that each mouse button remains held |
| `mouse_dx_raw` / `mouse_dy_raw` | Sum of recorded relative mouse counts |
| `wheel_vertical` / `wheel_horizontal` | Sum of wheel deltas |
| `events` | Exact transitions and motion with offsets from the anchor |
| `keys_at_end` / `buttons_at_end` | Held state after events before the interval end |

A short tap remains present even if its start and end held states are empty.
Modifier transitions retain their scan codes and extended-key flags.
The builder does not clip mouse labels to the current controller's 24-count limit.
It does not round labels or apply the estimated image-response delay.
Raw device counts still require an actuator conversion before model deployment.

## Boundaries and eligibility

The builder handles each recording segment separately.
It rejects incomplete future windows, incomplete image coverage, stale histories, and excessive capture gaps.
It rejects segments with missing boundaries or unknown event markers, including character-change markers.
The source must report one known character with fixed selection.
The source archive must pass the recording integrity check.

These checks do not detect unmarked death, chat, menus, or character changes from images.
Pause before chat or menus.
End the session before a character or map change.
Pausing alone does not update the declared character or map.
A later human review must check the actual gameplay content.
The builder cannot prove stable character identity from a metadata declaration alone.

Interface and calibration sessions require `--allow-interface-check`.
They can enter only the `diagnostic` split.
Guided protocol events also force diagnostic handling if metadata changes.
The builder never marks calibration clips as training demonstrations.

Every output currently has `training_ready: false`.
It lists human review and actuator conversion as remaining requirements.
A valid slice is a candidate training sample, rather than proof of task quality.

## Build diagnostic samples

Run this command from the repository root in PowerShell:

```powershell
.local\windows-venv\Scripts\python -m vimm8bdm.dataset `
  --session runs/mouse-timing/session --output data/timing-slices `
  --split diagnostic --allow-interface-check
```

Use a new output directory.
The builder creates `dataset.json` and `samples.jsonl`.
It references the source archive rather than copying game images.
Keep the source and output directories together when moving the dataset.
The source reference is relative to the output directory.
Keep the source and output on the same filesystem for portable relative references.

The manifest records checksums for the source manifest, summary, event stream, image index, and archive.
Check those checksums before later training or export:

```powershell
.local\windows-venv\Scripts\python -m vimm8bdm.dataset `
  --verify-source data/timing-slices
```

The Python `load_history` helper loads referenced RGB images and rejects future-image references.
Checksum verification is a separate session check, rather than a full archive hash for every sample.

## Record task demonstrations later

Use the normal recorder with explicit demonstration metadata:

```powershell
.local\windows-venv\Scripts\python -m vimm8bdm.recording `
  --window-id 0xYOUR_WINDOW_ID --run-dir data/navigation-session `
  --purpose demonstration --task navigation --protocol free `
  --character grenademan --class-selection fixed --map MM1CUT `
  --engine-version 3.3-alpha-r260925-2352 --hz 20 --seconds 120
```

The recorder requires a fixed known character, known map, task, and free protocol for this purpose.
The source still needs human review.
Use actual navigation or combat actions rather than timing-test camera movements.
Start with short sessions before collecting large datasets.

A later demonstration build can use `--split train`, `validation`, or `test`.
Assign whole sessions to one split.
Do not randomly divide adjacent frames between training and evaluation.
Check all dataset manifests together before training:

```powershell
.local\windows-venv\Scripts\python -m vimm8bdm.dataset `
  --check-splits data/navigation-train data/navigation-validation
```

The check rejects a shared source session or identical image archive across different splits.
Changes to task metadata do not create a new session identity.
Keep source recordings, labels, and previews in excluded local directories.

## Verification status

Tests cover short taps, exact interval boundaries, held-state carry, paired modifiers, and capture jitter.
They also cover future-image rejection, pause boundaries, character markers, gaps, and partial tails.
Archive tests check image loading, source mutation, and session overlap between splits.
Live timing recordings serve only as diagnostic build inputs.
The builder does not establish a behavior-cloning policy or a physical-to-injected input conversion.


## First real builds

Each of the two complete timing recordings produced 756 diagnostic samples.
The builder retained all 1376 motion events in the first recording and all 698 in the repeat.
The omitted tail lacked a complete future window or image coverage.
Neither build entered a training or evaluation split.

The audit checked every sample for causal images and valid event offsets.
The resulting future windows were contiguous in both recordings.
Windows loaded first, middle, and final histories from each Linux-built dataset.
Linux also loaded a Windows-built dataset.
Source checksum checks passed in both cases.
The largest newest-image age in the first audited dataset was 20.39 milliseconds.
Actual timestamps remain available for later model preprocessing.

Linux and Windows each passed 92 automated tests.
See the [sanitized dataset record](evidence/causal-dataset.json).
These results establish slice construction and image loading, rather than learned gameplay.

Use the [action audit](action-mapping.md) to identify labels that the current executor cannot represent.
An offline candidate does not certify game bindings or mouse-unit equivalence.

## Reviewed image exclusions

Use `--exclude-frames` when a reviewed overlay obscures part of a recording.
Supply a private JSON file with this structure:

```json
{
  "source_session_id": "copy the exact source identity from the original dataset manifest",
  "reason": "Reviewed desktop notification",
  "frame_ranges": [[100, 130]]
}
```

Frame indices start at zero, and both range endpoints are inclusive.
Ranges must be ordered, disjoint, and inside the source archive.
The source identity must match the recording.
The builder rejects samples whose history-to-action interval overlaps an excluded capture interval.
This rule covers both observed images and the following action window.
Use reviewed margins when an overlay moves or fades near its visible boundary.
The tool does not identify notifications automatically.

Build a new dataset directory instead of changing the original:

```bash
python -m vimm8bdm.dataset \
  --session data/corner-recording --output data/corner-clean-slices \
  --split train --exclude-frames runs/corner-review/frame-exclusions.local.json
```

The new manifest embeds the exclusion record and reports the excluded sample count.
Retained samples preserve their original identifiers, labels, and timestamps.
The builder does not join the intervals on opposite sides of an exclusion.
The original recording remains unchanged.
The training loader reconstructs the exclusions before comparing the stored samples with the source.
Use only the reviewed derived dataset for that experiment.
Do not combine the full dataset and its filtered copy as independent training examples.

The [first navigation demonstration](navigation-demo.md) also produced 2005 causal candidate slices.
Its action audit exposes fast-turn and within-window timing limits in the current executor.
