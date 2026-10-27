# Demo files and training data

Use synchronized image and input recording for the first behavior-cloning dataset.
Keep the user's existing demos for a later extraction experiment.
Milestone 0 does not implement a demo parser.
The [Windows recorder](human-recording.md) now implements the initial image-and-input collection path.
Live validation and dataset preparation remain separate steps.

## What the source establishes

Zandronum has different recording paths for local demos and client network demos.
Check the file signature before selecting a parser.
A filename extension alone does not identify the exact format.

The examined client-demo source writes yaw, roll, pitch, buttons, vertical movement, forward movement, and sideways movement.
It also records server data and local commands.
This establishes recoverable game commands in that source version.
It does not establish the original physical keyboard keys or mouse counts.
See the [public source mirror](https://github.com/doomtech/zandronum/blob/master/src/cl_demo.cpp).

The local comparison source is a separate 3.3 development tree.
The audit read that tree without changing it.
It is not the tested stock 3.2.1 binary's verified source revision.
Its source digest appears in the experiment evidence.
Check the exact recorder version before implementing a parser.

| Requested information | Evidence and limit |
| --- | --- |
| Movement | The examined tic command contains forward, sideways, and vertical movement values. Bindings and aliases cannot generally be recovered. |
| Turning | The examined command contains yaw and pitch. These are engine values, not raw mouse deltas. |
| Primary attack | The examined command contains a buttons field. Decode its bit definitions for the exact version. |
| Weapon switching | Do not assume that it appears in the movement command. Check weapon messages and local commands in the exact replay protocol. |
| Class skills | Do not assume complete recovery from one byte of buttons. Check alternate attack, reload, extra buttons, inventory, and aliases separately. |
| Corresponding images | Replay can render the recorded player's view if the correct engine and assets are available. Camera settings can differ. |
| Hidden state for training | A training tool could recover additional state. This remains outside the final deployment observation contract. |

## Version dependence

The examined header stores an engine version, recording version, random seed, asset names, and authentication checksums.
The local source rejects a different engine revision string.
Other versions can enforce different checks.
Do not assume cross-version playback.

Keep the following information with each demo:

1. Record the exact client version and executable digest.
2. Record the IWAD and ordered MOD list.
3. Record a digest for each asset.
4. Record the map, character, server settings, and player perspective.
5. Preserve the original recording file.

The project found all requested target packages locally.
Their digests identify the supplied copies.
The digests do not independently establish upstream authenticity.
The project did not test a user demo or demo playback.

## Pair images with actions

A replay can support behavior cloning in principle.
The extractor must pair the displayed image before an action with the command applied after that image.
It must account for game ticks, client prediction, replay timing, and camera changes.
Do not treat the replay container as a uniform sequence of movement records.
Client demos also contain variable-length protocol data.

Use only the recorded player's commands as action labels.
Do not use another player's actions as labels for the current image.
Do not add hidden opponent positions to a deployment observation.

Demo yaw and pitch require a separate conversion to the external action contract.
Bindings, mouse sensitivity, acceleration, filtering, and SR50 aliases can change that relationship.
Store semantic actions when possible.
Do not claim recovery of the user's original device events.

## First recorder

Run capture and input recording on the game's operating system.
Record raw input events and capture timestamps on the same monotonic clock.
Preserve press and release events rather than only the key state at frame boundaries.
Record relative mouse motion where the OS permits it.
Test pointer recentering before using absolute cursor differences as aim labels.

For each image, store these fields:

| Field | Purpose |
| --- | --- |
| Session and frame IDs | Identify a complete recording and its images |
| Capture start and end | Measure image timing and capture duration |
| Input event timestamps | Align key, button, wheel, and mouse events |
| Held input state | Reconstruct actions over each policy interval |
| Input source | Distinguish human events from AI-generated events |
| Character and bindings | Interpret class actions consistently |
| Client and asset digests | Reproduce mechanisms and content |
| Render and network settings | Explain visual and timing differences |
| Takeover and menu events | Exclude invalid training intervals |

Keep original event timing when resampling data.
Measure timestamp skew rather than assuming one frame equals one game tick.
Start with short, controlled sessions.
Check image/action alignment through playback before collecting hours of data.

## Behavior-cloning sequence

1. Record movement and turning with one character.
2. Record aim and primary attack in a local combat scenario.
3. Add jump, weapon changes, and resource decisions.
4. Add alternate attack and one class skill at a time.
5. Test on maps and sessions excluded from training.

Use binary outputs for independent held buttons.
Use continuous outputs for mouse motion.
Use categorical outputs for weapon events where appropriate.
Measure idle, attack, and movement frequencies before selecting loss weights.
Do not let idle frames dominate the model without a task-based check.

Evaluate closed-loop behavior after offline action prediction.
A low prediction error does not prove stable gameplay.
Use RL only after the policy and action timing work in the actual client.
