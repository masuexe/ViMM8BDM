# Roadmap

## Milestone 0 status

The source audit and route decision are complete.
The target assets are available.
The stock 3.2.1 client loaded the complete target stack.
Direct window capture returned game images.
ContinuousDoom passed strict loading and CPU inference without ViZDoom.
The visual control interface passed 12 offline tests.

The live input experiment did not pass the focus condition.
It sent no input.
M1 remains incomplete.
See the [M1 progress record](milestone-1-progress.md) for later window and movement tests.
The experiment report separates tested functions from source-supported functions.

## Milestone 1: establish reliable local control

The estimate is three to seven engineering days after stable desktop access.
This estimate excludes combat training and unresolved upstream permissions.

### Step 1: select and identify the deployment host

Use the operating system that will run the official client.
Use official `3.3-alpha-r260925-2352` for primary acceptance tests.
Keep the 3.2.1 results as a separate baseline.
The normal-play environment determines the primary revision rather than the stable or alpha label alone.
For Windows, use Windows Python for capture and input.
For Linux, use the tested X11 path first.

Record the executable digest and target asset digests.
Use a separate project configuration.
Disable unrelated autoload packages for the test.
Preserve this load order:

1. Use `megagame.wad` as the IWAD.
2. Load `MM8BDM-v6a.pk3`.
3. Load `classes-v9fh.pk3`.
4. Load `gondolapatchv2.7h2.pk3`.

Do not substitute v6b or v6c during target acceptance tests.
If a separate v6b comparison becomes necessary, use the required v6b fixes for that comparison.

Acceptance: the stock client loads the exact stack without startup errors.
This passed once on the local Linux 3.2.1 client.
Preserve the startup record for repetition.

### Step 2: resolve focus and capture behavior

Add a window selector that identifies the actual client window.
Do not select the startup dialog or a window decoration.
Check the active window on each action interval.
Display the selected window ID and capture size before enabling input.

Test WSLg activation after the user selects the window.
Determine whether the compositor changes or clears the X11 focus ID.
Do not remove the focus condition to make a test pass.

For X11, use window capture if MSS returns a black root image.
For Windows, test a normal window-capture method.
Check blank frames, client resize, minimization, and lost windows.
Stop input if the capture is invalid.

Acceptance: capture 60 seconds of visible gameplay with stable dimensions and valid timestamps.
Check the captured images against the visible game.
Do not infer success from a nonempty pixel buffer alone.

### Step 3: check normal input with a rule controller

Use a local scenario that permits a short control test.
Reset the local map through normal game controls when necessary.
Do not connect to a public server for this step.

Test each action separately:

1. Move forward and stop.
2. Move sideways and stop.
3. Turn horizontally in both directions.
4. Aim vertically in both directions.
5. Attack and release the attack button.
6. Jump and change weapons.

Check movement and aim through displayed images.
Do not read hidden position or angle values.
Test short mouse deltas before continuous aim.
Measure the relationship between mouse counts and visible camera movement.

Acceptance: every action starts and stops as commanded.
The player must not continue moving or firing after a stop command.
The operator observed movement, both aim axes, attack, and jump in the Windows alpha test.
The [short mouse test](mouse-calibration.md) measured local image displacement on both axes.
Weapon selection and conversion from model actions to view angles remain unverified.
See the [M1 progress record](milestone-1-progress.md) for the test evidence.

### Step 4: check human takeover

Keep the controller paused at startup.
Test F8 pause and F9 exit while movement and attack are active.
Test focus loss during a held action.
Test stale observations and a stalled policy.
Test listener failure and normal process exit.

Add an independent input process before long sessions.
Give its input lease an explicit expiry time.
Use an independent user pause signal.
Check recovery after forced termination of the inference process.

Acceptance: the input process releases owned inputs within the measured deadline.
Set an initial target of 250 ms for focus loss and heartbeat expiry.
Record the observed maximum, not only the mean.
The independent-worker results appear in the [watchdog report](input-watchdog.md).
They cover inference failure and do not establish recovery after an input-worker failure.
The [focus test](focus-check.md) records held-input release and paused behavior after focus returns.

### Step 5: compare model and rule policies

The [Windows model test](model-control.md) passed the image-to-inference-to-input interface check.
The model selected attack throughout the short run.
Target-game task performance remains unverified.

Use the tested ContinuousDoom checkpoint only as a research baseline until its reuse terms are clear.
Keep strict loading enabled.
Record preprocessing, action scale, and policy interval.
Compare repeated runs against the rule probe in the same local scenario.

ContinuousDoom cannot supply vertical aim or class skills.
Keep the capability test separate from the full gameplay acceptance test.
If the checkpoint repeatedly produces unsuitable actions, start target-game BC.
Do not extend the old model merely to preserve its checkpoint.

Acceptance: the model consumes live images and produces bounded inputs for 60 seconds.
Record pauses, deaths, invalid images, and action counts.
This does not require a win rate.

### Step 6: measure latency and stability

Record capture, preprocessing, inference, input-call, and complete loop durations separately.
Use monotonic timestamps.
Exclude model loading from steady-state inference statistics.
Report warm-up separately.
Report sample count, median, 95th percentile, and maximum.

Measure visible response delay separately from input-call duration.
The return of an OS input call does not prove that the game processed the event.
Use a video or image sequence for the response measurement.

Set an initial 20 Hz control target for the new policy.
Keep ContinuousDoom near its original 8.75 Hz interval for the baseline comparison.
At 20 Hz, aim for a loop 95th percentile below 50 ms.
These values are engineering targets rather than measured guarantees.

Acceptance: complete three local sessions without stuck inputs or uncontrolled actions.
Include at least one takeover and one focus-loss event per session.
Keep the reports and per-frame logs.

## Milestone 2: collect target-game demonstrations

The [first Windows recorder](human-recording.md) passed a short image-and-input collection check.
The session remains interface-check data.
Check the raw mouse units and label timing before building training samples.
Start with one character and two maps.
Check press/release timing and relative mouse labels.
Preserve character, skill bindings, and asset digests.
Test one supplied demo against its exact engine and MOD versions.
Implement a demo parser only if that test demonstrates useful recoverable labels.

The [causal sample builder](causal-dataset.md) passed diagnostic archive and cross-platform loading checks.
Human review and physical-to-actuator conversion remain necessary before training.

Acceptance: replay a short recording with aligned images and actions.
Create separate training and evaluation sessions.

## Milestone 3: train and improve the policy

Train behavior cloning before RL.
Measure navigation, aim, attack timing, and takeover frequency separately.
Add jump and one class skill at a time.
Use actual character descriptions and read-only asset evidence for skill metadata.
Do not guess Gondola mechanisms from base-game behavior.

Add local bot matches after the policy can operate reliably.
Reuse SF components when RL requires them.
Use only observations that the final policy can obtain.
Evaluate each character on maps excluded from training.

Acceptance: meet task-specific movement and combat thresholds defined before each experiment.
Do not select thresholds after observing results.

## Milestone 4: evaluate multiplayer play

Start with controlled bot matches and separate agent instances.
Add self-play when independent instances and resets work reliably.
Test several maps, characters, frame rates, and network delays.
Distinguish prediction error, policy error, and network delay in the reports.

Use human servers only when their operators permit AI players.
The client must remain official and unmodified.
The MOD files must remain unchanged.
The server must not need an additional component.
The user must retain the pause control.

Acceptance: a recorded permitted-server session satisfies all final deployment conditions.
M0 does not establish this result.

## Resources still required

The target game assets are already available locally.
A compatible user demo remains necessary for the demo feasibility test.
ContinuousDoom reuse terms remain unresolved.
A stable foreground-window test remains necessary for real input validation.
The final Windows host needs a separate test if Windows will run the client.
