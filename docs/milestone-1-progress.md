# Milestone 1 progress

## Window selection and focus checks

The CLI now lists visible game windows without sending input.
The default list omits window titles.
It excludes client areas smaller than 160 by 120 pixels.
It can identify game windows from their class, title, or shared process ID.
Use the complete local list when those game identifiers are absent.

The focus check accepts the selected client or a child window.
It rejects unrelated windows and unknown focus values.
It rejects hidden or small client windows.
It also rejects a changed process ID or window class.
The process check uses metadata from the operating system, not game memory.

The Windows source uses the parent relationship rather than the popup-owner relationship.
This prevents an owned popup from matching solely through its owner.
No native Windows test ran.

## Local checks

The expanded offline suite passed 25 tests before the live attempt.
The tests cover child focus, unrelated focus, cycles, hidden windows, and changed process identity.
The earlier 12 control tests remain in the suite.
Their historical M0 evidence remains unchanged.

A native X11 test created an unmapped diagnostic window and one child.
It checked dimensions, process metadata, and the actual parent relationship.
The CLI focus probe sampled that window twice.
It rejected input because the window was not visible.
The test sent no input and did not change desktop focus.

A stock Linux client then loaded the target packages through the new local startup script.
The window list found one visible game client at 640 by 480 pixels.
It obtained the matching process ID.
The startup script used a new run directory and a separate configuration.
It did not join a remote server.

## Publication boundaries

Public evidence omits machine paths, account names, process IDs, window IDs, and raw desktop titles.
Local records remain under the excluded `runs/` directory.
The public M0 evidence retains package versions, artifact digests, and measured results.
The publication check examines the staged file set before a commit.

## Remaining acceptance checks

Window discovery does not establish reliable game control.
The first M1 live test sent 77 movement and horizontal-aim actions through the OS input receiver.
The operator reported visible automatic forward movement.
The log contains three enable/pause cycles.
This establishes a basic movement test, not complete combat control.
No accepted action in that sample contained vertical aim or attack.
It must also check real focus loss during held movement and attack.
The accepted input calls took 1.52 ms at the median and 4.26 ms at the 95th percentile.
These durations do not measure visible response delay.
The previous M0 live attempt remains a failed focus test with no input.
Do not treat the new offline tests as a replacement for that live test.

An independent input process remains necessary before long sessions.
Accurate mouse calibration, menu detection, and Windows live behavior also remain unverified.


## Hotkey correction

The operator also reported the game message toggle after controller input stopped.
The old test configuration retained the game's F8 binding.
The old controller also treated Esc as process exit.
The trace ended after approximately 31.89 seconds, before the 55-second limit.
The old trace does not establish the exact exit cause.

The controller now uses Esc for pause and F9 for exit.
It reports frame-limit, duration-limit, key-exit, and interrupt reasons.
The separate local configuration removes the game's F8 and F9 bindings.
This does not change an existing installation configuration.
The controller also explains that F8 cannot control a stopped process.

The expanded suite passed 30 offline tests after this correction.
The new tests check Escape pause, key repeat, exit, and blocked focus.
At this stage, live regression of the corrected keys remained a separate check.


A separate stock-client startup checked the new configuration.
The saved game configuration contained no F8 or F9 binding.
That test sent no control input.
The test client stopped after the check.
At this stage, live testing of Escape pause and F8 re-enabling remained unverified.

The [sanitized M1 evidence](evidence/milestone-1.json) records these results.
The [current offline test log](evidence/milestone-1-tests.txt) records the 32-test suite.

The final suite passed 32 tests, including checks against the actual Git index contents.

## Later input and runtime tests

The separate input-check policy exercises each control in a short sequence.
It checks movement, both aim axes, attack, release, jump, and a weapon key.
It pauses after the sequence.
A new enable command starts a new sequence.
Black images and changed client dimensions block input.

The 3.2.1 loopback test accepted 268 actions across two sequences.
The operator reported visible movement, jump, camera movement, and attack.
The operator also reported that control could be enabled again after a pause.
See the [baseline input evidence](evidence/input-check-321.json).

The Linux alpha test accepted 154 actions.
The log records Escape pause followed by re-enabling.
The operator reported camera flicker with automatic and physical mouse movement.
The physical-mouse symptom continued after capture and control stopped.
The project did not establish its exact cause.
Aim acceptance failed for this environment.
See the [Linux alpha evidence](evidence/input-check-alpha.json).

The project then tested official Windows alpha with Windows native Python.
The operator reported normal physical and automatic mouse behavior without flicker.
The sequence completed with 119 accepted actions.
The Windows log also records Escape pause without early process exit.
See the [Windows alpha evidence](evidence/input-check-windows-alpha.json).

The primary runtime now uses Windows native capture and relative `SendInput` movement.
The exact engine revision is `3.3-alpha-r260925-2352`.
The [engine report](engine-and-mouse.md) explains this decision and its limits.

The first Windows trace used a clock with approximately 16 ms steps.
Its zero-duration samples do not establish zero latency.
The code now uses `perf_counter_ns` for capture, action age, and stage timing.
Windows reports a 100 ns counter resolution in the local check.
This is a clock property, not measured visible response latency.
The summary separates accepted-action timing from inactive or rejected frames.

The current Linux and Windows suites each passed 42 tests.
Those runs did not test independent watchdog recovery.
Calibration and long-session acceptance remained incomplete.

## Independent input ownership

The input worker now owns desktop input, focus checks, hotkeys, and the action lease.
The capture and policy process cannot enable live input through its command pipe.
Generation checks reject results computed before a user pause and re-enabling.

Real process tests cover a stalled or forcibly stopped inference parent.
The stock Windows alpha test released input during a 750 ms parent stall.
Release calls completed 217.35 ms after the last accepted action in that run.
The operator reported that input stopped and could be enabled again.

A later desktop test exited the parent without normal cleanup.
The worker completed release calls 12.77 ms after the recorded fault marker and then exited.
Its release record contains no held keys or attack button.
The operator observed one automatic primary attack.
No visible movement-stop latency measurement occurred.

The first exit attempt sent no input and therefore supplied no release evidence.
The controller now waits separately for initial user enabling before starting the run budget.
This change also prevents idle waiting from consuming the frame budget.

See the [input watchdog report](input-watchdog.md) for evidence and failure limits.
Calibrated aim, held-input focus-loss timing, and long-session acceptance remain incomplete.


## Visual mouse measurement

The stock Windows alpha short test completed 24 small pulses on both axes.
The image matcher accepted all 24 pulse measurements.
All 529 captured frames produced accepted actions over 26.62 seconds.
The operator observed small horizontal and vertical view movements.
The worker released input and closed with no dropped audit events.

The local median ratios were -1.0 horizontal and -0.6875 vertical image pixels per mouse count.
These ratios do not measure engine angles or visible response latency.
The calibration program now waits for the first F8 without a default timeout.
A regression test covers initial waiting beyond 180 seconds.
See the [mouse calibration report](mouse-calibration.md).
Extended capture, held-input focus-loss timing, and model control remain incomplete.
