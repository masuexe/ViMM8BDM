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
Live regression of the corrected keys remains a separate check.


A separate stock-client startup checked the new configuration.
The saved game configuration contained no F8 or F9 binding.
That test sent no control input.
The test client stopped after the check.
Live testing of Escape pause and F8 re-enabling remains unverified.

The [sanitized M1 evidence](evidence/milestone-1.json) records these results.
The [current offline test log](evidence/milestone-1-tests.txt) records the 32-test suite.

The final suite passed 32 tests, including checks against the actual Git index contents.
