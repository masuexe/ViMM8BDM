# Mouse-only Cutman tracking prototype

The `body_aim` policy proposes small mouse movements toward a visible Cutman candidate.
It never proposes movement keys or attack.
The existing pause, focus, result-age, and input-watchdog checks remain active.
The first attended aim test produced one small mouse dispatch.
The operator did not observe automatic turning.

## Candidate checks

The detector keeps its existing acceptance score of 0.5.
The aim stage separately requires a score of at least 0.8.
This stricter score can reject a valid body.

The aim stage checks the current standard magenta Cutman appearance.
It requires at least three percent magenta pixels in the candidate region.
Their vertical extent must cover at least 45 percent of the region.
The check examines a reduced region with dimensions no greater than 96 pixels.
It does not classify actor life state or enemy identity.
Damage flashes, occlusion, and other palette settings can cause rejection.
Other characters require a different appearance profile.

The box width must be at least eight pixels, and its height must be at least twelve pixels.
Its width-to-height ratio must remain between 0.2 and 1.1.
Its area must not exceed 35 percent of the image.
These conditions restrict this prototype to upright body candidates.
A wider crouched or partly hidden body can fail them.

The stage removes overlapping duplicate candidates at box overlap above 0.3.
It retains the higher-scoring candidate.
If multiple distinct candidates remain, tracking stops.

## Tracking and mouse limits

Tracking requires three consecutive matching observations.
The maximum interval between observations is 200 milliseconds.
A match requires an area ratio between 0.5 and 2.
It also requires box overlap of at least 0.1 or center separation of at most 48 pixels.
These spatial checks do not prove actor identity.

A missing target, ambiguous targets, a large position change, or an invalid result resets confirmation.
The controller does not predict a target through missing images.
Pause and enable operations also reset confirmation.

Only the calibrated 640 by 480 capture size permits mouse proposals.
The target point is the box center.
The center tolerance is eight horizontal pixels and six vertical pixels.
The proportional gain is 0.15, with vertical conversion divided by 0.625.
The [mouse calibration](mouse-calibration.md) supplied approximate local pixel-to-count ratios.
These values are not engine angles.
They depend on the saved local view and mouse settings.

The controller limits horizontal motion to sixty counts per second and vertical motion to forty counts per second.
Each proposal uses at most 100 milliseconds of elapsed time.
Thus, one proposal cannot exceed six horizontal counts or four vertical counts.
The input worker performs its existing fractional-count conversion.
The first live test must check the resulting feedback response.

## Offline result

The replay uses every second image from the second recording and the saved detector scan.
It checks 596 images at approximately ten observations per second.
It uses a recording that already informed development.
It is not an independent test.
The replay sends no input.
Its scene does not respond to the proposed mouse movements.
It therefore cannot prove tracking stability, convergence, or aim accuracy.

The final replay retains eligible candidates in 195 images and proposes mouse movement in 55 images.
All proposals remain within the six-count and four-count limits.
None contain movement keys or attack.
The seventeen reviewed validation images retain eight of nine body locations and no candidates in their eight negative images.
This count does not establish the quality of the other replay images.

An initial appearance-only version still accepted a wide foreground candidate in frame 677.
The final extent check rejects that candidate.
It also rejects the large foreground candidates in frame 808.
In frame 447, it removes the combined body-and-projectile box and retains the smaller body box.
In frame 965, the stricter aim score rejects the visible body too.
These examples show both useful rejection and a valid-target loss.
The earlier replay remains available privately.
The [replay script](../scripts/replay_aim.py) records its source checks and writes new result files.
The [aggregate evidence](evidence/aim-only.json) retains the replay and implementation checks.
Automatic attack remains disabled.

Linux and native Windows each passed all 241 tests without skips.
A native Windows check also processed five archived images through the actual model and aim policy.
It produced three nonzero mouse proposals and sent no input.
This check does not measure live feedback.


## First attended aim result

The native Windows run completed 298 images in approximately thirty seconds.
The input worker accepted all action messages and stopped normally.
Only 31 images supplied an eligible target.
Thirty images requested further confirmation, and one image reached the tracking condition.
That image produced six horizontal counts and minus four vertical counts after input conversion.
The other messages contained no mouse movement.
No message contained movement keys or attack.

The operator did not observe automatic turning and reported frequent deaths from Cutman attacks.
The result does not establish useful aim feedback.
Most images failed to supply an eligible target, so this test produced little motion to observe.
The runtime did not save image ground truth for these frames.
The report cannot assign all missing targets to death, occlusion, or model error.
The [attended evidence](evidence/aim-first-attended.json) retains the sparse-motion result.

## Local practice conditions

A separate private loopback server now requests `sv_cheats 1` at startup.
Its client uses a new copy of the saved configuration and binds F7 to `god`.
These settings belong to a local interface test.
They must not become a deployment requirement or evidence of normal combat survival.
The game executable and original packages remain unchanged.

The engine repository's [inherited console reference](https://github.com/doomtech/zandronum/blob/master/docs/console.html) describes network cheat permission and the god-mode toggle.
Permission alone does not enable god mode.
The reference describes protection against normal attacks and notes that telefragging can still kill the player.
The operator reported the enabled-mode message with the supplied alpha and loaded MODs.
The operator also reported that ordinary Cutman attacks no longer reduce health.
This observation does not establish protection from every damage type.

After joining as a living player, press F7 once and inspect the game message.
Do not repeatedly press F7 because it toggles the mode.
Check ordinary Cutman damage before the next aim run.
The next comparison keeps the detector and aim thresholds unchanged.


## Practice replay preparation

The next attended run keeps the same model, appearance conditions, confirmation count, gains, and movement limits.
It enables optional diagnostic image output with `--image-output NEW_PRIVATE_DIRECTORY`.
The image directory must be new.
The runtime saves PNG images only for enabled, nonblank frames.
Each runtime row records the matching image filename and write-stage duration.
The saved image contains the original captured pixels rather than an annotated overlay.

Image writing occurs after input submission.
The complete loop duration includes the write stage, while the input duration excludes it.
This instrumentation adds processing cost.
The comparison cannot isolate its timing effect from other session differences.
The images contain AI-assisted play and do not establish human imitation labels.
Keep these files in private ignored storage.

Linux and native Windows each saved three synthetic images with exact RGB agreement.
Both checks rejected an existing image directory and retained the earlier image hashes.
These checks sent no desktop input.
Both platforms also passed all 241 regression tests without skips.
The [practice preflight evidence](evidence/aim-practice-preflight.json) records the operator report and diagnostic checks.
The next attended aim result remains pending.
