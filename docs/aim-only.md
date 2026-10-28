# Mouse-only Cutman tracking prototype

The `body_aim` policy proposes small mouse movements toward a visible Cutman candidate.
It never proposes movement keys or attack.
The existing pause, focus, result-age, and input-watchdog checks remain active.
The first attended aim test remains pending.

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
