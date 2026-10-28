# First practice aim result

The operator observed some automatic following.
The runtime records sixteen nonzero mouse dispatches.
This is partial mouse-following evidence, not a complete aim-quality result.
The controller proposed no movement keys or attack.

## Active and paused intervals

The run lasted approximately thirty seconds and captured 297 loop records.
Only 72 records performed model inference.
The first result was too old and triggered the existing safety pause.
The remaining active interval began after a later manual enable command and lasted approximately 7.2 seconds.
The worker accepted 71 action messages during that interval.
It dispatched nonzero mouse motion in sixteen of them.
The longest consecutive motion sequence contains thirteen images.
The runtime then stopped at its duration limit with no dropped trace events.

The first policy call took 157.65 milliseconds.
Its result age reached 172.13 milliseconds and exceeded the existing 150-millisecond limit.
The input worker correctly rejected it and paused control.
The first PNG write then took another 58.19 milliseconds.
That write occurred after input submission and did not cause the earlier stale-frame rejection.
The root cause of the slow inference remains unproved.

Most records therefore describe paused control.
Their aggregate policy median is not a neural-inference speed measurement.
Among the 71 accepted records, median policy processing took 43.56 milliseconds.
The accepted complete-loop median was 61.76 milliseconds, including diagnostic image writing.
These measurements do not establish thirty seconds of uninterrupted tracking.

## Visual review

The runtime retained 72 private image pairs.
The agent inspected all sixteen motion images with selected-box overlays.
Those boxes cover visible Cutman bodies in the reviewed images.
The review does not provide complete ground truth, actor identity, or life-state labels.
The target moves rapidly and sometimes becomes large or leaves the image.
The current mouse limits do not establish accurate tracking through those changes.

The operator checked god mode before the run.
The test did not continuously measure protection from damage.
It does not establish normal combat survival or uninterrupted invulnerability.
The [aggregate evidence](evidence/aim-practice-result.json) retains the partial result and initial safety pause.
No private image enters the public repository.

## Readiness change

The body policies now perform one dummy inference after two seconds without inference, while control is inactive.
This work keeps the model ready during the initial user wait and later pauses.
It sends no input and does not create target observations.
A new runtime field counts these idle warmup calls.
Normal inference updates the same readiness timestamp.

The runtime also initializes PNG encoding in memory before input-worker setup.
It does not create an extra diagnostic image for that initialization.
The result-age limit and watchdog remain unchanged.
An old result must still stop control.
The change aims to reduce startup delay, but it cannot guarantee a deadline under every scheduling condition.
A new attended run remains necessary to check startup and sustained following.


## Readiness checks

Linux and native Windows each passed all 243 tests without skips.
The tests check warmup spacing, absence of target observations during warmup, and initial waiting behavior.

A native Windows smoke check used one archived image after two separate ten-second waits.
Without periodic warmup, processing took 39.93 milliseconds.
With four idle warmup calls, processing took 32.48 milliseconds.
Neither result was stale, and the check sent no input.
This short check did not reproduce the original slow first frame or its much longer user wait.
It does not prove that idle warmup resolves every startup delay.

The PNG initialization path also saved three exact synthetic RGB images on native Windows.
It rejected an existing output directory and retained the saved image hashes.
This check used no game capture or desktop input.
The next attended result remains pending.
