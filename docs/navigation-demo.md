# First navigation demonstration

The operator recorded ground navigation with Grenade Man on MM1CUT.
The source uses the official Windows alpha client with the three target packages.
The recorder sent no game input.
The operator ended the session with F9 after approximately 100 seconds.

## Collection result

| Measurement | Result |
| --- | --- |
| Images | 1997 |
| Events | 5895 |
| Segments | 1 |
| Dropped items | 0 |
| Archive check | Pass |
| Median capture duration | 8.21 ms |
| Maximum interval between images | 52.14 ms |
| Candidate samples | 2005 |
| Source action events retained in labels | 5892 of 5892 |

The input contains W, A, S, D, and relative mouse motion.
It contains no attack or jump events.
The operator reported no jumping or ladder navigation.
This session does not establish support for those actions.

Six image checks showed gameplay in different parts of the map.
These checks did not cover every image.
The recording contains no automatic annotation of deaths, menus, or other interruptions.
The private preview supports further human review.

## Candidate samples

Each sample contains four references to past images and the next 50 milliseconds of input.
The builder retained all source action events without clipping or a response-delay correction.
One final window lacked complete future coverage and produced no sample.

The dataset has a provisional `train` split designation.
No separate navigation validation or test session exists yet.
The dataset retains `training_ready: false`.
Human review and mouse-unit checks remain incomplete.

## Executor limits

The offline audit used experimental mouse scales of 1.06 horizontally and 0.92 vertically.
These scales do not establish mouse-unit equivalence.
The audit found 1390 representable candidates among 2005 samples.
The remaining 615 samples encounter at least one executor limit.

| Condition | Affected samples |
| --- | --- |
| Mouse request exceeds the current limit | 456 |
| Key transition occurs inside the action window | 225 |
| Multiple transitions require timing | 1 |

The condition counts overlap.
Do not add them to calculate the number of blocked samples.
The audit did not send input or remove samples.
These limits describe the current executor, not a recording failure.

## Next implementation

Extend action execution before training on this session:

1. Preserve press and release times inside each action window.
2. Divide larger mouse requests into bounded packets without losing their total motion.
3. Preserve fractional mouse remainders between packets.
4. Cancel pending packets on pause, focus loss, or an expired input lease.
5. Test timing and cancellation before an attended game check.

Queued packets must not renew the lease after the parent stops sending valid actions.
The [timed execution path](timed-actions.md) now passes offline checks for this session.
The attended game check remains pending.
Keep the original recording and labels for later checks.
Do not clip fast turns or remove short taps to fit the current executor.

The [evidence file](evidence/navigation-demo.json) contains public aggregate results.
Raw recordings, previews, local configuration, and machine paths remain private.
