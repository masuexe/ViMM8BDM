# Navigation validation collection

Keep the first navigation demonstration in the provisional training split.
Record a separate session for validation before training the first navigation policy.
Do not divide adjacent windows from one session between these splits.

## First validation session

Use the same official Windows client, original packages, fixed Grenade Man selection, and MM1CUT map.
Keep the mouse settings and key bindings unchanged.
The target duration is 90 seconds at 20 frames per second.
The recorder sends no game input.
The player controls all movement and turning.

1. Select the game window.
2. Check that Grenade Man is active.
3. Close menus and release gameplay keys.
4. Press F8 to start recording.
5. Follow a different route through the accessible ground-level rooms.
6. Include ordinary forward movement, sideways movement, turns, and short stops.
7. Omit attacks, jumps, ladders, weapon changes, and skills for this collection.
8. Continue until the recorder stops automatically.

Use F8 to pause before an interruption.
Esc or focus loss also pauses recording.
Use F9 to end the recording early if necessary.
Do not change the character or map during the session.
Stopping the recorder does not stop human control of the player.

## Check the result

Check archive integrity, source hashes, image timing, input transitions, and operator reports.
Build this session with `--split validation` in a new dataset directory.
Use `--check-splits` with both datasets before training or comparing policy results.
Retain the raw input events without clipping fast turns or removing short taps.
Keep recordings and previews private.

This session checks another recording on the training map.
It does not establish navigation on a new map.
Collect a separate map session later for that test.
The validation designation does not establish mouse-unit equivalence or training readiness.
Do not use validation labels to train the policy.

## First collection result

The operator completed the 90-second recording with no reported interruption or extra action.
The recorder stopped at its duration limit.
It saved 1788 images and 6098 events in one segment, with zero dropped items.
The archive check passed.
The image span was 89.93 seconds.
The median capture duration was 7.41 milliseconds.
The maximum interval between images was 51.35 milliseconds.

The input contains W, A, D, and relative mouse motion.
It contains no backward key, attack, jump, or weapon events.
The operator reported no ladder navigation, death, chat, menu, character change, or long wall obstruction.
Five image checks showed gameplay in different parts of the map.
A basic scan of all images found no nearly black frames or frames with very low texture.
These checks do not provide automatic menu or death detection.

The dataset builder produced 1794 candidate validation samples.
It retained 6094 of 6095 source action events in the labels.
One final horizontal mouse count occurred after the last complete candidate window.
That event remains in the original recording.
The builder rejected one window for insufficient history and two windows for incomplete future coverage.
It did not move event times, clip turns, or fabricate missing actions.

The operator still held W and D when the recording ended.
The recorder saved this final state without sending releases.
This boundary is valid for a human recording.
Any later replay must release owned inputs when the replay ends.

All validation samples passed timed-action compilation with candidate scales 1.06 and 0.92.
They produced 9912 packets, with at most 20 packets in one window.
This offline check sent no desktop input.
It does not establish mouse-unit equivalence.

Both datasets passed source-hash checks.
The split check found two distinct source sessions with no session or identical-archive overlap across splits.
The recorded asset digests and pointer-setting snapshots matched between sessions.
Those snapshots do not prove every active engine setting.
The two datasets contain 2005 training candidates and 1794 validation candidates.

The source files and dataset manifests remain unchanged after operator review.
Both datasets retain `training_ready: false` and the builder's default review blockers.
The separate operator records contain the later session reviews.
No independent test session exists yet.
The [evidence file](evidence/navigation-validation.json) contains the public aggregate results.
