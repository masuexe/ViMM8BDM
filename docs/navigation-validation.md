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
