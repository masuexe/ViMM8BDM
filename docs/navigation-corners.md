# Navigation demonstrations near corners

The first model did not beat the constant median-mouse baseline.
This collection adds short navigation examples near visible corners and walls.
It does not assume that data imbalance caused every model error.
Keep the first recordings and model result as a reference.

## Collection scope

Use fixed Grenade Man selection on MM1CUT with the same engine, packages, bindings, and mouse settings.
The training collection lasts at most 120 seconds at 20 frames per second.
The recorder sends no game input.
Use accessible ground-level areas only.
Full map coverage is unnecessary.
Omit jumps, ladders, attacks, weapon changes, and skills.

Demonstrate these situations when the visible route requires them:

- Approach a corner and turn toward its clear passage.
- Use both left and right corners during the recording.
- Release forward movement when a wall blocks the route.
- Move sideways to align with a passage.
- Use brief backward movement to leave a close wall, then continue along a clear route.

Select situations with an obvious clear passage where possible.
Avoid arbitrary route choices at junctions with several equally clear exits.
Do not hold forward movement throughout every turn.
Do not remain against a wall for a long interval.
Do not add random turns or artificial key presses to meet a numerical quota.
The input must correspond to the visible navigation situation.

## Recording procedure

1. Wait for the recorder's ready message.
2. Select the game window.
3. Check the active character.
4. Close menus and release gameplay keys.
5. Press F8 to start recording.
6. Demonstrate short routes through nearby corners and passages.
7. Continue until the recorder stops automatically.

F8 pauses recording or starts another segment.
Esc and focus loss pause recording.
Use F9 to end the session early if necessary.
Pause before chat, menus, death, or unrelated activity.
End the session before a character or map change.
Recording stops without changing human control of the player.

Brief wall recovery is part of this task.
Report long obstructions, unintended extra actions, or other interruptions during review.
The recorder does not identify these situations automatically.

## Evaluation plan

Store this recording as a new training source.
Collect a separate validation session with the same task before another model comparison.
Do not move the existing validation recording into the training split.
Do not divide adjacent windows from this recording between training and validation.

Check archive integrity, source hashes, operator review, and causal labels.
Report key occupancy, mouse direction, and timed-action compilation for each split.
Keep all original mouse counts and press/release events.
Compare the next model with constants derived only from its training data.
Repeat the image-correspondence check.
Keep model predictions offline until the results support a separate attended control experiment.

See the [first model report](behavior-cloning.md) for the baseline result and target limitations.
