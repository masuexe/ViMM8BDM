# Visual navigation rule baseline

This rule provides a local navigation baseline while the learned event model remains unsuitable for control.
It uses game images and its own commanded actions.
It reads no engine memory, map geometry, packets, or hidden player state.
It does not load a neural checkpoint.

## Behavior

The rule first waits for the displayed scene to settle.
It then holds W for a forward probe.
After 0.6 seconds, it compares sampled grayscale pixels with the probe's starting view.
If the image changes enough, it continues forward with a new reference view.
If the image stays nearly static, it releases W and attempts a small right turn.
It then lets the scene settle before another forward probe.

The image sample covers the central upper scene, away from Grenade Man's current weapon and left HUD.
The rule removes the largest ten percent of pixel differences before calculating its change score.
This reduces isolated animation effects.
A score below two triggers a turn attempt.

Each turn requests 64 horizontal mouse counts, with at most eight counts in one action.
The nominal turn duration is 0.4 seconds, followed by 0.3 seconds of settling.
Slow capture can extend the turn rather than increase its total motion.
The requests specify relative mouse counts.
Their angle conversion remains uncalibrated.
Capture gaps above 150 milliseconds, repeated timestamps, or size changes clear the policy reference.
Re-enabling input also resets the rule.

The change score is only a movement proxy.
Scene animation can resemble motion, and a featureless view can resemble a blocked player.
The rule cannot identify ledges, death screens, menus, enemies, or route objectives.
Its sampling region currently targets the inspected Grenade Man setup.
It sends no attack, jump, weapon, or skill actions.

## Run an attended check

Use the stock Windows alpha client, the original packages, and the private local server.
Select fixed Grenade Man and an open ground-level area of MM1CUT.
Enable Windows Do Not Disturb before the check.
Keep the game window fully visible and close menus.

Run this command with the current window identifier and new output files:

```powershell
.local\windows-venv\Scripts\python -m vimm8bdm `
  --live --capture window --window-id 0x12345 --policy visual_navigation `
  --seconds 30 --frames 1200 --hz 20 --arm-timeout 0 `
  --output runs/visual-navigation-check-1/actions.local.jsonl `
  --input-trace runs/visual-navigation-check-1/worker.local.jsonl
```

1. Select the game window.
2. Release gameplay keys and leave the real mouse still.
3. Press F8 to enable control.
4. Observe forward movement and any turn attempts near walls.
5. Use Esc or F8 to pause if necessary.
6. Check that control ends after the 30-second limit.

F9 exits the process.
Focus loss pauses control, and returning does not resume it.
The existing independent input lease remains active.
The rule does not advance its internal state during an input pause.
The run timer starts after the first F8 command and includes later pauses.
The initial wait has no timeout.

## Evidence status

Linux and Windows each passed 152 automated tests after both the event model and rule changes.
Rule tests check static and changed views, HUD exclusion, bounded turns, capture resets, and pause behavior.
A synthetic command-line check also completed without desktop input.
The attended game check remains pending.
These checks do not establish collision detection or navigation success in the actual game.
The [evidence file](evidence/visual-navigation.json) records the current status.
Keep logs and screenshots private.
