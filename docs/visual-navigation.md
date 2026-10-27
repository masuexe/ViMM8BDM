# Visual navigation rule baseline

This rule provides a local navigation baseline while the learned event model remains unsuitable for control.
It uses game images and its own commanded actions.
It reads no engine memory, map geometry, packets, or hidden player state.
It does not load a neural checkpoint.

## Behavior

The rule first waits for the displayed scene to settle.
It then holds W for a forward probe.
After 0.2 seconds, it compares sampled grayscale pixels with the probe's starting view.
If the image changes enough, it continues forward with a new reference view.
If the image stays nearly static, it releases W and sends a positive horizontal turn request.
It then lets the scene settle before another forward probe.

The image sample covers the central upper scene, away from Grenade Man's current weapon and left HUD.
The rule removes the largest ten percent of pixel differences before calculating its change score.
This reduces isolated animation effects.
A score below two triggers a turn attempt.

The first turn requests 256 horizontal mouse counts.
Repeated blockage increases the turn budget by 256 counts, with a limit of 1024 counts.
Nine consecutive changed views clear the blockage history.
This keeps the nominal reset interval at 1.8 seconds.
Each action requests at most eight counts, at a nominal rate of 160 counts per second.
Each turn ends with 0.3 seconds of settling.
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

Linux and Windows each passed 157 automated tests after the turn adjustment.
Rule tests check static and changed views, HUD exclusion, bounded turns, capture resets, and pause behavior.
A synthetic command-line check also completed without desktop input.
Movement and turn execution worked in the first attended check.
The operator reported a left turn followed by repeated wall contact.
The log recorded seven turns of 64 counts and 593 accepted actions.
The 30-second run ended at its duration limit.
This check failed to establish useful navigation.
The second attended check used the larger turn budgets.
The operator reported much better movement, with a delay before turns at walls.
The log recorded four turn attempts and 592 accepted actions.
The first turn packet followed each static-view decision by approximately 50 milliseconds.
The log does not establish the physical collision time.
The rule now uses 0.2-second probes instead of 0.6-second probes.
A synthetic moving-to-static test checks the first turn within 0.45 seconds at 20 Hz.
This bound applies to the synthetic images, not all game scenes.
The shorter probes require another attended check.
Slow movement can produce small image changes and cause false blockage decisions.
These checks do not establish collision detection or navigation success in the actual game.
The [evidence file](evidence/visual-navigation.json) records the current status.
Keep logs and screenshots private.
