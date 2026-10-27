# ContinuousDoom runtime check

This experiment connects the external ContinuousDoom checkpoint to images from the stock Windows client.
The model remains a research baseline.
The experiment does not establish target-game combat skill.
The upstream source and weights remain outside the public repository.
The upstream reuse terms remain unresolved.

## Check the model files

Use upstream revision `ae10f186d3f936dab2b4b669797c2a3c0d410e71`.
The checkpoint must have SHA-256 `a728bb348321062ede790c9b13f153d1500926310ba3170a08d5ced8d46466de`.
The file contains 21,373,884 bytes.
The adapter loads the state dictionary with `weights_only=True` and strict parameter checks.
Three CPU warm-up passes occur before the input worker starts.

Install CPU PyTorch, NumPy, and Pillow in the Windows project environment.
Use the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/).
Keep the original files in the excluded `.local/` directory.
The project does not download executable source during normal startup.

## Action mapping correction

The earlier adapter mapped three movement outputs to incorrect keys.
The source audit found the error before this Windows live model test.
The corrected mapping follows the pinned [training scenario](https://github.com/bebeal/ContinuousDoom/blob/ae10f186d3f936dab2b4b669797c2a3c0d410e71/VizDoomGym/scenarios/defend_the_center.cfg).
The upstream [button construction](https://github.com/bebeal/ContinuousDoom/blob/ae10f186d3f936dab2b4b669797c2a3c0d410e71/utils.py) adds idle before the five button choices.

| Model choice | Original action | Adapter output |
| --- | --- | --- |
| 0 | Idle | No key or attack |
| 1 | Attack | Mouse button 1 |
| 2 | Move right | D |
| 3 | Move left | A |
| 4 | Move forward | W |
| 5 | Move backward | S |

The model also returns horizontal yaw.
The adapter multiplies yaw by the selected experimental scale.
The input worker limits each mouse delta to 24 counts.
The desktop backend rounds each delta to an integer.
Small values can therefore produce no mouse movement.
The current adapter does not accumulate fractional counts.
The local image calibration does not establish the correct scale for ViZDoom yaw units.

The discrete choices remain mutually exclusive.
The model cannot combine attack with movement or select pitch, jump, weapons, or class skills.

## Observe before control

Use an attended local game and a new pair of output files.
Use Windows Python on the same desktop as the game.
Run this command from the repository root in PowerShell:

```powershell
.local\windows-venv\Scripts\python -m vimm8bdm --live --capture window `
  --window-id 0xYOUR_WINDOW_ID --policy continuousdoom `
  --continuous-source .local/continuousdoom --checkpoint .local/weights/agent.pt `
  --yaw-scale 2 --hz 8.75 --seconds 15 --frames 132 --arm-timeout 0 `
  --observe-only --output runs/model-observe.jsonl --input-trace runs/model-observe-worker.jsonl
```

Wait for the model and input-worker ready messages.
Select the game window.
Close the game menu.
Press F8 to start.
The model predicts from displayed images, but the worker receives empty actions.
The player must not move, turn, or fire because of the model.
The initial waiting phase has no timeout.
F8 and Esc pause input.
F9 stops the controller.

The frame log records `model_output`, the proposed `action`, and `dispatched_mouse_counts`.
The summary marks `policy_actions_suppressed` as true.
Its accepted-action count refers to empty lease-renewal actions during observation.
It does not count executed model actions.
Input cleanup can still release inputs that the worker owns.

Check inference times, image age, model choice frequency, and rounded mouse counts.
Check the independent worker trace for expiry, focus loss, and complete shutdown.
The model does not detect game menus or death screens.
The operator must prepare a valid gameplay view for this experiment.

## Short control test

Complete the observation check first.
Start a new run without `--observe-only` for a short local control test.
Use a ten-second limit for the first run.
Retain the same model, preprocessing, mouse scale, and control protections.
Observe the actual action sequence without assuming intelligent behavior.
Stop with Esc or F9 if necessary.

Record interface success separately from navigation or combat success.
Repeated attack output can exercise the interface while failing the game task.


## Windows observation result

The model loaded and completed CPU warm-up before the input worker became ready.
The operator then enabled observation with F8.
The run processed 131 nonblank window images and reached its 15-second duration limit.
The model selected attack for every image.
The policy stage had a median duration of 20.15 milliseconds.
Its maximum duration was 44.72 milliseconds.

The worker accepted 111 empty actions.
Another 20 frames occurred during a user pause.
The controller dispatched zero model actions and zero nonzero mouse deltas.
The independent audit ended with no held inputs and zero dropped events.
The operator reported no visible action, followed by disappearance of the background terminal indicator.
This result matches observation mode and normal program exit.
It does not indicate a failed F8 hotkey.

All 61 automated tests passed on Linux and Windows.
The tests cover the corrected six-choice mapping and suppression of movement, both mouse axes, and attack.
See the [sanitized runtime record](evidence/model-runtime.json).
## Windows control result

The subsequent run enabled model actions for ten seconds.
All 87 captured frames produced accepted model actions.
The model selected attack on every frame and varied its horizontal yaw output.
The operator observed rightward turning with primary attack.
The run reached its duration limit, released attack, and closed normally.
The worker audit ended with no held inputs and zero dropped events.

The median policy duration was 20.31 milliseconds.
The median capture-to-dispatch loop duration was 28.18 milliseconds.
The loop duration reached 58.19 milliseconds at its maximum.
These timings exclude the game's visible response delay.
No navigation, enemy tracking, or combat success measurement occurred.
The model made no movement-key choice in this run.

The test establishes the screenshot-to-model-to-input path on the stock Windows client.
It also exposes persistent attack output in the tested target-game views.
Target-game demonstrations and behavior cloning are the next useful learning step.
The result does not justify a large engine fork.
