# ViMM8BDM

Public repository: [masuexe/ViMM8BDM](https://github.com/masuexe/ViMM8BDM).

ViMM8BDM will control an unmodified Zandronum client through game images and normal keyboard and mouse input.
The target stack is MM8BDM v6a, CBM v9fh, and Gondola Patch v2.7h2.
The AI must join permitted servers as a normal Player.
The user must be able to stop AI input at any time.

The primary runtime target is the official `3.3-alpha-r260925-2352` build used for normal play.
Zandronum 3.2.1 remains a compatibility baseline.
Test results apply only to the engine revision that produced them.
The project does not use an MCP engine for the deployment path.

Run the commands from the repository root.

## Milestone 0 result

Use route B: an external adapter for images, model inference, and normal input.
Add a small data recorder for route C when behavior cloning requires target-game data.
Do not create an engine fork at this stage.

The research found no complete product that meets the final acceptance conditions.
ContinuousDoom is the closest tested model for an initial adapter.
Its checkpoint loads with strict parameter checks and runs without ViZDoom.
Its original action space has no vertical aim, jump, weapon selection, or class skills.
The Windows alpha test accepted 87 live model actions over ten seconds.
The operator observed horizontal turning and primary attack.
The model selected attack throughout that run.
Target-game combat ability remains unverified.

Read these documents:

- [Existing projects and model evidence](docs/existing-projects-evaluation.md)
- [Architecture and route comparison](docs/architecture.md)
- [Milestone 1 plan and later work](docs/roadmap.md)
- [Demo files and training data](docs/demo-and-data.md)
- [Environment checks and experiment results](docs/experiments.md)
- [Source revisions and checkpoint entries](docs/evidence/repositories.json)
- [Technical terms](docs/glossary.md)
- [Public files and local records](docs/publication.md)
- [Milestone 1 progress](docs/milestone-1-progress.md)
- [Independent input watchdog](docs/input-watchdog.md)
- [Mouse calibration](docs/mouse-calibration.md)
- [Held-input focus check](docs/focus-check.md)
- [Pretrained model runtime check](docs/model-control.md)
- [Human gameplay recorder](docs/human-recording.md)
- [Mouse input and image alignment](docs/input-alignment.md)
- [Causal behavior-cloning samples](docs/causal-dataset.md)

The code includes a visual control probe and an experimental ContinuousDoom adapter.
The project does not contain upstream model code, model weights, or game assets.
The upstream license remains unresolved for ContinuousDoom.
The external source files serve this research experiment only.

## Run the offline probe

Use the system Python for the probe without optional packages:

```bash
mkdir -p runs
PYTHONPATH=src python3 -m vimm8bdm --doctor
PYTHONPATH=src python3 -m vimm8bdm --frames 300 --output runs/probe.jsonl
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

The default probe uses synthetic images.
It sends no desktop input.
Its policy follows a red marker and produces movement, two mouse axes, and attack.
It does not identify MM8BDM enemies.
The output file must not exist before the command starts.

## Install desktop packages

Use Python on the same operating system and desktop as the game.
For a Windows game, use Windows Python.
WSL screenshot and input packages do not directly control Windows game windows.

For a Linux game, use these commands:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[desktop]'
```

For a Windows game, use these commands:

```powershell
py -3.12 -m venv .local/windows-venv
.local\windows-venv\Scripts\python -m pip install -e ".[desktop]"
```

Use a separate Windows environment when the repository already contains a Linux environment.
The primary runtime uses Windows native capture and relative mouse input.
The Linux/WSLg alpha test produced camera flicker even after the controller stopped.
The Windows native test did not reproduce that flicker.
See the [engine and mouse report](docs/engine-and-mouse.md).

The experiment used a project environment with Python 3.13.15.
The initial system Python was 3.12.3.
See the tested package versions in the experiment report.
The Windows alpha input test passed the operator's visual check.
Native Wayland capture and control are outside this prototype.

## Primary Windows input test

Run these commands from the repository root in PowerShell:

```powershell
.local\windows-venv\Scripts\python -m vimm8bdm --list-windows
.local\windows-venv\Scripts\python -m vimm8bdm --live --capture window `
  --window-id 0xYOUR_WINDOW_ID --policy input_check `
  --seconds 90 --frames 1800 --hz 20 --output runs/input-check.jsonl
```

Select the game window rather than the server console.
Press F8 to start a sequence.
Press Esc to pause.
Press F9 to stop the controller.
The `window` capture backend uses the current client position and dimensions.
It does not capture another foreground application while waiting for the selected game.
Relative mouse events use normal Win32 `SendInput` calls.

## Run a local desktop probe

First complete the local checks in the Milestone 1 plan.
Select the game client window ID.
Use the client area for the capture region.
List visible Zandronum windows without sending input:

```bash
PYTHONPATH=src .venv/bin/python -m vimm8bdm --list-windows
```

The list excludes small startup windows.
Window titles remain absent unless you add `--include-window-titles`.
Use `--all-windows` if the game has no recognizable class or process metadata.
The output contains local window and process IDs.
Keep that output outside public reports.

Check the selected window without capture or input:

```bash
PYTHONPATH=src .venv/bin/python -m vimm8bdm --check-window \
  --window-id 0xYOUR_WINDOW_ID --seconds 10 --frames 200 --hz 20
```

Select the game during the check.
The result reports how many samples matched the selected window or its children.
Unknown focus, unrelated windows, hidden windows, and changed process identity block input.
These diagnostics do not activate a window or send keyboard or mouse events.

Replace the example values below:

```bash
PYTHONPATH=src .venv/bin/python -m vimm8bdm --live \
  --window-id 0xYOUR_WINDOW_ID --region LEFT TOP WIDTH HEIGHT \
  --frames 200 --seconds 10 --hz 20 --output runs/live.jsonl
```

WSLg can return a black image for the desktop root.
Use direct window capture for the tested X11 configuration:

```bash
PYTHONPATH=src .venv/bin/python -m vimm8bdm --live --capture x11-window \
  --window-id 0xYOUR_WINDOW_ID --frames 100 --seconds 10 \
  --output runs/window.jsonl
```

The controller starts paused.
Select the game window.
Press F8 to enable or pause AI input.
Press Esc to pause before using the game menu.
Press F9 to exit.
The default run limit is 10 seconds after the first enable command.
The separate initial waiting limit is 180 seconds.
Ctrl+C also releases the inputs that this process holds.

The controller pauses after focus loss, an old image, or an expired input lease.
An image is old after 150 ms.
An input lease expires after 200 ms without a valid action.
Press F8 again to enable input after a pause.
The probe assumes WASD movement and mouse-button-1 attack.
It does not change the user's game configuration.

Use the separate input-check policy for a local interface test:

```bash
PYTHONPATH=src .venv/bin/python -m vimm8bdm --live --capture x11-window \
  --window-id 0xYOUR_WINDOW_ID --policy input_check \
  --seconds 90 --frames 1800 --hz 20 --output runs/input-check.jsonl
```

Press F8 to start the sequence.
It checks movement, both aim axes, attack, jump, and a weapon key separately.
It releases inputs between actions and pauses after approximately seven seconds.
Press F8 to start a new sequence after a pause.
This policy checks the interface rather than enemy detection or combat skill.
Black captured images block input.
A changed client size also blocks input until a new run selects the window again.

Use a separate configuration when starting a local test client:

```bash
python3 scripts/launch_local.py --engine .local/engine/zandronum \
  --iwad .local/game/megagame.wad --base .local/game/MM8BDM-v6a.pk3 \
  --classes .local/game/classes-v9fh.pk3 --gondola .local/game/gondolapatchv2.7h2.pk3 \
  --run-dir runs/local-session
```

Replace these example paths with your installation paths.
The run directory must not exist before startup.
The script does not join a remote server.
The launcher enables client sound and music by default.
Add `--mute` for a quiet client test.
It writes the configuration and startup records into the new run directory.
The separate configuration removes the game bindings for F8 and F9.
This prevents the game message toggle from sharing the controller key.
The controller cannot suppress those keys in an unrelated game configuration.
The final summary reports the exit reason.
After exit, F8 cannot enable the stopped controller.
The Linux startup path passed a local test.
The local Windows startup path passed the alpha test.

Add `--loopback-server` for a stock local server without campaign bots.
The server binds to `127.0.0.1` and does not join a public session.
Linux requires `zandronum-server` in the engine directory.
Windows uses `zandronum-server.exe` if present, or the stock `zandronum.exe` server mode.
The script records both process IDs in the excluded run directory.
Stop both test processes after the experiment.

Live control now uses a separate input process.
Its input lease expires independently from capture and inference.
Only the input worker owns the control hotkeys and desktop input.
The [watchdog report](docs/input-watchdog.md) describes its failure tests and limits.
Use the current code for local experiments only.

## Test the external pretrained model

Supply the trusted upstream files separately.
Use ContinuousDoom revision `ae10f186d3f936dab2b4b669797c2a3c0d410e71`.
The source directory must contain `actor_critic.py` and `modules.py`.
The adapter imports these Python files.
Install CPU PyTorch, NumPy, and Pillow in the project environment.

```bash
PYTHONPATH=src .venv/bin/python -m vimm8bdm \
  --policy continuousdoom --continuous-source .local/continuousdoom \
  --checkpoint .local/weights/agent.pt --yaw-scale 2 --frames 35 \
  --output runs/model.jsonl
```

This command uses synthetic images and sends no desktop input.
For a local game test, add `--live`, `--window-id`, and the capture options.
Use `--hz 8.75` to approximate the training interval of four game ticks.
The value `2` is an experimental mouse scale.
It is not a calibrated conversion between game angle and mouse counts.

The adapter uses CPU inference and strict checkpoint loading.
It reads four grayscale images at 320 by 240 pixels.
It divides pixel values by 255.
It repeats the first image to fill the initial history.
It preserves the original six-choice action head.
A live game test must check image scaling, mouse sensitivity, timing, and policy behavior.
