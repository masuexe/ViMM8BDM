# Milestone 0 experiments

Test date: 2026-10-26.
The [machine-readable record](evidence/experiments.json) contains versions, digests, results, and raw test references.
The [offline test log](evidence/offline-tests.txt) contains the final unit-test output.

## Result

The project tested model loading, image capture, and inference on a stock client.
It did not establish keyboard or mouse control of the game.
The live input test did not pass the focus condition.
The test sent no game input.
Milestone 1 remains incomplete.

## Environment

| Item | Result |
| --- | --- |
| Operating system | Ubuntu 24.04.1 under WSL2 |
| Kernel | `6.18.54.1-microsoft-standard-WSL2` |
| Initial Python | 3.12.3 |
| Project environment | Python 3.13.15, selected by the installed `uv` |
| Display | X11 `:0`, WSLg with `wayland-0` |
| GPU after access changed | Available, approximately 8 GiB of memory |
| Model execution | CPU PyTorch 2.14.1, four CPU threads |
| Capture/input packages | MSS 10.2.0, pynput 1.8.2, python-xlib 0.33 |
| Image packages | NumPy 2.5.4, Pillow 13.0.0 |
| ViZDoom package | Not installed in the project environment |

Initial socket requests returned `EPERM`.
X11 access failed and GPU access reported an OS restriction.
The environment permissions later changed during this turn.
Network, X11, and GPU checks then passed.
The reports use the later results for current availability.

Package installation affected the project environment and a temporary package cache.
It did not change another project.
The complete package list appears in the evidence record.

## Target assets

The audit read the supplied files without changes.
It did not copy the files into the project.
It checked archive entries and configuration lumps.
These digests identify the supplied copies.
They do not independently prove upstream authenticity.

| Asset filename | Bytes | SHA256 |
| --- | --- | --- |
| `megagame.wad` | 139550 | `ba6eadf3d054107d449a6e1cc6add58cf2ba33d49ca0f0f91fa47a4a403f1ad0` |
| `MM8BDM-v6a.pk3` | 165333603 | `a39c670536b945724fb7a6aec1b9e60ed669d39a0e9e98a999f34d93da656076` |
| `classes-v9fh.pk3` | 8478308 | `420d7077dd81a28c8a43cef2a10e369b3e459a923a06ca38c3081391e5894d23` |
| `gondolapatchv2.7h2.pk3` | 16704339 | `f108214fa71c39403c536b0e3092da2eec02632618770c181d80819c687f035b` |

MM8BDM GAMEINFO selects `megagame.wad`.
CBM and Gondola contain their own KEYCONF and ACS-loading lumps.
The Gondola configuration exposes an extra reload/attack control.
The archive scan found no path containing `zscript` in these three packages.
This scan does not prove compatibility with another engine.

## Source audit

The GitHub connector returned complete recursive trees for the audited projects.
The audit then read selected implementation files at fixed revisions.
The source manifest records file paths, blob digests, and checkpoint sizes.
The project did not rely only on README statements.

The browser could not read the ContinuousDoom Drive URL.
The GitHub connector also refused binary checkpoint content.
Direct HTTP downloads succeeded after network access changed.
These earlier tool errors do not establish that the remote files are unavailable.

## ContinuousDoom checkpoint

The author links the checkpoint from the repository README.
The direct request used this [Drive download](https://drive.google.com/uc?export=download&id=1XvftbgcGp8dwgWLwqJRd8W2ftGFaVzjc).
The original file remains in a local research directory outside the public files.

| Check | Result |
| --- | --- |
| Downloaded size | 21,373,884 bytes |
| SHA256 | `a728bb348321062ede790c9b13f153d1500926310ba3170a08d5ced8d46466de` |
| Deserialization | `torch.load(..., weights_only=True)` |
| Checkpoint type | Ordered state dictionary with 35 keys |
| Model load | All keys matched with `strict=True` |
| Input shape | `[1, 4, 240, 320]` |
| Outputs | Six categorical choices and one continuous yaw value |
| ViZDoom dependency during inference | None |

The experiment imported two inspected upstream files from a temporary directory.
The project did not copy those files into its source tree.
Their digests appear in the evidence record.
The adapter omitted random parameter initialization before strict loading.
Strict loading then replaced every parameter.
The experiment did not change upstream source files.

### Independent forward test

The test used zero-valued images.
It executed five warm-up passes and 30 measured passes.
The median forward duration was 10.15 ms.
The 95th percentile was 11.72 ms.
One deterministic output selected attack and yaw approximately 1.41.
This establishes compatible inference, not useful game behavior.

### Adapter test without desktop input

The CLI processed 35 synthetic frames with the checkpoint.
The median policy stage was 13.48 ms.
The 95th percentile was 16.62 ms.
That stage includes image conversion, image history, and inference.
The input receiver remained an in-memory test object.
The test sent no OS input.

## Stock Zandronum startup

The experiment downloaded the official Linux archive from the [Zandronum download host](https://zandronum.com/downloads/zandronum3.2.1-linux-x86_64.tar.bz2).
The archive contains 8,355,843 bytes.
Its SHA256 is `9f4b93ed99ab15d2406e697e2a06d956eb808c9ac2e53e62ae9f1affb53c86c0`.
The project extracted the archive into a temporary directory.
It made no changes to the executable or engine package.

The client reported `Zandronum 3.2.1 - 250804-2140 - SDL version`.
The experiment used a separate project configuration and disabled unrelated autoload packages.
It loaded the IWAD and the three requested packages in order.
It entered `MM1CUT - CUT MAN` with local campaign bots.
It did not join a remote server.

The renderer reported llvmpipe.
Thus, the accessible GPU did not render this game test.
The session later reached the campaign loss screen while the player remained idle.

The local startup record contains the exact arguments.
The local engine log records package loading and startup.
The Git ignore file excludes those records under `runs/`.
The public evidence contains versions, asset digests, and results without machine paths.

## Capture tests

MSS returned a black image from the desktop root.
Direct X11 window capture returned the rendered game image at 640 by 480 pixels.
The inspected format was 24-bit TrueColor with least significant byte first.
The new capture backend rejects unsupported formats.

The local game image displays the map and bots.
The later loss image displays the campaign result.
The public repository does not include these images.

### Client capture and forward test

The test used five warm-up iterations and 30 measured iterations.
It captured the client window and called the pretrained policy.
It did not execute the returned actions.

| Stage | Median | 95th percentile |
| --- | --- | --- |
| Window capture | 1.68 ms | 4.79 ms |
| Preprocessing and inference | 15.30 ms | 17.75 ms |
| Capture through action result | 17.30 ms | 21.08 ms |

The policy selected attack in all 30 measured frames.
It selected no movement key in that sample.
The campaign reached its loss screen around this test.
Treat these values as interface timing, not an active-combat benchmark.
They do not establish enemy detection or skill transfer.

### Paused CLI test

The full CLI processed 44 client images at a requested 8.75 Hz.
The controller remained paused.
The accepted-action count was zero.

| Stage | Median | 95th percentile |
| --- | --- | --- |
| Window capture | 1.56 ms | 4.88 ms |
| Preprocessing and inference | 14.65 ms | 22.55 ms |
| Complete loop | 17.03 ms | 23.71 ms |

The log also measures the input gate.
That value is not OS input execution latency because the gate rejected all actions.
The evidence record includes maximum durations for this test.
Neither test measures visible game response latency.

## Live input attempt

The test script waited up to 50 seconds for the selected game window to receive focus.
The focus condition did not pass.
The script exited with code 3 before its first input event.
It did not execute its map-reset, movement, aim, attack, or pause-key tests.

The source contains those control paths.
Their live game behavior remains unverified.
Do not disable the focus condition as a workaround.
M1 must resolve client-window selection and WSLg activation, or use a Windows desktop test.

## Offline control tests

All 12 tests passed with Python 3.13.15 after the final code changes.
An earlier run also passed with system Python 3.12.3.
The tests cover these failure conditions and control functions:

- Default pause blocks input.
- Takeover releases movement and attack.
- A late inference result cannot restore input after takeover.
- Focus loss releases input and requires explicit re-enabling.
- Another foreground window prevents enabling input.
- Stale and future image timestamps disarm input.
- Watchdog expiry releases input without a policy heartbeat.
- A late action cannot renew an expired lease.
- Partial input failure releases owned state.
- Invalid model actions disarm input.
- Synthetic markers produce both aim axes and attack.
- Invalid image data causes rejection.

These tests use a simulated input receiver.
They do not establish real hotkey delivery or OS input release.

## DoomBot binary checks

Two downloaded SB3 archives passed ZIP CRC checks.
The experiment inspected JSON metadata and loaded `policy.pth` with `weights_only=True`.
It did not deserialize cloudpickle objects or run SB3 inference.

| Checkpoint | SB3 version | Policy input | Action output |
| --- | --- | --- | --- |
| `ppo_doom_center.zip` | 2.7.1 | 20 features | 16 discrete actions |
| `ppo_multimap_yolo_ft.zip` | 2.7.1 | 18 features | 18 discrete actions |

The different dimensions require different observation and action contracts.
The file name does not establish compatibility with the current source.
Both checkpoints use feature vectors rather than raw images.
The evidence record includes file digests and tensor shapes.

## Unverified functions

The project did not establish any of these results:

- Actual game movement, aim, or attack from external OS input
- Real human takeover during game input
- Calibrated mouse counts or accurate relative mouse behavior
- Effective navigation, projectile avoidance, or combat transfer
- Class skills, weapon management, or Gondola mechanisms
- Demo playback or action extraction
- Behavior-cloning training, RL, or self-play
- A permitted human-server session

No game or MOD file changed.
No engine fork, Git commit, or remote push occurred.
The project stopped its local test client after the experiments.
A final digest check established that the stock executable remained unchanged.
