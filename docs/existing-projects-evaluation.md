# Existing projects evaluation

Research date: 2026-10-26.
The audit checked source files, dependency files, license files, checkpoint entries, and release lists.
The [evidence manifest](evidence/repositories.json) records the specific source revisions.
The experiment report records local tests.

No audited project meets all final deployment conditions without an adapter.
ContinuousDoom passed an independent inference test after environment access changed.
No test established effective combat in CBM or Gondola.

## Evidence classes

| Class | Meaning |
| --- | --- |
| Tested | This project executed the test and recorded the result. |
| Source | The source implements the function, but this project did not execute it. |
| Upstream claim | The author reports the function or result. |
| Inference | The proposed function requires a new experiment. |

A public checkpoint entry does not prove that a model runs.
A successful forward pass does not prove that a model plays the target game well.

## Requested projects

| Project and revision | Checkpoint evidence | Observation and action | Decision |
| --- | --- | --- | --- |
| [ViZDoom](https://github.com/Farama-Foundation/ViZDoom/tree/ceb89752e79c250cc34718df9b5a71988601a252) `ceb8975` | No policy checkpoint appeared in the audited main tree. | Images are optional inputs. The API also exposes internal state. It supports binary and DELTA actions. | Reuse training examples and interfaces where applicable. It does not control an existing stock client. |
| [ContinuousDoom](https://github.com/bebeal/ContinuousDoom/tree/ae10f186d3f936dab2b4b669797c2a3c0d410e71) `ae10f18` | A 21,373,884-byte checkpoint downloaded from the author's Drive link. Strict loading passed. | Four grayscale images. One continuous yaw output and one six-choice action output. | Best tested candidate for the initial adapter. License and target-game skill remain unresolved. |
| [tetsuo-doom](https://github.com/tetsuo-ai/tetsuo-doom/tree/c09cb3e5b3f67072c8de662b789e6e59398bf24c) `c09cb3e` | No local trained policy or checkpoint appeared. | Screenshots, objects, depth, map data, and variables. Numeric DELTA actions. | Default observations violate the deployment contract. Do not use it as the combat controller. |
| [Player-1](https://github.com/TencentARC/Player-1/tree/63c5a8d65fca75b78b9011e73ecf10aae8c7a7e6) `63c5a8d` | The tree contains a README, a license, and two images. Code and weights remain unpublished. | The author reports screenshot input, keyboard and mouse output, SFT, and RL. | Monitor releases. Do not make the project depend on an unpublished model. |
| [DoomBot](https://github.com/JoelJimenez23/DoomBot/tree/ecf45c99ff86224399f3d78fa694627f3b830fe8) `ecf45c9` | Seven SB3 ZIP entries exist. Two files passed binary structure checks. | Engine labels and game variables form a feature vector. Actions are discrete combinations. | It is not a direct screenshot policy. More changes than an input adapter are necessary. |

### ViZDoom

The audit read `ViZDoomGame.cpp`, `setup.py`, and the learning, spectator, and recording examples.
`DoomGame` controls its own engine process.
It is not an API for an existing Zandronum window.
An image-only configuration is possible.
This does not make every ViZDoom policy image-only.
Original ViZDoom code uses MIT terms.
The engine and third-party components have separate terms.
See the [API source](https://github.com/Farama-Foundation/ViZDoom/blob/ceb89752e79c250cc34718df9b5a71988601a252/src/lib/ViZDoomGame.cpp) and [license statement](https://github.com/Farama-Foundation/ViZDoom/tree/ceb89752e79c250cc34718df9b5a71988601a252#license).

The spectator example records human actions.
The replay example uses `get_last_action()` after an advance step.
These functions can supply imitation data.
They do not form a complete behavior-cloning trainer.
They do not establish compatibility with Zandronum client demos.
See the [spectator example](https://github.com/Farama-Foundation/ViZDoom/blob/ceb89752e79c250cc34718df9b5a71988601a252/examples/python/spectator.py) and [replay example](https://github.com/Farama-Foundation/ViZDoom/blob/ceb89752e79c250cc34718df9b5a71988601a252/examples/python/record_episodes.py).

### ContinuousDoom

`train.py` selects `num_continuous=1`.
The input contains four grayscale images.
The actor selects idle, attack, right, left, forward, or backward.
The adapter now follows the exact order in the pinned training scenario.
See the [runtime correction](model-control.md) for the earlier mapping error.
It also produces a horizontal turn value.
The discrete choices are mutually exclusive.
The original model cannot move and attack simultaneously.
It has no pitch, jump, weapon, or class-skill output.
An adapter cannot add a learned action that the model does not produce.
See the [training code](https://github.com/bebeal/ContinuousDoom/blob/ae10f186d3f936dab2b4b669797c2a3c0d410e71/train.py) and [action conversion](https://github.com/bebeal/ContinuousDoom/blob/ae10f186d3f936dab2b4b669797c2a3c0d410e71/utils.py).

The saved environment specifies Python 3.8, PyTorch 1.11, and CUDA 11.3.
The experiment instead used Python 3.13.15 and CPU PyTorch 2.14.1.
All 35 checkpoint keys matched the model with `strict=True`.
A forward pass accepted shape `[1, 4, 240, 320]` without ViZDoom.
The measured CPU median was 10.15 ms for 30 passes after five warm-up passes.
This test used zero-valued images.
It does not measure gameplay performance.
See the [dependency file](https://github.com/bebeal/ContinuousDoom/blob/ae10f186d3f936dab2b4b669797c2a3c0d410e71/doomEnv.yml) and [prediction function](https://github.com/bebeal/ContinuousDoom/blob/ae10f186d3f936dab2b4b669797c2a3c0d410e71/actor_critic.py).

The [author's checkpoint URL](https://drive.google.com/file/d/1XvftbgcGp8dwgWLwqJRd8W2ftGFaVzjc/view) failed through the research browser.
The direct download succeeded after network permissions changed.
The audit found no project LICENSE or COPYING file.
The project stores neither upstream source nor weights.
Resolve reuse terms before distribution or adoption as a maintained dependency.

### tetsuo-doom

The audit read `game_manager.py`, `state.py`, `actions.py`, and `pyproject.toml`.
Dependencies include Python 3.10 or later, fastmcp, ViZDoom, Pillow, and NumPy.
State extraction reads `state.objects` and uses labels for visibility.
Its controller can use information outside the displayed image.
DELTA values enter ViZDoom rather than the operating system's mouse interface.
The tree contains no PPO or BC trainer, checkpoint, or project license file.
The action names are useful references.
The default state-driven controller does not meet deployment constraints.
See the [state code](https://github.com/tetsuo-ai/tetsuo-doom/blob/c09cb3e5b3f67072c8de662b789e6e59398bf24c/src/doom_mcp/state.py) and [controller](https://github.com/tetsuo-ai/tetsuo-doom/blob/c09cb3e5b3f67072c8de662b789e6e59398bf24c/src/doom_mcp/game_manager.py).

### Player-1

The recursive tree contains four files.
The README lists model, data, environment, and training-code releases as incomplete.
Screenshot control and SFT/RL results remain upstream claims.
The available LICENSE contains MIT terms.
It does not establish the terms or usability of unpublished weights.
The audit found no evidence for local inference speed on the proposed 9B or 35B-A3B models.
See the [specific tree](https://github.com/TencentARC/Player-1/tree/63c5a8d65fca75b78b9011e73ecf10aae8c7a7e6) and [license file](https://github.com/TencentARC/Player-1/blob/63c5a8d65fca75b78b9011e73ecf10aae8c7a7e6/LICENSE).

### DoomBot

The default detection path can read engine labels.
`TaskAimShoot.make_obs_and_info` also reads health and ammunition through `core.get_vars(st)`.
Its default policy input has 20 features.
It is not a pixel CNN.
The YOLO branch does not remove the other direct state reads.
It also contains mixed dictionary and attribute access for configuration.
The audit did not execute that branch.
See the [core](https://github.com/JoelJimenez23/DoomBot/blob/ecf45c99ff86224399f3d78fa694627f3b830fe8/doom_env/core/vizdoom_core.py) and [policy observation](https://github.com/JoelJimenez23/DoomBot/blob/ecf45c99ff86224399f3d78fa694627f3b830fe8/doom_env/tasks/aim_shoot.py).

The evaluation entry calls SB3 `PPO.load`.
This establishes a loading path, not checkpoint compatibility.
The audit found no YOLO checkpoint, dependency manifest, or project license file.
The GitHub connector refused the binary checkpoint because it accepts UTF-8 text only.
After network access changed, two ZIP files passed CRC checks.
The center checkpoint contains a 20-feature input and 16-action output.
The YOLO checkpoint contains an 18-feature input and 18-action output.
Their tensor metadata establishes different feature contracts.
Neither is a screenshot policy.
Both files record SB3 version 2.7.1.
The experiment report records their digests.
This project did not execute an SB3 policy forward pass.
See the [evaluation entry](https://github.com/JoelJimenez23/DoomBot/blob/ecf45c99ff86224399f3d78fa694627f3b830fe8/eval/bot_evaluation.py) and [checkpoint directory](https://github.com/JoelJimenez23/DoomBot/tree/ecf45c99ff86224399f3d78fa694627f3b830fe8/models).

## Additional projects

| Project | Evidence | Recommendation |
| --- | --- | --- |
| [Sample Factory](https://github.com/alex-petrenko/sample-factory/tree/7a837983d3625671650e75c9e9d1c1a99f7b7454) | MIT. APPO/PPO and multiple environments. Public HF checkpoints. Default deathmatch includes game-variable inputs. | Prefer its algorithm components when RL becomes necessary. |
| [Arnold](https://github.com/glample/Arnold/tree/86af06d2fdb35c4bf552ecacfe8fe6ac1abd8cd4) | Five actual PTH entries, approximately 9.8–52.2 MB. DQN/DRQN. Deathmatch uses health and selected ammunition. No project license file appeared. | Keep it as a historical reference. Old APIs and unresolved reuse terms reduce its priority. |
| [doom-arena](https://github.com/wendlerc/doom-arena/tree/363570905f532cbf4507029271e94c1ae744493f) | HF downloader, `act(obs)`, recordings, and multiplayer examples. It needs a modified SF branch. No project license file appeared. | Study its inference and recording flow. Do not assume checkpoint compatibility. |
| [ZandronumMCP](https://github.com/rc4l/ZandronumMCP/tree/0a943b67bc187f5ce86d3a0a7f813337bacbab86) | MIT. Its bridge script changes engine source. | Exclude it from final deployment. This project did not call its engine tools. |

Sample Factory's standard deathmatch adds a `measurements` vector.
The vector includes health, ammunition, kills, attack readiness, player count, and weapon inventory.
The encoder combines that vector with the image features.
The current HUD does not reliably expose every field.
Zero values do not preserve the original observation distribution.
See the [additional-input wrapper](https://github.com/alex-petrenko/sample-factory/blob/7a837983d3625671650e75c9e9d1c1a99f7b7454/sf_examples/vizdoom/doom/wrappers/additional_input.py).

The standard action space combines movement, weapon selection, attack, sprint, and continuous yaw.
It does not supply the complete pitch, jump, and CBM skill actions.
See the [action implementation](https://github.com/alex-petrenko/sample-factory/blob/7a837983d3625671650e75c9e9d1c1a99f7b7454/sf_examples/vizdoom/doom/action_space.py).

The [HF deathmatch directory](https://huggingface.co/edbeeching/doom_deathmatch_bots_2222/tree/main/checkpoint_p0) lists three checkpoints of approximately 41.4 MB each.
Its [configuration](https://huggingface.co/edbeeching/doom_deathmatch_bots_2222/blob/main/cfg.json) specifies 128 by 72 pixels, LSTM size 512, and four-tick actions.
It records training revision `9da68b57eecd73c3c884c1be2d938b46aa7a7f49`.
Its [model card](https://huggingface.co/edbeeching/doom_deathmatch_bots_2222/blob/main/README.md) has no explicit license field and marks the reported metric unverified.
This project did not load those weights.

The doom-arena wrapper sets `skip_measurements_head=True` and loads weights with `strict=False`.
Its SF branch implements that option.
The original HF training revision includes a measurements encoder.
Removal of that encoder can change the LSTM input size.
`strict=False` does not repair different shapes for matching tensor names.
Check tensor names, shapes, and loading results before accepting the wrapper.
See the [wrapper](https://github.com/wendlerc/doom-arena/blob/363570905f532cbf4507029271e94c1ae744493f/doom_arena/agent.py), [branch](https://github.com/wendlerc/sample-factory/blob/3793cde8bbbefea9b09c5cec423a588a8e96ae6c/sf_examples/vizdoom/doom/doom_model.py), and [original encoder](https://github.com/alex-petrenko/sample-factory/blob/9da68b57eecd73c3c884c1be2d938b46aa7a7f49/sf_examples/vizdoom/doom/doom_model.py).

Arnold's deathmatch configuration selects health and selected ammunition as network inputs.
Its model implements embeddings for those variables.
Visual auxiliary targets during training do not make the complete deployment input image-only.
See the [scenario](https://github.com/glample/Arnold/blob/86af06d2fdb35c4bf552ecacfe8fe6ac1abd8cd4/src/doom/scenarios/deathmatch.py) and [network construction](https://github.com/glample/Arnold/blob/86af06d2fdb35c4bf552ecacfe8fe6ac1abd8cd4/src/model/utils.py).

Searches also found miniworld, OpenJev, and Laya.
These remain search leads rather than audited recommendations.
Their demonstrations do not establish stock Zandronum compatibility.

## Answers to the nine required questions

1. **Which pretrained models run?** ContinuousDoom passed strict loading and independent inference here. Arnold, DoomBot, and SF have checkpoint entries and loading paths. Their inference remains untested here.
2. **Which models use only images?** The audited ContinuousDoom path does. ViZDoom and SF permit image-only training. Their default deathmatch policies can use extra state. DoomBot, tetsuo, and Arnold deathmatch use extra state.
3. **Which models produce continuous mouse and keyboard actions?** ContinuousDoom produces yaw and one discrete choice. SF produces yaw and combined actions. tetsuo accepts DELTA values. These are engine actions until an adapter converts them to OS input.
4. **Can inference run without ViZDoom?** ContinuousDoom did so here. SF and SB3 forward passes can be separated in principle. Their exact preprocessing, recurrent state, and extra inputs remain necessary.
5. **Which projects support imitation and RL?** ViZDoom supplies demonstration capture and replay. ContinuousDoom, SF, Arnold, and DoomBot implement RL. Player-1 describes SFT and RL but did not release code. None supplies a complete MM8BDM BC system.
6. **Which directly control stock Zandronum?** None of the audited upstream projects supplies the required external adapter. The new ViMM8BDM adapter is an experiment. ZandronumMCP requires engine changes.
7. **Can an adapter transfer models to CBM and Gondola?** It can connect compatible images and actions. Useful combat transfer remains unverified. New visuals, projectiles, jumps, HUDs, maps, and skills will probably require target-game training.
8. **Which code merits direct reuse?** Prefer MIT components from SF and ViZDoom. Preserve their license notices. Resolve missing licenses before copying other projects. The repository contains only the new adapter code.
9. **Is a dedicated training environment necessary?** M1 and BC do not require a new engine. Target-game recordings are necessary. A small stock-client wrapper can serve first. Large-scale RL requires measured reset, reward, and throughput results before route C or D becomes justified.

## Search scope

Queries covered the six requested search terms:

- Doom AI player
- Zandronum AI
- Doom imitation learning
- ViZDoom pretrained model
- Doom visual agent
- Doom bot reinforcement learning

Further queries used screen, pixel, keyboard, mouse, and pretrained terms.
The audit does not claim that no other suitable project exists.
Public release queries returned empty lists for ContinuousDoom, tetsuo-doom, DoomBot, and Arnold.

## Remaining model checks

First measure ContinuousDoom on actual gameplay images and OS input.
Then compare its actions against the simple probe under the same conditions.
Keep the test separate from claims about combat skill.
Resolve its license before selecting it as the maintained base.

If a model needs incompatible state or actions, start target-game BC.
Do not spend weeks repairing an old Doom checkpoint solely to avoid a small BC model.
Use SF when parallel RL becomes a measured requirement.
