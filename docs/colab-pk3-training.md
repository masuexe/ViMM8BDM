# Version-agnostic PK3 extraction and Colab GPU training

The detector training pipeline now extracts upright character views and foreground projectile or hit-smoke occluders from any ordered PK3 stack and packages them into a self-contained `.tar.gz` training bundle for local or Google Colab GPU execution.

## Version-agnostic PK3 sprite and occluder resolution

Different MM8BDM releases store base skin sprites under different prefixes.
`MM8BDM-v6a.pk3` stores Cutman skin frames under `CUTM`, whereas `mm8bdm-v6b.pk3` and `mm8bdm-v6c.pk3` declare `sprite = $CUT` in `SKININFO.txt` and store frames `A` through `H` as `$CUTA1.PNG` through `$CUTH8H2.PNG` while keeping `CUTMI` and `CUTMJ`.
`build_bank` in `src/vimm8bdm/sprite_synthesis.py` now parses `SKININFO` and `SKININFO.txt` across the supplied PK3 load order and normalizes both `CUTM` and `$CUT` frames so later addon packages override earlier packages.
Full `v6a` Gondola stacks (`MM8BDM-v6a.pk3` plus `classes-v9fh.pk3` and `gondolapatchv2.7h2.pk3`) supply all 17 upright poses (`136` views); `classes-v9fh.pk3` and `gondolapatchv2.7h2.pk3` target `MM8BDM-v6a.pk3` and cannot be launched with `mm8bdm-v6b.pk3`.
Base-only `v6b` stacks (`mm8bdm-v6b.pk3` plus `8bdm-v6b-miscfixes-v6.pk3`) can pass `--allow-partial-poses` to extract the eight upright base poses (`64` views), and future `v6b`-compatible class/Gondola releases will resolve `$CUT` base frames alongside addon frames automatically.

`build_occluder_bank` extracts foreground projectile and hit-smoke sprites (`BUST`, `BUS2`, `CBUS`, `MBU2`, `MMDX`, `MMFX`, `MXP1`, `MXP2`, `MXP3`, `MXP4`, `NEDR`, `PCOI`, `RCTC`, `RCUT`, `W_0B`) across the same PK3 load order.
When `build_dataset` runs with `--profile occlusion` and `--occluders`, `composite` places foreground projectile or explosion sprites either partially in front of the target body (updating the ground-truth box to the remaining visible body pixels) or as separate foreground distractors in the scene.

## Post-attack occlusion grace, width-scaled association, and bounded horizontal search in `StableAim`

In `runs/body-aim-4` and `runs/body-aim-5`, single-frame self-projectiles and hit smoke immediately after a tap-fire pulse briefly occluded the target, and close/medium-range strafing or camera turns shifted consecutive boxes by `50-155 px` (`IoU < 0.1`) or varied partially occluded box area by up to `0.35x-2.65x`.
`StableAim` now associates consecutive single-candidate frames (`interval <= 0.2 s`) when `1/3 <= w*h/(pw*ph) <= 3` and either `IoU >= 0.1` or center displacement is at most `max(96, 1.35 * max(w, pw))` pixels.
When `enable_attack=True`, `StableAim` holds the prior target box and match count for at most one missing frame (`reason="post_attack_occlusion"`, `interval <= 0.2 s`) immediately after a frame that pulsed `attack=True`, advances the tracker timestamp to the occluded frame, and allows up to `max(144, 2.0 * max(w, pw))` pixels of two-frame displacement when the target reappears on the next frame (`interval <= 0.2 s`).
The occluded frame still emits `Action()` (zero mouse motion and `attack=False`).
When `enable_search=True` (`--enable-search`), `StableAim` records the horizontal side of the most recent valid single candidate (`search_direction = -1` when `x + w / 2 < width / 2`, else `1`), widens the horizontal tracking rate limit to `160 * dt` (`16 px/frame`), waits one `missing_target` frame (`missing_streak == 1`) with zero motion, and on consecutive missing frames `2 <= missing_streak <= 120` emits a bounded horizontal-only turn `Action(mouse_dx=16 * search_direction, mouse_dy=0, attack=False)` (`reason="search_target"`, `matches=0`) before stopping at `missing_streak > 120`.
Any reappearing candidate or ambiguous frame immediately resets `missing_streak` to `0`.
When `enable_search=True` and a candidate reappears during an active search sweep (`2 <= missing_streak <= 120`), `StableAim` immediately applies a horizontal search-brake alignment turn (`mouse_dx = max(-16, min(16, error_x * 0.15))`, `search_brake=True`) on the first confirmation frame (`matches == 1`), engages full 2D `track_target` on the second velocity-associated frame (`matches >= 2`), and still requires `matches >= 3` before unlocking `attack` pulses.
`filter_candidates` also allows upright close-range Cutman body boxes up to `68%` of the frame area (`0.2 <= w/h <= 1.1`), preventing point-blank targets from being rejected as `box_extent`.
`BodyShadowPolicy.__init__` prewarms both the full `480x640` frame shape and the `320x320` local-zoom crop shape on CUDA so the first `refine_with_local_zoom` call never incurs a cuDNN JIT compilation spike.
Replaying all 294 frames of `runs/body-aim-4` increases `track_target` frames from `67` to `93` and `attack=True` pulses from `7` to `9`, while both post-attack single-frame occlusions (`115 -> 116 -> 117 -> 118 -> 119` and `133 -> 134 -> 135 -> 136`) remain in `track_target` and two consecutive missing frames (`263 -> 264 -> 265` and `284 -> 285 -> 286`) reset confirmation to zero.
In attended 30-second live combat validation (`runs/body-aim-5`, `298/298` accepted frames on CUDA, `policy_ms` median `19.52 ms`, `loop_ms` median `39.28 ms`), all three dispatched attack pulses (`126`, `167`, `280`) triggered immediate `post_attack_occlusion` grace on the following frame (`127`, `168`, `281`) with `CUTMAN ENEMY` confirmed on the HUD; replaying `runs/body-aim-5` with the width-scaled association bounds eliminates all `17` consecutive-detection match resets, recovers both `167 -> 168 -> 169 -> 170 -> 171` and `280 -> 281 -> 282 -> 283 -> 284` into continuous `track_target`, and increases `track_target` frames from `33` to `54` (`7` attack pulses).
In subsequent closed-loop live validation with the width-scaled association tracker (`runs/body-aim-6`, `298/298` accepted frames on CUDA, `policy_ms` median `20.84 ms`, `loop_ms` median `41.54 ms`), consecutive-detection match resets dropped to `0` in live play, `track_target` frames rose to `85` (`83` nonzero mouse dispatches, `26`-frame longest continuous tracking run), and the policy dispatched `7` on-target attack pulses (`27`, `104`, `127`, `144`, `186`, `271`, `290`) with seamless post-shot tracking recovery.
In closed-loop live validation with `--enable-search` and `320x320` local-zoom CUDA prewarming (`runs/body-aim-8`, `298/298` accepted frames on CUDA, `policy_ms` median `17.98 ms` and maximum `51.14 ms`, `loop_ms` median `36.91 ms` and maximum `80.02 ms`), the policy executed `188` directional `search_target` frames (`39` left at `mouse_dx=-16`, `149` right at `mouse_dx=16`), achieved `16` search-to-target reacquisitions across corridor and behind-player exits (including 70-frame and 55-frame sweeps that turned around to reacquire Cutman), and dispatched `3` on-target attack pulses (`44`, `264`, `281`) with `CUTMAN ENEMY` confirmed on the HUD.
In closed-loop live validation with search-brake alignment, 2-frame tracking reacquisition, and close-range body area allowance (`runs/body-aim-9`, `298/298` accepted frames on CUDA, `policy_ms` median `17.67 ms` and maximum `68.76 ms`, `loop_ms` median `36.14 ms` and maximum `86.86 ms`), `box_extent` rejections dropped from `22` to `2`, all `12` search reacquisitions triggered immediate `search_brake` alignment (`4`, `17`, `32`, `42`, `48`, `53`, `58`, `73`, `167`, `194`, `213`, `220`), nonzero mouse dispatches rose to `257` (`93`-frame longest continuous motion run), `in_attack_window` frames doubled from `3` to `6` (`33`, `34`, `74`, `75`, `197`, `207`), and the policy dispatched `4` on-target attack pulses (`34`, `75`, `197`, `207`) with `CUTMAN ENEMY` confirmed on the HUD.

## Export a portable Colab GPU bundle

Export a self-contained `.tar.gz` bundle from any PK3 stack and reviewed background recording:

```bash
PYTHONPATH=src .venv/bin/python -m vimm8bdm.colab_bundle export \
  --packages .local/wads/MM8BDM-v6a.pk3 .local/wads/classes-v9fh.pk3 .local/wads/gondolapatchv2.7h2.pk3 \
  --initial .local/perception/torchvision/ssdlite320_mobilenet_v3_large_coco-a79551df.pth \
  --recording .local/recordings/background-session \
  --frames 0 76 153 230 307 384 461 538 \
  --train-count 512 --validation-count 128 \
  --output .local/bundles/cutman-occlusion-bundle.tar.gz
```

To package an existing reviewed mixed dataset (`data/cutman-real-mixed-3`) with the `v6a` + Gondola stack, augment its training split with foreground occluders extracted from the PK3 stack (`--augment-occluders`), and start from an existing transfer checkpoint:

```bash
PYTHONPATH=src .venv/bin/python -m vimm8bdm.colab_bundle export \
  --packages .local/wads/MM8BDM-v6a.pk3 \
             .local/wads/classes-v9fh.pk3 .local/wads/gondolapatchv2.7h2.pk3 \
  --initial .local/weights/projectile-transfer-model.pt --initial-transfer \
  --dataset data/cutman-real-mixed-3 --augment-occluders 64 \
  --output .local/bundles/cutman-v6a-mixed-bundle.tar.gz
```

The exported archive contains `dataset/`, `initial/checkpoint.pt`, `upright_bank/bank.json`, `occluder_bank/occluders.json`, `colab_manifest.json`, `src/vimm8bdm/`, and a standalone `train_colab.py` script.
It excludes raw `.pk3` and `.vrec` archives so it can be uploaded directly to Google Drive or a Colab session.

## Run GPU training on Google Colab or local CUDA

Inside a Google Colab GPU runtime (A100, L4, or T4) or on a local CUDA machine, run training directly from the `.tar.gz` archive:

```bash
PYTHONPATH=src python3 -m vimm8bdm.colab_bundle train \
  --bundle cutman-occlusion-bundle.tar.gz \
  --output trained-detector \
  --device cuda --epochs 8 --batch-size 16 --learning-rate 0.0005
```

Or unpack the archive and run the bundled script:

```bash
tar -xzf cutman-occlusion-bundle.tar.gz
python3 train_colab.py --output trained-detector --device cuda --epochs 8 --batch-size 16
```

`train_from_bundle` checks every SHA256 digest in `colab_manifest.json` before starting SGD, rejects unsafe archive paths, writes `trained-detector/model.pt` and `trained-detector/colab_bundle_provenance.local.json`, and produces a checkpoint compatible with `BodyShadowPolicy` and `BodyAimPolicy`.
