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

## Post-attack occlusion grace in `StableAim`

In `runs/body-aim-4`, single-frame self-projectiles and hit smoke immediately after a tap-fire pulse briefly occluded the target and forced a full three-frame re-confirmation.
When `enable_attack=True`, `StableAim` now holds the prior target box and match count for at most one missing frame (`reason="post_attack_occlusion"`, `interval <= 0.2 s`) immediately after a frame that pulsed `attack=True`, advances the tracker timestamp to the occluded frame, and allows up to `max(96, 1.35 * max(w, pw))` pixels of two-frame displacement when the target reappears on the next frame (`interval <= 0.2 s`).
The occluded frame still emits `Action()` (zero mouse motion and `attack=False`).
Replaying all 294 frames of `runs/body-aim-4` confirms that both post-attack single-frame occlusions (`115 -> 116 -> 117` and `133 -> 134 -> 135`) resume `track_target` immediately (`matches=7` and `matches=16`) instead of resetting to `matches=1`, while two consecutive missing frames (`263 -> 264 -> 265`) reset confirmation to zero.
In attended 30-second live combat validation (`runs/body-aim-5`, `298/298` accepted frames on CUDA, `policy_ms` median `19.52 ms`, `loop_ms` median `39.28 ms`), all three dispatched attack pulses (`126`, `167`, `280`) triggered immediate `post_attack_occlusion` grace on the following frame (`127`, `168`, `281`) with `CUTMAN ENEMY` confirmed on the HUD, and sequence `167 -> 168 -> 169` recovered directly into `track_target` (`matches=8`) through residual hit smoke.

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
