# Sprite perception baseline

This experiment tests a small visual reference before combat automation.
It uses original player sprites and the existing OpenCV template matcher.
It sends no game input and does not change the current navigation rule.
The baseline did not pass its conditions.

## Static source audit

The supplied Gondola actor definition names Cutman as a ClassBase player and uses the CUTM sprite prefix.
Its normal standing state uses frame A.
Walking, jumping, attacks, and missing-cutter states also use other frames.
The initial bank deliberately covers only the standing frame.
It does not cover all visible states of this character.

The source audit used the exact v6a, v9fh, and Gondola v2.7h2 packages in their normal load order.
The bank resolves the supplied standing sprite lumps and expands mirrored rotation pairs.
It supplies eight standing views from five original images.
Later packages contain other Cutman frames but do not replace those five standing images in this stack.
The bank records the source packages, member names, image hashes, and mirrored views.
The original package hashes remain unchanged.

Raw sprites contain blue and cyan colors that the game can replace.
The supplied CBM color source defines replacements for the Cutman color translation.
The original palette and translation source therefore matter even when the body shape remains fixed.
The experiment does not claim complete engine palette emulation.
Actual rendered colors remain a separate verification task.

The neutral mask retains opaque pixels whose largest and smallest RGB values differ by at most eight.
This keeps black and nearly white details while excluding the original saturated colors.
The opaque-mask control keeps all opaque pixels.
Neither mask establishes immunity to lighting, antialiasing, team colors, or custom skins.

## Reused library

The optional vision dependency is opencv-python-headless 4.13.0.92, with OpenCV 4.13.0.
The matcher calls masked TM_CCORR_NORMED and minMaxLoc.
The [official tutorial](https://docs.opencv.org/4.13.0/de/da9/tutorial_template_matching.html) describes this image-search method.
The audit also checked the tagged [matching implementation](https://github.com/opencv/opencv/blob/4.13.0/modules/imgproc/src/templmatch.cpp), including its mask handling and normalization.
Nonfinite scores become rejected scores, so a black region cannot become a perfect candidate through division by zero.

The [OpenCV top-level license](https://github.com/opencv/opencv/blob/4.13.0/LICENSE) is Apache 2.0.
The matching source also retains an Intel permissive notice.
The [Python wrapper license](https://github.com/opencv/opencv-python/blob/4.x/LICENSE.txt) is MIT.
Installed wheels include additional third-party notices.
The project calls the library without copying its implementation into the repository.
No pretrained weights are necessary for this baseline.

## Protocol and results

The matcher resizes the input to 320 by 240 pixels.
It searches the upper central scene at five template heights: 12, 16, 24, 32, and 48 pixels.
Each mask variant makes 40 template searches per frame.
The search excludes much of the HUD, weapon, and outer scene.
It returns one sprite-like candidate with a correlation score, not an enemy probability.
It cannot distinguish a teammate, enemy, decorative image, or other matching shape.

The experiment used 32 sampled map backgrounds and 48 artificial positive images per mask.
A contact-sheet review identified no other players in those sampled backgrounds.
This review does not establish every frame of the source recording.
The artificial cases place known standing sprites onto recorded backgrounds.
They vary rotation, brightness, interpolation, and simple source-backed color replacements.
They do not reproduce all engine rendering behavior.

Half the artificial cases use template heights present in the bank.
The other half use heights 20, 28, and 40, which are absent from the bank.
A successful case requires correlation of at least 0.97 and box intersection-over-union of at least 0.5.
The conditions also require zero negative candidates, recall of 0.9 at bank scales, recall of 0.8 at other scales, and median processing below 50 milliseconds.
The experiment recorded these conditions before evaluation.

| Mask | Candidates in 32 backgrounds | Recall at bank scales | Recall at other scales | Linux median ms |
| --- | --- | --- | --- | --- |
| Neutral | 0 | 0.625 | 0 | 132.99 |
| Opaque | 0 | 0.375 | 0 | 139.79 |

Neither variant passed the conditions.
The threshold remained at 0.97 after evaluation.
The neutral mask helped these artificial cases, but scale and rendering differences still caused failures.
A native Windows check on eight recorded images took 193.46 milliseconds at the median.
That preliminary check started immediately after a separate local client and server startup.
It is not a controlled comparison between operating systems.
It also did not establish the 50-millisecond processing target.

Linux and Windows each passed 194 tests after the addition.
New tests cover sprite precedence, mirrored views, neutral masks, known box recovery, black images, and changed asset hashes.
The matcher remains outside the live controller.
The [evidence file](evidence/sprite-perception.json) records source identities, case results, and timing.

## Reproduce

Install the optional dependency in the project environment:

```bash
python -m pip install -e ".[vision]"
```

Build a new private bank from the original packages:

```bash
python -m vimm8bdm.sprite_perception \
  --packages assets/MM8BDM-v6a.pk3 assets/classes-v9fh.pk3 assets/gondolapatchv2.7h2.pk3 \
  --output .local/perception/new-cutman-bank
```

Run the offline comparison with a new output directory:

```bash
python scripts/evaluate_sprite_perception.py \
  --recording data/wall-recovery-demo-1 --bank .local/perception/new-cutman-bank \
  --base-game assets/MM8BDM-v6a.pk3 --output runs/sprite-perception-new
```

The example asset directory is a local placeholder.
Keep extracted sprites, screenshots, and reference banks private.
Do not commit game art or original packages.

## Real appearance collection

The earlier map recordings supply no reviewed positive examples of another player.
A new stock local server loads the original target stack and one original Cutman Bot.
The normal client joins as an ordinary Player with fixed Grenade Man.
A client combat message established that Cutman was active after startup.
No custom server component or engine change is necessary.

The new window has the local caption `ViMM8BDM Cutman capture` to distinguish it from the older empty-map client.
The recorder remains paused until F8.
It records human play for 60 seconds and never controls the player.
You can move, jump, and attack normally.
Pause before menus or death and respawn transitions when possible.
Review any remaining interruptions before action training.

The primary purpose is real player-appearance data for visual box review.
Input labels remain provisional until a separate combat review.
This is not an automated bot match or a successful aiming policy.
The recording remains pending.

The first collection is complete.
The [visible-body review](visible-body-calibration.md) supplies candidate COCO boxes and real-image matcher results.
Both masks located zero of nine reviewed positive images.
