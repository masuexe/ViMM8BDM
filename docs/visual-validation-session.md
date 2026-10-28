# Separate visual validation recording

The second human combat recording is complete.
It supplies a separate session for later visual validation.
The source remains outside gradient training.
The intake review used no detector predictions.
The later [real-image transfer](real-body-transfer.md) tested a fixed final checkpoint on the reserved export.

## Recording result

The stock Windows alpha client loaded the original target packages and joined the stock local Cutman Bot server as an ordinary Player.
The scene used MM1CUT and the saved fixed Grenade Man configuration.
The recorder initially recorded the character source as client configuration.
After recording, the operator reported visible Cutman and Grenade Man throughout the session.
The operator report remains a separate review record.
The original recording files remain unchanged.

| Check | Result |
| --- | --- |
| Saved images | 1,192 |
| Saved input events | 4,613 |
| Recording segments | 1 |
| Dropped items | 0 |
| Archive integrity | Pass |
| Median capture duration | 7.53 ms |
| Maximum capture duration | 24.08 ms |
| Maximum interval between image starts | 53.18 ms |
| Automatic player input | None |

The archive checker verified all indexed PNG images, record counts, event sequence numbers, dimensions, offsets, and capture intervals.
The writer reported no write or cleanup error.
The session ended at the sixty-second duration limit.
The protocol permitted death and respawn images for visual analysis.
The recording does not establish action-label alignment or eligibility for behavior cloning.

## Candidate image review

The review selected twenty-four equally spaced images before detector output.
A development agent inspected the samples and positive box overlays.
Nine images contain one clear Cutman body each.
Eight images contain no clear body.
The review excluded seven images with uncertain body extent, severe projectile occlusion, translucent sprites, or death fragments.
The operator has not reviewed the box coordinates.
These annotations remain candidate labels.

The private COCO export contains seventeen original PNG images and nine boxes.
It preserves occlusion and image-edge clipping flags.
Its split is `validation`, and `reserved_for_validation` is true.
The export records `training_ready: false` and `action_training_eligible: false`.
The source hashes still match after export.

The reviewer did not use HUD names or frag text as body locations.
A frag message alone also does not establish that the local player is currently dead.
The message can coexist with a displayed health value after a transition.
The labels describe visible body extent and do not establish team membership or actor life state.

## Reservation

The recording identity and PNG archive hash differ from the first combat recording.
Its identity also differs from the synthetic background recording.
This separation supports a later check on a different session.
It does not establish performance on a different map or character.
Only sampled images have candidate body annotations.
The review does not label every frame in the complete recording.

The image exporter now preserves both `calibration` and `validation` review splits.
The current synthetic training loader rejects the real validation export.
The evaluator requires an explicit `--validation` option and complete project training provenance for this reserved split.
These checks prevent accidental use through those existing commands.
They do not replace a provenance check in any future real-image training loader.

Train the next experimental detector with reviewed images from the first recording.
Fix the training protocol and checkpoint-selection rule before the reserved prediction run.
Keep this second recording outside gradient training and checkpoint selection.
Report all accepted positive and negative samples in the later validation result.
Keep excluded images available for a separate difficult-case analysis.

The [aggregate evidence](evidence/visual-validation-session.json) records intake and reservation status.
Keep images, complete labels, previews, input traces, and local settings private.

Linux and Windows each passed all 214 tests without skips.
New tests check preservation of the validation split and rejection by the synthetic training loader.
