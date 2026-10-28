# Transfer with reviewed real body images

This experiment adds reviewed images from the first combat recording to the existing synthetic detector training.
The fixed final model then runs on the separate second recording.
It locates seven of nine candidate body boxes, the same count as the frozen synthetic baseline.
The new model does not pass the required recall condition.
It remains outside the live controller.

## Training images and review

The first recording supplies thirteen reviewed positive images and twenty-four reviewed negative images across its two calibration exports.
The development agent checked those images and body overlays before this experiment.
An explicit image-training review binds the export metadata and annotation hashes.
It permits experimental body-image gradients and excludes action training.
The operator has not reviewed the box coordinates.
The labels do not establish team membership or actor life state.

The original calibration exports remain unchanged.
Their earlier scores remain historical results from models without real-image gradients.
Those exports no longer supply held-out evaluation for this new model.
The training builder rejects any export with a reserved validation split, even when an image-training review names it.

| Training component | Images |
| --- | --- |
| Existing first-profile synthetic images | 512 |
| Thirteen real positives with sixteen appearance variants each | 208 |
| Sprites on twenty-four reviewed empty real images, eight variants each | 192 |
| Complete training mixture | 912 |

The real-positive variants preserve image geometry and apply only small saturation, contrast, and brightness changes.
The first variant retains the resized original RGB appearance.
Repeated variants do not constitute additional independent real examples.
The empty real images require a pasted body because the reused SSD loss supplies no gradient for an entirely empty image.
These pasted bodies remain outside the upper-left HUD region.
This placement preserves the original HUD as background for the existing hard-negative selection.
The composites do not reproduce full game geometry or depth occlusion.

The mixture keeps the previous 128 synthetic validation images for training monitoring.
They supply no real-session checkpoint selection.
Generation seed nineteen determines the additional training images.
All complete image sets, labels, model weights, and source reviews remain private.

## Fixed training protocol

The run starts from the first synthetic checkpoint.
It retains the SSDLite320 architecture and all trainable parameters.
The protocol fixes four epochs, batches of four images, four CPU threads, and random seed zero.
SGD uses a learning rate of 0.0005, momentum of 0.9, and weight decay of 0.0005.
The gradient-norm limit remains ten.
The final fourth epoch supplies the checkpoint.
No real validation score selects a checkpoint or changes a threshold.

The Linux CPU run completed 912 optimizer steps in 215.32 seconds.
Mean training loss decreased from 0.7628 to 0.4222.
The final checkpoint located 95 of 128 synthetic monitoring bodies.
It also produced four unmatched predictions in those positive monitoring images.
Training loss and monitoring results do not establish independent real-image quality.

The checkpoint records both project training-recording identities and their image-archive hashes.
The program rejects an inherited training recording absent from the new dataset provenance.
It also rejects changed real annotations or source images that do not match the explicit image review.
The source recordings remain unchanged.

## Separate-session validation

The second [visual recording](visual-validation-session.md) supplies seventeen accepted images: nine positive and eight negative.
Its recording identity, image archive, export metadata, and candidate annotation hashes match the reservation made before training.
The recording remains outside gradients and checkpoint selection.
The images come from the same map and player class as development.
This experiment does not measure an unseen map or character.
The annotations remain development-agent candidates without human box-coordinate review.

Evaluation requires both `--transfer` and `--validation` for the reserved split.
It checks the complete project training-recording history.
A matching recording identity or matching image-archive hash stops evaluation before prediction.
The archive check also rejects a copied recording with different event metadata.
The script does not infer independence from a validation label alone.

The legacy synthetic baseline has only one recording identity in its checkpoint.
Its exact original synthetic dataset metadata supplies the missing archive provenance.
The loader requires the metadata digest recorded inside that checkpoint.
It also checks the source identity and synthetic format before completing the provenance.
The original checkpoint bytes remain unchanged.

All runs retain a score threshold of 0.5 and a minimum box intersection-over-union of 0.5.
The conditions require recall of at least 0.8, zero candidate images among the negatives, and median processing within fifty milliseconds.

| Model | Located positive images | Negative images with candidates | Total candidate boxes |
| --- | --- | --- | --- |
| Frozen synthetic baseline | 7 of 9 | 0 of 8 | 7 |
| Synthetic and reviewed real mixture | 7 of 9 | 0 of 8 | 8 |

Both models fail the recall condition.
The new model recovers one previously missed body beside the HUD.
Another body moves from box overlap 0.508 to 0.492, crossing the fixed threshold.
The candidate box coordinates did not change after prediction.
The near-threshold result is sensitive to annotation extent and does not establish a stable improvement.
One small, partly occluded body remains undetected by both models.
The box-overlap rule leaves the additional candidate unmatched.
It does not automatically establish another actor in the scene.

Linux median processing for the new model took 26.33 milliseconds, with a maximum of 35.07 milliseconds.
The first native Windows check reproduced the detection counts while unit-test startup could share CPU resources.
The separate native timing check took 37.83 milliseconds at the median and 54.47 milliseconds at the maximum.
That repeat used the same frozen model and thresholds without the concurrent unit-test job.
Processing measurements include tensor conversion, inference, postprocessing, and filtering after five warmup calls.
They exclude screenshot capture, disk decoding, model loading, and input execution.

The [aggregate evidence](evidence/real-body-transfer.json) records training, provenance checks, and validation results.
Linux and Windows each passed all 222 tests without skips.
The added tests cover mixed data provenance, reserved data rejection, changed annotations, inherited sources, and legacy metadata matching.

The experiment does not establish automatic aim, attack, or multiplayer readiness.
Keep the frozen synthetic model as the offline comparison baseline.
Add reviewed training examples from the first recording before the next detector revision.
Retain the current validation result when reporting later experiments.

## Reproduce

Create an explicit private body-image training review for the first recording's calibration exports.
The review must bind both metadata and annotation hashes.
It must permit image gradients and exclude action training.
Then create a new mixture and output directory:

```bash
python -m vimm8bdm.real_body_data \
  --synthetic data/cutman-synthetic-1 \
  --calibrations data/cutman-body-calibration-1 data/cutman-body-calibration-extra-1 \
  --bank .local/perception/cutman-upright-1 \
  --review runs/real-body-transfer-1/image-training-review.local.json \
  --output data/new-real-body-mixture
python -m vimm8bdm.sprite_transfer \
  --dataset data/new-real-body-mixture \
  --initial runs/sprite-transfer-1/model.pt --initial-transfer \
  --learning-rate 0.0005 --epochs 4 --output runs/new-real-body-transfer
python scripts/evaluate_pretrained_detection.py \
  --dataset data/cutman-body-validation-1 \
  --checkpoint runs/new-real-body-transfer/model.pt --transfer --validation \
  --output runs/new-real-body-validation
```

For the original synthetic baseline, also supply its exact metadata with `--provenance-dataset data/cutman-synthetic-1/dataset.json`.
Use new directories for all outputs.
Do not transfer any reserved validation export into the training mixture.

A later [annotation review](annotation-expansion.md) identified a missed body in the first recording and excluded one uncertain strip.
New exports contain the revised training labels.
The earlier tables retain their original candidate annotations and results.
