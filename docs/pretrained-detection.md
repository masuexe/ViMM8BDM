# Pretrained object detector check

The official SSDLite320 COCO weights loaded and ran on recorded game images.
The unchanged model did not identify any reviewed Cutman body at the fixed score threshold.
The model remains outside the live controller.

## Reuse choice

The initial comparison considered SSDLite320 and YOLOX Nano.
Neither model supplies a Cutman category.
A COCO person prediction cannot establish an enemy or a living game actor.

| Candidate | Source inspection | Local experiment | Decision |
| --- | --- | --- | --- |
| TorchVision SSDLite320 MobileNetV3 | Builder, preprocessing, output, checkpoint entry, and license | Strict weight loading and image inference | Use first for an offline transfer check |
| YOLOX Nano | Model configuration, Hub entry, requirements, and license | Not installed or run | Retain as an alternative |

SSDLite320 has a small official weight file and uses the existing PyTorch environment.
Its public implementation supports image inference and supervised box training.
The [tagged builder](https://github.com/pytorch/vision/blob/v0.29.1/torchvision/models/detection/ssdlite.py) defines the architecture and COCO weight URL.
The [SSD implementation](https://github.com/pytorch/vision/blob/v0.29.1/torchvision/models/detection/ssd.py) accepts image tensors without engine variables.
It returns boxes, labels, and scores during evaluation.
Training accepts image boxes and category labels.
The code also provides the matching, loss, and postprocessing functions.
A project-specific detector does not require a new detection algorithm.

The inspected [YOLOX Nano configuration](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/exps/default/yolox_nano.py) uses depthwise blocks and a 416-pixel input.
The [Hub entry](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/hubconf.py) exposes Nano and custom models.
The [requirements](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/requirements.txt) include TorchVision, OpenCV, COCO tools, and export tools.
These training dependencies exceed the requirements of this first inference check.
This inspection does not establish YOLOX compatibility or speed in the project environment.

## License and weight identity

TorchVision code uses the [BSD 3-Clause license](https://github.com/pytorch/vision/blob/v0.29.1/LICENSE).
YOLOX code uses the [Apache 2.0 license](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/LICENSE).
The code license alone does not establish all pretrained-weight terms.
The [TorchVision model documentation](https://docs.pytorch.org/vision/stable/models.html) identifies possible dataset-derived terms for pretrained models.
The current experiment keeps the official checkpoint private and does not redistribute weights.
It does not assign the code license to those weights.

The checkpoint came from the official URL in the tagged builder:

```text
https://download.pytorch.org/models/ssdlite320_mobilenet_v3_large_coco-a79551df.pth
```

The file contains 14,069,355 bytes.
Its full SHA-256 is `a79551df90c79834bcd3bb3845ef9d966b5449a3a9b2833ae8404778ca5d65d2`.
The prefix agrees with the official filename.
The loader requires this full hash and uses strict state loading.
It loads tensor weights on the CPU with `weights_only=True`.
It does not download code or weights implicitly.

## Protocol

The check uses the same twenty private [calibration images](visible-body-calibration.md).
Nine images have one candidate Cutman body box each.
Eleven images have no clear body box.
A development agent supplied these annotations.
The operator has not reviewed the coordinates.
These images are not an independent test set.

The evaluator uses the complete RGB image, including the weapon and HUD.
It converts pixels to floating-point values between zero and one.
The existing model performs its own normalization and fixed-size image transform.
It retains COCO category 1, person, at a score of at least 0.5.
A successful location requires box intersection-over-union of at least 0.5.
These thresholds remained unchanged after the result.

The preliminary conditions require at least 0.8 recall, no negative-image candidates, and median processing within 50 milliseconds.
The check runs one image per call, four CPU threads, and five warmup calls.
Measured processing includes tensor conversion, inference, postprocessing, and person filtering.
It excludes disk decoding, model loading, screenshot capture, and game input.
No training or game control occurs.

## Result

| Measurement | Linux CPU | Native Windows CPU |
| --- | --- | --- |
| Located positive images | 0 of 9 | 0 of 9 |
| Negative images with person predictions | 0 of 11 | 0 of 11 |
| Median processing | 39.09 ms | 55.60 ms |
| 95th percentile processing | 48.19 ms | 60.59 ms |
| Maximum processing | 48.63 ms | 72.32 ms |

The Linux run passed the preliminary processing condition but failed the recall condition.
The Windows run failed both conditions.
These runs do not isolate operating-system effects from other system activity.
No frame produced a person prediction at the fixed threshold.
This result rejects direct use of these unchanged weights for Cutman localization in this recording.
It does not establish that supervised adaptation will fail.
It also does not establish detection performance on other characters or maps.

Both experiments used Torch 2.14.1+cpu and TorchVision 0.29.1+cpu.
The installed native nonmaximum-suppression operation also passed a one-box check.
The GPU is visible to the host, but this CPU-only Torch installation cannot use CUDA.
The result does not measure GPU inference.
The [aggregate evidence](evidence/pretrained-detection.json) records checkpoint identity and processing results.

## Reproduce

Use a project Python environment with the tested CPU packages:

```bash
python -m pip install torch==2.14.1 torchvision==0.29.1 \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[detector]"
```

Download the official checkpoint to a private local directory before evaluation.
Supply a new output directory:

```bash
python scripts/evaluate_pretrained_detection.py \
  --dataset data/cutman-body-calibration-1 \
  --checkpoint .local/perception/torchvision/ssdlite320_mobilenet_v3_large_coco-a79551df.pth \
  --output runs/new-ssdlite-calibration
```

Unit tests do not require a checkpoint or network access.
They check score and category filtering, invalid values, image bounds, checkpoint identity, and box overlap.
Linux and Windows each passed all 203 tests without skips.

## Next step

Reuse the existing detector for a small supervised transfer experiment.
Start with private synthetic images from supplied upright player sprites and recorded map backgrounds.
A read-only audit of the supplied Gondola Cutman actor identified seventeen normal-action frames.
The target package stack supplies 85 sprite lumps for those frames.
These include standing, walking, jumping, attacking, and missing-cutter poses.
The proposed positive set excludes the H and U frames shared by pain and death entries.
The audit does not establish that pasted sprites reproduce complete game rendering.
Keep the real calibration images outside that synthetic training set.
Then measure whether transfer improves body location without excessive false candidates.
A later independent session remains necessary before a live recognition test.
Automatic aim and attack remain disabled.
