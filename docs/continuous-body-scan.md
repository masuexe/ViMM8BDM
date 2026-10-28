# Continuous body-detection scan

This experiment processes the complete second combat recording in order.
It retains the expanded-image checkpoint and conditional local zoom.
It performs no training and sends no game input.
The source recording already supplied repeated validation images.
This scan is not a new independent test.

## Measurements

The [scanner](../scripts/scan_body_recording.py) checks archive integrity and recording provenance before model inference.
It rejects a recording whose identity or image-archive hash appears in the checkpoint training sources.
It checks the source hashes again after processing.
It retains all per-frame results privately.

Each image receives one full-image model call.
A weak proposal can request one additional crop call under the existing local-zoom rules.
The score threshold remains 0.5.
The scanner records candidate boxes, crop coordinates, decoding time, and processing time.
It uses four CPU threads and five warmup calls.

The scan also counts changes between frames with and without candidates.
It compares the highest-scoring box in adjacent frames when both contain candidates.
A low overlap does not prove an identity change or a false detection.
Camera movement, occlusion, animation, death, and another body can change the visible box.
These counters identify sequences for visual review.
They do not measure recall or the false-positive rate.

The saved archive measures disk access and image decoding rather than current screenshot capture.
The report separately estimates capture plus processing from the original capture durations and current inference durations.
This estimate does not include queue delay or prove live control-loop latency.
The scan does not replay player actions.

## Result

Both platforms process all 1,192 images and retain the original source hashes.
They produce candidates in the same 557 images and produce 568 boxes in total.
Eleven images contain multiple candidates.
Conditional local zoom adds 136 candidate-bearing images beyond the 421 images accepted by the full-image pass.
This count does not prove that all added boxes are valid bodies.

The scan starts 333 additional crop calls.
Candidate presence changes 181 times between adjacent frames.
Among 466 adjacent pairs with candidates, 33 have top-box overlap below 0.1.
These diagnostic counts do not establish tracking quality.

| Processing measurement | Linux CPU | Native Windows CPU |
| --- | --- | --- |
| Median | 29.37 ms | 39.15 ms |
| 95th percentile | 61.36 ms | 80.64 ms |
| Maximum | 78.00 ms | 97.44 ms |
| Frames above 50 ms | 271 | 338 |

Native Windows archive access, decoding, and inference together have a median of 42.81 milliseconds.
The estimated original capture plus current inference has a median of 47.05 milliseconds and a maximum of 101.86 milliseconds.
Neither value measures a current live capture loop.
Candidate decisions and crop windows match across platforms.
Small numeric differences remain in scores and box coordinates.

## Visual failure review

The agent inspected 43 selected images with box overlays.
The selection includes sparse crop acceptances, all multiple-box images, and sparse adjacent box jumps.
Full-size inspection also checked three specific failure examples.
This model-assisted selection cannot estimate the overall false-positive rate.

Frame 965 contains a visible Cutman body and a separate black-and-white round projectile.
The projectile receives a body candidate with a score of approximately 0.867.
This is a clear non-body candidate.
Frame 447 has a large candidate box that covers both the body and a foreground projectile.
Its high score does not establish good body localization.
Frame 677 contains large foreground sprite candidates while visible HUD text reports local player death.
The detector does not establish those candidates as live enemies.

The earlier eight negative validation images therefore do not cover all real combat failure cases.
Their zero-candidate result remains valid for those images only.
The continuous scan demonstrates additional projectile and foreground confusion.
Automatic aim and attack remain disabled.
The next step reviews projectile negatives from the first training recording.
The second recording will remain outside gradient training.
A later independent recording remains necessary for broader quality measurement.
The [aggregate evidence](evidence/continuous-body-scan.json) retains the counts, processing times, and failure examples.

Linux and native Windows each passed all 226 regression tests without skips.
The complete archive scans provide separate execution evidence for the scanner.
