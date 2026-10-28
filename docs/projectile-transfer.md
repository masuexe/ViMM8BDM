# Targeted projectile-image training

The continuous scan found a separate projectile with a body label in the second recording.
This experiment adds reviewed examples from the first recording only.
The second recording remains outside gradient training.
All later comparisons reuse that recording for development checks.
They are not new independent tests.

## Targeted review

The expanded-image detector scanned all 1,193 first-recording images without local zoom.
This scan is training-data inspection, not validation.
The private mining copy explicitly records training resubstitution.
The public evaluation scanner continues to reject training recordings.

The selection removes all 142 previously reviewed indices.
It selects every fifth remaining candidate-bearing image and all remaining multiple-candidate images.
This produces seventy images for contact-sheet inspection.
A bounded subset receives full-size review and accepted training labels.
The other selected images do not receive accepted labels in this batch.

Five accepted images contain a visible Cutman body with a projectile, explosion, or nearby resource.
Two accepted images contain no visible Cutman body.
One negative contains an explosion and a yellow resource.
The other contains a large blue-and-white projectile.
The review inspects the accepted box overlays and retains occlusion flags.
One positive contains a partly occluded body.
The agent supplied these candidate coordinates.
The operator has not reviewed the boxes.
These image labels do not establish valid player actions or live enemy identity.

## Training protocol

The new export adds seven distinct first-recording frames.
The combined real image set contains 49 positives and 65 negatives.
The mixture retains sixteen appearance variants per real positive and eight sprite composites per real negative.
It also retains the earlier 512 synthetic training images.
This produces 1,816 training images.
The 128 synthetic monitoring images remain unchanged.
They do not constitute independent real validation.

Training starts from the same frozen synthetic checkpoint as the expanded-image experiment.
It uses four epochs, batches of four images, four CPU threads, and random seed zero.
The learning rate remains 0.0005.
Momentum, weight decay, and gradient clipping remain unchanged.
The final epoch supplies the checkpoint without validation-based selection.
The larger dataset also changes the number and order of optimizer steps.
The experiment cannot isolate the effect of each additional image.

Before training, the loader checked every generated image hash and box boundary.
The reserved recording hashes remained unchanged.
Neither its recording identity nor its image-archive hash appears in the training sources.
The raw recordings and earlier exports remain unchanged.
Automatic aim and attack remain disabled.

## Result

Four training epochs completed 1,816 optimizer steps in 414.66 seconds.
The final synthetic monitor located 90 of 128 bodies and produced three unmatched predictions.
These values do not measure independent real-image quality.

Repeated validation located eight of nine bodies on Linux and native Windows.
Neither platform produced a candidate in the eight negative images.
Each produced nine boxes in total.
Frame 259 contains two overlapping boxes for its one annotated body.
The small partly hidden body in frame 362 remains undetected.
The three existing sample conditions pass, but they do not reject extra boxes on positive images.
The result therefore does not establish sufficient combat quality.

The previously observed projectile in frame 965 no longer receives a body candidate.
The visible body in that frame still receives a candidate, with a score of approximately 0.564.
This is one known development example, not a new independent test.

Both platforms also scan all 1,192 second-recording images.
They produce 535 boxes in 523 candidate-bearing images.
Twelve images contain multiple candidates.
The scan starts 267 additional crop calls and records 187 candidate-presence changes.
Compared with the preceding model, 59 images lose all candidates and 25 images gain candidates.
These counts do not distinguish valid detections from errors.
They do not establish a lower false-positive rate.

The agent inspected 41 selected comparison images with overlays.
The selection includes sparse lost detections, sparse gained detections, all multiple-box frames, and earlier failure examples.
The large combined body-and-projectile box in frame 447 remains.
The large foreground candidates during local player death in frames 677 and 808 also remain.
The review shows continuing confusion.

| Continuous processing measurement | Linux CPU | Native Windows CPU |
| --- | --- | --- |
| Median | 28.18 ms | 30.99 ms |
| 95th percentile | 57.12 ms | 63.01 ms |
| Maximum | 181.25 ms | 89.15 ms |

The Linux maximum is an observed slow frame and remains in the report.
The review does not establish its cause.
The different timing results do not establish that the new weights make inference faster.
The model architecture, thread count, and local-zoom procedure remain unchanged.
These measurements exclude current screenshot capture.

The raw recordings retain their saved hashes after evaluation.
No model, game resource, image, or raw recording enters the public repository.
The [aggregate evidence](evidence/projectile-transfer.json) records training, repeated validation, continuous scans, and the known projectile diagnostic.
Automatic aim and attack remain disabled.
The next step adds an observation-only runtime that measures actual capture and inference without game input.
