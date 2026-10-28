# Expanded and corrected body-image review

The next detector experiment uses more reviewed images from the first combat recording.
It also corrects one missed body in an earlier training export.
The second recording remains outside gradient training.
The experiment retains the earlier validation result and reports the next use as repeated validation.

## Review method

The review selects every twelfth frame from index six and removes all previously sampled indices.
This selection supplies 95 additional images from the 1,193-frame first recording.
The frozen synthetic baseline supplies up to two preliminary boxes per image at a score of at least 0.2.
These boxes assist annotation only.
They do not enter the accepted labels without visual review.
The development agent inspected the complete contact sheets, ambiguous full-size views, small-region enlargements, and accepted box overlays.
The operator has not reviewed these coordinates.

The review accepts thirty positive images and forty-one negative images.
It excludes twenty-four images with uncertain body extent, severe effects, or death fragments.
Some accepted bodies have no baseline proposal.
Some high-scoring baseline proposals cover projectiles rather than bodies and do not enter the body labels.
The labels remain experimental visible-body candidates.
They do not establish actor life state, team membership, or valid player actions.

A color scan assists the second review of negative examples.
It locates small magenta regions for visual inspection.
It does not classify those regions automatically.
Several regions are ordinary yellow pickups with magenta borders.
The review retains these resource images as negatives.
It also identifies small real bodies and ambiguous sprite strips that a reduced contact sheet can hide.

## Earlier annotation correction

Frame 1011 contains a small Cutman body partly behind the arrow sign.
The earlier review incorrectly labelled this image as negative.
The first real-image training mixture therefore used it as an empty background when adding a synthetic body.
This creates a false-negative training label for the original small body.
The review does not establish how much this error changed the earlier model's predictions.

The new export labels frame 1011 as positive.
A second change excludes frame 1037 because a tiny magenta strip behind a door has uncertain body extent.
The original reviews, exports, raw images, and model results remain unchanged.
The corrected labels use new review files and export directories.
Earlier score tables remain results under their original candidate annotations.
The separate validation recording has no annotation change.

## Combined first-recording review

| Reviewed group | Positive images | Negative images | Excluded images |
| --- | --- | --- | --- |
| Corrected first sample group | 9 | 10 | 5 |
| Corrected additional sample group | 5 | 12 | 6 |
| New dense sample group | 30 | 41 | 24 |
| Complete reviewed set | 44 | 63 | 35 |

The 142 reviewed indices are distinct.
The accepted export contains 107 real images.
Adjacent training frames remain correlated and do not count as independent sessions.

## Training protocol

The mixture builder retains 512 synthetic training images.
It adds sixteen appearance variants for each of the 44 real positives and eight sprite composites for each of the 63 real negatives.
This produces 1,720 training images.
The previous 128 synthetic monitoring images remain unchanged.
All variants and composites remain private.

The run starts from the same frozen synthetic checkpoint as the first real-image experiment.
It uses four epochs, a learning rate of 0.0005, batches of four images, four CPU threads, and random seed zero.
It retains momentum of 0.9, weight decay of 0.0005, and the gradient-norm limit of ten.
The final epoch supplies the checkpoint.
The larger dataset also increases the number of optimizer steps.
Any result cannot isolate label count, label correction, and additional optimization steps.

The score threshold remains 0.5, and the minimum box intersection-over-union remains 0.5.
The validation recording stays outside training and checkpoint selection.
Its annotation and image hashes remain unchanged.
However, the earlier validation result already informed model development.
The next report therefore uses `--validation-reuse` and records `independent_test: false`.
A separate training-recording history still supports `independent_session_validation: true`.
This distinction prevents a repeated comparison from appearing to be a new blind test.
A later independent test remains necessary before a broader quality claim.


## Result

The four-epoch Linux CPU run completed 1,720 optimizer steps in 397.84 seconds.
The final checkpoint located 96 of 128 synthetic monitoring bodies and produced six additional unmatched predictions there.
These monitoring results do not measure independent real-image quality.

On the reused real validation images, Linux and native Windows each located seven of nine bodies.
Neither run produced a candidate in the eight negative images.
Each run produced seven candidate boxes in total.
The model still fails the required recall of at least 0.8.
The score and box-overlap thresholds remain unchanged.

Linux median processing took 23.01 milliseconds, with a maximum of 33.52 milliseconds.
Native Windows median processing took 38.64 milliseconds, with a maximum of 47.65 milliseconds.
These values exclude screenshot capture, disk decoding, model loading, and input execution.
The same final checkpoint and thresholds supplied both runs.

The HUD-adjacent body and the small partly hidden body remain below the score threshold.
The previously near-threshold location now exceeds the box-overlap threshold.
These changes do not increase the total accepted location count.
More annotations and the label correction therefore do not establish a better accepted detector in this comparison.
The review cannot isolate which changes affected each prediction.

Linux and Windows each passed all 222 tests without skips.
The original recording files, original review files, and reserved validation annotations retain their saved hashes.
The [aggregate evidence](evidence/annotation-expansion.json) records the correction, data counts, training, and repeated validation.
Automatic aim and attack remain disabled.
The next experiment will test local image enlargement for weak candidates and measure its processing cost.
