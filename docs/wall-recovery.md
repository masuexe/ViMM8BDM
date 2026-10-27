# Human wall recovery recording and training

The operator reported completion of a 120-second human recording on MM1CUT.
The session metadata specifies fixed Grenade Man and the official Windows alpha client.
The executable and target asset hashes match the earlier directed training recording.
The recorder sent no game input.
The operator reported completion of wall approaches and turns after slow movement along angled walls.
The operator reported no death, chat, menu, character change, or long wall obstruction.

## Recording checks

| Measurement | Result |
| --- | --- |
| Active duration | 120.024 seconds |
| Images | 2385 |
| Events | 3615 |
| Input events | 3612 |
| Segments | 1 |
| Dropped items | 0 |
| Median capture duration | 7.7673 ms |
| Maximum interval between images | 51.7085 ms |
| Candidate 50 ms samples | 2395 |
| Input events retained in labels | 3612 of 3612 |

The archive passed length, sequence, image, and timestamp checks.
Two final candidate windows lacked complete future coverage.
The builder omitted these windows without changing the source files.
The new training source has no session or image-archive overlap with the validation recording.

All images passed a basic 32-by-24 grayscale scan for nearly black or low-texture content.
The scan found 26 identical adjacent image grids.
It did not remove them or identify them as recording failures.
Six sampled images contained game views with the expected weapon and HUD.
Those six images contained no visible menu or desktop notification.
These checks do not establish every wall approach, recovery, or gameplay interruption.

The input contains 77 W events, four A events, and 3531 mouse events.
It contains no S, D, jump, attack, or wheel events.
The average held fractions are approximately 88.86 percent for W and 0.25 percent for A.
The last complete label retains W as held.
The recorder does not release human-held keys when it stops.
This state does not indicate stuck AI input.

## Training from multiple sessions

The offline trainer now accepts additional training datasets and their separate review files.
It checks every source before concatenating images and labels.
It rejects duplicate source sessions, duplicate image archives, and inconsistent maps or characters.
The existing split check rejects an archive that appears in training and validation.
The trainer requires one review for each additional dataset.

This experiment combines 1879 filtered directed-navigation samples with the 2395 new samples.
The combined training set contains 4274 samples.
The separate validation set still contains 1796 samples.
The combined W held fraction is approximately 63.96 percent.
The earlier session preserves the A, S, D, and stop examples missing from the new session.
No rule-generated actions enter training.

The run uses 50-millisecond targets, motion features, 12 epochs, seed 0, and batches of 64.
It uses four CPU threads and Adam with a learning rate of 0.001.
The architecture and loss remain unchanged.
Normalization uses the combined training labels only.
Epoch 7 supplied the selected checkpoint.
The complete training command took approximately 61.38 seconds.

At training time, full-session operator confirmation was pending for the new recording.
The operator supplied that confirmation after training.
A separate review record preserves this later report without changing the training-time review.
The validation recording still needs full-session operator confirmation.
The explicit experimental mode permits this candidate-data diagnosis.
All source and model status fields remain unsuitable for deployment.

## Results

The new model did not pass the pre-run decision conditions.
Its combined score of 0.500020 beat its own constant baseline score of 0.519782.
Both mouse errors still exceeded the zero-mouse baseline.
A, S, and D remained below the held-fraction decision threshold throughout validation.
The checkpoint remains offline.

The combined training data changes the mouse normalization scales.
The earlier model's original score of 0.617206 therefore cannot directly measure improvement against the new score.
The following comparison uses the new training scales for both models.
Lower error scores indicate smaller errors.

| Predictor | Common-scale score | Key cross-entropy | Mouse X MAE | Mouse Y MAE | Forward held-state F1 |
| --- | --- | --- | --- | --- | --- |
| Earlier motion model | 0.455559 | 0.260341 | 9.5630 | 0.4239 | 0.701706 |
| Combined-data model | 0.500020 | 0.287745 | 9.8543 | 0.4875 | 0.680247 |
| Combined-training constant | 0.519782 | 0.357587 | 7.7233 | 0.3630 | 0.538055 |

Mouse errors use raw counts per 50-millisecond window.
The new model performed worse than the earlier model on these validation measurements.
Adding this session did not establish an improvement in autonomous navigation.
This result does not show that the recording lacks useful information.

Shuffled images increased the new combined score to 0.642271.
Repeating only the latest image increased it to 0.573022.
These results show an image-history dependency, but they do not establish steering success.
The thresholded prediction sequence matched none of the 25 forward presses in their original windows.
It matched one of the 25 releases and produced 73 press predictions and 73 release predictions.
Occupancy F1 must not substitute for exact transition quality.

The validation images still show human actions rather than reactions to model commands.
The recording also supplies no independent labels for successful wall recovery.
The evaluation is not a live rollout, an unseen-map test, or an independent final test.
Keep the current visual rule unchanged with its unresolved operator feedback.

## Verification and reproduction

Linux and Windows each passed 167 tests after the training extension.
The tests check combined inputs, duplicate sources, character consistency, missing reviews, and split contamination.
A training test changes validation labels and checks identical combined-source parameter updates and normalization.
The saved checkpoint reproduced its predictions after loading.
Recordings, screenshots, review files, and weights remain private.
The [evidence file](evidence/wall-recovery.json) records the measurements and evaluation results.

```bash
python -m vimm8bdm.behavior_cloning \
  --train data/navigation-corners-train-clean-slices-1 \
  --extra-train data/wall-recovery-slices-1 \
  --extra-train-review runs/wall-recovery-demo-1/collection-review.local.json \
  --validation data/navigation-corners-validation-slices-1 \
  --train-review runs/navigation-corners-train-1/operator-review.local.json \
  --validation-review runs/navigation-corners-validation-1/collection-review.local.json \
  --output runs/wall-recovery-bc-1/new-artifacts --experimental \
  --features motion --action-ms 50 --epochs 12 --seed 0 \
  --batch-size 64 --threads 4 --learning-rate 0.001
```

Use a new output directory for repetition.
Do not overwrite the source recording or assign it to validation after training.
The new session now has the operator's full-session quality report.
This report does not establish mouse conversion or model performance.
Inspect steering targets and transition errors before requesting another recording or enabling model input.
