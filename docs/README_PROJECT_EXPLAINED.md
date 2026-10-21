# Project walkthrough

## What the project does

The repository experiments with image-based saffron authenticity
classification. It treats a **physical sample** as the independent unit:
several uploaded or captured views may belong to one sample, and every view
must remain together during train/validation/test splitting and scoring.

The deep architecture combines:

1. **ConvNeXt-Tiny**, which creates spatial visual features.
2. **Grid graph attention**, which passes messages between neighbouring
   feature-map patches.
3. **HAFM**, which combines the global CNN feature with the graph features.
4. **Wombat-OA / weighted beam decoding**, which tunes and combines predictions
   from multiple heads and test-time views using validation data.

This describes the implementation, not evidence that the model learned
saffron-specific cues.

## Why the evaluation is cautious

The dataset has fewer independent samples than images. Camera model, flash,
ISO, exposure, image size, colour, and post-capture editing can correlate with
the label. A random image split could place different views of the same
physical sample on both sides and inflate scores, so the scripts use the
manifest `group` field for splits, aggregation, bootstrap intervals, and
permutation tests.

Shortcut baselines are deliberately included. They answer whether simple
metadata, size, or colour features achieve comparable performance without
the deep architecture. The controlled-manifest tools and capture protocol
are intended to reduce confounds in future data collection, not to repair
confounded evidence after the fact.

## Typical workflow

1. Prepare a manifest with stable physical-sample group IDs and inspect
   duplicates and capture metadata.
2. Run audit and shortcut baselines before training.
3. Run grouped cross-validation on a GPU; preserve fold predictions,
   configuration, weights, and thresholds.
4. Generate reports only from complete result files that pass provenance
   checks. Smoke runs and incomplete folds are rejected.
5. Compare the deep model with shortcut baselines and report camera,
   flash/ISO, session, and date strata when data permit.
6. Calibrate abstaining inference and test it on a separately collected,
   hash-locked prospective set.

Useful commands are in the root [README.md](../README.md). For safe
prospective evaluation, see [preregistration.md](./preregistration.md) and
[capture_protocol.md](./capture_protocol.md).

## Reading an app output

- **P(fake)** is the model's score, not a probability of laboratory-confirmed
  adulteration.
- **REAL** or **FAKE** is issued only when the required model, calibration,
  capture-support, image-count, uncertainty, OOD, and fold-agreement checks
  pass.
- **INCONCLUSIVE** means the app lacks adequate evidence for a primary
  decision. It is not a real/fake classification.
- The colour-statistics mode is a baseline-only comparison and never issues a
  primary verdict.

The app is a research interface. Do not use it as a substitute for chemical
analysis or independent authentication.
