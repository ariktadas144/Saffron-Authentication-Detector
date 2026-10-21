# Prospective evaluation preregistration

This protocol must be completed and time-stamped before opening the labels or
running inference on a new prospective test set. It does not turn the existing
cross-validation data into an independent test set.

## Research question and scope

The primary question is whether the frozen, versioned Saffron model can
distinguish real from adulterated samples in a newly collected, controlled
capture set. The target is the physical sample, not an individual photo.
Results apply only to the phones, sessions, sample sources, and capture
conditions actually represented in that test set.

The existing dataset has known capture/class confounding, including the
SM-J600G ISO/flash association. It must not be used to tune the prospective
capture protocol, threshold, model, or decoder after the test labels are
available.

## Freeze before testing

Before the first inference, archive and hash:

1. The model run directory, including weights, per-fold configuration, decoder
   parameters, thresholds, and OOD references.
2. The calibration JSON and the manifest used to build it.
3. The prospective image files, capture log, and locked manifest.
4. The software revision and package versions, plus the exact command.

Use `scripts/lock_test_set.py` to create a write-protected, hash-checked
manifest. The evaluator in `scripts/evaluate_locked.py` claims that locked set
exactly once. A failed attempt consumes the claim; do not reset the marker or
retry on the same set. Preserve the signed report and HMAC key separately.

## Collection requirements

- Use at least three phones and at least 30 independent physical samples per
  class per phone.
- Photograph both classes interleaved in the same sessions, with flash off
  and fixed/logged ISO and exposure. Keep background, distance, framing, and
  illumination consistent within a phone/session block.
- Record a stable sample ID, label, phone, date/session, ISO, exposure, flash,
  white balance, and background for every image.
- Do not manually crop, selectively re-save, or apply class-dependent
  processing. Link all views of one physical sample to one group ID.
- Keep the prospective set inaccessible to model developers until the locked
  evaluation has completed.

See [capture_protocol.md](./capture_protocol.md) for the operational
collection steps.

## Frozen analysis

- Primary unit: one physical sample (`group`); multiple views are aggregated
  within that sample.
- Primary metrics: sample-level balanced accuracy and ROC AUC. Report
  sensitivity/recall for fake and specificity for real, plus coverage,
  abstention rate, and the number of samples and images.
- Report all metrics with 95% group-bootstrap intervals. Show camera,
  session/date, flash, and ISO strata where estimable; do not merge away empty
  or one-class strata.
- Report `INCONCLUSIVE` predictions separately. Do not count abstentions as
  correct classifications. Provide both conditional accuracy among decided
  samples and coverage (fraction receiving REAL/FAKE).
- Compare against frozen colour and capture-feature shortcut baselines on the
  same held-out groups. Do not tune thresholds or choose a model using
  prospective test labels.
- Keep all failed loads, missing metadata, and unsupported capture conditions
  in the audit log. Never silently exclude samples after inspecting outcomes.

## Decision and deviations

The model, decoder, calibration, subgroup definitions, metrics, and bootstrap
seed are frozen before testing. Any deviation must be logged with its reason
before labels are examined. If a deviation depends on prospective outcomes,
the set is exploratory and cannot be presented as a confirmatory evaluation.

No result should be described as generalization to new cameras, sessions,
lighting, or adulterant types unless that exact shift was represented in the
locked test.
