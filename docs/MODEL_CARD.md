# Model card: Saffron authenticity research prototype

## Summary

This repository contains a research prototype for classifying saffron images
as real or adulterated. Its deep path uses ConvNeXt-Tiny, a grid
graph-attention network (GNN), a hybrid attention fusion module (HAFM), and an
optional Wombat-OA tuned weighted-beam decoder. The repository also contains
colour and capture-feature baselines. A classifier output is not a chemical
or laboratory authentication result.

The supported user-facing decision contract is **REAL**, **FAKE**, or
**INCONCLUSIVE**. The app must abstain when calibration, model provenance,
capture support, image support, model agreement, or uncertainty checks are
missing or out of range. The colour baseline is exploratory and must not issue
a primary REAL/FAKE verdict.

## Intended and out-of-scope use

**Intended:** controlled research comparisons on images and physical samples
matching the documented dataset and capture conditions.

**Not intended:** commercial grading, food-safety decisions, regulatory
certification, medical or safety-critical decisions, or claims about unseen
phones, sessions, locations, lighting, saffron sources, or adulterant types.
Do not use a single image as a definitive authentication result.

## Training/evaluation data and known limitations

The project manifest describes 311 images from 59 sample groups. The known
camera/class and exposure associations make pooled accuracy potentially
misleading. The SM-J600G subset is explicitly confounded: the supplied audit
found fake and real samples separated by ISO and observed flash imbalance.
SM-G973F is less affected by the specific flash/ISO pattern, but colour
statistics remain a strong shortcut there. These are dataset properties, not
proof that a model learned saffron-specific visual features.

The earlier 100% deep-model result has not been reproduced from a complete
eligible result set in this workspace. Partial checkpoints and smoke tests
are not evidence of model performance. Check the generated, provenance-gated
report before citing any numeric model result.

## Model and inference

- Image preprocessing is a centre-square crop followed by resize, matching
  the training loader.
- The deep model extracts CNN, GNN, and HAFM scores across test-time views and
  combines them with the saved decoder configuration.
- At inference, the app checks model/data fingerprints, fold completeness,
  capture support, OOD distance, MC uncertainty, fold consistency, and a
  group-level conformal prediction set. A failed or missing check yields
  INCONCLUSIVE.
- Current conformal calibration is estimated from out-of-fold predictions.
  Its observed coverage is empirical on those OOF samples; it is not a
  prospective guarantee for distribution shift or new capture conditions.
- The deployment threshold and calibration must match the exact model bundle
  and manifest fingerprint. Any mismatch invalidates calibration.

## Evaluation status

No complete, provenance-eligible fine-tuned run is available locally for a
performance claim. The CPU environment has no GPU, and prior CPU training was
stopped before producing complete fold results. Smoke runs are explicitly
excluded from reports. Do not infer performance from the architecture,
training logs, baseline results, or incomplete fold directories.

Before reporting performance, run the complete planned GPU evaluation,
generate the report from saved OOF predictions, include shortcut baselines and
camera/session/flash strata, and conduct an independent locked prospective
test. Consult [preregistration.md](./preregistration.md).

## Known risks

- Camera, flash, ISO, exposure, file size, resolution, colour balance,
  background, cropping, and re-saving can act as class shortcuts.
- There are few independent physical sample groups relative to the model
  complexity; estimates and decoder tuning may be unstable.
- A source/session or adulterant-type shift can dominate performance.
- Images may not show enough diagnostic structure to determine authenticity;
  the app's confidence-like probabilities are not calibrated truth.
- Missing EXIF metadata can prevent support checks and cause abstention.

## Safe interpretation

Treat any REAL/FAKE output as a research-model prediction conditioned on the
available validation and calibration evidence—not as proof of authenticity.
Report abstention and coverage alongside classification metrics. Never claim
generalization beyond the exact held-out cameras, sessions, and sample types
that were tested.

## Maintenance

Every released model should be accompanied by the manifest hash, run
configuration, package versions, per-fold weights and decoder parameters,
calibration report, eligible OOF report, shortcut comparison, and the
prospective-test protocol and signed result when available.
