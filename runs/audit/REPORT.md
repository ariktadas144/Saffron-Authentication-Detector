# Phase 1 — Shortcut audit (no training)

## Scope and reproducibility

This report covers only Phase 1. No model training, label-shuffle test,
controlled-manifest evaluation, Grad-CAM, LOCO run, or stress test was
performed. Nothing in this report is a result on a new camera or session.

The audit was run on the repository's current `data/manifest.csv`: 311 image
rows, 59 `group` identifiers, and five pre-existing folds. The manifest does
not contain a camera column, so camera identity was derived from the original
image EXIF `Make` and `Model`. The EXIF audit used `orig_path`, normalizing
Windows and POSIX separators. All 311 original paths existed; metadata was
not inferred from working copies for missing originals.

Exact audit command:

```powershell
.\.venv\Scripts\python.exe -u scripts\audit_shortcuts.py --manifest data\manifest.csv
```

Baseline details: Random Forest, 100 trees, minimum leaf size 2, balanced
class weights; five `StratifiedGroupKFold` seeds (42–46), up to five folds,
never splitting a `group`. Results below first average image probabilities
within each physical-sample group using logit aggregation, then calculate
metrics on the held-out groups. Mean and sample standard deviation are over
the 25 seed-fold evaluations (5 seeds × 5 folds), not one pooled out-of-fold
prediction set. The unknown-camera-only scope contains real samples only and
is therefore excluded from per-camera discrimination.

Package/runtime versions:

| Package | Version |
|---|---:|
| Python | 3.11.6 |
| Pillow | 12.3.0 |
| pandas | 3.0.6 |
| NumPy | 2.4.6 |
| scikit-learn | 1.9.1 |
| TensorFlow | 2.21.0 |
| Keras | 3.15.1 |
| SciPy | 1.17.1 |

## A. EXIF and capture-confound counts

`data/audit_exif.csv` contains one row per manifest row, the original path
resolution/existence result, requested EXIF fields, image dimensions, file
size, estimated JPEG quality, and derived numeric/categorical features.
JPEG quality is an estimate from the JPEG luminance quantization table, not
the original editor setting.

### Camera × class

Counts are **images** (`distinct groups` in parentheses):

| Camera | Fake | Real |
|---|---:|---:|
| Samsung SM-G973F | 40 (8) | 101 (17) |
| Samsung SM-J600G | 100 (20) | 60 (12) |
| Unknown | 0 (0) | 10 (2) |

### Flash × camera × class

Counts are images:

| Camera | Flash | Fake | Real |
|---|---|---:|---:|
| Samsung SM-G973F | Off | 40 | 101 |
| Samsung SM-G973F | On | 0 | 0 |
| Samsung SM-J600G | Off | 35 | 60 |
| Samsung SM-J600G | On | 65 | 0 |
| Unknown | Unknown | 0 | 10 |

Each observed group has one camera and one flash status. On SM-J600G, the
65 flash-on fake images are 13 groups; the 35 flash-off fake images are 7
groups; the 60 flash-off real images are 12 groups. Thus flash alone is not
perfect on this camera because some fake samples are also flash-off, but
flash-on is perfectly class-specific in these observed SM-J600G samples.

### Date × camera × class

Image counts:

| Camera | DateTimeOriginal date | Fake | Real |
|---|---|---:|---:|
| SM-G973F | 2021:04:05 | 15 | 41 |
| SM-G973F | 2021:04:06 | 25 | 60 |
| SM-J600G | 2021:04:14 | 100 | 60 |
| Unknown | 2021:03:10 | 0 | 5 |
| Unknown | 2021:03:11 | 0 | 5 |

Dates are camera/session-associated in this dataset: all SM-J600G images share
one date, and the two SM-G973F dates occur only on that camera. Both classes
are present on the known-camera dates, so date is not by itself perfectly
class-specific within those camera/date cells. It remains a potential session
and processing confound.

### ISO × camera × class

Image counts, with ISO buckets `<=100`, `101–400`, `401–800`, and `missing`:

| Camera | ISO bucket | Fake | Real |
|---|---|---:|---:|
| SM-G973F | <=100 | 10 | 25 |
| SM-G973F | 101–400 | 30 | 76 |
| SM-J600G | <=100 | 100 | 0 |
| SM-J600G | 101–400 | 0 | 20 |
| SM-J600G | 401–800 | 0 | 40 |
| Unknown | missing | 0 | 10 |

On SM-J600G, image-level median ISO/exposure time are 40 / 0.00251 seconds
(about 1/398 s) for fake and 500 / 0.045 seconds (about 1/22 s) for real.
These are measurements from the 311-row manifest's original files, not
estimated from the model's working images.

The JPEG quantization-table quality estimate was 90 for 304 images and 97
for 7. EXIF Software was `Windows Photo Editor 10.0.10011.16384` for 304
images and `G973FXXU9FUBD` for 7. This is consistent with substantial
re-saving, but does not establish how every image was cropped or whether
background/session conditions were shared.

## B. Shortcut-only grouped-CV baselines

Metrics are **sample-level** after grouped out-of-fold prediction
aggregation. Entries are mean ± standard deviation over 25 held-out fold
evaluations. `n/run` is the mean number of held-out samples per fold; the two
groups with unknown camera are excluded from EXIF-only pooled features when
those features are all missing.

| Scope | Features | AUC | Accuracy | Balanced accuracy | n/run |
|---|---|---:|---:|---:|---:|
| Pooled | Camera only | 0.668 ± 0.130 | 0.663 ± 0.131 | 0.666 ± 0.130 | 11.8 |
| Pooled | RGB+HSV mean/std | 0.942 ± 0.069 | 0.887 ± 0.097 | 0.887 ± 0.098 | 11.8 |
| Pooled | ISO/exposure/brightness/aperture/focal length/flash | 0.929 ± 0.077 | 0.821 ± 0.114 | 0.825 ± 0.112 | 11.4 |
| Pooled | Flash only | 0.738 ± 0.082 | 0.738 ± 0.073 | 0.723 ± 0.081 | 11.4 |
| Pooled | Width/height/pixels/file size/bits per pixel | 0.878 ± 0.087 | 0.763 ± 0.102 | 0.763 ± 0.101 | 11.8 |
| SM-G973F | Camera only | 0.500 ± 0.000 | 0.516 ± 0.205 | 0.500 ± 0.000 | 5.0 |
| SM-G973F | RGB+HSV mean/std | 0.987 ± 0.046 | 0.927 ± 0.100 | 0.905 ± 0.141 | 5.0 |
| SM-G973F | ISO/exposure/brightness/aperture/focal length/flash | 0.753 ± 0.212 | 0.617 ± 0.162 | 0.598 ± 0.204 | 5.0 |
| SM-G973F | Flash only | 0.500 ± 0.000 | 0.516 ± 0.205 | 0.500 ± 0.000 | 5.0 |
| SM-G973F | Width/height/pixels/file size/bits per pixel | 0.710 ± 0.314 | 0.685 ± 0.190 | 0.637 ± 0.248 | 5.0 |
| SM-J600G | Camera only | 0.500 ± 0.000 | 0.501 ± 0.140 | 0.500 ± 0.000 | 6.4 |
| SM-J600G | RGB+HSV mean/std | 0.995 ± 0.025 | 0.974 ± 0.060 | 0.975 ± 0.062 | 6.4 |
| SM-J600G | ISO/exposure/brightness/aperture/focal length/flash | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 6.4 |
| SM-J600G | Flash only | 0.825 ± 0.102 | 0.780 ± 0.127 | 0.825 ± 0.102 | 6.4 |
| SM-J600G | Width/height/pixels/file size/bits per pixel | 0.997 ± 0.017 | 0.940 ± 0.075 | 0.938 ± 0.081 | 6.4 |

Full-precision results, per-scope run counts, and seeds are in
`runs/audit/shortcut_baselines.csv`. The camera-only model is at chance
within each camera but above chance pooled; camera identity alone is not
enough to explain the classifier's performance. In contrast, high within-
camera color and capture-feature scores, including perfect SM-J600G
capture-feature CV on this dataset, are direct evidence that non-saffron
signals can predict the label. This audit alone cannot prove that the deep
model used those signals.

## C. Code review: decoder and colour baseline

Files reviewed: `src/wombat_beam.py`, `src/color_baseline.py`, and the
`scripts/run_cv.py` integration used to trace fitting/scoring.

| Finding | Evidence | Minimal fix / interpretation |
|---|---|---|
| No direct test-label threshold tuning was found. | `wombat_beam.py:387-403`: random/Wombat tuning receives `val`; only decoded predictions use `test`. `wombat_beam.py:406-411`: image and sample metrics are calculated against `test.y`. `make_fitness` at `330-342` uses only the cache passed to tuning. | No metric-pipeline change indicated for test-label leakage in this path. Keep validation and test objects separate. |
| Group aggregation silently overwrites a group's label with the last row's label. | `wombat_beam.py:307-314`, especially line 313 (`ys[inv] = y`). | Validate each group has exactly one unique label and raise on conflicts before aggregation. The dataset-preparation code checks group label consistency; this silent behavior was not shown to affect the current manifest. |
| The validation threshold is tuned for image-level predictions, then reused for sample-level metrics. | `wombat_beam.py:400-403` tunes from `val`; `409-411` aggregates test predictions by sample but passes the original image-level threshold to `metrics`. | For a sample-level headline, aggregate validation predictions by group and select/use a validation-only sample threshold; alternatively label the current sample metrics as using an image-tuned threshold. This is a threshold-calibration mismatch, not test-label leakage. |
| `bootstrap_ci` resamples rows, not independent groups. | `wombat_beam.py:317-324`. | For image-level confidence intervals, resample group IDs and retain all images in each selected group. If inputs are already one row per sample, document that contract. No call from the current `evaluate_all` path was found. |
| The color-feature cache is keyed by the ordered path list, not image contents. | `color_baseline.py:9-20`; `run_cv.py:64`. | Include content hashes or at least file size/mtime and a feature-version key, or disable cache reuse when files can be replaced in place. This is a reproducibility risk if source images change without paths changing; no stale-cache mismatch was established in this audit. |
| CV cache ordering is partially checked. | `run_cv.py:74-77` asserts group and label order before scoring. | Also persist/compare test paths in cache metadata if caches must survive manifest reorderings; within-group reorderings can share group and label sequences. No ordering mismatch was established in this audit. |

`color_baseline.py:14-21` extracts deterministic per-image RGB+HSV means and
standard deviations. `fit_predict` at `24-26` fits on the supplied training
features and predicts the supplied test features; no use of test labels was
found. `run_cv.py:75` selects training/test rows by manifest index from the
feature matrix generated from that same manifest order. Its explicit
group/label assertion at line 76 provides a useful cache alignment check.

## D. What the evidence says about 100%

**Is the 100% explained by shortcuts?** The evidence shows that shortcuts are
a credible and strong explanation: on this 311-image/59-group dataset, color
alone reached sample-level grouped-CV AUC 0.942 pooled and 0.995 on SM-J600G;
SM-J600G capture features reached AUC, accuracy, and balanced accuracy of
1.000 ± 0.000; and size/file/compression-related features reached AUC 0.878
pooled and 0.997 on SM-J600G. Flash-on is also exclusive to fake images in
the observed SM-J600G set.

However, **the local checkout does not contain the original grouped-CV
summary or per-fold predictions**: `runs/cv1` currently contains a
`colour_features.npz`, fold-0 weights, and `config.json`, but no
`summary_sample.csv`, `ablation_test.csv`, `ablation_sample_level.csv`, or
`preds_test.csv`. Therefore the reported 100% model result was not
independently reproduced in Phase 1. This audit cannot conclude that
shortcuts fully explain the deep model's 100%, nor that it learned
saffron-specific features. It establishes a strong shortcut-only signal and
a need for controlled evaluation.

## Files produced

- `scripts/audit_shortcuts.py`
- `tests/test_audit_shortcuts.py`
- `data/audit_exif.csv`
- `runs/audit/shortcut_baselines.csv`
- `runs/audit/REPORT.md`

The audit only read the manifest, originals under `data/extracted`, and
working images. It did not modify extracted files or the source zip archives.

## Safe claims for the mentor

**Can safely claim from Phase 1:** on this particular 311-image, 59-group
manifest, metadata-only and simple colour/size baselines predict labels well
under sample-grouped CV; capture metadata alone achieves perfect CV metrics
within SM-J600G; and the observed camera, flash, date, and class counts are
imbalanced as tabulated above. These are evidence that shortcut confounding
is plausible and must be controlled.

**Cannot claim yet:** that the deep model's reported 100% has been
independently reproduced from this checkout; that shortcuts fully explain
that reported score; that the model does or does not learn saffron-specific
visual features; or that performance generalizes to new cameras, sessions,
lighting setups, or fake types. No controlled evaluation or deep-model
sanity test has been run in Phase 1.

## Questions for the dataset owner

1. Were fake SM-J600G samples intentionally photographed with flash more
   often/on, or were flash and lighting changed by sample class?
2. Were the real and fake images for each camera captured in the same
   session, against the same background, and with the same phone settings?
3. How were crops selected and applied, and were crop dimensions or framing
   chosen differently by class?
4. Were all images re-saved through Windows Photo Editor? If not, what
   processing applies to the 7 files reporting camera software?
5. What do the two camera-unknown real groups represent, and can their
   camera/session metadata be recovered?
6. Are any physical samples or acquisition sessions shared across the two
   phones even where the manifest's current `group` values differ?

## Verification and commands

Synthetic EXIF/path utility tests:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_audit_shortcuts.py -q
```

Result: 3 passed. Syntax check:

```powershell
.\.venv\Scripts\python.exe -m py_compile scripts\audit_shortcuts.py tests\test_audit_shortcuts.py
```

Result: passed. Phase 1 did not run `run_cv.py` for deep model training or
evaluate the deep model.

Full requested non-slow suite:

```powershell
.\.venv\Scripts\python.exe -m pytest -m "not slow" -q
```

Result: 23 passed, 1 slow test deselected.
