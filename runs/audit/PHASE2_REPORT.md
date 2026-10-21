# Phase 2 — Evaluation fixes, grouped-fold shortcuts, and reproduction status

## Scope and outcome

This report covers Phase 2 only. The evaluation-code fixes and group-level
label-shuffle utility are implemented. Shortcut baselines were measured on the
manifest's existing five folds. The deep-model reproduction and the three
label-shuffle training runs did **not** complete, so this report contains no
deep-model or shuffled-label metrics.

The dataset is the current `data/manifest.csv` (311 images, 59 physical-sample
groups, five preassigned folds). All split and bootstrap units use the
manifest's `group` field, not filename IDs. No files under `data/extracted`
or source archives were modified.

## A. Evaluation fixes and code review

| Issue | Change |
|---|---|
| `aggregate_by_group` silently accepted conflicting labels within a group. | `src/wombat_beam.py:310-333` now validates array lengths and raises `ValueError` when any group has more than one label. |
| Sample metrics reused a threshold tuned for image-level predictions. | `src/wombat_beam.py:376-384, 495-514` selects a threshold using validation predictions aggregated by group, then uses it for test sample-level metrics. Existing image-level metrics remain and use their original image-level thresholds. |
| Bootstrap intervals sampled image rows independently. | `src/wombat_beam.py:333-375, 485-513` resamples group IDs and retains all images in each selected group. `scripts/decode.py` passes group IDs to its CI call. |
| Cache/manifest order checks did not detect swapped paths within one group. | `src/wombat_beam.py:112-132` persists optional cache paths; `src/saffron_cnn_gnn.py:311-326, 360-366` writes them in deterministic manifest order; `scripts/run_cv.py:290-312` validates validation/test paths, groups, and labels before scoring. Old caches without path metadata must be rebuilt with `--retrain`. |
| The model's test labels were being used for tuning or threshold selection. | No direct test-label use for model/decoder tuning was found. Decoder optimization and sample-threshold selection use validation predictions; test labels are used for final scoring only. |
| Colour-baseline fitting could use test labels. | No such use was found: the baseline is fit on training features/labels and predicts validation/test features. Its cache remains keyed by ordered paths, so replacing image contents in place without invalidating that cache is still a reproducibility risk. |

The existing image-level result remains available in `ablation_test.csv`;
sample-level results are in `ablation_sample_level.csv`. `evaluate_all`
returns group-bootstrap CIs for the image and sample metrics, and `run_cv.py`
writes them per fold to `ci_image.csv` and `ci_sample.csv`.

## B. Deep-model reproduction

Requested default-settings fold-0 command:

```powershell
python scripts\run_cv.py --manifest data\manifest.csv --out runs\cv_repro --folds 0
```

The run started on CPU. TensorFlow reported no available GPU. Fold 0's split
was 188 training images / 36 groups, 64 validation images / 12 groups, and 59
test images / 11 groups. I stopped the run during the first warm-up epoch
after the initial CPU batches were taking several minutes. No completed
checkpoint, cache, fold metrics, or predictions were produced in
`runs/cv_repro`; therefore there is no real-label fold-0 result to report.
This is an incomplete attempt, not a reproduced score.

The package versions for the Phase 2 `python` interpreter were:

| Package | Version |
|---|---:|
| Python | 3.11.6 |
| NumPy | 2.4.2 |
| pandas | 2.2.3 |
| scikit-learn | 1.6.1 |
| SciPy | 1.15.2 |
| Pillow | 12.2.0 |
| TensorFlow | 2.20.0 |
| Keras | 3.13.0 |
| GPU devices | none |

No image-level or sample-level deep-model metrics, confidence intervals,
colour-row comparison against the deep model, or camera/flash/ISO deep
breakdowns are available. Do not interpret missing output as a low or high
score.

## C. Shortcut baselines on the exact manifest folds

These were run independently of neural-network training but on the same
preassigned train/validation/test group folds, with the same held-out groups
for each method. For each fold, the sample threshold was selected using
validation predictions aggregated by group; image metrics use threshold 0.5.
Reported mean ± standard deviation is across the five held-out folds. Each
fold's metric-specific 95% percentile CI uses 1,000 group-cluster bootstrap
resamples and is recorded in the per-fold CSV; those intervals are not a
single pooled CI.

Exact command:

```powershell
python scripts\run_shortcut_cv.py --manifest data\manifest.csv --audit data\audit_exif.csv --out runs\audit\phase2_shortcut_same_folds.csv --seed 42 --ci-samples 1000
```

### Sample-level shortcut results (mean ± std over five folds)

| Baseline | Accuracy | Balanced accuracy | Fake recall | Specificity | Fake F1 | AUC |
|---|---:|---:|---:|---:|---:|---:|
| RGB+HSV colour mean/std | 0.833 ± 0.050 | 0.829 ± 0.049 | 0.787 ± 0.137 | 0.871 ± 0.138 | 0.815 ± 0.062 | 0.909 ± 0.063 |
| Exposure/ISO/brightness/aperture/focal length/flash | 0.830 ± 0.088 | 0.826 ± 0.089 | 0.747 ± 0.107 | 0.905 ± 0.087 | 0.805 ± 0.105 | 0.945 ± 0.040 |
| Size/file size/bpp | 0.695 ± 0.046 | 0.687 ± 0.040 | 0.607 ± 0.283 | 0.767 ± 0.325 | 0.631 ± 0.097 | 0.882 ± 0.109 |

### Image-level shortcut results (mean ± std over five folds)

| Baseline | Accuracy | Balanced accuracy | Fake recall | Specificity | Fake F1 | AUC |
|---|---:|---:|---:|---:|---:|---:|
| RGB+HSV colour mean/std | 0.859 ± 0.041 | 0.862 ± 0.045 | 0.889 ± 0.114 | 0.836 ± 0.085 | 0.849 ± 0.047 | 0.924 ± 0.059 |
| Exposure/ISO/brightness/aperture/focal length/flash | 0.870 ± 0.091 | 0.881 ± 0.086 | 0.967 ± 0.075 | 0.796 ± 0.130 | 0.872 ± 0.090 | 0.950 ± 0.035 |
| Size/file size/bpp | 0.701 ± 0.057 | 0.701 ± 0.043 | 0.691 ± 0.230 | 0.712 ± 0.300 | 0.666 ± 0.045 | 0.842 ± 0.102 |

Sample-level accuracy/AUC by fold, to show the fold variation:

| Fold | Colour accuracy / AUC | Exposure accuracy / AUC | Size accuracy / AUC |
|---:|---:|---:|---:|
| 0 | 0.909 / 0.967 | 0.727 / 0.900 | 0.636 / 0.833 |
| 1 | 0.833 / 0.944 | 0.917 / 1.000 | 0.750 / 1.000 |
| 2 | 0.769 / 0.929 | 0.846 / 0.940 | 0.692 / 1.000 |
| 3 | 0.818 / 0.900 | 0.909 / 0.967 | 0.727 / 0.800 |
| 4 | 0.833 / 0.806 | 0.750 / 0.917 | 0.667 / 0.778 |

Full per-fold image/sample metrics and their bootstrap interval columns are in
`runs/audit/phase2_shortcut_same_folds.csv`; fold means and standard deviations
are in `runs/audit/phase2_shortcut_same_folds_summary.csv`. The deep model
cannot be compared against these rows until the deep runs finish.

The Phase 1 grouped-CV audit, measured with a different set of grouped folds
and five seeds, separately reported pooled colour AUC 0.942 and
SM-J600G-only exposure/ISO/flash AUC 1.000. Those numbers are Phase 1 results,
not the fixed-fold measurements above; see `REPORT.md`.

## D. Group-level label-shuffle sanity test

`scripts/label_shuffle_test.py:21-63` permutes labels across physical-sample
groups independently within each non-test fold, preserves group-level class
counts in each fold, and verifies that test-fold labels are unchanged. The
utility's synthetic test passed. It was **not run against a trained model**,
because it requires the same full model training that could not be completed
on CPU here.

| Run | Shuffle seed | Sample AUC | Sample accuracy |
|---|---:|---:|---:|
| Real labels, fold 0 | — | Not available | Not available |
| Shuffled labels, fold 0 | 101 | Not run | Not run |
| Shuffled labels, fold 0 | 202 | Not run | Not run |
| Shuffled labels, fold 0 | 303 | Not run | Not run |

No conclusion about label leakage from model training can be drawn from the
synthetic utility test. If completed shuffled-label AUCs are clearly above
0.5, investigate leakage or evaluation bugs; AUC near 0.5 would indicate
that this training/evaluation pipeline is not exploiting the randomized
training/validation labels, but would not prove saffron-specific learning.

## E. Per-camera / flash / ISO deep breakdown

Not available: no held-out deep predictions were produced. The requested
SM-J600G flash-off fake recall (35 images / 7 groups in Phase 1 metadata),
flash-on fake recall (65 images / 13 groups), ISO-low breakdown, camera
metrics, and deep-vs-shortcut comparison must wait for a completed
reproduction. Phase 1's counts are descriptive metadata, not model results.

## F. Verification and exact follow-up commands

Completed:

```powershell
python -m py_compile scripts\run_shortcut_cv.py scripts\label_shuffle_test.py scripts\run_cv.py src\wombat_beam.py src\saffron_cnn_gnn.py
python -m pytest -m "not slow" -q
```

The syntax check passed. The non-slow suite passed: **28 passed, 1
deselected**. TensorFlow/Keras emitted 212 upstream Future/Deprecation
warnings; no test failed.

To finish the requested deep-model work on a machine with adequate GPU
compute, first run fold 0 with the default settings:

```powershell
python scripts\run_cv.py --manifest data\manifest.csv --out runs\cv_repro --folds 0 --seed 42
```

After fold 0 completes, run the remaining folds:

```powershell
python scripts\run_cv.py --manifest data\manifest.csv --out runs\cv_repro --folds 1 2 3 4 --seed 42
```

Then run the three label-shuffle training runs:

```powershell
python scripts\label_shuffle_test.py --fold 0 --seeds 101 202 303
```

The exact default image size, epoch counts, batch size, MC sample count,
optimizer budget, and seed are carried by these commands; no settings were
reduced to force a score.

## Conclusion: what these results show and what they do not show

On these five manifest folds, simple colour statistics and exposure/ISO/flash
features alone achieved sample-level mean AUCs of 0.909 and 0.945,
respectively. This is measured evidence that non-saffron capture signals can
predict labels strongly in this dataset and that the reported perfect deep
score could plausibly be affected by shortcuts.

The deep result has not been reproduced here, and there are no completed
label-shuffle, per-camera, flash, ISO, or controlled-evaluation model
predictions. Therefore we cannot say that shortcuts fully explain the deep
model's reported 100%, that the model learned saffron-specific features, or
that it generalizes to new cameras, sessions, lighting conditions, or fake
types. **The honest Phase 2 conclusion is inconclusive about what the deep
model learned; shortcut confounding is demonstrably strong enough that the
perfect score is not trustworthy as evidence of saffron-feature learning.**
