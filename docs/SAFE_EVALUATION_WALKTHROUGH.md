# Safe evaluation walkthrough

## Status at this checkout

### VERIFIED here

- `python -m compileall -q app scripts src tests` completed successfully.
- `pytest -m "not slow" -q` completed: **48 passed, 1 deselected**. The
  remaining output was TensorFlow/Keras deprecation/future warnings.
- TensorFlow reports no GPU devices in this environment.
- The current data-side G973F manifests exist: `manifest_g973f.csv` contains
  141 images / 25 groups (17 real groups, 8 fake groups);
  `manifest_g973f_matched.csv` contains 60 images / 12 groups (6 groups per
  class). These are manifest counts, not model results.
- `scripts/session_analysis.py` completed after its group/date validation
  fix. It excluded two groups without a supported mixed-class capture date
  and wrote `data/manifest_leave_one_date_out.csv`.
- The synthetic report-builder test created a report from synthetic
  predictions and verified that it contains run and data fingerprints.

### NOT VERIFIED / no performance claim

- `runs/cv_repro` has no completed fold outputs. No full, eligible
  fine-tuned model result or OOF report can be produced from it.
- `runs/cv1` contains only a partial `fold0` checkpoint/configuration; it has
  no complete fold ensemble, calibration, or eligible OOF predictions.
- The earlier `SMOKE_TEST_NOT_A_RESULT_*` attempt started CPU training and
  was stopped before completing. Its outputs are explicitly excluded from
  research reporting.
- No full GPU training, label-shuffle training, exposure counterfactual,
  saved-weight decoder comparison, prospective locked-test evaluation, or
  Grad-CAM result was run in this environment.
- The frozen-embedding evaluation is implemented but has not produced an
  eligible result table here.
- No conclusion about saffron-specific learning, new-phone generalization,
  new sessions, lighting shifts, or unseen adulterants is supported.

The prior Phase 2b report contained an incorrect statement that no G973F
data existed because it inspected the manifest without joining the EXIF
audit. That section is corrected in `runs/audit/PHASE2B_REPORT.md`. The
camera-joined G973F manifests above are the current data-side evidence.

## Runtime recorded in this environment

Command:

```powershell
python -c "import importlib.metadata as m, sys; print('Python', sys.version.split()[0]); names=['numpy','pandas','scikit-learn','scipy','tensorflow','keras','Pillow','streamlit']; [print(name, m.version(name)) for name in names]"
```

Observed versions:

| Component | Version |
|---|---:|
| Python | 3.11.6 |
| NumPy | 2.4.2 |
| pandas | 2.2.3 |
| scikit-learn | 1.6.1 |
| SciPy | 1.15.2 |
| TensorFlow | 2.20.0 |
| Keras | 3.13.0 |
| Pillow | 12.2.0 |
| Streamlit | 1.54.0 |

GPU check:

```powershell
python -c "import tensorflow as tf; print(tf.__version__, tf.config.list_physical_devices('GPU'))"
```

Observed GPU list: `[]`. The package versions above are from this checkout's
current Python environment; do not merge them with version tables recorded
for earlier runs in other environments.

For Colab, `notebooks/run_on_colab.ipynb` pins NumPy 2.2.6, SciPy 1.16.3,
TensorFlow 2.20.0, Keras 3.13.0, pandas 2.2.3, scikit-learn 1.6.1, and
Pillow 12.2.0. It installs these in a dedicated step and requires a runtime
restart before importing TensorFlow/SciPy, then checks the versions and GPU.
This pinned set is a compatibility precaution for Colab's Python 3.13 runtime;
it has not been verified by a completed Colab run in this checkout.

## Verification commands

```powershell
python -m compileall -q app scripts src tests
pytest -m "not slow" -q
python scripts/session_analysis.py
```

The Phase 2b CPU smoke attempt is documented in
`runs/audit/PHASE2B_REPORT.md`; it was not completed and is not a result.

## GPU run order

Run on a GPU host after placing the original archives in `data/` if the
prepared manifest and working images are not already available. Do not
overwrite the original archive files or `data/manifest.csv`.

```powershell
python scripts/run_cv.py --manifest data/manifest.csv --out runs/cv_repro --folds 0 --img-size 256 --batch 16 --epochs-warm 3 --epochs-ft 25 --mc-samples 16 --seed 42 --pop 20 --iters 25
python scripts/label_shuffle_test.py --manifest data/manifest.csv --real-out runs/cv_repro --out runs/audit/label_shuffle --fold 0 --seeds 101 202 303 --img-size 256 --batch 16 --epochs-warm 3 --epochs-ft 25 --mc-samples 16 --seed 42 --pop 20 --iters 25
python scripts/run_cv.py --manifest data/manifest.csv --out runs/cv_repro --folds 1 2 3 4 --img-size 256 --batch 16 --epochs-warm 3 --epochs-ft 25 --mc-samples 16 --seed 42 --pop 20 --iters 25
python scripts/run_cv.py --manifest data/manifest.csv --out runs/cv_repro --aggregate-only
python scripts/collect_results.py --cv-dir runs/cv_repro --shuffle-dir runs/audit/label_shuffle --manifest data/manifest.csv --out runs/audit/collected_phase2_results.csv
python scripts/build_report.py --run runs/cv_repro --bootstrap 1000
python scripts/calibrate_inference.py --run runs/cv_repro --manifest data/manifest.csv --audit data/audit_exif.csv
```

The first command persists the run configuration and completes fold zero.
The label-shuffle command reads fold zero's real-label result, then trains
three group-shuffled folds with the test-fold labels untouched. The third
command resumes the same run configuration and finishes folds one through
four. Do not change settings between these commands. Report results only
after every fold and shuffle seed has completed; the report and collector
refuse missing or ineligible inputs.

Optional additional GPU analyses, only after complete saved fold weights and
OOF files exist:

```powershell
python scripts/exposure_counterfactual.py --manifest data/manifest.csv --audit data/audit_exif.csv --run runs/cv_repro --out runs/cv_repro/exposure_counterfactual.csv
python scripts/matched_subset_eval.py --run runs/cv_repro --manifest data/manifest.csv --audit data/audit_exif.csv --out runs/audit/matched_subset_results.csv
```

These commands are not substitutes for the locked prospective test below.

## CPU frozen-feature check

The frozen-embedding script does not fine-tune the network, but it still
loads ImageNet ConvNeXt weights and performs forward passes over the images.
No result is reported here because it has not been completed in this
environment. To run it:

```powershell
python scripts/frozen_embedding_cv.py --manifest data/manifest.csv --audit data/audit_exif.csv --cache runs/audit/frozen_embeddings.npz --out runs/audit/frozen_embedding_cv.csv --seed 42 --batch-size 4 --shuffle-seeds 101 202 303
```

Its group-shuffle rows are a generic frozen-feature permutation check. A
strong frozen-feature result would indicate image separability by generic
features and capture cues; it would not show that the fine-tuned model learned
saffron-specific cues.

## New controlled prospective test

Only after collection and preregistration, create a locked dataset and run it
exactly once:

```powershell
python scripts/lock_test_set.py --images data/prospective/images --capture-log data/prospective/capture_log.csv --out data/prospective/locked
$env:SAFFRON_REPORT_HMAC_KEY = "<store a private signing key outside the repository>"
python scripts/evaluate_locked.py --locked-dir data/prospective/locked --run runs/cv_repro --out runs/prospective/evaluation
```

Do not use the example key literally or commit it. A failed evaluation
consumes the exactly-once claim by design. Follow
`docs/preregistration.md`; do not tune the system using the locked test labels.

## Interpretation boundary

The source dataset and shortcut baselines show that capture and colour cues
can be predictive. The available partial checkpoints do not establish whether
the deep model learned saffron features. The correct current conclusion is
that **the reported perfect deep-model result is not reproduced here**, and
the model's learned signal remains unknown until a complete GPU run,
group-level sanity checks, confound-aware comparisons, and an independent
prospective test are completed.
