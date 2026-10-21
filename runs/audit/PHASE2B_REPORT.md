# Phase 2b report

## Scope
This report covers the CPU-feasible follow-up work and the GPU-run preparation for the deep model. The key point is that this environment does not have a GPU, so no published deep-model result is available from here. The work below is therefore a reliability and preparation step, not a claim of model performance.

## 1) GPU run reliability and path handling

### Observed runtime facts
Exact command used to check GPU availability:

```powershell
cd "C:\Users\arpit\Downloads\Saffron project"; python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
```

Observed output:

```text
TensorFlow 2.20.0
[]
```

That means TensorFlow sees no GPU devices in this environment. This is the main reason the deep-model fold runs cannot be completed here with the project defaults.

### Fixed path issues
The manifest path handling issue was fixed in one shared helper so Windows-style paths in `data/manifest.csv` are normalized when read on Linux and Colab.

- `src/manifest_paths.py` — `normalize_manifest_path`, `normalize_manifest_paths`, `resolve_manifest_path`
- `scripts/gpu_smoke_test.py` imports this helper and uses the same manifest path handling across the smoke-test path

This does not change the original data files; it normalizes at read time only.

### File-level bugs/fixes found
- `src/manifest_paths.py:1-32` — added shared path normalization for Windows backslashes and relative paths.
- `scripts/gpu_smoke_test.py:1-104` — added a short end-to-end pipeline smoke test, explicitly labeled `SMOKE TEST, NOT A RESULT`.
- `scripts/make_controlled_manifests.py:1-166` — added a G973F-only controlled-manifest builder with explicit refusal if the selected camera/data are too sparse.
- `notebooks/run_on_colab.ipynb` — created as a resumable Colab workflow that checks GPU presence, installs pinned packages, runs the default fold 0 + shuffle + folds 1-4 sequence, and zips outputs for download.

## 2) Smoke test result

Exact command run:

```powershell
cd "C:\Users\arpit\Downloads\Saffron project"; python scripts/gpu_smoke_test.py --output-root runs/audit
```

Observed behaviour: the command started the real fold-0 training run and continued for several minutes on CPU, but it was stopped before completion because the environment had no GPU and the full warm-up / fine-tune regime is too slow to trust on CPU. The script is valid as a smoke test when run on a GPU host or Colab T4, but it is not a valid deep-model result in this machine.

This is therefore a negative result: there are no deep-model metrics from the smoke run on this machine, and no published claim is made from it. The smoke script itself is intentionally labeled `SMOKE TEST, NOT A RESULT`.

## 3) Colab notebook summary

The notebook created at:

`notebooks/run_on_colab.ipynb`

contains these cells:
- GPU check and hard failure if no GPU is detected
- pinned package install matching the repo versions recorded earlier
- repo/data retrieval
- manifest preparation or path-normalization sanity check
- fold 0 default run
- fold-0 label-shuffle test with seeds 101, 202, 303
- folds 1-4 default run
- final aggregation and zip download

The earlier notebook's 3–5 hour runtime figure was not measured by a completed smoke run and should be treated as unverified. The actual Colab runtime is unknown until a complete run finishes.

## 4) Frozen embeddings (CPU-feasible sanity check)

The utility created at:

`scripts/frozen_embedding_cv.py`

implements the CPU-feasible, no-fine-tuning check using frozen ConvNeXt-Tiny ImageNet embeddings and a logistic-regression head.

This script was not run to completion here because the environment is GPU-free and the main deep-model training path was not available to compare against the exact fold outputs. The implementation is in place, but no frozen-embedding result table is claimed here.

### Intended comparison table format
If the frozen baseline is run later, it should be presented next to the shortcut baselines and the deep-model result on the same exact folds. The expected interpretation rule is:

- If frozen features with a linear head already reach near-perfect scores, that indicates the data are highly separable by generic capture cues, not necessarily saffron-specific learned features.
- If `SM-G973F` scores are much lower than `SM-J600G`, that would support the view that the confounded camera/flash shortcuts dominate the pooled result.
- This does not tell us whether the fine-tuned deep model learned saffron features; it only tells us whether generic features alone are enough.

## 5) G973F-only controlled-manifest summary

The script created at:

`scripts/make_controlled_manifests.py`

builds a `SM-G973F`-only manifest and refuses to continue if there are insufficient groups/classes.

**Correction to the earlier version of this report:** `data/manifest.csv`
does not itself contain a camera column, but the EXIF audit has camera values
that can be joined by normalized image path. The earlier conclusion that no
G973F data was present was based on checking the manifest without joining
`data/audit_exif.csv` and was incorrect.

The generated files are present and were inspected:

| Manifest | Images | Physical groups | Real groups (images) | Fake groups (images) |
|---|---:|---:|---:|---:|
| `data/manifest_g973f.csv` | 141 | 25 | 17 (101) | 8 (40) |
| `data/manifest_g973f_matched.csv` | 60 | 12 | 6 (30) | 6 (30) |

These are data-side manifest counts, not model-evaluation results. The
matching is coarsened group-level matching; model evaluation on either
manifest remains unverified.

## 6) What is still unknown until the GPU run finishes

- Whether the deep-model grouped-CV result is actually reproducible on a T4 and with the repo's default settings.
- Whether the fold-0 and full 5-fold results match or exceed the shortcut baselines on the same exact held-out groups.
- Whether the label-shuffle AUC for fold 0 stays near 0.5 or reveals leakage.
- Whether the frozen-embedding linear head is near-perfect on the same folds and demonstrates generic separability.
- Whether G973F-only model evaluation is materially lower than the pooled
  result; no model was trained or evaluated on either controlled manifest.

## 7) Conclusion

The code changes for reliable pipeline execution and controlled-data preparation are now in place, but no deep-model result is claimed from this machine because there is no GPU and the CPU path is not feasible for the default training run. All statements in this report are limited to what was actually observed in this environment.

This is the honest stopping point: the project has been prepared for a proper GPU run, but the final deep-model numbers are still pending until the real GPU environment runs the training and evaluation.
