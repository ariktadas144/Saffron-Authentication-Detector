# Saffron authenticity: real vs fake
ConvNeXt-Tiny + grid graph-attention (GNN) + HAFM fusion, with Wombat-OA-tuned weighted-beam decoding and log-probability fusion,
evaluated with **sample-level cross-validation**.

## 0. Read this first: what is wrong with the original split
Your files are named `<id>[ <TYPE>]_<shot>.jpg`. Several photos (shots) show the SAME physical sample, and **all 57 test samples
also appear in the training folder** (shots 1-3 in Training, shots 4-5 in Testing). Four photos are exact duplicates stored under
different sample IDs. Results on the original Training/Testing split would therefore be inflated. `prepare_dataset.py` rebuilds
the splits so that every sample (and every duplicate-linked pair) lives in exactly one fold.

Also measured on your data (group-CV, trivial features only): image size/aspect/file size -> AUC 0.79; mean/std of colour -> AUC 0.94.
So colour alone is a strong baseline (included as a row in every table) and image resolution is a shortcut (the code uses square crops,
resolution jitter, and capped working copies to reduce it).

## 1. Folder structure
```
data/                        <- put the two dataset .zip files here
├── extracted/               (created)  unzipped originals
├── images/{fake,real}/      (created)  working copies, shorter side <= 768 px, EXIF-rotated
└── manifest.csv             (created)  path,label,cls,type,sample,group,shot,orig_split,fold,orig_path
src/       saffron_cnn_gnn.py (TensorFlow) | wombat_beam.py (NumPy) | color_baseline.py
scripts/   prepare_dataset.py | run_cv.py | train_fold.py | decode.py, train.py, check_dataset.py, split_dataset.py (folder workflow)
tests/     pytest suite
notebooks/ run_saffron.ipynb (Colab: same steps as below)
runs/      outputs
requirements.txt
```

## 2. Setup
```bash
python -m venv .venv && source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt                           # NVIDIA GPU on Linux/WSL2: pip install "tensorflow[and-cuda]>=2.16"
python -c "import tensorflow as tf; print(tf.__version__, tf.config.list_physical_devices('GPU'))"
```
Native Windows has no TensorFlow GPU support: use WSL2 or Google Colab (see the notebook). CPU-only training is very slow.

## 3. Run (your dataset)
```bash
# 1) copy Test_dataset-....zip and train_dataset-....zip into data/
python scripts/prepare_dataset.py --src data/Test_dataset-*.zip data/train_dataset-*.zip --folds 5

# 2) smoke test, minutes: fold 0, tiny settings, random weights (numbers are NOT results)
python scripts/run_cv.py --manifest data/manifest.csv --out runs/smoke --folds 0 --img-size 128 --epochs-warm 1 --epochs-ft 1 --mc-samples 4 --no-imagenet --pop 8 --iters 4

# 3) real run: 5 folds, ImageNet weights (resumable)
python scripts/run_cv.py --manifest data/manifest.csv --out runs/cv1 --img-size 256

# 4) summaries again at any time
python scripts/run_cv.py --out runs/cv1 --aggregate-only
```
Each fold: test = fold k, validation = fold k+1 (early stopping and Wombat-OA tuning), train = the other three. Each fold trains in
its own process to keep memory flat. Options: `--img-size 384`, `--batch 8` (out of memory), `--epochs-ft 40`, `--mc-samples 32`.

## 4. Outputs (`runs/cv1/`)
`fold*/ablation_test.csv` (image level), `fold*/ablation_sample_level.csv` (all shots of a sample fused by averaging logits),
`fold*/preds_test.csv`, `fold*/wombat_params.json`, `summary_image.csv`, `summary_sample.csv` (mean ± std over folds),
`pooled_oof_sample_level.csv` (every sample tested exactly once). Rows go from single heads, through TTA and log-prob fusion without
beam, to the weighted beam (untuned / random search / Wombat-OA) and the colour-statistics baseline.

## 5. How to check that it works
**Code works (about 40 s, no dataset):** `pytest -m "not slow" -q` -> 14 passed (optimizer, beam == exhaustive search, GNN has no
cross-sample leakage, MC-dropout only when sampling, weights reload, leak-free manifest builder, sample-level aggregation).
**Pipeline works (about 5 min, CPU):** `pytest -m slow -q` -> 1 passed (the folder-based `train.py` / `decode.py` workflow).
**Your data works:** step 2 above finishes and writes `runs/smoke/fold0/ablation_test.csv`; `prepare_dataset.py` prints `[LEAK]` and `[DUP]`
lines and per-fold counts of about 59-65 images / 11-13 samples.
**Red flags:** deep models far below the colour baseline after the full run (more epochs / larger images / check ImageNet weights loaded);
train AUC ~1.0 with val AUC low (overfitting); very different fold results (expected with 11-13 samples per fold: report mean ± std).

## 6. Honest limits
* 59 independent samples is small. Tuning 13 decoder parameters on a validation fold of about 12 samples is noisy; the random-search
  and untuned rows exist so you can see whether Wombat-OA genuinely helps.
* The meaning of the C / S / M fake types is not documented in the files; confirm it before analysing per-type results.
* For two classes, log-probability fusion equals weighted logit averaging, so gains must come from the tuned weights, view reliability
  and diversity-aware beam selection, or the optimiser; the ablation rows isolate each.

## 6c. Harder evaluations (after the first results)
**Leave-one-camera-out** (train on one phone, validate on held-out samples of that phone, test on the other phone):
```bash
python scripts/make_loco_manifests.py --manifest data/manifest.csv       # writes data/manifest_loco_<A>_to_<B>.csv for both directions
python scripts/run_cv.py --manifest data/manifest_loco_SMJ600G_to_SMG973F.csv --out runs/loco_J600G_to_G973F --folds 0
python scripts/run_cv.py --manifest data/manifest_loco_SMG973F_to_SMJ600G.csv --out runs/loco_G973F_to_J600G --folds 0
```
Roles are encoded in the `fold` column (0 = test camera, 1 = validation, 2 = train), so the normal runner and tables are reused. Training on SM-G973F is the harder direction:
its fakes include only types `-` and `M`, so types `C` and `S` appear only in the test camera (device shift and unseen-adulterant shift at once), and its training part has only ~19 samples.
The 10 images with no camera tag are excluded. Validation parts are tiny (6-8 samples), so Wombat-OA tuning is noisy: report that.

**Stress test** (degraded TEST images, saved weights, no retraining or retuning; needs `final.weights.h5` on disk, i.e. the Colab session still alive):
```bash
python scripts/stress_test.py --manifest data/manifest.csv --run runs/cv1                     # all folds, 8 perturbations
python scripts/stress_test.py --manifest data/manifest.csv --run runs/cv1 --perturb clean blur gray
```
Perturbations: clean, blur, jpeg (quality 20), noise, lowres (4x), gray, hue shift, dark. Writes `stress_results.csv` and prints accuracy/AUC tables per method x perturbation.
This is the setting where fusion / beam / Wombat can differ from a plain average when clean scores are all 100%.

## 7. Other data (folder workflow)
`scripts/split_dataset.py`, `scripts/check_dataset.py`, `scripts/train.py`, `scripts/decode.py` work on a plain `data/{train,val,test}/{real,fake}` tree.

## 8. Single-image Streamlit app
Install the frontend and inference dependencies, then run the app from the project root:
```bash
pip install -r app/requirements_app.txt
streamlit run app/streamlit_app.py
```
The app loads trained fold checkpoints from `runs/cv1/fold*/`. Saved `wombat_params.json` and `thresholds.json` enable the tuned decoder; if those files are missing but weights and `config.json` exist, the deep model runs with the default decoder and a 0.5 threshold. If no usable deep checkpoint is found, the app trains a colour-statistics Random Forest from `data/manifest.csv`; without that manifest it uses a simple colour heuristic. Dataset archives, extracted images, manifests, and run outputs are intentionally excluded from Git.
