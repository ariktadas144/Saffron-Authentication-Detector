#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import sys, io
# Force UTF-8 on Windows consoles so Unicode chars print correctly
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
"""build_colab_notebook.py  --  generate colab/saffron_colab.ipynb

Reads every project source file and embeds it byte-for-byte as a %%writefile cell.
The generated notebook is self-contained: upload only the .ipynb to Colab.

Usage:
    python colab/build_colab_notebook.py            # build the notebook
    python colab/build_colab_notebook.py --check    # re-read sources and verify SHA-256 matches
    python colab/build_colab_notebook.py --ast-check# also AST-compile every code cell
"""
import argparse, ast, hashlib, json, os, re, sys, textwrap, uuid
import nbformat

# ---------------------------------------------------------------------------
# Repository root (this script lives in  <root>/colab/build_colab_notebook.py)
# ---------------------------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT  = os.path.join(ROOT, "colab", "saffron_colab.ipynb")

# Files to embed as %%writefile cells.  (dst_rel, src_rel_to_ROOT)
EMBED_FILES = [
    ("src/saffron_cnn_gnn.py",          "src/saffron_cnn_gnn.py"),
    ("src/wombat_beam.py",               "src/wombat_beam.py"),
    ("src/color_baseline.py",            "src/color_baseline.py"),
    ("scripts/_path.py",                 "scripts/_path.py"),
    ("scripts/prepare_dataset.py",       "scripts/prepare_dataset.py"),
    ("scripts/run_cv.py",                "scripts/run_cv.py"),
    ("scripts/train_fold.py",            "scripts/train_fold.py"),
    ("scripts/train.py",                 "scripts/train.py"),
    ("scripts/decode.py",                "scripts/decode.py"),
    ("scripts/check_dataset.py",         "scripts/check_dataset.py"),
    ("scripts/split_dataset.py",         "scripts/split_dataset.py"),
    ("scripts/make_loco_manifests.py",   "scripts/make_loco_manifests.py"),
    ("scripts/stress_test.py",           "scripts/stress_test.py"),
    ("requirements.txt",                 "requirements.txt"),
    ("tests/conftest.py",                "tests/conftest.py"),
    ("tests/test_dataset_prep.py",       "tests/test_dataset_prep.py"),
    ("tests/test_decoder.py",            "tests/test_decoder.py"),
    ("tests/test_model_tf.py",           "tests/test_model_tf.py"),
    ("tests/test_pipeline_e2e.py",       "tests/test_pipeline_e2e.py"),
    ("tests/test_loco.py",               "tests/test_loco.py"),
    ("pytest.ini",                       "pytest.ini"),
]

COLAB_ROOT = "/content/saffron-authenticity"


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def uid() -> str:
    return uuid.uuid4().hex[:16]


# ---------------------------------------------------------------------------
# Cell builders
# ---------------------------------------------------------------------------
def md(source: str) -> dict:
    return {"cell_type": "markdown", "id": uid(), "metadata": {}, "source": source}


def code(source: str, tags: list[str] | None = None) -> dict:
    meta: dict = {}
    if tags:
        meta["tags"] = tags
    return {"cell_type": "code", "execution_count": None, "id": uid(),
            "metadata": meta, "outputs": [], "source": source}


def writefile_cell(dst_rel: str, content: str) -> dict:
    """One %%writefile cell.  The dst path is relative to COLAB_ROOT."""
    dst_abs = f"{COLAB_ROOT}/{dst_rel}"
    src = f"%%writefile {dst_abs}\n{content}"
    return code(src, tags=["writefile"])


def _load(src_rel: str) -> str:
    p = os.path.join(ROOT, src_rel.replace("/", os.sep))
    with open(p, encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Notebook content — every section is a function returning list[dict]
# ---------------------------------------------------------------------------

def section_title() -> list[dict]:
    return [md("""\
# 🌿 Saffron Authenticity: Real vs Fake — end-to-end sample-level cross-validation

**Model**: ConvNeXt-Tiny + Grid Graph-Attention (GNN) + Hybrid Attention Fusion (HAFM), three logit heads,
MC-Dropout uncertainty, 8 colour-preserving test-time views, Wombat-OA weighted beam decoding.

**Novelty**: Wombat-OA-tuned diversity-penalised beam search over test-time views + log-probability fusion.

## 📋 Table of Contents

| § | Section | Skip? |
|---|---------|-------|
| 1 | GPU & environment check | Never |
| 2 | Config | Edit only this cell |
| 3 | Drive mount | Skip if running locally |
| 4 | Write project files | Idempotent, run once |
| 5 | Data preparation | Run once; idempotent |
| 5a | Dataset audit | After §5 |
| 6 | Colour-statistics baseline | Before training |
| 7 | Smoke test | Recommended once (SMOKE=True) |
| 8 | Resume logic | Run on reconnect |
| 9 | Full cross-validation | Main job (~hours on GPU) |
| 10 | Aggregate results | After §9 |
| 11 | Training curves | After §9 |
| 12 | Per-fold results | After §9 |
| 13 | Pooled out-of-fold analysis | After §9 |
| 14 | Tuned decoder parameters | After §9 |
| 15 | Uncertainty check | After §9 |
| 16 | Per-type breakdown | After §9 |
| 17 | Optimiser study | After §9 |
| 18 | Headline dashboard | After §9 |
| 19 | Package results | After §9 |
| 20 | Interpretation guide | Read anytime |
| 21 | Troubleshooting | Read if stuck |
| 22 | Parameters reference | Read anytime |
| 24 | Leave-one-camera-out evaluation | After §9 |
| 25 | Stress-test robustness | After §9 |
| 26 | What to do next | Read anytime |

> **Runtime → Change runtime type → T4 GPU** (or better).
> **Restart-safe**: re-running any cell is safe. Finished folds are skipped automatically (§9).
> **MUST NOT** report results on the original Training/Testing split — it leaks all test samples into training.
""")]


def section_gpu_check() -> list[dict]:
    return [
        md("## § 1 · GPU & Environment Check"),
        code("""\
# § 1 — GPU & Environment Check
# Checks GPU, TF/Keras versions, installs missing packages. Never imports TF in the kernel.
import os, subprocess, sys

ALLOW_CPU = False   # set True in Config cell (§2) — overridden there

# ---- nvidia-smi ----
r = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
if r.returncode != 0:
    msg = "⚠ No GPU found. Set ALLOW_CPU=True in §2 to continue on CPU (very slow)."
    if not ALLOW_CPU:
        raise RuntimeError(msg)
    print(msg)
else:
    print(r.stdout[:600])

# ---- TF / Keras version check via subprocess (never import TF in kernel) ----
_ver = subprocess.run(
    [sys.executable, "-c",
     "import tensorflow as tf, keras; "
     "print(tf.__version__, keras.__version__, "
     "len(tf.config.list_physical_devices('GPU')))"],
    capture_output=True, text=True
)
if _ver.returncode != 0:
    print("TensorFlow not yet installed — will install in §2 / §4.")
    _tf_ver, _keras_ver, _ngpu = None, None, 0
else:
    _parts = _ver.stdout.strip().split()
    _tf_ver, _keras_ver, _ngpu = _parts[0], _parts[1], int(_parts[2])
    print(f"TensorFlow {_tf_ver}  |  Keras {_keras_ver}  |  GPUs visible: {_ngpu}")

# ---- Version requirements ----
from packaging.version import Version
_need_install = False
if _tf_ver and (Version(_tf_ver) < Version("2.16") or Version(_keras_ver) < Version("3.3")):
    print("⚠  TF < 2.16 or Keras < 3.3 detected. Will pip-install below. RESTART RUNTIME ONCE after install.")
    _need_install = True

if _need_install:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q",
         "tensorflow>=2.16", "keras>=3.3"],
        check=True
    )
    print("✅ Installed. Please RESTART THE RUNTIME, then re-run from §1.")
    raise SystemExit("Restart the runtime now.")

# ---- GPU memory growth (if GPU present) ----
subprocess.run(
    [sys.executable, "-c",
     "import tensorflow as tf; "
     "[tf.config.experimental.set_memory_growth(g, True) "
     " for g in tf.config.list_physical_devices('GPU')]"],
    capture_output=True
)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
print("§1 complete ✅")
"""),
    ]


def section_config() -> list[dict]:
    return [
        md("## § 2 · Config (the only cell you edit)"),
        code("""\
# § 2 — Config: edit this cell only.
# All other cells read these variables.

# ── Paths ──────────────────────────────────────────────────────────────────────────────
DRIVE_DATA_DIR  = "/content/drive/MyDrive/saffron_data"   # Drive folder containing the two .zip files (or extracted folders)
DRIVE_RUNS_DIR  = "/content/drive/MyDrive/saffron_runs"   # Drive folder for checkpoints (created automatically)

# ── Dataset / training ─────────────────────────────────────────────────────────────────
N_FOLDS       = 5
IMG_SIZE      = 256   # 384 needs BATCH=8 and more GPU RAM
BATCH         = 16    # use 8 if GPU < 16 GB
EPOCHS_WARM   = 3
EPOCHS_FT     = 25
MC_SAMPLES    = 16
POP           = 20    # Wombat-OA population
ITERS         = 25    # Wombat-OA iterations
SEED          = 42

# ── Run control ────────────────────────────────────────────────────────────────────────
SMOKE             = False   # True  → fold 0 only, tiny settings (pipeline test; numbers meaningless)
ALLOW_CPU         = False   # True  → skip GPU check (training will be very slow)
SAVE_WEIGHTS_TO_DRIVE = False  # True → include *.weights.h5 in Drive rsync (~300 MB per fold)

# ── Smoke-mode overrides (applied automatically when SMOKE=True) ────────────────────────
if SMOKE:
    N_FOLDS    = 5          # keep manifest folds, but only run fold 0
    IMG_SIZE   = 128
    BATCH      = 8
    EPOCHS_WARM = 1
    EPOCHS_FT  = 1
    MC_SAMPLES = 4
    POP        = 8
    ITERS      = 4
    print("⚠  SMOKE mode: fold 0 only, tiny settings. Numbers produced are NOT results.")

# ── Derived paths ──────────────────────────────────────────────────────────────────────
import os
COLAB_ROOT  = "/content/saffron-authenticity"
LOCAL_DATA  = os.path.join(COLAB_ROOT, "data")
LOCAL_RUNS  = os.path.join(COLAB_ROOT, "runs", "cv1")
FIGURES_DIR = os.path.join(LOCAL_RUNS, "figures")
os.makedirs(LOCAL_DATA,  exist_ok=True)
os.makedirs(LOCAL_RUNS,  exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

print(f"Config: IMG_SIZE={IMG_SIZE}  BATCH={BATCH}  EPOCHS={EPOCHS_WARM}+{EPOCHS_FT}  "
      f"MC={MC_SAMPLES}  POP={POP}  ITERS={ITERS}  SMOKE={SMOKE}")
print(f"COLAB_ROOT = {COLAB_ROOT}")
print(f"DRIVE_DATA = {DRIVE_DATA_DIR}")
print(f"DRIVE_RUNS = {DRIVE_RUNS_DIR}")
print("§2 complete ✅")
"""),
    ]


def section_drive_mount() -> list[dict]:
    return [
        md("## § 3 · Google Drive Mount"),
        code("""\
# § 3 — Drive mount.  Skip gracefully when running outside Colab (e.g. local test).
_IN_COLAB = False
try:
    from google.colab import drive as _drive
    _drive.mount("/content/drive", force_remount=False)
    _IN_COLAB = True
    print("Drive mounted ✅")
except ModuleNotFoundError:
    print("⚠ Not running in Colab — Drive mount skipped. Using local paths only.")
    # Override Drive paths to local equivalents for CPU-only local testing
    import os
    DRIVE_DATA_DIR = os.path.join(COLAB_ROOT, "local_data_input")
    DRIVE_RUNS_DIR = os.path.join(COLAB_ROOT, "local_runs_backup")
    os.makedirs(DRIVE_DATA_DIR, exist_ok=True)
    os.makedirs(DRIVE_RUNS_DIR, exist_ok=True)
    print(f"Using local data dir: {DRIVE_DATA_DIR}")
print("§3 complete ✅")
"""),
    ]


def section_writefile_intro() -> list[dict]:
    return [md("""\
## § 4 · Write Project Files

> **Generated — do not edit these cells manually.**
> They recreate the entire project under `/content/saffron-authenticity/` with the exact same layout as the repository.
> Re-running is idempotent (%%writefile overwrites but does not change logic).
""")]


def section_writefiles() -> list[dict]:
    """One %%writefile cell per file, preceded by mkdir cells."""
    cells: list[dict] = []
    # mkdir for all directories we'll need
    dirs_needed = sorted({
        os.path.dirname(f"{COLAB_ROOT}/{dst}") for dst, _ in EMBED_FILES
        if os.path.dirname(dst)
    })
    dirs_code = "\n".join(f'os.makedirs("{d}", exist_ok=True)' for d in dirs_needed)
    cells.append(code(f"import os\n{dirs_code}\nprint('directories ready')"))

    for dst_rel, src_rel in EMBED_FILES:
        content = _load(src_rel)
        cells.append(writefile_cell(dst_rel, content))
        cells.append(code(f"print('wrote {COLAB_ROOT}/{dst_rel}')"))
    return cells


def section_data() -> list[dict]:
    return [
        md("## § 5 · Data Preparation"),
        code("""\
# § 5 — Copy zips from Drive → local /content disk, then run prepare_dataset.py
# Drive I/O is slow; we always train from /content local disk.
import os, glob, shutil, subprocess, sys, time

COLAB_ROOT   = globals().get("COLAB_ROOT",  "/content/saffron-authenticity")
LOCAL_DATA   = globals().get("LOCAL_DATA",  os.path.join(COLAB_ROOT, "data"))
DRIVE_DATA_DIR = globals().get("DRIVE_DATA_DIR", "")
N_FOLDS      = globals().get("N_FOLDS", 5)
SEED         = globals().get("SEED", 42)

os.makedirs(LOCAL_DATA, exist_ok=True)

# ── Subprocess output helper ───────────────────────────────────────────────────────────
import subprocess, sys
def run(cmd, cwd=COLAB_ROOT, env_extra=None):
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "TF_CPP_MIN_LOG_LEVEL": "2"}
    if env_extra: env.update(env_extra)
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, bufsize=1, env=env)
    for line in proc.stdout:
        print(line, end="", flush=True)
    proc.wait()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)

# ── Locate source data ─────────────────────────────────────────────────────────────────
manifest_path = os.path.join(LOCAL_DATA, "manifest.csv")
if os.path.exists(manifest_path):
    print(f"manifest.csv already exists ({manifest_path}) — skipping data copy and prepare step.")
    print("Delete data/manifest.csv and re-run this cell to force a fresh preparation.")
else:
    # Find inputs: zips or folders in DRIVE_DATA_DIR
    zips = sorted(glob.glob(os.path.join(DRIVE_DATA_DIR, "*.zip")))
    folders = [d for d in glob.glob(os.path.join(DRIVE_DATA_DIR, "*"))
               if os.path.isdir(d) and os.path.basename(d).lower() in ("training","testing","train","test")]

    if len(zips) == 2:
        print(f"Found 2 zip files: {[os.path.basename(z) for z in zips]}")
        local_zips = []
        for z in zips:
            dst = os.path.join(LOCAL_DATA, os.path.basename(z))
            if not os.path.exists(dst):
                print(f"  copying {os.path.basename(z)} to local disk ...", end="", flush=True)
                shutil.copy2(z, dst); print(" done")
            else:
                print(f"  {os.path.basename(z)} already on local disk")
            local_zips.append(dst)
        src_args = local_zips
    elif len(zips) > 2:
        raise RuntimeError(
            f"Expected exactly 2 zip files in {DRIVE_DATA_DIR}, found {len(zips)}: {[os.path.basename(z) for z in zips]}\\n"
            "Please keep only the Training zip and the Testing zip in that folder.")
    elif folders:
        print(f"No zips found; using folders: {[os.path.basename(f) for f in folders]}")
        src_args = folders
    else:
        try:
            from google.colab import files as _cfiles
            print("No zips found in Drive. Please upload the two dataset zips now:")
            uploaded = _cfiles.upload()
            src_args = []
            for fname in uploaded:
                dst = os.path.join(LOCAL_DATA, fname)
                with open(dst, "wb") as f: f.write(uploaded[fname])
                src_args.append(dst)
            if len(src_args) != 2:
                raise RuntimeError(f"Expected 2 zips, got {len(src_args)}: {src_args}")
        except (ModuleNotFoundError, RuntimeError) as e:
            raise RuntimeError(
                f"No dataset found in {DRIVE_DATA_DIR} and not in Colab.\\n"
                f"Put Test_dataset-*.zip and train_dataset-*.zip in {DRIVE_DATA_DIR} and re-run.") from e

    print("\\nRunning prepare_dataset.py ...")
    run([sys.executable, "scripts/prepare_dataset.py",
         "--src", *src_args,
         "--data", LOCAL_DATA,
         "--folds", str(N_FOLDS),
         "--seed", str(SEED)])

print("\\n§5 preparation complete ✅")
"""),
        code("""\
# § 5 — Display: fold table, montage of random real/fake images
import os, pandas as pd, matplotlib.pyplot as plt, matplotlib.image as mpimg
import numpy as np, collections

COLAB_ROOT   = globals().get("COLAB_ROOT", "/content/saffron-authenticity")
LOCAL_DATA   = globals().get("LOCAL_DATA", os.path.join(COLAB_ROOT, "data"))
FIGURES_DIR  = globals().get("FIGURES_DIR", os.path.join(COLAB_ROOT, "runs","cv1","figures"))
os.makedirs(FIGURES_DIR, exist_ok=True)

man = pd.read_csv(os.path.join(LOCAL_DATA, "manifest.csv"))
print(f"Manifest: {len(man)} images  |  {man.group.nunique()} physical samples  |  "
      f"{(man.label==1).sum()} fake / {(man.label==0).sum()} real")

# Per-fold summary
rep = man.groupby("fold").agg(
    images=("path","size"), samples=("group","nunique"), fake_frac=("label","mean")
)
rep["types(fake)"] = [
    dict(collections.Counter(man[(man.fold==k)&(man.label==1)].type)) for k in rep.index
]
print("\\nPer-fold summary (each fold is the TEST set once):")
print(rep.round(3).to_string())

# 2×5 montage
rng = np.random.default_rng(42)
fig, axes = plt.subplots(2, 5, figsize=(14, 5.5))
for col, (label, name) in enumerate([(0,"Real"),(1,"Fake")]):
    pool = man[man.label==label].path.values
    chosen = rng.choice(pool, min(5, len(pool)), replace=False)
    for i, p in enumerate(chosen):
        row = 0 if label==0 else 1
        ax = axes[row, i]
        try:
            img = mpimg.imread(p)
            ax.imshow(img); ax.axis("off")
            ax.set_title(f"{name} {i+1}", fontsize=8)
        except Exception as e:
            ax.text(0.5,0.5,str(e),ha="center",va="center",fontsize=6); ax.axis("off")
plt.suptitle("Sample images from the working copies", fontsize=11)
plt.tight_layout()
fn = os.path.join(FIGURES_DIR, "montage.png")
plt.savefig(fn, dpi=120, bbox_inches="tight"); plt.show()
print(f"Saved → {fn}")
print("§5 display complete ✅")
"""),
    ]


def section_dataset_audit() -> list[dict]:
    return [
        md("""\
## § 5a · Dataset Audit

Checks for the known dataset problems described in `README.md § 0`:
- **Leak**: how many of the 57 original test samples also appear in the training folder (expected: all 57)
- **Duplicates**: exact-duplicate files
- **Image-size histogram**: shows whether image resolution is a shortcut cue (real photos tend to be larger)
- **Shortcut AUC**: group-CV AUC from (a) size/aspect/file-size features, (b) colour statistics, (c) both combined

⚠ **Red flag**: colour-stats AUC > 0.9 means trivial colour features explain most of the task — any deep model must beat this.
"""),
        code("""\
# § 5a — Dataset audit: leak summary, duplicates, size histogram, shortcut AUC
import os, sys, subprocess, collections
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

COLAB_ROOT  = globals().get("COLAB_ROOT", "/content/saffron-authenticity")
LOCAL_DATA  = globals().get("LOCAL_DATA", os.path.join(COLAB_ROOT, "data"))
FIGURES_DIR = globals().get("FIGURES_DIR", os.path.join(COLAB_ROOT,"runs","cv1","figures"))
sys.path.insert(0, os.path.join(COLAB_ROOT, "src"))
sys.path.insert(0, os.path.join(COLAB_ROOT, "scripts"))
os.makedirs(FIGURES_DIR, exist_ok=True)

man = pd.read_csv(os.path.join(LOCAL_DATA, "manifest.csv"))

# ── Leak summary (re-compute from orig_split column) ────────────────────────────────────
tr_samp = set(man[man.orig_split=="train"].sample)
te_samp = set(man[man.orig_split=="test"].sample)
n_leaked = len(tr_samp & te_samp)
n_te     = len(te_samp)
print(f"[LEAK] {n_leaked}/{n_te} original test samples also in training (different shots)")
assert n_te > 0, "orig_split column missing test rows — re-run §5 with the original zips"

# Bar
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
axes[0].bar(["Only train", "Only test", "BOTH (leak)"],
            [len(tr_samp)-n_leaked, n_te-n_leaked, n_leaked],
            color=["steelblue","orange","crimson"])
axes[0].set_title("Sample leak: original split"); axes[0].set_ylabel("# samples")

# ── Image-size histogram ─────────────────────────────────────────────────────────────────
widths, heights = [], []
file_sizes = []
for _, row in man.iterrows():
    try:
        with Image.open(row.path) as im:
            w, h = im.size
        widths.append(w); heights.append(h)
        file_sizes.append(os.path.getsize(row.path))
    except Exception:
        widths.append(0); heights.append(0); file_sizes.append(0)
man = man.copy()
man["img_w"], man["img_h"] = widths, heights
man["img_area"] = np.array(widths) * np.array(heights)
man["file_bytes"] = file_sizes
man["aspect"] = np.array(widths) / np.maximum(np.array(heights), 1)

for label, name, c in [(0,"Real","steelblue"),(1,"Fake","crimson")]:
    sub = man[man.label==label]
    axes[1].hist(sub.img_w.clip(0,3000)/1000, bins=25, alpha=0.6, label=name, color=c)
axes[1].set_xlabel("Width (k px)"); axes[1].set_ylabel("Count")
axes[1].set_title("Image-width distribution by class"); axes[1].legend()

# ── Shortcut AUC: size/aspect/file-size and colour ─────────────────────────────────────
import color_baseline as cb

X_col = cb.features(man.path.tolist())
X_size = np.column_stack([man.img_w, man.img_h, man.img_area, man.aspect, man.file_bytes])

K = man.fold.max() + 1
results = {"size_feat": [], "colour": [], "both": []}
for k in range(K):
    kv = (k + 1) % K
    tr_idx = man[~man.fold.isin([k, kv])].index.values
    te_idx = man[man.fold == k].index.values
    ytr, yte = man.label.values[tr_idx], man.label.values[te_idx]
    for key, X in [("size_feat", X_size), ("colour", X_col), ("both", np.hstack([X_size, X_col]))]:
        p = cb.fit_predict(X[tr_idx], ytr, X[te_idx])
        results[key].append(roc_auc_score(yte, p) if len(np.unique(yte)) > 1 else float("nan"))

print(f"\\nShortcut AUC (image-level, group-CV):")
for k, v in results.items():
    print(f"  {k:15s}  per-fold: {[round(x,3) for x in v]}  mean={np.nanmean(v):.3f}")

axes[2].boxplot([results["size_feat"], results["colour"], results["both"]],
                tick_labels=["Size/aspect", "Colour", "Both"])
axes[2].set_ylim(0.4, 1.05); axes[2].axhline(0.5, ls="--", c="grey")
axes[2].set_title("Shortcut AUC (group-CV)"); axes[2].set_ylabel("AUC")

plt.tight_layout()
fn = os.path.join(FIGURES_DIR, "dataset_audit.png")
plt.savefig(fn, dpi=120, bbox_inches="tight"); plt.show()
print(f"Saved → {fn}")

assert np.nanmean(results["colour"]) > 0.5, \
    "Colour AUC ≤ 0.5 — unexpected. Check that manifest labels are correct."
print(f"\\nColour AUC {np.nanmean(results['colour']):.3f} — this is the bar every deep model must beat.")
print("§5a complete ✅")
"""),
    ]


def section_colour_baseline() -> list[dict]:
    return [
        md("""\
## § 6 · Colour-Statistics Baseline (CPU)

Uses the same 5 folds as `run_cv.py` (test=fold k, val=fold k+1, train=rest) so the AUC is directly comparable to deep-model results.
The colour features are RGB mean/std + HSV mean/std on a 64×64 centre crop — 12 numbers per image.

**Why this matters**: colour alone reaches AUC ~0.94 on this dataset. Any deep model row in the final table that falls below this baseline is not learning anything beyond colour.
"""),
        code("""\
# § 6 — Colour-statistics baseline under the same 5 folds as the deep model
import os, sys
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score

COLAB_ROOT  = globals().get("COLAB_ROOT", "/content/saffron-authenticity")
LOCAL_DATA  = globals().get("LOCAL_DATA", os.path.join(COLAB_ROOT, "data"))
LOCAL_RUNS  = globals().get("LOCAL_RUNS", os.path.join(COLAB_ROOT, "runs", "cv1"))
FIGURES_DIR = globals().get("FIGURES_DIR", os.path.join(LOCAL_RUNS, "figures"))
sys.path.insert(0, os.path.join(COLAB_ROOT, "src"))
sys.path.insert(0, os.path.join(COLAB_ROOT, "scripts"))
os.makedirs(FIGURES_DIR, exist_ok=True)

import color_baseline as cb

man = pd.read_csv(os.path.join(LOCAL_DATA, "manifest.csv"))
cache_file = os.path.join(LOCAL_RUNS, "colour_features.npz")
X = cb.features(man.path.tolist(), cache=cache_file)
K = man.fold.max() + 1

fold_aucs, fold_accs = [], []
print(f"{'Fold':>5}  {'AUC':>6}  {'Acc':>6}")
for k in range(K):
    kv = (k + 1) % K
    tr_idx = man[~man.fold.isin([k, kv])].index.values
    te_idx = man[man.fold == k].index.values
    ytr, yte = man.label.values[tr_idx], man.label.values[te_idx]
    p = cb.fit_predict(X[tr_idx], ytr, X[te_idx])
    auc = roc_auc_score(yte, p) if len(np.unique(yte)) > 1 else float("nan")
    acc = accuracy_score(yte, p >= 0.5)
    fold_aucs.append(auc); fold_accs.append(acc)
    print(f"  {k:3d}    {auc:.4f}  {acc:.4f}")

print(f"  mean   {np.nanmean(fold_aucs):.4f}  {np.mean(fold_accs):.4f}")
print(f"  std    {np.nanstd(fold_aucs):.4f}  {np.std(fold_accs):.4f}")

fig, ax = plt.subplots(1, 1, figsize=(6, 3.5))
ax.bar(range(K), fold_aucs, color="steelblue", alpha=0.8)
ax.axhline(np.nanmean(fold_aucs), ls="--", c="navy", label=f"mean {np.nanmean(fold_aucs):.3f}")
ax.set_xticks(range(K)); ax.set_xticklabels([f"Fold {k}" for k in range(K)])
ax.set_ylim(0, 1.05); ax.set_ylabel("AUC"); ax.set_title("Colour-statistics baseline AUC per fold")
ax.legend(); plt.tight_layout()
fn = os.path.join(FIGURES_DIR, "baseline_auc.png")
plt.savefig(fn, dpi=120, bbox_inches="tight"); plt.show(); print(f"Saved → {fn}")

print(f"\\n⚠  The bar to beat: colour baseline mean AUC = {np.nanmean(fold_aucs):.3f}")
print("§6 complete ✅")
"""),
    ]


def section_smoke() -> list[dict]:
    return [
        md("""\
## § 7 · Smoke Test (optional, runs only when `SMOKE=True`)

Runs fold 0 with tiny settings and random weights. **Numbers here are NOT results** — this is a pipeline sanity check only.
Expected runtime: ~5–15 min on GPU, ~30–60 min on CPU.
"""),
        code("""\
# § 7 — Smoke test: fold 0, tiny settings, no ImageNet weights
import os, sys, subprocess, time
SMOKE      = globals().get("SMOKE", False)
COLAB_ROOT = globals().get("COLAB_ROOT", "/content/saffron-authenticity")
LOCAL_DATA = globals().get("LOCAL_DATA", os.path.join(COLAB_ROOT, "data"))
IMG_SIZE   = globals().get("IMG_SIZE", 128)
MC_SAMPLES = globals().get("MC_SAMPLES", 4)
POP        = globals().get("POP", 8)
ITERS      = globals().get("ITERS", 4)
SEED       = globals().get("SEED", 42)

def run(cmd, cwd=COLAB_ROOT, env_extra=None):
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "TF_CPP_MIN_LOG_LEVEL": "2"}
    if env_extra: env.update(env_extra)
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, bufsize=1, env=env)
    for line in proc.stdout:
        print(line, end="", flush=True)
    proc.wait()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)

if not SMOKE:
    print("SMOKE=False — skipping smoke test. Set SMOKE=True in §2 to enable.")
else:
    print("Running smoke test (fold 0, no ImageNet, tiny settings) ...")
    t0 = time.time()
    run([sys.executable, "scripts/run_cv.py",
         "--manifest", os.path.join(LOCAL_DATA, "manifest.csv"),
         "--out", os.path.join(COLAB_ROOT, "runs", "smoke"),
         "--folds", "0",
         "--img-size", str(IMG_SIZE),
         "--epochs-warm", "1",
         "--epochs-ft", "1",
         "--mc-samples", str(MC_SAMPLES),
         "--no-imagenet",
         "--pop", str(POP),
         "--iters", str(ITERS),
         "--seed", str(SEED)])
    elapsed = time.time() - t0
    smoke_csv = os.path.join(COLAB_ROOT, "runs", "smoke", "fold0", "ablation_test.csv")
    assert os.path.exists(smoke_csv), f"Smoke test failed: {smoke_csv} not created"
    print(f"\\n✅ Smoke test passed in {elapsed/60:.1f} min — pipeline is healthy.")
    print("Numbers above are NOT results (random weights, tiny settings).")
print("§7 complete ✅")
"""),
    ]


def section_resume() -> list[dict]:
    return [
        md("## § 8 · Resume Logic (re-run after a Colab disconnect)"),
        code("""\
# § 8 — If Drive has a previous runs/cv1, rsync it to local disk before training.
# Re-running this cell on a fresh session restores all finished folds automatically.
import os, shutil, subprocess

COLAB_ROOT     = globals().get("COLAB_ROOT", "/content/saffron-authenticity")
LOCAL_RUNS     = globals().get("LOCAL_RUNS", os.path.join(COLAB_ROOT, "runs", "cv1"))
DRIVE_RUNS_DIR = globals().get("DRIVE_RUNS_DIR", "")

drive_cv1 = os.path.join(DRIVE_RUNS_DIR, "cv1")
os.makedirs(LOCAL_RUNS, exist_ok=True)

if os.path.isdir(drive_cv1):
    print(f"Found previous run in Drive: {drive_cv1} — syncing to local disk ...")
    # Try rsync first, fall back to shutil
    rsync_ok = subprocess.run(
        ["rsync", "-a", "--exclude=*.weights.h5", drive_cv1 + "/", LOCAL_RUNS + "/"],
        capture_output=True
    ).returncode == 0
    if rsync_ok:
        print("rsync complete ✅")
    else:
        print("rsync not available — using shutil.copytree fallback ...")
        # Copy only non-weight files
        for dp, dns, fns in os.walk(drive_cv1):
            rel = os.path.relpath(dp, drive_cv1)
            dst_dp = os.path.join(LOCAL_RUNS, rel)
            os.makedirs(dst_dp, exist_ok=True)
            for fn in fns:
                if fn.endswith(".weights.h5"): continue
                src_f = os.path.join(dp, fn)
                dst_f = os.path.join(dst_dp, fn)
                if not os.path.exists(dst_f):
                    shutil.copy2(src_f, dst_f)
        print("shutil copy complete ✅")

    # Report what folds are already done
    done = [d for d in os.listdir(LOCAL_RUNS)
            if d.startswith("fold") and os.path.exists(os.path.join(LOCAL_RUNS, d, "cache_test.npz"))]
    print(f"Folds already complete: {sorted(done)}")
else:
    print(f"No previous run found in {drive_cv1} — starting fresh.")
print("§8 complete ✅")
"""),
    ]


def section_training() -> list[dict]:
    return [
        md("""\
## § 9 · Full Cross-Validation

Trains all folds. Each fold spawns a fresh subprocess so TensorFlow GPU memory is released between folds.
Finished folds (both `cache_val.npz` and `cache_test.npz` present) are skipped automatically.
After each fold the results are synced to Drive (excluding weights unless `SAVE_WEIGHTS_TO_DRIVE=True`).

Training log: one line per epoch (Keras `verbose=2`). ETA is updated after each fold.
"""),
        code("""\
# § 9 — Full cross-validation: train → cache → Wombat-OA decode, for every fold
import os, sys, subprocess, time, shutil

COLAB_ROOT            = globals().get("COLAB_ROOT", "/content/saffron-authenticity")
LOCAL_DATA            = globals().get("LOCAL_DATA", os.path.join(COLAB_ROOT, "data"))
LOCAL_RUNS            = globals().get("LOCAL_RUNS", os.path.join(COLAB_ROOT, "runs", "cv1"))
DRIVE_RUNS_DIR        = globals().get("DRIVE_RUNS_DIR", "")
N_FOLDS               = globals().get("N_FOLDS", 5)
IMG_SIZE              = globals().get("IMG_SIZE", 256)
BATCH                 = globals().get("BATCH", 16)
EPOCHS_WARM           = globals().get("EPOCHS_WARM", 3)
EPOCHS_FT             = globals().get("EPOCHS_FT", 25)
MC_SAMPLES            = globals().get("MC_SAMPLES", 16)
POP                   = globals().get("POP", 20)
ITERS                 = globals().get("ITERS", 25)
SEED                  = globals().get("SEED", 42)
SMOKE                 = globals().get("SMOKE", False)
SAVE_WEIGHTS_TO_DRIVE = globals().get("SAVE_WEIGHTS_TO_DRIVE", False)
ALLOW_CPU             = globals().get("ALLOW_CPU", False)

import pandas as pd
man = pd.read_csv(os.path.join(LOCAL_DATA, "manifest.csv"))
K   = int(man.fold.max()) + 1
folds_to_run = [0] if SMOKE else list(range(K))

os.makedirs(LOCAL_RUNS, exist_ok=True)
drive_cv1 = os.path.join(DRIVE_RUNS_DIR, "cv1")

def run(cmd, cwd=COLAB_ROOT, env_extra=None):
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "TF_CPP_MIN_LOG_LEVEL": "2"}
    if env_extra: env.update(env_extra)
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, bufsize=1, env=env)
    for line in proc.stdout:
        print(line, end="", flush=True)
    proc.wait()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)

def sync_to_drive():
    if not DRIVE_RUNS_DIR:
        return
    os.makedirs(drive_cv1, exist_ok=True)
    exclude = [] if SAVE_WEIGHTS_TO_DRIVE else ["--exclude=*.weights.h5"]
    r = subprocess.run(
        ["rsync", "-a", *exclude, LOCAL_RUNS + "/", drive_cv1 + "/"],
        capture_output=True
    )
    if r.returncode != 0:
        # fallback: shutil
        for dp, _, fns in os.walk(LOCAL_RUNS):
            rel = os.path.relpath(dp, LOCAL_RUNS)
            ddp = os.path.join(drive_cv1, rel); os.makedirs(ddp, exist_ok=True)
            for fn in fns:
                if fn.endswith(".weights.h5") and not SAVE_WEIGHTS_TO_DRIVE: continue
                shutil.copy2(os.path.join(dp, fn), os.path.join(ddp, fn))
    try:
        os.sync()
    except AttributeError:
        pass  # Windows

fold_times = []
t_total_start = time.time()
for fold in folds_to_run:
    od = os.path.join(LOCAL_RUNS, f"fold{fold}")
    if (os.path.exists(os.path.join(od, "cache_val.npz")) and
            os.path.exists(os.path.join(od, "cache_test.npz"))):
        print(f"\\n→ Fold {fold}: caches found, skipping training.")
        continue

    t0 = time.time()
    print(f"\\n{'='*60}")
    print(f"FOLD {fold} / {K-1} — training starting")
    print(f"{'='*60}")
    cmd = [sys.executable, "scripts/train_fold.py",
           "--manifest", os.path.join(LOCAL_DATA, "manifest.csv"),
           "--out", od,
           "--fold", str(fold),
           "--img-size", str(IMG_SIZE),
           "--batch",    str(BATCH),
           "--epochs-warm", str(EPOCHS_WARM),
           "--epochs-ft",   str(EPOCHS_FT),
           "--mc-samples",  str(MC_SAMPLES),
           "--seed",        str(SEED)]
    if SMOKE:
        cmd.append("--no-imagenet")
    try:
        run(cmd)
    except subprocess.CalledProcessError as e:
        print(f"\\n❌ Fold {fold} FAILED: {e}")
        print("Next action: check the error above. Common causes: OOM (reduce BATCH or IMG_SIZE), TF version mismatch.")
        raise

    elapsed = time.time() - t0
    fold_times.append(elapsed)
    mean_t = sum(fold_times) / len(fold_times)
    remaining = [f for f in folds_to_run
                 if not (os.path.exists(os.path.join(LOCAL_RUNS,f"fold{f}","cache_val.npz")) and
                         os.path.exists(os.path.join(LOCAL_RUNS,f"fold{f}","cache_test.npz")))]
    eta = mean_t * max(len(remaining) - 1, 0)
    print(f"\\n✅ Fold {fold} complete in {elapsed/60:.1f} min  |  ETA remaining: {eta/60:.0f} min")

    # Wombat-OA decoding for this fold
    import color_baseline as cb
    import wombat_beam as wb
    import json, numpy as np

    val_c  = wb.Cache.load(os.path.join(od, "cache_val.npz"))
    test_c = wb.Cache.load(os.path.join(od, "cache_test.npz"))
    kv = (fold + 1) % K
    te_idx = man[man.fold == fold].index.values
    tr_idx = man[~man.fold.isin([fold, kv])].index.values
    X_col  = cb.features(man.path.tolist(), cache=os.path.join(LOCAL_RUNS, "colour_features.npz"))
    p_col  = cb.fit_predict(X_col[tr_idx], man.label.values[tr_idx], X_col[te_idx])
    assert (man.loc[te_idx, "group"].values == test_c.groups).all(), "cache/manifest order mismatch"
    assert (man.loc[te_idx, "label"].values == test_c.y).all(), "cache/manifest label mismatch"
    table, info = wb.evaluate_all(val_c, test_c, pop=POP, iters=ITERS, seed=fold,
                                  extra_rows={"colour-statistics random forest (baseline)": (p_col, 0.5)})
    table.to_csv(os.path.join(od, "ablation_test.csv"))
    info["sample_table"].to_csv(os.path.join(od, "ablation_sample_level.csv"))
    pd.DataFrame({
        **{m: p for m, p in info["probs"].items()},
        "group": test_c.groups, "y": test_c.y, "fold": fold
    }).to_csv(os.path.join(od, "preds_test.csv"), index=False)
    json.dump(info["thr"], open(os.path.join(od, "thresholds.json"), "w"))
    json.dump({q: (v.tolist() if hasattr(v,"tolist") else v) for q,v in info["params"].items()},
              open(os.path.join(od, "wombat_params.json"), "w"), indent=1)

    KEYS = ["acc","bal_acc","f1_fake","recall_fake","specificity","auc","ece"]
    print(f"\\n--- Fold {fold} ablation (image level) ---")
    print(table[KEYS].round(3).to_string())

    print(f"\\nSyncing fold {fold} to Drive ...")
    sync_to_drive()

total_elapsed = time.time() - t_total_start
print(f"\\n{'='*60}")
print(f"All folds complete. Total time: {total_elapsed/60:.1f} min")
print(f"{'='*60}")
print("§9 complete ✅")
"""),
    ]


def section_aggregate() -> list[dict]:
    return [
        md("## § 10 · Aggregate Results & Summary Tables"),
        code("""\
# § 10 — Aggregate results across folds; load summary tables; styled display + bar chart
import os, sys, subprocess, json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from IPython.display import display

COLAB_ROOT  = globals().get("COLAB_ROOT", "/content/saffron-authenticity")
LOCAL_DATA  = globals().get("LOCAL_DATA", os.path.join(COLAB_ROOT, "data"))
LOCAL_RUNS  = globals().get("LOCAL_RUNS", os.path.join(COLAB_ROOT, "runs", "cv1"))
FIGURES_DIR = globals().get("FIGURES_DIR", os.path.join(LOCAL_RUNS, "figures"))
os.makedirs(FIGURES_DIR, exist_ok=True)
sys.path.insert(0, os.path.join(COLAB_ROOT, "src"))
sys.path.insert(0, os.path.join(COLAB_ROOT, "scripts"))

def run(cmd, cwd=COLAB_ROOT, env_extra=None):
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "TF_CPP_MIN_LOG_LEVEL": "2"}
    if env_extra: env.update(env_extra)
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, bufsize=1, env=env)
    for line in proc.stdout: print(line, end="", flush=True)
    proc.wait()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)

# Re-run aggregate step to get fresh summary CSVs
run([sys.executable, "scripts/run_cv.py",
     "--manifest", os.path.join(LOCAL_DATA, "manifest.csv"),
     "--out", LOCAL_RUNS,
     "--aggregate-only"])

KEYS = ["acc","bal_acc","f1_fake","recall_fake","specificity","auc","ece"]
PROPOSED = "PROPOSED: Wombat-OA + weighted beam + log-prob fusion"
BASELINE = "colour-statistics random forest (baseline)"

for level, fname in [("IMAGE level", "summary_image.csv"),
                     ("SAMPLE level", "summary_sample.csv"),
                     ("Pooled OOF (SAMPLE)", "pooled_oof_sample_level.csv")]:
    fp = os.path.join(LOCAL_RUNS, fname)
    if not os.path.exists(fp):
        print(f"⚠ {fname} not found — run §9 first"); continue
    df = pd.read_csv(fp, index_col=0)
    # For summary files, values are strings "mean ± std"; for pooled they are floats
    is_float = df.dtypes.iloc[0] != object
    if is_float:
        cols = [c for c in KEYS if c in df.columns]
        styled = df[cols].style \
            .highlight_max(axis=0, color="lightgreen") \
            .format("{:.4f}")
    else:
        cols = [c for c in KEYS if c in df.columns]
        styled = df[cols].style
    print(f"\\n=== {level} ===")
    display(styled)

# ── Grouped bar chart: mean AUC and F1(fake) with std bars ─────────────────────────────
mean_fp = os.path.join(LOCAL_RUNS, "summary_sample_mean.csv")
if os.path.exists(mean_fp):
    mean_df = pd.read_csv(mean_fp, index_col=0)
    # Compute std from per-fold data
    folds = sorted(d for d in os.listdir(LOCAL_RUNS)
                   if d.startswith("fold") and os.path.exists(os.path.join(LOCAL_RUNS, d, "ablation_sample_level.csv")))
    tabs = [pd.read_csv(os.path.join(LOCAL_RUNS, d, "ablation_sample_level.csv"), index_col=0)
            for d in folds]
    cat = pd.concat(tabs)
    std_df = cat.groupby(level=0, sort=False)[KEYS].std(ddof=0)

    methods = mean_df.index.tolist()
    x = np.arange(len(methods))
    fig, axes = plt.subplots(1, 2, figsize=(max(12, len(methods)*0.6), 5))
    colors = plt.cm.tab20(np.linspace(0, 1, len(methods)))
    for ai, col in enumerate(["auc", "f1_fake"]):
        ax = axes[ai]
        vals = mean_df[col].values if col in mean_df.columns else np.zeros(len(methods))
        stds = std_df[col].values  if col in std_df.columns  else np.zeros(len(methods))
        bars = ax.bar(x, vals, color=colors, alpha=0.85, yerr=stds, capsize=3)
        # Highlight baseline
        if BASELINE in methods:
            bi = methods.index(BASELINE)
            bars[bi].set_edgecolor("crimson"); bars[bi].set_linewidth(2.5)
        ax.set_xticks(x)
        ax.set_xticklabels([m[:35] for m in methods], rotation=45, ha="right", fontsize=7)
        ax.set_ylabel(col.upper())
        ax.set_title(f"Mean ± std {col.upper()} (sample level, 5-fold CV)")
        ax.set_ylim(0, 1.05)
    plt.tight_layout()
    fn = os.path.join(FIGURES_DIR, "summary_bar.png")
    plt.savefig(fn, dpi=120, bbox_inches="tight"); plt.show(); print(f"Saved → {fn}")

print("§10 complete ✅")
"""),
    ]


def section_training_curves() -> list[dict]:
    return [
        md("""\
## § 11 · Training Curves

Per-fold fine-tuning history from `history_finetune.csv`. Useful to spot overfitting (train AUC ~ 1.0 while val AUC is low)
or under-training (AUC still rising at the last epoch — increase `EPOCHS_FT`).
"""),
        code("""\
# § 11 — Training curves: loss and AUC per fold from history_finetune.csv
import os
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

COLAB_ROOT  = globals().get("COLAB_ROOT", "/content/saffron-authenticity")
LOCAL_RUNS  = globals().get("LOCAL_RUNS", os.path.join(COLAB_ROOT, "runs", "cv1"))
FIGURES_DIR = globals().get("FIGURES_DIR", os.path.join(LOCAL_RUNS, "figures"))
os.makedirs(FIGURES_DIR, exist_ok=True)

folds = sorted(d for d in os.listdir(LOCAL_RUNS)
               if d.startswith("fold") and os.path.exists(os.path.join(LOCAL_RUNS, d, "history_finetune.csv")))
if not folds:
    print("No history_finetune.csv files found — run §9 first."); raise SystemExit()

K = len(folds)
plot_cols = []
sample_hist = pd.read_csv(os.path.join(LOCAL_RUNS, folds[0], "history_finetune.csv"))
for c in sample_hist.columns:
    if any(kw in c for kw in ("loss","auc","acc")) and "val_" not in c:
        plot_cols.append(c)

nplots = min(len(plot_cols), 4)
fig, axes = plt.subplots(K, nplots, figsize=(4*nplots, 3*K), squeeze=False)
for fi, fold_dir in enumerate(folds):
    hist = pd.read_csv(os.path.join(LOCAL_RUNS, fold_dir, "history_finetune.csv"))
    for pi, col in enumerate(plot_cols[:nplots]):
        ax = axes[fi][pi]
        ax.plot(hist[col], label="train")
        vcol = "val_" + col
        if vcol in hist.columns:
            ax.plot(hist[vcol], ls="--", label="val")
        ax.set_title(f"{fold_dir} — {col}", fontsize=8)
        ax.set_xlabel("epoch"); ax.legend(fontsize=6)

plt.suptitle("Fine-tuning history per fold", fontsize=11)
plt.tight_layout()
fn = os.path.join(FIGURES_DIR, "training_curves.png")
plt.savefig(fn, dpi=100, bbox_inches="tight"); plt.show(); print(f"Saved → {fn}")
print("§11 complete ✅")
"""),
    ]


def section_perfold_results() -> list[dict]:
    return [
        md("""\
## § 12 · Per-Fold Results: Proposed vs Colour Baseline

Shows paired per-fold delta (proposed − baseline) for AUC and F1(fake). With only 5 folds and 11–13 samples each,
no significance test is meaningful — report the delta pattern and note if the proposed method consistently beats the baseline or not.

**Red flag**: proposed method consistently below the colour baseline → deep model is not learning beyond colour.
"""),
        code("""\
# § 12 — Per-fold results table: proposed vs baseline, with deltas
import os
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from IPython.display import display

COLAB_ROOT  = globals().get("COLAB_ROOT", "/content/saffron-authenticity")
LOCAL_RUNS  = globals().get("LOCAL_RUNS", os.path.join(COLAB_ROOT, "runs", "cv1"))
FIGURES_DIR = globals().get("FIGURES_DIR", os.path.join(LOCAL_RUNS, "figures"))
os.makedirs(FIGURES_DIR, exist_ok=True)

PROPOSED = "PROPOSED: Wombat-OA + weighted beam + log-prob fusion"
BASELINE = "colour-statistics random forest (baseline)"
KEYS     = ["auc", "f1_fake", "acc", "recall_fake", "specificity"]

folds = sorted(d for d in os.listdir(LOCAL_RUNS)
               if d.startswith("fold") and os.path.exists(os.path.join(LOCAL_RUNS, d, "ablation_sample_level.csv")))
if not folds:
    print("No ablation_sample_level.csv found — run §9 first."); raise SystemExit()

rows_prop, rows_base = [], []
for fd in folds:
    df = pd.read_csv(os.path.join(LOCAL_RUNS, fd, "ablation_sample_level.csv"), index_col=0)
    for rows, name in [(rows_prop, PROPOSED), (rows_base, BASELINE)]:
        if name in df.index:
            rows.append({k: df.loc[name, k] for k in KEYS if k in df.columns})
        else:
            rows.append({k: float("nan") for k in KEYS})

prop_df = pd.DataFrame(rows_prop, index=folds)
base_df = pd.DataFrame(rows_base, index=folds)
delta_df = prop_df - base_df

combined = pd.concat([prop_df.add_suffix("_prop"), base_df.add_suffix("_base"), delta_df.add_suffix("_Δ")], axis=1)
print("Per-fold results (sample level): proposed | baseline | Δ(proposed − baseline)")
display(combined.round(4))

# Line plot
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
K = len(folds)
for ai, col in enumerate(["auc", "f1_fake"]):
    ax = axes[ai]
    ax.plot(range(K), prop_df[col].values, "o-", label="Proposed", color="steelblue")
    ax.plot(range(K), base_df[col].values, "s--", label="Colour baseline", color="crimson")
    ax.set_xticks(range(K)); ax.set_xticklabels(folds, rotation=15)
    ax.set_ylabel(col.upper()); ax.set_title(f"{col.upper()} per fold (sample level)")
    ax.set_ylim(0, 1.05); ax.legend()
plt.tight_layout()
fn = os.path.join(FIGURES_DIR, "perfold_comparison.png")
plt.savefig(fn, dpi=120, bbox_inches="tight"); plt.show(); print(f"Saved → {fn}")
print("§12 complete ✅")
"""),
    ]


def section_pooled_analysis() -> list[dict]:
    return [
        md("""\
## § 13 · Pooled Out-of-Fold Analysis

Every sample is tested exactly once (different fold). Uses each fold's tuned threshold (as `run_cv.py` does) before pooling.
Shows ROC, PR curves, calibration diagram, and sample-level confusion matrices for 4 key methods.

**Why pooled OOF matters**: fold-level tables have ~11–13 samples each (very noisy). The pooled OOF table uses all samples
and is the most reliable summary when folds are small.
"""),
        code("""\
# § 13 — Pooled OOF: ROC, PR, calibration, confusion matrices (4 methods, sample level)
import os, sys, json
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
from scipy.special import expit, logit
from IPython.display import display

COLAB_ROOT  = globals().get("COLAB_ROOT", "/content/saffron-authenticity")
LOCAL_RUNS  = globals().get("LOCAL_RUNS", os.path.join(COLAB_ROOT, "runs", "cv1"))
FIGURES_DIR = globals().get("FIGURES_DIR", os.path.join(LOCAL_RUNS, "figures"))
sys.path.insert(0, os.path.join(COLAB_ROOT, "src"))
os.makedirs(FIGURES_DIR, exist_ok=True)
import wombat_beam as wb

PROPOSED = "PROPOSED: Wombat-OA + weighted beam + log-prob fusion"
BASELINE = "colour-statistics random forest (baseline)"
TTA      = "TTA mean-logit, hafm"
BEST_SGL = "single view, hafm"
METHODS  = [BASELINE, BEST_SGL, TTA, PROPOSED]

folds = sorted(d for d in os.listdir(LOCAL_RUNS)
               if d.startswith("fold") and os.path.exists(os.path.join(LOCAL_RUNS, d, "preds_test.csv")))
if not folds:
    print("preds_test.csv not found — run §9 first."); raise SystemExit()

parts = []
for fd in folds:
    q  = pd.read_csv(os.path.join(LOCAL_RUNS, fd, "preds_test.csv"))
    th = json.load(open(os.path.join(LOCAL_RUNS, fd, "thresholds.json")))
    for m in th:
        if m not in q.columns: continue
        pm = q[m].clip(1e-6, 1 - 1e-6)
        t  = min(max(th[m], 1e-6), 1 - 1e-6)
        q[m] = expit(logit(pm) - logit(t))   # shift so threshold = 0.5
    parts.append(q)
P = pd.concat(parts, ignore_index=True)

# Sample-level aggregation
def agg(col):
    ps, ys = wb.aggregate_by_group(P[col].values, P.y.values, P.group.values)
    return ps, ys

fig, axes = plt.subplots(2, 4, figsize=(18, 9))

for mi, m in enumerate(METHODS):
    if m not in P.columns:
        print(f"⚠ Method '{m}' not in pooled predictions — skip"); continue
    ps, ys = agg(m)
    label = m[:30]

    # ROC
    fpr, tpr, _ = roc_curve(ys, ps)
    roc_auc = auc(fpr, tpr)
    axes[0][mi].plot(fpr, tpr, lw=2, label=f"AUC={roc_auc:.3f}")
    axes[0][mi].plot([0,1],[0,1],"k--",lw=0.8)
    axes[0][mi].set_xlabel("FPR"); axes[0][mi].set_ylabel("TPR")
    axes[0][mi].set_title(f"ROC: {label}", fontsize=8)
    axes[0][mi].legend(fontsize=7)

    # PR
    prec, rec, _ = precision_recall_curve(ys, ps)
    pr_auc = auc(rec, prec)
    axes[1][mi].plot(rec, prec, lw=2, label=f"AP={pr_auc:.3f}")
    axes[1][mi].set_xlabel("Recall"); axes[1][mi].set_ylabel("Precision")
    axes[1][mi].set_title(f"PR: {label}", fontsize=8)
    axes[1][mi].legend(fontsize=7)

plt.tight_layout()
fn = os.path.join(FIGURES_DIR, "roc_pr_curves.png")
plt.savefig(fn, dpi=120, bbox_inches="tight"); plt.show(); print(f"Saved → {fn}")

# Calibration + confusion matrices
fig2, axes2 = plt.subplots(2, 4, figsize=(18, 9))
for mi, m in enumerate(METHODS):
    if m not in P.columns: continue
    ps, ys = agg(m)
    bins = np.linspace(0, 1, 11)
    bin_ids = np.digitize(ps, bins) - 1
    mean_conf, mean_acc = [], []
    for b in range(10):
        mask = bin_ids == b
        if mask.sum() > 0:
            mean_conf.append(ps[mask].mean()); mean_acc.append(ys[mask].mean())
    axes2[0][mi].plot([0,1],[0,1],"k--",lw=0.8,label="perfect")
    axes2[0][mi].scatter(mean_conf, mean_acc, s=40, c="steelblue")
    axes2[0][mi].set_xlabel("Mean confidence"); axes2[0][mi].set_ylabel("Fraction positive")
    axes2[0][mi].set_title(f"Calibration: {m[:28]}", fontsize=7)
    axes2[0][mi].legend(fontsize=6)

    cm = confusion_matrix(ys, (ps >= 0.5).astype(int))
    axes2[1][mi].imshow(cm, cmap="Blues", vmin=0)
    for r in range(2):
        for c in range(2):
            axes2[1][mi].text(c, r, cm[r,c], ha="center", va="center", fontsize=12, color="black")
    axes2[1][mi].set_xticks([0,1]); axes2[1][mi].set_yticks([0,1])
    axes2[1][mi].set_xticklabels(["Real","Fake"]); axes2[1][mi].set_yticklabels(["Real","Fake"])
    axes2[1][mi].set_xlabel("Predicted"); axes2[1][mi].set_ylabel("True")
    axes2[1][mi].set_title(f"Confusion: {m[:28]}", fontsize=7)

plt.tight_layout()
fn2 = os.path.join(FIGURES_DIR, "calibration_confusion.png")
plt.savefig(fn2, dpi=120, bbox_inches="tight"); plt.show(); print(f"Saved → {fn2}")
print("§13 complete ✅")
"""),
    ]


def section_decoder_params() -> list[dict]:
    return [
        md("""\
## § 14 · Tuned Decoder Parameters

Wombat-OA finds 13 parameters per fold (head weights `a_cnn/gnn/hafm`, temperatures `τ`, view reliability `β`,
diversity penalty `γ`, length-normalisation `α`, beam width `B`, depth `L`, sharpness `κ`, threshold `θ`).
High variance across folds means the optimiser is overfitting a small val set — the "untuned" rows in the ablation table provide a sanity check.
"""),
        code("""\
# § 14 — Tuned Wombat-OA decoder parameters per fold
import os, json
import numpy as np, pandas as pd
from IPython.display import display

COLAB_ROOT  = globals().get("COLAB_ROOT", "/content/saffron-authenticity")
LOCAL_RUNS  = globals().get("LOCAL_RUNS", os.path.join(COLAB_ROOT, "runs", "cv1"))
FIGURES_DIR = globals().get("FIGURES_DIR", os.path.join(LOCAL_RUNS, "figures"))
os.makedirs(FIGURES_DIR, exist_ok=True)

folds = sorted(d for d in os.listdir(LOCAL_RUNS)
               if d.startswith("fold") and os.path.exists(os.path.join(LOCAL_RUNS, d, "wombat_params.json")))
if not folds:
    print("wombat_params.json not found — run §9 first."); raise SystemExit()

rows = {}
for fd in folds:
    p = json.load(open(os.path.join(LOCAL_RUNS, fd, "wombat_params.json")))
    flat = {}
    for k, v in p.items():
        if hasattr(v, "__iter__") and not isinstance(v, str):
            for i, vi in enumerate(v):
                flat[f"{k}[{i}]"] = round(float(vi), 4)
        elif isinstance(v, (int, float)):
            flat[k] = round(float(v), 4)
    rows[fd] = flat

df = pd.DataFrame(rows).T
print("Tuned Wombat-OA parameters per fold:")
display(df.style.highlight_max(axis=0, color="lightgreen").highlight_min(axis=0, color="lightyellow")
        .format("{:.4f}"))

print(f"\\nParameter std across folds (stability check):")
print(df.std(ddof=0).round(4).to_string())
print("\\nHigh std → optimiser overfits the small val set. Use 'untuned' row as sanity check.")
print("§14 complete ✅")
"""),
    ]


def section_uncertainty() -> list[dict]:
    return [
        md("""\
## § 15 · Uncertainty Check

Tests whether MC-Dropout uncertainty (std of logits over T=16 samples) predicts misclassification.
- **AUROC ≈ 0.5**: uncertainty is uninformative — the MC-Dropout claim is unsupported on this data.
- **AUROC > 0.6**: uncertainty is somewhat predictive — useful for risk-coverage curves.

Also shows a risk-coverage curve at the sample level: what accuracy you get if you refer the most uncertain X% of samples to a human expert.
"""),
        code("""\
# § 15 — Uncertainty check: AUROC of (MC std predicts misclassification) + risk-coverage
import os, sys
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

COLAB_ROOT  = globals().get("COLAB_ROOT", "/content/saffron-authenticity")
LOCAL_RUNS  = globals().get("LOCAL_RUNS", os.path.join(COLAB_ROOT, "runs", "cv1"))
FIGURES_DIR = globals().get("FIGURES_DIR", os.path.join(LOCAL_RUNS, "figures"))
sys.path.insert(0, os.path.join(COLAB_ROOT, "src"))
os.makedirs(FIGURES_DIR, exist_ok=True)
import wombat_beam as wb

folds = sorted(d for d in os.listdir(LOCAL_RUNS)
               if d.startswith("fold") and os.path.exists(os.path.join(LOCAL_RUNS, d, "cache_test.npz")))
if not folds:
    print("cache_test.npz not found — run §9 first."); raise SystemExit()

aucs_unc = []
ps_all, ys_all, unc_all = [], [], []
for fd in folds:
    c = wb.Cache.load(os.path.join(LOCAL_RUNS, fd, "cache_test.npz"))
    N, V, H = c.shape
    # mean uncertainty across views and heads (hafm only = head index 2)
    unc = c.U.mean(axis=(1,2))   # (N,) — mean MC std
    # default decode for probability
    from wombat_beam import default_params, unpack, decode
    p_obj = unpack(default_params(H, V), H, V)
    prob = decode(c, p_obj)
    misclf = (((prob >= 0.5).astype(int)) != c.y).astype(int)
    if len(np.unique(misclf)) > 1:
        auc_unc = roc_auc_score(misclf, unc)
    else:
        auc_unc = float("nan")
    aucs_unc.append(auc_unc)
    ps_all.append(prob); ys_all.append(c.y); unc_all.append(unc)

print(f"AUROC(uncertainty → misclassification) per fold: {[round(a,3) for a in aucs_unc]}")
print(f"Mean: {np.nanmean(aucs_unc):.3f}  |  ~0.5 means uncertainty is uninformative.")

# Risk-coverage curve (pooled)
ps_pool = np.concatenate(ps_all); ys_pool = np.concatenate(ys_all); unc_pool = np.concatenate(unc_all)
# Sample level
all_unc = unc_pool
order = np.argsort(all_unc)   # most certain first
accs, coverages = [], []
for frac in np.linspace(0.1, 1.0, 20):
    n_covered = max(1, int(frac * len(order)))
    idx = order[:n_covered]
    pred = (ps_pool[idx] >= 0.5).astype(int)
    acc = (pred == ys_pool[idx]).mean()
    accs.append(acc); coverages.append(frac)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].bar(range(len(folds)), aucs_unc, color="steelblue")
axes[0].axhline(0.5, ls="--", c="grey", label="chance"); axes[0].legend()
axes[0].set_xticks(range(len(folds))); axes[0].set_xticklabels(folds)
axes[0].set_ylabel("AUROC"); axes[0].set_title("Uncertainty → misclassification AUROC per fold")

axes[1].plot(coverages, accs, "o-", color="steelblue")
axes[1].set_xlabel("Coverage (fraction of samples kept)"); axes[1].set_ylabel("Accuracy")
axes[1].set_title("Risk-coverage curve (most certain first)")
axes[1].set_ylim(0, 1.05); axes[1].invert_xaxis()

plt.tight_layout()
fn = os.path.join(FIGURES_DIR, "uncertainty.png")
plt.savefig(fn, dpi=120, bbox_inches="tight"); plt.show(); print(f"Saved → {fn}")
print("§15 complete ✅")
"""),
    ]


def section_per_type() -> list[dict]:
    return [
        md("""\
## § 16 · Per-Type Breakdown (fake samples only)

Fake saffron in this dataset is labelled by type (`-`, `C`, `S`, `M` from the filename).
The meaning of C/S/M is **not documented in the dataset files** — confirm before drawing conclusions.
This breakdown shows whether the model struggles more with one fake type than another.
"""),
        code("""\
# § 16 — Per-type breakdown for fake samples (types -, C, S, M) at sample level
import os, sys, json
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.special import expit, logit
from IPython.display import display

COLAB_ROOT  = globals().get("COLAB_ROOT", "/content/saffron-authenticity")
LOCAL_DATA  = globals().get("LOCAL_DATA", os.path.join(COLAB_ROOT, "data"))
LOCAL_RUNS  = globals().get("LOCAL_RUNS", os.path.join(COLAB_ROOT, "runs", "cv1"))
FIGURES_DIR = globals().get("FIGURES_DIR", os.path.join(LOCAL_RUNS, "figures"))
sys.path.insert(0, os.path.join(COLAB_ROOT, "src"))
os.makedirs(FIGURES_DIR, exist_ok=True)
import wombat_beam as wb

PROPOSED = "PROPOSED: Wombat-OA + weighted beam + log-prob fusion"
BASELINE = "colour-statistics random forest (baseline)"

man = pd.read_csv(os.path.join(LOCAL_DATA, "manifest.csv"))
# sample → type (fake only)
samp_type = man[man.label==1].groupby("group")["type"].first()

folds = sorted(d for d in os.listdir(LOCAL_RUNS)
               if d.startswith("fold") and os.path.exists(os.path.join(LOCAL_RUNS, d, "preds_test.csv")))
if not folds:
    print("preds_test.csv not found — run §9 first."); raise SystemExit()

parts = []
for fd in folds:
    q  = pd.read_csv(os.path.join(LOCAL_RUNS, fd, "preds_test.csv"))
    th = json.load(open(os.path.join(LOCAL_RUNS, fd, "thresholds.json")))
    for m in th:
        if m not in q.columns: continue
        pm = q[m].clip(1e-6, 1 - 1e-6); t = min(max(th[m],1e-6),1-1e-6)
        q[m] = expit(logit(pm) - logit(t))
    parts.append(q)
P = pd.concat(parts, ignore_index=True)

# Sample-level aggregation
results = {}
for m in [PROPOSED, BASELINE]:
    if m not in P.columns: continue
    ps, ys = wb.aggregate_by_group(P[m].values, P.y.values, P.group.values)
    groups, _ = np.unique(P.group.values, return_inverse=True)
    # unique groups from agg
    u_groups = np.unique(P.group.values)
    # map group → type for fake samples
    type_map = samp_type.to_dict()
    results[m] = []
    for g, p_g, y_g in zip(u_groups, ps, ys):
        t = type_map.get(g, "real")
        results[m].append({"group":g,"p":p_g,"y":y_g,"type":t})

for m, rows in results.items():
    df = pd.DataFrame(rows)
    fake_df = df[df.y==1].copy()
    fake_df["correct"] = (fake_df.p >= 0.5).astype(int)
    print(f"\\n=== {m[:50]} — fake sample recall by type ===")
    print(fake_df.groupby("type")["correct"].agg(["sum","count","mean"]).rename(
        columns={"sum":"correct","count":"total","mean":"recall"}
    ).to_string())
    print("⚠ Type meanings (C/S/M) are unconfirmed — interpret with caution.")

print("§16 complete ✅")
"""),
    ]


def section_optimizer_study() -> list[dict]:
    return [
        md("""\
## § 17 · Optimiser Study: Wombat-OA vs Random Search

Runs both optimisers with 10 independent seeds on each fold's validation cache, then pools results.
Shows convergence curves and a final box plot, and reports the one-sided Wilcoxon signed-rank p-value
(H1: Wombat-OA reaches lower fitness than random search with the same budget).

**Note**: with small val caches (~11–13 samples), convergence curves may be noisy. The Wilcoxon test requires 5+ seeds.
"""),
        code("""\
# § 17 — Optimiser study: Wombat-OA vs Random Search, 10 seeds × each fold's val cache
import os, sys, json
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.stats import wilcoxon

COLAB_ROOT  = globals().get("COLAB_ROOT", "/content/saffron-authenticity")
LOCAL_RUNS  = globals().get("LOCAL_RUNS", os.path.join(COLAB_ROOT, "runs", "cv1"))
FIGURES_DIR = globals().get("FIGURES_DIR", os.path.join(LOCAL_RUNS, "figures"))
POP  = globals().get("POP", 20)
ITERS= globals().get("ITERS", 25)
sys.path.insert(0, os.path.join(COLAB_ROOT, "src"))
os.makedirs(FIGURES_DIR, exist_ok=True)
import wombat_beam as wb

N_OPT_SEEDS = 10

folds = sorted(d for d in os.listdir(LOCAL_RUNS)
               if d.startswith("fold") and os.path.exists(os.path.join(LOCAL_RUNS, d, "cache_val.npz")))
if not folds:
    print("cache_val.npz not found — run §9 first."); raise SystemExit()

# Cache result to Drive to avoid re-running
cmp_cache = os.path.join(LOCAL_RUNS, "optimizer_comparison.json")
if os.path.exists(cmp_cache):
    print(f"Loading cached optimiser comparison from {cmp_cache}")
    cmp = json.load(open(cmp_cache))
    for k in ("wombat","random"):
        cmp[k] = np.array(cmp[k])
    cmp["hist_wombat"] = [np.array(h) for h in cmp["hist_wombat"]]
    cmp["hist_random"] = [np.array(h) for h in cmp["hist_random"]]
else:
    print(f"Running optimiser comparison: {N_OPT_SEEDS} seeds × {len(folds)} folds ...")
    all_fw, all_fr, all_hw, all_hr = [], [], [], []
    for fd in folds:
        val = wb.Cache.load(os.path.join(LOCAL_RUNS, fd, "cache_val.npz"))
        print(f"  {fd}: val shape {val.shape}", flush=True)
        res = wb.compare_optimizers(val, seeds=range(N_OPT_SEEDS), pop=POP, iters=ITERS)
        all_fw.extend(res["wombat"].tolist()); all_fr.extend(res["random"].tolist())
        all_hw.extend(res["hist_wombat"]); all_hr.extend(res["hist_random"])
    cmp = {"wombat": np.array(all_fw), "random": np.array(all_fr),
           "hist_wombat": all_hw, "hist_random": all_hr}
    cmp["wilcoxon_p"] = float(wilcoxon(cmp["wombat"], cmp["random"], alternative="less").pvalue) \
        if len(cmp["wombat"]) >= 5 and not np.allclose(cmp["wombat"], cmp["random"]) else float("nan")
    # Save
    save = {"wombat": cmp["wombat"].tolist(), "random": cmp["random"].tolist(),
            "hist_wombat": [h.tolist() for h in cmp["hist_wombat"]],
            "hist_random": [h.tolist() for h in cmp["hist_random"]],
            "wilcoxon_p": cmp["wilcoxon_p"]}
    json.dump(save, open(cmp_cache, "w"))
    print(f"Saved → {cmp_cache}")

print(f"\\nWombat-OA: {np.mean(cmp['wombat']):.4f} ± {np.std(cmp['wombat']):.4f}")
print(f"Random:    {np.mean(cmp['random']):.4f} ± {np.std(cmp['random']):.4f}")
print(f"Wilcoxon p (H1: Wombat < Random): {cmp.get('wilcoxon_p', float('nan')):.4g}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for h in cmp["hist_wombat"]: axes[0].plot(np.linspace(0,1,len(h)), h, c="tab:green", alpha=0.3)
for h in cmp["hist_random"]:  axes[0].plot(np.linspace(0,1,len(h)), h, c="tab:gray",  alpha=0.3)
axes[0].set_title("Convergence (green=Wombat-OA, gray=Random)")
axes[0].set_xlabel("Fraction of budget"); axes[0].set_ylabel("Best val fitness")
axes[1].boxplot([cmp["wombat"], cmp["random"]], tick_labels=["Wombat-OA","Random"])
axes[1].set_title(f"Final fitness ({N_OPT_SEEDS}×{len(folds)} seeds)")
plt.tight_layout()
fn = os.path.join(FIGURES_DIR, "optimizer_study.png")
plt.savefig(fn, dpi=120, bbox_inches="tight"); plt.show(); print(f"Saved → {fn}")
print("§17 complete ✅")
"""),
    ]


def section_dashboard() -> list[dict]:
    return [
        md("## § 18 · Headline Dashboard"),
        code("""\
# § 18 — Headline dashboard: best method, vs baseline, honest auto-generated summary
import os, sys
import numpy as np, pandas as pd
from IPython.display import display

COLAB_ROOT  = globals().get("COLAB_ROOT", "/content/saffron-authenticity")
LOCAL_RUNS  = globals().get("LOCAL_RUNS", os.path.join(COLAB_ROOT, "runs", "cv1"))
sys.path.insert(0, os.path.join(COLAB_ROOT, "src"))
import wombat_beam as wb

PROPOSED = "PROPOSED: Wombat-OA + weighted beam + log-prob fusion"
BASELINE = "colour-statistics random forest (baseline)"

mean_fp = os.path.join(LOCAL_RUNS, "summary_sample_mean.csv")
if not os.path.exists(mean_fp):
    print("summary_sample_mean.csv not found — run §10 first."); raise SystemExit()
mean_df = pd.read_csv(mean_fp, index_col=0)

folds = sorted(d for d in os.listdir(LOCAL_RUNS)
               if d.startswith("fold") and os.path.exists(os.path.join(LOCAL_RUNS, d, "ablation_sample_level.csv")))
if not folds:
    print("No fold ablation files — run §9 first."); raise SystemExit()
tabs = [pd.read_csv(os.path.join(LOCAL_RUNS, d, "ablation_sample_level.csv"), index_col=0) for d in folds]
cat = pd.concat(tabs)
std_df = cat.groupby(level=0, sort=False)[["auc","f1_fake","acc"]].std(ddof=0)

# Best method by mean AUC
best_row = mean_df["auc"].idxmax() if "auc" in mean_df.columns else None
prop_auc  = mean_df.loc[PROPOSED, "auc"]  if PROPOSED  in mean_df.index else float("nan")
base_auc  = mean_df.loc[BASELINE, "auc"]  if BASELINE  in mean_df.index else float("nan")
prop_f1   = mean_df.loc[PROPOSED, "f1_fake"] if PROPOSED in mean_df.index else float("nan")
base_f1   = mean_df.loc[BASELINE, "f1_fake"] if BASELINE in mean_df.index else float("nan")
prop_std  = std_df.loc[PROPOSED, "auc"]   if PROPOSED in std_df.index else float("nan")

print("=" * 60)
print("HEADLINE RESULTS (sample-level, 5-fold group CV)")
print("=" * 60)
print(f"Best method by mean AUC: {best_row}")
print(f"  Proposed mean AUC:     {prop_auc:.3f}  (std {prop_std:.3f})")
print(f"  Colour baseline AUC:   {base_auc:.3f}")
print(f"  Proposed mean F1(fake):{prop_f1:.3f}")
print(f"  Colour baseline F1:    {base_f1:.3f}")
print()

# Auto-generated honest summary
beats_auc = prop_auc > base_auc if not (np.isnan(prop_auc) or np.isnan(base_auc)) else None
beats_f1  = prop_f1 > base_f1  if not (np.isnan(prop_f1)  or np.isnan(base_f1))  else None
delta_auc = prop_auc - base_auc if beats_auc is not None else float("nan")
delta_f1  = prop_f1  - base_f1  if beats_f1  is not None else float("nan")

lines = []
if beats_auc is True:
    lines.append(f"✅ PROPOSED beats the colour baseline on mean sample-level AUC by {delta_auc:+.3f}.")
elif beats_auc is False:
    lines.append(f"⚠  PROPOSED does NOT beat the colour baseline on mean sample-level AUC ({delta_auc:+.3f}).")
else:
    lines.append("⚠  Could not compare proposed vs baseline (missing data).")

if beats_f1 is True:
    lines.append(f"✅ PROPOSED beats the colour baseline on mean F1(fake) by {delta_f1:+.3f}.")
elif beats_f1 is False:
    lines.append(f"⚠  PROPOSED does NOT beat the colour baseline on F1(fake) ({delta_f1:+.3f}).")

if not np.isnan(prop_std):
    if prop_std > 0.15:
        lines.append(f"⚠  High fold-to-fold std on AUC ({prop_std:.3f}) — results are noisy with only {len(folds)} folds of ~11–13 samples.")
    else:
        lines.append(f"ℹ  Fold std on AUC: {prop_std:.3f} (across {len(folds)} folds).")
lines.append("ℹ  59 independent samples is a small dataset — use mean ± std in the paper, not single folds.")

for line in lines:
    print(line)
print("=" * 60)
print("§18 complete ✅")
"""),
    ]


def section_package_results() -> list[dict]:
    return [
        md("## § 19 · Package Results"),
        code("""\
# § 19 — Zip csv/json/png results from runs/cv1 to Drive; offer files.download
import os, shutil, glob, time

COLAB_ROOT  = globals().get("COLAB_ROOT", "/content/saffron-authenticity")
LOCAL_RUNS  = globals().get("LOCAL_RUNS", os.path.join(COLAB_ROOT, "runs", "cv1"))
DRIVE_RUNS_DIR = globals().get("DRIVE_RUNS_DIR", "")

ts = time.strftime("%Y%m%d_%H%M")
zip_name = f"saffron_results_{ts}"
zip_path_base = os.path.join(COLAB_ROOT, zip_name)

# Collect only lightweight files (no weights)
tmp = os.path.join(COLAB_ROOT, "results_tmp"); os.makedirs(tmp, exist_ok=True)
for dp, dns, fns in os.walk(LOCAL_RUNS):
    for fn in fns:
        ext = os.path.splitext(fn)[1].lower()
        if ext not in (".csv", ".json", ".png"): continue
        src = os.path.join(dp, fn)
        rel = os.path.relpath(src, LOCAL_RUNS)
        dst = os.path.join(tmp, rel); os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)

zip_path = shutil.make_archive(zip_path_base, "zip", tmp)
print(f"Created results zip: {zip_path}")

# Copy to Drive
if DRIVE_RUNS_DIR:
    os.makedirs(DRIVE_RUNS_DIR, exist_ok=True)
    drive_zip = os.path.join(DRIVE_RUNS_DIR, os.path.basename(zip_path))
    shutil.copy2(zip_path, drive_zip)
    try: os.sync()
    except AttributeError: pass
    print(f"Copied to Drive: {drive_zip}")

# Offer download in Colab
try:
    from google.colab import files as _cfiles
    _cfiles.download(zip_path)
    print("Download triggered ✅")
except (ModuleNotFoundError, Exception) as e:
    print(f"Colab download not available ({e}). Find the zip at: {zip_path}")

shutil.rmtree(tmp, ignore_errors=True)
print("§19 complete ✅")
"""),
    ]


def section_interpretation() -> list[dict]:
    return [md("""\
## § 20 · Interpretation Guide

### How to read the tables
- **Primary table**: sample-level, group-CV (§10 summary, §18 dashboard). One decision per physical sample (all shots fused).
- **Secondary**: image-level table (less meaningful because the same sample appears multiple times).
- **Pooled OOF** (§13): every sample is tested exactly once — the most reliable aggregate with only 11–13 samples per fold.
- Compare **every deep-learning row** against the `colour-statistics random forest (baseline)` row.
  A deep model below this baseline is not learning anything beyond colour.

### Red flags
| Symptom | Likely cause |
|---------|-------------|
| Deep model far below colour baseline | ImageNet weights not loaded; too few epochs; overfitting |
| Train AUC ~1.0, val AUC low | Overfitting; try more dropout, fewer epochs, larger L2 |
| Huge fold variance (std > 0.15) | Expected with 11–13 samples/fold; report mean ± std, not single folds |
| 100% accuracy on any fold | Dataset leakage or label bug; re-check manifest |
| Uncertainty AUROC ≈ 0.5 | MC-Dropout claim unsupported; don't claim calibrated uncertainty |

### Honest limits (from README § 6)
- 59 independent samples is small. Tuning 13 decoder parameters on ~12 val samples is noisy.
- The meaning of C / S / M fake types is not documented — confirm before type-level analysis.
- For two classes, log-probability fusion equals weighted logit averaging. Gains must come from tuned weights, diversity-aware beam, or the optimiser — the ablation rows isolate each.
- Do **not** report results on the original Training/Testing split (leaked, §5 reports this).
""")]


def section_troubleshooting() -> list[dict]:
    return [md("""\
## § 21 · Troubleshooting

| Problem | Solution |
|---------|---------|
| **No GPU** | Runtime → Change runtime type → T4 GPU. If unavailable, set `ALLOW_CPU=True` in §2 (very slow). |
| **OOM during training** | Set `BATCH=8` and/or `IMG_SIZE=224` in §2. |
| **Drive zips not found** | Check `DRIVE_DATA_DIR` in §2 points to the folder with the two `.zip` files. |
| **Keras/TF version mismatch** | §1 installs the needed versions automatically. **Restart the runtime once** after the install message, then re-run from §1. |
| **Session disconnect** | Re-run §1–§3 (environment, config, drive), then §8 (resume: rsyncs Drive → local). §9 skips finished folds. |
| **`prepare_dataset.py` fails** | Check that the zip files contain `Training/{Fake,Real}` and `Testing/{Fake,Real}` subdirectories. |
| **"cache/manifest order mismatch"** | Delete `runs/cv1/fold*` and re-run §9. Manifest and caches must come from the same `prepare_dataset.py` run. |
| **Smoke test slow** | Expected — smoke uses CPU by default. On GPU it takes ~5–15 min. Numbers are not results. |
""")]


def section_params_reference() -> list[dict]:
    return [md("""\
## § 22 · Parameters Reference

| Variable | Default | Effect of changing |
|----------|---------|-------------------|
| `DRIVE_DATA_DIR` | `"/content/drive/MyDrive/saffron_data"` | Folder in Drive with the two dataset zips |
| `DRIVE_RUNS_DIR` | `"/content/drive/MyDrive/saffron_runs"` | Drive folder for checkpoint backup |
| `N_FOLDS` | `5` | Number of CV folds. Changing this invalidates cached caches. |
| `IMG_SIZE` | `256` | Input resolution. `384` gives better results but needs `BATCH=8`. |
| `BATCH` | `16` | Mini-batch size. Use `8` if GPU < 16 GB. |
| `EPOCHS_WARM` | `3` | Warm-up epochs (backbone frozen). Rarely needs changing. |
| `EPOCHS_FT` | `25` | Fine-tune epochs. Increase to `40` if val AUC is still rising. |
| `MC_SAMPLES` | `16` | MC-Dropout samples at cache time. `32` for better uncertainty estimates. |
| `POP` | `20` | Wombat-OA population size. |
| `ITERS` | `25` | Wombat-OA iterations. |
| `SEED` | `42` | Global random seed. |
| `SMOKE` | `False` | Pipeline test mode: fold 0 only, tiny settings. Numbers meaningless. |
| `ALLOW_CPU` | `False` | Allow running without GPU. Very slow. |
| `SAVE_WEIGHTS_TO_DRIVE` | `False` | Include `*.weights.h5` in Drive sync (~300 MB/fold). |
""")]


def section_loco() -> list[dict]:
    return [
        md("""\
## § 24 · Leave-One-Camera-Out (LOCO) Evaluation

Trains on ONE phone, validates on held-out samples of the same phone, and tests on the OTHER phone.
This is the strongest cross-device test available when both phones contain real and fake samples.

Roles encoded in the `fold` column so the standard runner is reused:
- `fold 0` = TEST (other camera)
- `fold 1` = VALIDATION (held-out groups of training camera)
- `fold 2` = TRAIN (rest of training camera)

> ⚠ Validation folds here are tiny (6–8 samples). Wombat-OA tuning is therefore noisy — always
> report results alongside the untuned-beam row.
"""),
        code("""\
# § 24 — Leave-one-camera-out evaluation (both cross-device directions)
import os, sys, subprocess, shutil
import pandas as pd
from IPython.display import display

COLAB_ROOT   = globals().get("COLAB_ROOT", "/content/saffron-authenticity")
LOCAL_DATA   = globals().get("LOCAL_DATA", os.path.join(COLAB_ROOT, "data"))
LOCAL_RUNS   = globals().get("LOCAL_RUNS", os.path.join(COLAB_ROOT, "runs", "cv1"))
DRIVE_RUNS_DIR = globals().get("DRIVE_RUNS_DIR", "")
IMG_SIZE     = globals().get("IMG_SIZE", 256)
POP          = globals().get("POP", 20)
ITERS        = globals().get("ITERS", 25)
SEED         = globals().get("SEED", 42)

def run(cmd, cwd=COLAB_ROOT, env_extra=None):
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "TF_CPP_MIN_LOG_LEVEL": "2"}
    if env_extra: env.update(env_extra)
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, bufsize=1, env=env)
    for line in proc.stdout:
        print(line, end="", flush=True)
    proc.wait()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)

# ── Step 1: generate LOCO manifests ─────────────────────────────────────────
manifest_path = os.path.join(LOCAL_DATA, "manifest.csv")
if not os.path.exists(manifest_path):
    print("manifest.csv not found — run §5 first."); raise SystemExit()

print("Generating LOCO manifests...")
run([sys.executable, "scripts/make_loco_manifests.py",
     "--manifest", manifest_path])

# Find generated manifests
import glob
loco_manifests = sorted(glob.glob(os.path.join(LOCAL_DATA, "manifest_loco_*_to_*.csv")))
print(f"Found {len(loco_manifests)} LOCO manifests: {[os.path.basename(m) for m in loco_manifests]}")

if not loco_manifests:
    print("No LOCO manifests generated — check that your data has >= 2 usable cameras."); raise SystemExit()

# ── Step 2: run cross-device test for each direction ─────────────────────────
loco_results = {}
for mf in loco_manifests:
    name = os.path.splitext(os.path.basename(mf))[0]   # manifest_loco_CAMA_to_CAMB
    parts = name.replace("manifest_loco_", "").split("_to_")
    out_name = f"loco_{parts[0]}_to_{parts[1]}"
    out_dir  = os.path.join(COLAB_ROOT, "runs", out_name)
    print(f"\\n{'='*60}")
    print(f"Running LOCO: {out_name}")
    print(f"{'='*60}")
    run([sys.executable, "scripts/run_cv.py",
         "--manifest", mf,
         "--out", out_dir,
         "--folds", "0",
         "--img-size", str(IMG_SIZE),
         "--pop", str(POP),
         "--iters", str(ITERS),
         "--seed", str(SEED)])
    summary_path = os.path.join(out_dir, "summary_sample.csv")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"LOCO summary was not generated: {summary_path}")
    if DRIVE_RUNS_DIR:
        drive_out = os.path.join(DRIVE_RUNS_DIR, out_name)
        os.makedirs(drive_out, exist_ok=True)
        shutil.copy2(summary_path, os.path.join(drive_out, "summary_sample.csv"))
        print(f"Synced LOCO summary to Drive: {drive_out}")
    loco_results[out_name] = out_dir

# ── Step 3: compare LOCO vs CV results ───────────────────────────────────────
print("\\n" + "="*60)
print("LOCO vs Standard CV — PROPOSED method, sample level")
print("="*60)
PROPOSED = "PROPOSED: Wombat-OA + weighted beam + log-prob fusion"
comparison = []
cv_summary = os.path.join(LOCAL_RUNS, "summary_sample.csv")
if not os.path.exists(cv_summary):
    raise FileNotFoundError(f"Standard CV summary was not found: {cv_summary}")
cv = pd.read_csv(cv_summary, index_col=0)
if PROPOSED not in cv.index:
    raise KeyError(f"PROPOSED method is missing from {cv_summary}")
cv_row = cv.loc[PROPOSED]
comparison.append({
    "Evaluation": "Standard CV",
    "AUC": cv_row.get("auc", "—"),
    "F1 (fake)": cv_row.get("f1_fake", "—"),
    "Accuracy": cv_row.get("acc", "—"),
})
for out_name, out_dir in loco_results.items():
    summary_path = os.path.join(out_dir, "summary_sample.csv")
    df = pd.read_csv(summary_path, index_col=0)
    if PROPOSED not in df.index:
        raise KeyError(f"PROPOSED method is missing from {summary_path}")
    row = df.loc[PROPOSED]
    comparison.append({
        "Evaluation": out_name,
        "AUC": row.get("auc", "—"),
        "F1 (fake)": row.get("f1_fake", "—"),
        "Accuracy": row.get("acc", "—"),
    })

display(pd.DataFrame(comparison).set_index("Evaluation"))

print("\\n§24 complete ✅")
"""),
    ]


def section_stress_test() -> list[dict]:
    return [
        md("""\
## § 25 · Stress Test — Robustness to Image Degradation

Re-evaluates the SAVED fold models on 8 types of degraded test images **without retraining or retuning**.
This reveals:
- What features the network relies on (colour? texture? resolution?)
- Whether Wombat-OA beam decoding and log-probability fusion are more robust than a plain average

Perturbations applied to TEST images only:
| Name | Transform |
|------|-----------|
| `clean` | No change |
| `blur` | Gaussian blur σ=2 |
| `jpeg` | JPEG quality 20 |
| `noise` | Additive Gaussian noise std=12 |
| `lowres` | 4× downsample then upsample |
| `gray` | Grayscale (colour removed) |
| `hue` | Hue rotation 0.08 |
| `dark` | 40% darkening |

> ⚠ Needs `final.weights.h5` on disk (i.e. Colab session still alive, or `SAVE_WEIGHTS_TO_DRIVE=True`).
"""),
        code("""\
# § 25 — Stress test: robustness of all methods to image degradation
import os, sys, subprocess
import pandas as pd
import matplotlib.pyplot as plt

COLAB_ROOT  = globals().get("COLAB_ROOT", "/content/saffron-authenticity")
LOCAL_DATA  = globals().get("LOCAL_DATA", os.path.join(COLAB_ROOT, "data"))
LOCAL_RUNS  = globals().get("LOCAL_RUNS", os.path.join(COLAB_ROOT, "runs", "cv1"))
FIGURES_DIR = globals().get("FIGURES_DIR", os.path.join(LOCAL_RUNS, "figures"))
os.makedirs(FIGURES_DIR, exist_ok=True)

manifest_path = os.path.join(LOCAL_DATA, "manifest.csv")
if not os.path.exists(manifest_path):
    print("manifest.csv not found — run §5 first."); raise SystemExit()

# Check at least one fold has weights
have_weights = any(
    os.path.exists(os.path.join(LOCAL_RUNS, f"fold{k}", "final.weights.h5")) or
    os.path.exists(os.path.join(LOCAL_RUNS, f"fold{k}", "best.weights.h5"))
    for k in range(5)
)
if not have_weights:
    print("⚠ No saved weights found in runs/cv1/fold*/. Weights are only kept if:")
    print("  (a) the Colab session is still alive (in-memory), or")
    print("  (b) you set SAVE_WEIGHTS_TO_DRIVE=True in §2 before training.")
    print("Skipping stress test.")
    raise SystemExit()

def run(cmd, cwd=COLAB_ROOT, env_extra=None):
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "TF_CPP_MIN_LOG_LEVEL": "2"}
    if env_extra: env.update(env_extra)
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, bufsize=1, env=env)
    for line in proc.stdout:
        print(line, end="", flush=True)
    proc.wait()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)

print("Running stress test (all 8 perturbations) ...")
run([sys.executable, "scripts/stress_test.py",
     "--manifest", manifest_path,
     "--run", LOCAL_RUNS])

# ── Display results ──────────────────────────────────────────────────────────
results_csv = os.path.join(LOCAL_RUNS, "stress_results.csv")
if not os.path.exists(results_csv):
    print("stress_results.csv not found."); raise SystemExit()

from IPython.display import display
R = pd.read_csv(results_csv)
PERTURBS = ["clean","blur","jpeg","noise","lowres","gray","hue","dark"]

for level in ("sample", "image"):
    for metric in ("auc", "acc"):
        T = R[R.level == level].pivot(index="method", columns="perturbation", values=metric)
        T = T[[c for c in PERTURBS if c in T.columns]]
        print(f"\\n=== {metric.upper()} — {level.upper()} level ===")
        display(T.round(3).style
                .background_gradient(cmap="RdYlGn", vmin=0, vmax=1)
                .format("{:.3f}"))

# ── Heatmap figure ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, max(4, len(R.method.unique()) * 0.45)))
for ai, metric in enumerate(("auc", "acc")):
    T = R[R.level == "sample"].pivot(index="method", columns="perturbation", values=metric)
    T = T[[c for c in PERTURBS if c in T.columns]]
    im = axes[ai].imshow(T.values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    axes[ai].set_xticks(range(len(T.columns)))
    axes[ai].set_xticklabels(T.columns, rotation=45, ha="right", fontsize=8)
    axes[ai].set_yticks(range(len(T.index)))
    axes[ai].set_yticklabels([m[:40] for m in T.index], fontsize=7)
    for r in range(T.shape[0]):
        for c in range(T.shape[1]):
            axes[ai].text(c, r, f"{T.values[r,c]:.2f}",
                          ha="center", va="center", fontsize=6,
                          color="black" if T.values[r,c] > 0.5 else "white")
    axes[ai].set_title(f"{metric.upper()} by perturbation (sample level)")
    plt.colorbar(im, ax=axes[ai], shrink=0.8)

plt.tight_layout()
fn = os.path.join(FIGURES_DIR, "stress_test_heatmap.png")
plt.savefig(fn, dpi=120, bbox_inches="tight"); plt.show()
print(f"Saved → {fn}")
print("§25 complete ✅")
"""),
    ]


def section_what_next() -> list[dict]:
    return [md("""\
## § 26 · What to Do Next

1. **Copy numbers**: take `runs/cv1/summary_sample.csv` mean ± std rows into your paper table.
   Use the sample-level table (one decision per physical sample), not the image-level table.
2. **Run ≥ 3 training seeds**: change `SEED` in §2 and re-run §5–§9 with a fresh `DRIVE_RUNS_DIR`.
   Report mean ± std over seeds × folds.
3. **Compare against colour baseline**: if the proposed method does not beat the colour baseline
   on AUC and F1(fake), investigate (more epochs, larger `IMG_SIZE`, check ImageNet weights loaded).
4. **Confirm fake-type meanings**: identify what C, S, M mean in your dataset before publishing §16 results.
5. **Ablation table interpretation**: the beam, diversity penalty, and Wombat-OA rows exist to isolate
   each contribution — cite the ablation rows, not just the final proposed row.
6. **LOCO results (§24)**: report cross-device AUC for both directions. The G973F→J600G direction
   is harder (unseen adulterant types in the test camera).
7. **Stress test (§25)**: if accuracy drops to ~50% under grayscale, the model relies on colour —
   discuss this as a limitation (or strength, if colour is a valid discriminator for your application).
""")]


# ---------------------------------------------------------------------------
# Build the full cell list
# ---------------------------------------------------------------------------
def build_cells() -> list[dict]:
    cells: list[dict] = []
    cells += section_title()
    cells += section_gpu_check()
    cells += section_config()
    cells += section_drive_mount()
    cells += section_writefile_intro()
    cells += section_writefiles()
    cells += section_data()
    cells += section_dataset_audit()
    cells += section_colour_baseline()
    cells += section_smoke()
    cells += section_resume()
    cells += section_training()
    cells += section_aggregate()
    cells += section_training_curves()
    cells += section_perfold_results()
    cells += section_pooled_analysis()
    cells += section_decoder_params()
    cells += section_uncertainty()
    cells += section_per_type()
    cells += section_optimizer_study()
    cells += section_dashboard()
    cells += section_package_results()
    cells += section_interpretation()
    cells += section_troubleshooting()
    cells += section_params_reference()
    cells += section_loco()
    cells += section_stress_test()
    cells += section_what_next()
    return cells


def build_notebook() -> nbformat.NotebookNode:
    cells = build_cells()
    nb = nbformat.v4.new_notebook()
    nb.cells = [nbformat.v4.new_markdown_cell(c["source"])
                if c["cell_type"] == "markdown"
                else nbformat.v4.new_code_cell(c["source"])
                for c in cells]
    # Assign unique ids
    seen: set[str] = set()
    for cell, orig in zip(nb.cells, cells):
        cid = orig["id"]
        while cid in seen:
            cid = uid()
        cell["id"] = cid
        seen.add(cid)
        if orig["cell_type"] == "code":
            cell["metadata"] = orig.get("metadata", {})

    nb.metadata.update({
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
        "gpuClass": "standard",
        "colab": {"provenance": [], "gpuType": "T4"},
    })
    return nb


# ---------------------------------------------------------------------------
# Check mode: re-read sources and verify SHA-256
# ---------------------------------------------------------------------------
def check_notebook(nb_path: str, ast_check: bool = False) -> bool:
    with open(nb_path, encoding="utf-8") as notebook_file:
        nb = nbformat.read(notebook_file, as_version=4)

    # Build expected {dst_rel: sha256_of_content}
    expected: dict[str, str] = {}
    for dst_rel, src_rel in EMBED_FILES:
        content = _load(src_rel)
        expected[dst_rel] = sha256(content)

    # Find %%writefile cells and extract their content
    embedded: dict[str, str] = {}
    errors: list[str] = []
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        src = cell.source
        if not src.startswith("%%writefile"):
            # AST check non-writefile code cells
            if ast_check:
                # Strip ! and % magic lines for AST parsing
                clean = "\n".join(
                    ln for ln in src.splitlines()
                    if not ln.startswith("!") and not ln.startswith("%")
                )
                try:
                    ast.parse(clean)
                except SyntaxError as e:
                    errors.append(f"SyntaxError in cell {cell.id}: {e}")
            continue
        first_line, _, rest = src.partition("\n")
        dst_abs = first_line.split(None, 1)[1].strip()
        if not dst_abs.startswith(COLAB_ROOT):
            continue
        dst_rel = os.path.relpath(dst_abs, COLAB_ROOT).replace("\\", "/")
        embedded[dst_rel] = sha256(rest)

    ok = True
    for dst_rel in expected:
        if dst_rel not in embedded:
            print(f"[MISSING] {dst_rel} not found in notebook"); ok = False
        elif embedded[dst_rel] != expected[dst_rel]:
            print(f"[MISMATCH] {dst_rel}: SHA-256 differs"); ok = False
        else:
            print(f"[OK] {dst_rel}")

    for e in errors:
        print(e); ok = False

    # Check unique ids and no outputs
    ids = [c["id"] for c in nb.cells]
    if len(ids) != len(set(ids)):
        print("[FAIL] duplicate cell ids"); ok = False
    for c in nb.cells:
        if c.cell_type == "code" and c.get("outputs", []):
            print(f"[FAIL] cell {c.id} has outputs"); ok = False

    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="verify notebook SHA-256 against source files")
    ap.add_argument("--ast-check", action="store_true",
                    help="additionally AST-compile every code cell")
    ap.add_argument("--out", default=OUT, help="output notebook path")
    args = ap.parse_args()

    if args.check or args.ast_check:
        nb_path = args.out
        if not os.path.exists(nb_path):
            print(f"Notebook not found: {nb_path}"); sys.exit(1)
        ok = check_notebook(nb_path, ast_check=args.ast_check)
        if ok:
            print("\n✅ All checks passed.")
        else:
            print("\n❌ Check failed — re-run without --check to regenerate.")
            sys.exit(1)
        return

    print(f"Building notebook -> {args.out}")
    nb = build_notebook()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    nbformat.write(nb, open(args.out, "w", encoding="utf-8"))
    n_cells = len(nb.cells)
    n_code = sum(1 for c in nb.cells if c.cell_type == "code")
    n_md   = sum(1 for c in nb.cells if c.cell_type == "markdown")
    size_kb = os.path.getsize(args.out) / 1024
    print(f"Written {args.out}")
    print(f"  {n_cells} cells ({n_code} code, {n_md} markdown)  |  {size_kb:.0f} KB")
    print(f"Run: python colab/build_colab_notebook.py --check  to verify SHA-256")


if __name__ == "__main__":
    main()
