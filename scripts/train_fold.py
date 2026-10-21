"""Train ONE fold in its own process (keeps TensorFlow memory from piling up across folds). Called by run_cv.py."""
import _path  # noqa
import argparse
import pandas as pd
import saffron_cnn_gnn as sg
from manifest_paths import normalize_manifest_paths

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", default="data/manifest.csv"); ap.add_argument("--out", required=True); ap.add_argument("--fold", type=int, required=True)
ap.add_argument("--img-size", type=int, default=256); ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--epochs-warm", type=int, default=3); ap.add_argument("--epochs-ft", type=int, default=25)
ap.add_argument("--mc-samples", type=int, default=16); ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--no-imagenet", action="store_true")
ap.add_argument("--capture-invariant", action="store_true")
a = ap.parse_args()
man = normalize_manifest_paths(pd.read_csv(a.manifest)); K = int(man.fold.max()) + 1
cfg = sg.Config(img_size=a.img_size, batch=a.batch, epochs_warm=a.epochs_warm, epochs_ft=a.epochs_ft,
                mc_samples=a.mc_samples, seed=a.seed, imagenet=not a.no_imagenet,
                capture_invariant=a.capture_invariant)
sg.run_fold(man, a.fold, K, cfg, a.out)
print("FOLD", a.fold, "DONE")
