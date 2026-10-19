"""Train the CNN+GNN+HAFM model, then cache MC-dropout outputs for val and test.
   python scripts/train.py --data data --out runs/run1
"""
import _path  # noqa
import argparse, os, json, dataclasses
import pandas as pd, keras
import saffron_cnn_gnn as sg

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data"); ap.add_argument("--out", default="runs/run1")
ap.add_argument("--img-size", type=int, default=256); ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--epochs-warm", type=int, default=3); ap.add_argument("--epochs-ft", type=int, default=25)
ap.add_argument("--mc-samples", type=int, default=16); ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--no-imagenet", action="store_true", help="random init (only for quick tests)")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
cfg = sg.Config(img_size=a.img_size, batch=a.batch, epochs_warm=a.epochs_warm, epochs_ft=a.epochs_ft,
                mc_samples=a.mc_samples, seed=a.seed, imagenet=not a.no_imagenet)
json.dump(dataclasses.asdict(cfg), open(f"{a.out}/config.json", "w"), indent=1)
keras.utils.set_random_seed(cfg.seed)
train_ds, val_ds, val_raw, test_raw = sg.make_datasets(a.data, cfg)
M = sg.build_models(cfg)
h1, h2 = sg.train(M, train_ds, val_ds, cfg, ckpt=f"{a.out}/best.weights.h5")
M["model"].save_weights(f"{a.out}/final.weights.h5")
pd.DataFrame(h2.history).to_csv(f"{a.out}/history_finetune.csv", index=False)
sg.extract_cache(M, val_raw, cfg).save(f"{a.out}/cache_val.npz")
sg.extract_cache(M, test_raw, cfg).save(f"{a.out}/cache_test.npz")
print("DONE ->", a.out)
