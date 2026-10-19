"""Leave-one-camera-out (LOCO) manifests: train on ONE phone, validate on held-out
samples of the SAME phone, test on the OTHER phone.

Why: inside each camera your cross-validation was already perfect, but that does
not show the model works on a device it has never seen. LOCO is the strongest
cross-device test your data allows (both phones contain real and fake samples).

   python scripts/make_loco_manifests.py --manifest data/manifest.csv
   python scripts/run_cv.py --manifest data/manifest_loco_<A>_to_<B>.csv --out runs/loco_<A>_to_<B> --folds 0 --img-size 256

Each output manifest re-uses the existing runner by encoding roles in the `fold` column:
   fold 0 = TEST (other camera) | fold 1 = VALIDATION (held-out groups of the training camera) | fold 2 = TRAIN (rest of the training camera)
Images with camera 'unknown' are excluded (they are all real and cannot form a two-class test).
Everything stays group-safe: a sample/group never appears in two roles.
"""
import _path  # noqa
import argparse, os, re
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", default="data/manifest.csv")
ap.add_argument("--out-dir", default=None)
ap.add_argument("--val-splits", type=int, default=4,
                help="1/N of the training camera's groups become validation")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--min-val-groups-per-class", type=int, default=2)
a = ap.parse_args()

man = pd.read_csv(a.manifest)
if "camera" not in man:
    raise SystemExit("manifest has no 'camera' column: re-run scripts/prepare_dataset.py "
                     "(it reads EXIF camera models)")

out_dir = a.out_dir or os.path.dirname(os.path.abspath(a.manifest))
man["camera"] = man["camera"].astype(str)
known = man[man.camera != "unknown"]
print(f"excluded {len(man) - len(known)} image(s) with unknown camera "
      f"(classes: { {k: int(v) for k, v in man[man.camera == 'unknown'].cls.value_counts().items()} })")

usable = [c for c, g in known.groupby("camera")
          if g.label.nunique() == 2 and g.group.nunique() >= 6]
print("cameras with both classes and >= 6 groups:", usable)
if len(usable) < 2:
    raise SystemExit("need at least two usable cameras for leave-one-camera-out")


def sanitize(s):
    return re.sub(r"[^A-Za-z0-9]+", "", s)


def split_val(tr):
    """Split training-camera data into train / validation, ensuring both classes
    have >= N groups in the validation part."""
    for seed in range(a.seed, a.seed + 50):
        sg = StratifiedGroupKFold(a.val_splits, shuffle=True, random_state=seed)
        i_tr, i_val = next(iter(sg.split(tr, tr.label, tr.group)))
        v, t = tr.iloc[i_val], tr.iloc[i_tr]
        gv = v.groupby("label").group.nunique()
        if (len(gv) == 2
                and gv.min() >= a.min_val_groups_per_class
                and t.label.nunique() == 2):
            return t, v, seed
    raise SystemExit("could not build a validation split with enough groups "
                     "of both classes")


for tr_cam in usable:
    for te_cam in usable:
        if tr_cam == te_cam:
            continue
        tr_all = known[known.camera == tr_cam]
        te = known[known.camera == te_cam]
        tr, va, seed_used = split_val(tr_all)
        full = pd.concat([
            te.assign(fold=0),
            va.assign(fold=1),
            tr.assign(fold=2),
        ], ignore_index=True)

        # safety check: no group appears in more than one role
        g_roles = full.groupby("group").fold.nunique()
        assert g_roles.max() == 1, "group appears in more than one role"

        name = f"manifest_loco_{sanitize(tr_cam)}_to_{sanitize(te_cam)}.csv"
        full.to_csv(os.path.join(out_dir, name), index=False)
        print(f"\n=== train on {tr_cam}  ->  test on {te_cam}   "
              f"(wrote {name}, val split seed {seed_used}) ===")
        for role, df in [("TRAIN", tr), ("VAL", va), ("TEST", te)]:
            ft = {k: int(v) for k, v in df[df.label == 1].type.value_counts().items()}
            print(f"  {role:5s} {len(df):4d} images / {df.group.nunique():3d} samples "
                  f"| real {int((df.label == 0).sum()):3d}  "
                  f"fake {int((df.label == 1).sum()):3d} | fake types {ft}")

        # warn about unseen fake types in the test camera
        seen = set(pd.concat([tr, va])[lambda d: d.label == 1].type)
        unseen = sorted(set(te[te.label == 1].type) - seen)
        if unseen:
            print(f"  [NOTE] fake type(s) {unseen} appear ONLY in the test "
                  f"camera: this direction mixes a device shift with an "
                  f"unseen-adulterant shift")

        print(f"  run: python scripts/run_cv.py "
              f"--manifest {os.path.join(out_dir, name)} "
              f"--out runs/loco_{sanitize(tr_cam)}_to_{sanitize(te_cam)} "
              f"--folds 0")
