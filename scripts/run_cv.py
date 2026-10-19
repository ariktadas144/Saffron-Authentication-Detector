"""Sample-level K-fold cross-validation:  train -> cache -> Wombat-OA beam decoding -> ablation, for every fold,
then aggregate (mean +- std over folds, plus pooled out-of-fold results).

   python scripts/run_cv.py --manifest data/manifest.csv --out runs/cv1                 # all folds
   python scripts/run_cv.py --manifest data/manifest.csv --out runs/cv1 --folds 0       # just fold 0 (quick check)
   python scripts/run_cv.py --manifest data/manifest.csv --out runs/cv1 --aggregate-only
Re-running resumes: folds whose caches already exist are not retrained (use --retrain to force).
"""
import _path  # noqa
import argparse, os, sys, json, subprocess
import numpy as np, pandas as pd
import wombat_beam as wb
import color_baseline as cb

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", default="data/manifest.csv"); ap.add_argument("--out", default="runs/cv1")
ap.add_argument("--folds", type=int, nargs="*", default=None)
ap.add_argument("--img-size", type=int, default=256); ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--epochs-warm", type=int, default=3); ap.add_argument("--epochs-ft", type=int, default=25)
ap.add_argument("--mc-samples", type=int, default=16); ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--pop", type=int, default=20); ap.add_argument("--iters", type=int, default=25)
ap.add_argument("--no-imagenet", action="store_true"); ap.add_argument("--retrain", action="store_true")
ap.add_argument("--aggregate-only", action="store_true")
a = ap.parse_args()
KEYS = ["acc", "bal_acc", "f1_fake", "recall_fake", "specificity", "auc", "ece"]


def aggregate(out):
    folds = sorted(d for d in os.listdir(out) if d.startswith("fold") and os.path.exists(f"{out}/{d}/ablation_test.csv"))
    if not folds: print("no finished folds in", out); return
    print(f"\n================ aggregate over {len(folds)} fold(s): {folds} ================")
    for level, fname in [("IMAGE level", "ablation_test.csv"), ("SAMPLE level (all shots of a sample fused)", "ablation_sample_level.csv")]:
        tabs = [pd.read_csv(f"{out}/{d}/{fname}", index_col=0) for d in folds if os.path.exists(f"{out}/{d}/{fname}")]
        if not tabs: continue
        cat = pd.concat(tabs); g = cat.groupby(level=0, sort=False)[KEYS]
        mean, std = g.mean(), g.std(ddof=0)
        disp = mean.round(3).astype(str) + " ± " + std.round(3).astype(str)
        print(f"\n--- {level}: mean ± std over folds (test folds) ---"); print(disp.to_string())
        mean.to_csv(f"{out}/summary_{'image' if 'IMAGE' in level else 'sample'}_mean.csv"); disp.to_csv(f"{out}/summary_{'image' if 'IMAGE' in level else 'sample'}.csv")
    parts = []
    for d in folds:
        f = f"{out}/{d}/preds_test.csv"
        if not os.path.exists(f): continue
        q = pd.read_csv(f); th = json.load(open(f"{out}/{d}/thresholds.json"))
        for m in th:                                   # shift logits so each method's tuned threshold becomes 0.5
            pm = q[m].clip(1e-6, 1 - 1e-6); t = min(max(th[m], 1e-6), 1 - 1e-6)
            q[m] = 1 / (1 + np.exp(-(np.log(pm / (1 - pm)) - np.log(t / (1 - t)))))
        parts.append(q)
    P = pd.concat(parts)
    methods = [c for c in P.columns if c not in ("group", "y", "fold")]
    rows = {}
    for m in methods:
        ps, ys = wb.aggregate_by_group(P[m].values, P.y.values, P.group.values)
        rows[m] = wb.metrics(ys, ps, 0.5)
    print("\n--- pooled out-of-fold, SAMPLE level, threshold 0.5 (every sample is tested exactly once) ---")
    print(pd.DataFrame(rows).T[["acc", "bal_acc", "f1_fake", "recall_fake", "specificity", "auc"]].round(3).to_string())
    pd.DataFrame(rows).T.to_csv(f"{out}/pooled_oof_sample_level.csv")


if not a.aggregate_only:
    man = pd.read_csv(a.manifest); K = int(man.fold.max()) + 1
    os.makedirs(a.out, exist_ok=True)
    X = cb.features(man.path.tolist(), cache=f"{a.out}/colour_features.npz")
    for k in (a.folds if a.folds is not None else range(K)):
        od = f"{a.out}/fold{k}"; os.makedirs(od, exist_ok=True)
        if a.retrain or not (os.path.exists(f"{od}/cache_val.npz") and os.path.exists(f"{od}/cache_test.npz")):
            cmd = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_fold.py"),
                   "--manifest", a.manifest, "--out", od, "--fold", str(k), "--img-size", str(a.img_size), "--batch", str(a.batch),
                   "--epochs-warm", str(a.epochs_warm), "--epochs-ft", str(a.epochs_ft), "--mc-samples", str(a.mc_samples),
                   "--seed", str(a.seed)] + (["--no-imagenet"] if a.no_imagenet else [])
            subprocess.run(cmd, check=True)                       # fresh process per fold -> no TF memory build-up
        val, test = wb.Cache.load(f"{od}/cache_val.npz"), wb.Cache.load(f"{od}/cache_test.npz")
        kv = (k + 1) % K
        te = man[man.fold == k].reset_index(drop=True); tr = man[~man.fold.isin([k, kv])]
        p_col = cb.fit_predict(X[tr.index.values], tr.label.values, X[man.index[man.fold == k].values])
        assert (te.group.values == test.groups).all() and (te.label.values == test.y).all(), "cache/manifest order mismatch"
        table, info = wb.evaluate_all(val, test, pop=a.pop, iters=a.iters, seed=k,
                                      extra_rows={"colour-statistics random forest (baseline)": (p_col, 0.5)})
        table.to_csv(f"{od}/ablation_test.csv"); info["sample_table"].to_csv(f"{od}/ablation_sample_level.csv")
        pd.DataFrame({**{m: p for m, p in info["probs"].items()}, "group": test.groups, "y": test.y, "fold": k}).to_csv(f"{od}/preds_test.csv", index=False)
        json.dump(info["thr"], open(f"{od}/thresholds.json", "w"))
        json.dump({q: (v.tolist() if hasattr(v, "tolist") else v) for q, v in info["params"].items()}, open(f"{od}/wombat_params.json", "w"), indent=1)
        print(f"\n=== fold {k} (test: {len(te)} images, {te.group.nunique()} samples) ===")
        print(table[["acc", "f1_fake", "recall_fake", "specificity", "auc"]].round(3).to_string())
aggregate(a.out)
