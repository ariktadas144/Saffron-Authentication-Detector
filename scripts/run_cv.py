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
from audit_shortcuts import BASELINE_FEATURES, _classifier
from manifest_paths import normalize_manifest_paths

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", default="data/manifest.csv"); ap.add_argument("--out", default="runs/cv1")
ap.add_argument("--folds", type=int, nargs="*", default=None)
ap.add_argument("--img-size", type=int, default=256); ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--epochs-warm", type=int, default=3); ap.add_argument("--epochs-ft", type=int, default=25)
ap.add_argument("--mc-samples", type=int, default=16); ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--pop", type=int, default=20); ap.add_argument("--iters", type=int, default=25)
ap.add_argument("--no-imagenet", action="store_true"); ap.add_argument("--retrain", action="store_true")
ap.add_argument("--aggregate-only", action="store_true")
ap.add_argument("--capture-invariant", action="store_true")
ap.add_argument("--restricted-decoder", action="store_true")
a = ap.parse_args()
KEYS = ["acc", "bal_acc", "f1_fake", "recall_fake", "specificity", "auc", "ece"]
METADATA_COLUMNS = {"path", "group", "y", "fold", "camera", "flash_status", "iso"}


def _shift_to_threshold(probability, threshold):
    p = np.clip(np.asarray(probability), 1e-6, 1 - 1e-6)
    t = min(max(float(threshold), 1e-6), 1 - 1e-6)
    return 1 / (1 + np.exp(-(np.log(p / (1 - p)) - np.log(t / (1 - t)))))


def _fit_shortcut(audit, train_idx, val_idx, test_idx, feature_names, seed):
    train = audit.iloc[train_idx]
    val = audit.iloc[val_idx]
    test = audit.iloc[test_idx]
    model = _classifier(feature_names, audit)
    model.set_params(randomforestclassifier__random_state=seed)
    model.fit(train[feature_names], train["label"].astype(int))
    fake_column = list(model.classes_).index(1)
    return (
        model.predict_proba(val[feature_names])[:, fake_column],
        model.predict_proba(test[feature_names])[:, fake_column],
    )


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
    parts, sample_parts = [], []
    for d in folds:
        f = f"{out}/{d}/preds_test.csv"
        if not os.path.exists(f): continue
        q = pd.read_csv(f)
        qs = q.copy()
        th = json.load(open(f"{out}/{d}/thresholds.json"))
        sample_th = json.load(open(f"{out}/{d}/sample_thresholds.json"))
        for m, t in th.items():
            q[m] = _shift_to_threshold(q[m], t)
        for m, t in sample_th.items():
            qs[m] = _shift_to_threshold(qs[m], t)
        parts.append(q)
        sample_parts.append(qs)
    P = pd.concat(parts)
    Ps = pd.concat(sample_parts)
    methods = [c for c in P.columns if c not in METADATA_COLUMNS]
    rows = {}
    for m in methods:
        ps, ys = wb.aggregate_by_group(Ps[m].values, Ps.y.values, Ps.group.values)
        rows[m] = wb.metrics(ys, ps, 0.5)
    print("\n--- pooled out-of-fold, SAMPLE level, threshold 0.5 (every sample is tested exactly once) ---")
    print(pd.DataFrame(rows).T[["acc", "bal_acc", "f1_fake", "recall_fake", "specificity", "auc"]].round(3).to_string())
    pd.DataFrame(rows).T.to_csv(f"{out}/pooled_oof_sample_level.csv")
    image_ci_rows, sample_ci_rows = [], []
    for method in methods:
        image_ci_rows.extend({
            "method": method,
            "metric": metric,
            "lower_95": interval[0],
            "upper_95": interval[1],
        } for metric, interval in wb.bootstrap_metrics_ci(
            P.y.to_numpy(), P[method].to_numpy(), n=1000,
            seed=42, groups=P.group.to_numpy(),
        ).items())
        sample_group_ids = np.unique(Ps.group.to_numpy())
        sample_probability, sample_labels = wb.aggregate_by_group(
            Ps[method].to_numpy(), Ps.y.to_numpy(), Ps.group.to_numpy()
        )
        sample_ci_rows.extend({
            "method": method,
            "metric": metric,
            "lower_95": interval[0],
            "upper_95": interval[1],
        } for metric, interval in wb.bootstrap_metrics_ci(
            sample_labels, sample_probability, n=1000,
            seed=42, groups=sample_group_ids,
        ).items())
    pd.DataFrame(image_ci_rows).to_csv(
        f"{out}/pooled_oof_image_ci.csv", index=False
    )
    pd.DataFrame(sample_ci_rows).to_csv(
        f"{out}/pooled_oof_sample_ci.csv", index=False
    )

    group_meta = Ps.groupby("group", sort=True).agg(
        y=("y", "first"),
        camera=("camera", "first"),
        flash_status=("flash_status", "first"),
        iso=("iso", "median"),
    )
    for column in ("y", "camera", "flash_status"):
        if (Ps.groupby("group")[column].nunique(dropna=False) > 1).any():
            raise ValueError(f"metadata column {column!r} is inconsistent within a group")
    sample_predictions = group_meta.reset_index()
    for method in methods:
        sample_probability, sample_labels = wb.aggregate_by_group(
            Ps[method].to_numpy(), Ps.y.to_numpy(), Ps.group.to_numpy()
        )
        if not np.array_equal(sample_labels, group_meta.y.to_numpy()):
            raise ValueError(f"sample prediction order does not match group metadata: {method}")
        sample_predictions[method] = sample_probability

    camera_class_counts = (
        P.groupby(["camera", "y"], dropna=False)
        .agg(n_images=("path", "size"), n_groups=("group", "nunique"))
        .reset_index()
    )
    camera_class_counts.to_csv(f"{out}/camera_class_counts.csv", index=False)
    print("\n--- camera x class counts across held-out test folds ---")
    print(camera_class_counts.to_string(index=False))

    for scope_name, image_scope, sample_scope in [
        (
            "camera",
            P.groupby("camera", dropna=False),
            sample_predictions.groupby("camera", dropna=False),
        ),
        (
            "J600G flash",
            P[P.camera.astype(str).str.contains("SM-J600G", na=False)].groupby(
                "flash_status", dropna=False
            ),
            sample_predictions[
                sample_predictions.camera.astype(str).str.contains("SM-J600G", na=False)
            ].groupby("flash_status", dropna=False),
        ),
    ]:
        for level_name, scope in [("image", image_scope), ("sample", sample_scope)]:
            summary = []
            for label, subset in scope:
                if subset.empty:
                    continue
                for method in methods:
                    result = wb.metrics(
                        subset.y.to_numpy(), subset[method].to_numpy(), 0.5
                    )
                    ci = wb.bootstrap_metrics_ci(
                        subset.y.to_numpy(), subset[method].to_numpy(),
                        n=1000, seed=42, groups=subset.group.to_numpy(),
                    )
                    summary.append({
                        scope_name: label,
                        "method": method,
                        "n_images": len(subset) if level_name == "image" else np.nan,
                        "n_groups": subset.group.nunique(),
                        **result,
                        **{
                            f"{metric}_lower_95": interval[0]
                            for metric, interval in ci.items()
                        },
                        **{
                            f"{metric}_upper_95": interval[1]
                            for metric, interval in ci.items()
                        },
                    })
            if summary:
                target = "per_camera" if scope_name == "camera" else "j600g_flash"
                summary_table = pd.DataFrame(summary)
                summary_table.to_csv(
                    f"{out}/{target}_{level_name}_level.csv", index=False
                )
                print(
                    f"\n--- {scope_name}: pooled test {level_name}s "
                    "(threshold adjusted using validation) ---"
                )
                print(summary_table.groupby(
                    [scope_name, "method"], dropna=False
                )[["acc", "bal_acc", "recall_fake", "specificity", "auc", "n_groups"]]
                      .first().round(3).to_string())

    j600 = P[P.camera.astype(str).str.contains("SM-J600G", na=False)]
    fake_j600 = j600[j600.y == 1]
    subgroup_rows = []
    j600_samples = sample_predictions[
        sample_predictions.camera.astype(str).str.contains("SM-J600G", na=False)
        & (sample_predictions.y == 1)
    ]
    for name, image_mask, sample_mask in [
        (
            "flash_off", fake_j600.flash_status == "off",
            j600_samples.flash_status == "off",
        ),
        (
            "flash_on", fake_j600.flash_status == "on",
            j600_samples.flash_status == "on",
        ),
        (
            "iso_le_100", pd.to_numeric(fake_j600.iso, errors="coerce") <= 100,
            pd.to_numeric(j600_samples.iso, errors="coerce") <= 100,
        ),
        (
            "iso_gt_100_or_missing",
            pd.to_numeric(fake_j600.iso, errors="coerce").isna()
            | (pd.to_numeric(fake_j600.iso, errors="coerce") > 100),
            pd.to_numeric(j600_samples.iso, errors="coerce").isna()
            | (pd.to_numeric(j600_samples.iso, errors="coerce") > 100),
        ),
    ]:
        for level_name, data, mask in [
            ("image", fake_j600, image_mask),
            ("sample", j600_samples, sample_mask),
        ]:
            subset = data.loc[mask]
            for method in methods:
                ci = wb.bootstrap_metrics_ci(
                    np.ones(len(subset), dtype=int),
                    subset[method].to_numpy(),
                    n=1000, seed=42, groups=subset.group.to_numpy(),
                )
                subgroup_rows.append({
                    "level": level_name,
                    "subgroup": name,
                    "method": method,
                    "n_fake_images": len(subset) if level_name == "image" else np.nan,
                    "n_fake_groups": subset.group.nunique(),
                    "recall_fake": (
                        float((subset[method] >= 0.5).mean()) if len(subset) else np.nan
                    ),
                    "recall_fake_lower_95": ci.get("recall_fake", (np.nan, np.nan))[0],
                    "recall_fake_upper_95": ci.get("recall_fake", (np.nan, np.nan))[1],
                })
    subgroup_table = pd.DataFrame(subgroup_rows)
    subgroup_table.to_csv(f"{out}/j600g_fake_subgroups.csv", index=False)
    subgroup_table[subgroup_table.level == "image"].to_csv(
        f"{out}/j600g_fake_subgroups_image_level.csv", index=False
    )
    subgroup_table[subgroup_table.level == "sample"].to_csv(
        f"{out}/j600g_fake_subgroups_sample_level.csv", index=False
    )
    print("\n--- SM-J600G fake-only pooled test recall by image and sample ---")
    print(subgroup_table.query(
        "method == 'PROPOSED: Wombat-OA + weighted beam + log-prob fusion'"
    ).to_string(index=False))


if not a.aggregate_only:
    man = normalize_manifest_paths(pd.read_csv(a.manifest)); K = int(man.fold.max()) + 1
    os.makedirs(a.out, exist_ok=True)
    run_metadata = {
        "manifest": str(a.manifest), "fold_count": K, "img_size": a.img_size,
        "batch": a.batch, "epochs_warm": a.epochs_warm, "epochs_ft": a.epochs_ft,
        "mc_samples": a.mc_samples, "seed": a.seed, "imagenet": not a.no_imagenet,
        "capture_invariant": a.capture_invariant,
        "restricted_decoder": a.restricted_decoder,
        "pop": a.pop, "iters": a.iters,
    }
    run_metadata_path = os.path.join(a.out, "cv_run.json")
    if os.path.exists(run_metadata_path):
        with open(run_metadata_path, encoding="utf-8") as stream:
            previous = json.load(stream)
        if previous != run_metadata:
            raise ValueError(
                f"run configuration differs from existing {run_metadata_path}; "
                "use a new output directory to avoid mixing runs"
            )
    else:
        with open(run_metadata_path, "w", encoding="utf-8") as stream:
            json.dump(run_metadata, stream, indent=2)
    X = cb.features(man.path.tolist(), cache=f"{a.out}/colour_features.npz")
    audit_path = "data/audit_exif.csv"
    if not os.path.exists(audit_path):
        raise FileNotFoundError(
            f"{audit_path} is required for exposure/size shortcuts; run "
            "`python scripts/audit_shortcuts.py --manifest data/manifest.csv` first."
        )
    audit = normalize_manifest_paths(pd.read_csv(audit_path), columns=("path",))
    if not audit[["path", "group"]].equals(man[["path", "group"]]):
        raise ValueError("audit_exif.csv row order/identity does not match the CV manifest")
    audit["label"] = man["label"].to_numpy()
    for k in (a.folds if a.folds is not None else range(K)):
        od = f"{a.out}/fold{k}"; os.makedirs(od, exist_ok=True)
        if a.retrain or not (os.path.exists(f"{od}/cache_val.npz") and os.path.exists(f"{od}/cache_test.npz")):
            cmd = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_fold.py"),
                   "--manifest", a.manifest, "--out", od, "--fold", str(k), "--img-size", str(a.img_size), "--batch", str(a.batch),
                   "--epochs-warm", str(a.epochs_warm), "--epochs-ft", str(a.epochs_ft), "--mc-samples", str(a.mc_samples),
                   "--seed", str(a.seed)] + (["--no-imagenet"] if a.no_imagenet else [])
            if a.capture_invariant:
                cmd.append("--capture-invariant")
            subprocess.run(cmd, check=True)                       # fresh process per fold -> no TF memory build-up
        val, test = wb.Cache.load(f"{od}/cache_val.npz"), wb.Cache.load(f"{od}/cache_test.npz")
        kv = (k + 1) % K
        va = man[man.fold == kv].reset_index(drop=True)
        te = man[man.fold == k].reset_index(drop=True); tr = man[~man.fold.isin([k, kv])]
        for split_name, cache_name, frame, cache in (
            ("validation", "cache_val.npz", va, val),
            ("test", "cache_test.npz", te, test),
        ):
            if cache.paths is None:
                raise ValueError(
                    f"{od}/{cache_name} has no path metadata; "
                    "rerun this fold with --retrain to rebuild aligned caches"
                )
            if not np.array_equal(frame.path.astype(str).to_numpy(), cache.paths.astype(str)):
                raise ValueError(f"{split_name} cache path order does not match the manifest")
            if cache.groups is None or not np.array_equal(
                frame.group.astype(str).to_numpy(), cache.groups.astype(str)
            ):
                raise ValueError(f"{split_name} cache group order does not match the manifest")
            if not np.array_equal(frame.label.to_numpy(), cache.y):
                raise ValueError(f"{split_name} cache labels do not match the manifest")
        va_idx = man.index[man.fold == kv].to_numpy()
        te_idx = man.index[man.fold == k].to_numpy()
        tr_idx = man.index[~man.fold.isin([k, kv])].to_numpy()
        baseline_indices = np.concatenate([va_idx, te_idx])
        p_col_all = cb.fit_predict(
            X[tr_idx], tr.label.values, X[baseline_indices], seed=a.seed + k
        )
        p_col_val, p_col = np.split(p_col_all, [len(va_idx)])
        extra_rows = {
            "colour-statistics random forest (baseline)": (p_col, 0.5),
        }
        extra_val_probs = {
            "colour-statistics random forest (baseline)": p_col_val,
        }
        for feature_set in ("exposure_iso_flash", "size_file_bpp"):
            p_val, p_test = _fit_shortcut(
                audit, tr_idx, va_idx, te_idx,
                BASELINE_FEATURES[feature_set], a.seed + k,
            )
            name = f"{feature_set.replace('_', ' ')} random forest (shortcut)"
            extra_rows[name] = (p_test, 0.5)
            extra_val_probs[name] = p_val
        table, info = wb.evaluate_all(
            val, test, pop=a.pop, iters=a.iters, seed=k,
            extra_rows=extra_rows, extra_val_probs=extra_val_probs,
            restricted_decoder=a.restricted_decoder,
        )
        table.to_csv(f"{od}/ablation_test.csv"); info["sample_table"].to_csv(f"{od}/ablation_sample_level.csv")
        pd.DataFrame({
            **{m: p for m, p in info["probs"].items()},
            "path": te.path.to_numpy(),
            "group": test.groups,
            "y": test.y,
            "fold": k,
            "camera": audit.iloc[te_idx].camera.to_numpy(),
            "flash_status": audit.iloc[te_idx].flash_status.to_numpy(),
            "iso": audit.iloc[te_idx].iso.to_numpy(),
            "capture_date": audit.iloc[te_idx].capture_date.to_numpy(),
            "type": te.type.to_numpy(),
        }).to_csv(f"{od}/preds_test.csv", index=False)
        json.dump(info["thr"], open(f"{od}/thresholds.json", "w"))
        json.dump(info["sample_thr"], open(f"{od}/sample_thresholds.json", "w"))
        for filename, key in (("ci_image.csv", "image_ci"), ("ci_sample.csv", "sample_ci")):
            ci_rows = [
                {"method": method, "metric": metric, "lower_95": interval[0], "upper_95": interval[1]}
                for method, values in info[key].items()
                for metric, interval in values.items()
            ]
            pd.DataFrame(ci_rows).to_csv(f"{od}/{filename}", index=False)
        json.dump({q: (v.tolist() if hasattr(v, "tolist") else v) for q, v in info["params"].items()}, open(f"{od}/wombat_params.json", "w"), indent=1)
        json.dump({
            q: (v.tolist() if hasattr(v, "tolist") else v)
            for q, v in info["params_random"].items()
        }, open(f"{od}/wombat_params_random.json", "w"), indent=1)
        if info.get("params_restricted") is not None:
            json.dump({
                q: (v.tolist() if hasattr(v, "tolist") else v)
                for q, v in info["params_restricted"].items()
            }, open(f"{od}/wombat_params_restricted.json", "w"), indent=1)
        print(f"\n=== fold {k} (test: {len(te)} images, {te.group.nunique()} samples) ===")
        print(table[["acc", "f1_fake", "recall_fake", "specificity", "auc"]].round(3).to_string())
aggregate(a.out)
