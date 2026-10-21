"""Evaluate Phase 1 shortcut baselines on the manifest's exact CV folds."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import color_baseline
import wombat_beam as wb
from audit_shortcuts import BASELINE_FEATURES, _classifier
from manifest_paths import normalize_manifest_paths


def evaluate_fold(
    manifest: pd.DataFrame,
    audit: pd.DataFrame,
    colour_features: np.ndarray,
    fold: int,
    seed: int,
    ci_samples: int = 1000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit shortcut-only models on the supplied fold's train groups and score test."""
    n_folds = int(manifest["fold"].max()) + 1
    fold_seed = seed + fold
    val_fold = (fold + 1) % n_folds
    train_idx = np.flatnonzero(~manifest.fold.isin([fold, val_fold]).to_numpy())
    val_idx = np.flatnonzero((manifest.fold == val_fold).to_numpy())
    test_idx = np.flatnonzero((manifest.fold == fold).to_numpy())
    train_groups = set(manifest.iloc[train_idx].group.astype(str))
    val_groups = set(manifest.iloc[val_idx].group.astype(str))
    test_groups = set(manifest.iloc[test_idx].group.astype(str))
    if train_groups & val_groups or train_groups & test_groups or val_groups & test_groups:
        raise ValueError("a physical-sample group appears in more than one split")

    val_test_idx = np.concatenate([val_idx, test_idx])
    y_train = manifest.iloc[train_idx].label.astype(int).to_numpy()
    y_val = manifest.iloc[val_idx].label.astype(int).to_numpy()
    y_test = manifest.iloc[test_idx].label.astype(int).to_numpy()
    val_groups_order = manifest.iloc[val_idx].group.astype(str).to_numpy()
    test_groups_order = manifest.iloc[test_idx].group.astype(str).to_numpy()

    predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    colour_probability = color_baseline.fit_predict(
        colour_features[train_idx],
        y_train,
        colour_features[val_test_idx],
        seed=fold_seed,
    )
    colour_val, colour_test = np.split(colour_probability, [len(val_idx)])
    predictions["colour_mean_std"] = (colour_val, colour_test)

    for feature_set in ("exposure_iso_flash", "size_file_bpp"):
        names = BASELINE_FEATURES[feature_set]
        model = _classifier(names, audit)
        model.set_params(randomforestclassifier__random_state=fold_seed)
        model.fit(audit.iloc[train_idx][names], y_train)
        fake_index = list(model.classes_).index(1)
        probability = model.predict_proba(audit.iloc[val_test_idx][names])[:, fake_index]
        val_probability, test_probability = np.split(probability, [len(val_idx)])
        predictions[feature_set] = (val_probability, test_probability)

    rows = []
    prediction_rows = pd.DataFrame({
        "fold": fold,
        "group": test_groups_order,
        "y": y_test,
    })
    for method, (val_probability, test_probability) in predictions.items():
        prediction_rows[method] = test_probability
        threshold = wb.select_group_threshold(
            y_val, val_probability, val_groups_order
        )
        image_metrics = wb.metrics(y_test, test_probability, 0.5)
        sample_probability, sample_labels = wb.aggregate_by_group(
            test_probability, y_test, test_groups_order
        )
        sample_metrics = wb.metrics(sample_labels, sample_probability, threshold)
        image_ci = wb.bootstrap_metrics_ci(
            y_test,
            test_probability,
            0.5,
            n=ci_samples,
            seed=fold_seed,
            groups=test_groups_order,
        )
        sample_ci = wb.bootstrap_metrics_ci(
            sample_labels,
            sample_probability,
            threshold,
            n=ci_samples,
            seed=fold_seed,
            groups=np.unique(test_groups_order),
        )
        for level, metrics, intervals in (
            ("image", image_metrics, image_ci),
            ("sample", sample_metrics, sample_ci),
        ):
            rows.append({
                "fold": fold,
                "seed": seed,
                "method": method,
                "level": level,
                "validation_sample_threshold": threshold,
                "n_images": len(test_idx) if level == "image" else np.nan,
                "n_groups": len(test_groups),
                **metrics,
                **{
                    f"{metric}_lower_95": interval[0]
                    for metric, interval in intervals.items()
                },
                **{
                    f"{metric}_upper_95": interval[1]
                    for metric, interval in intervals.items()
                },
            })
    return pd.DataFrame(rows), prediction_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/manifest.csv")
    parser.add_argument("--audit", default="data/audit_exif.csv")
    parser.add_argument("--out", default="runs/audit/shortcut_cv_same_folds.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ci-samples", type=int, default=1000)
    args = parser.parse_args()

    manifest_path = ROOT / args.manifest
    audit_path = ROOT / args.audit
    manifest = normalize_manifest_paths(pd.read_csv(manifest_path))
    audit = normalize_manifest_paths(pd.read_csv(audit_path), columns=("path",))
    required = {"path", "group", "label", "fold"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"manifest is missing required columns: {sorted(missing)}")
    if not audit[["path", "group"]].equals(manifest[["path", "group"]]):
        raise ValueError("EXIF audit rows do not match manifest row order and group ids")
    if (manifest.groupby("group").fold.nunique() != 1).any():
        raise ValueError("each physical-sample group must belong to exactly one fold")
    if (manifest.groupby("group").label.nunique() != 1).any():
        raise ValueError("each physical-sample group must have exactly one label")
    audit["label"] = manifest["label"].to_numpy()
    colour = color_baseline.features(manifest.path.tolist())

    evaluated = [
        evaluate_fold(manifest, audit, colour, fold, args.seed, args.ci_samples)
        for fold in sorted(manifest.fold.unique())
    ]
    results = pd.concat([result for result, _ in evaluated], ignore_index=True)
    predictions = pd.concat([prediction for _, prediction in evaluated], ignore_index=True)
    output = ROOT / args.out
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)
    predictions_path = output.with_name(f"{output.stem}_predictions.csv")
    predictions.to_csv(predictions_path, index=False)

    metric_columns = [
        "acc", "bal_acc", "recall_fake", "specificity", "f1_fake", "auc",
    ]
    summary = (
        results.groupby(["level", "method"], sort=True)[metric_columns]
        .agg(["mean", "std"])
    )
    summary_path = output.with_name(f"{output.stem}_summary.csv")
    summary.to_csv(summary_path)
    print(f"Wrote per-fold shortcut results: {output}")
    print(f"Wrote held-out predictions: {predictions_path}")
    print(f"Wrote mean ± std by level: {summary_path}")
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
