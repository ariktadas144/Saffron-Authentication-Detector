"""Validate and collect complete Phase 2 GPU outputs into one report table."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import wombat_beam as wb
from manifest_paths import (
    normalize_manifest_paths, validate_prediction_manifest_identity,
)

DEEP_METHOD = "PROPOSED: Wombat-OA + weighted beam + log-prob fusion"
METHODS = {
    DEEP_METHOD: "Deep model",
    "colour-statistics random forest (baseline)": "Colour baseline",
    "exposure iso flash random forest (shortcut)": "Exposure/ISO/flash",
    "size file bpp random forest (shortcut)": "Size/file size/bpp",
}
METRIC_NAMES = {
    "auc": "AUC",
    "acc": "Accuracy",
    "bal_acc": "Balanced accuracy",
    "recall_fake": "Recall (fake)",
    "specificity": "Specificity",
    "f1_fake": "F1 (fake)",
}


def _shift(probability: np.ndarray, threshold: float) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    t = float(np.clip(threshold, 1e-6, 1 - 1e-6))
    return 1 / (1 + np.exp(-(np.log(p / (1 - p)) - np.log(t / (1 - t)))))


def _required_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required result input is missing: {path}. "
            "Complete the requested GPU runs before collecting results."
        )
    return path


def _add_metric_rows(
    output: list[dict],
    section: str,
    method: str,
    level: str,
    scope: str,
    fold_values: list[dict],
    pooled_values: dict,
    pooled_ci: dict,
    n_samples: int,
    n_images: int,
) -> None:
    for metric, label in METRIC_NAMES.items():
        values = [float(row[metric]) for row in fold_values if np.isfinite(row[metric])]
        output.append({
            "section": section,
            "method": method,
            "level": level,
            "scope": scope,
            "metric": label,
            "fold_mean": float(np.mean(values)) if values else np.nan,
            "fold_std": float(np.std(values)) if values else np.nan,
            "pooled_oof": pooled_values.get(metric, np.nan),
            "ci95_low": pooled_ci.get(metric, (np.nan, np.nan))[0],
            "ci95_high": pooled_ci.get(metric, (np.nan, np.nan))[1],
            "n_groups": n_samples,
            "n_images": n_images,
            "n_samples": n_samples,
            "seed": "",
        })


def collect(
    cv_dir: Path, shuffle_dir: Path, manifest_path: Path
) -> pd.DataFrame:
    """Collect only after all five deep folds and the three shuffle seeds exist."""
    pred_frames = []
    fold_thresholds = {}
    fold_sample_thresholds = {}
    fold_ids = list(range(5))
    required_columns = {"path", "group", "y", "fold", "camera", "flash_status", "iso"}
    for fold in fold_ids:
        fold_dir = cv_dir / f"fold{fold}"
        prediction_path = _required_file(fold_dir / "preds_test.csv")
        _required_file(fold_dir / "ablation_test.csv")
        _required_file(fold_dir / "ablation_sample_level.csv")
        threshold_path = _required_file(fold_dir / "thresholds.json")
        sample_threshold_path = _required_file(fold_dir / "sample_thresholds.json")
        predictions = pd.read_csv(prediction_path)
        missing = required_columns - set(predictions.columns)
        if missing:
            raise ValueError(f"{prediction_path} is missing columns: {sorted(missing)}")
        if set(METHODS) - set(predictions.columns):
            raise ValueError(
                f"{prediction_path} is missing prediction methods: "
                f"{sorted(set(METHODS) - set(predictions.columns))}"
            )
        if predictions.fold.nunique() != 1 or int(predictions.fold.iloc[0]) != fold:
            raise ValueError(f"{prediction_path} contains unexpected fold IDs")
        if predictions.groupby("group").y.nunique().gt(1).any():
            raise ValueError(f"{prediction_path} has conflicting labels within a group")
        if predictions.groupby("group").camera.nunique(dropna=False).gt(1).any():
            raise ValueError(f"{prediction_path} has inconsistent camera metadata")
        fold_thresholds[fold] = json.loads(threshold_path.read_text(encoding="utf-8"))
        fold_sample_thresholds[fold] = json.loads(
            sample_threshold_path.read_text(encoding="utf-8")
        )
        if set(METHODS) - set(fold_thresholds[fold]):
            raise ValueError(f"{threshold_path} is missing thresholds for required methods")
        if set(METHODS) - set(fold_sample_thresholds[fold]):
            raise ValueError(
                f"{sample_threshold_path} is missing sample thresholds for required methods"
            )
        pred_frames.append(predictions)

    images = pd.concat(pred_frames, ignore_index=True)
    manifest = normalize_manifest_paths(pd.read_csv(manifest_path))
    validate_prediction_manifest_identity(manifest, images)
    if images.group.duplicated().any():
        # A physical sample may have multiple image rows but must be in exactly one fold.
        if (images.groupby("group").fold.nunique() > 1).any():
            raise ValueError("a physical-sample group occurs in multiple held-out folds")
    if set(images.fold.astype(int)) != set(fold_ids):
        raise ValueError("not all five held-out folds are represented")

    output_rows: list[dict] = []
    sample_scores_by_method: dict[str, pd.DataFrame] = {}
    for method, display in METHODS.items():
        image_parts = []
        sample_parts = []
        image_fold_metrics = []
        sample_fold_metrics = []
        for fold in fold_ids:
            data = images[images.fold == fold].copy()
            image_p = _shift(data[method].to_numpy(), fold_thresholds[fold][method])
            image_parts.append(pd.DataFrame({
                "fold": fold,
                "group": data.group.astype(str).to_numpy(),
                "y": data.y.to_numpy(),
                "probability": image_p,
            }))
            image_fold_metrics.append(wb.metrics(data.y.to_numpy(), image_p, 0.5))

            sample_p = _shift(
                data[method].to_numpy(), fold_sample_thresholds[fold][method]
            )
            group_probability, group_labels = wb.aggregate_by_group(
                sample_p, data.y.to_numpy(), data.group.astype(str).to_numpy()
            )
            groups = np.unique(data.group.astype(str).to_numpy())
            if len(groups) != len(group_labels):
                raise AssertionError("sample prediction order mismatch during collection")
            sample_parts.append(pd.DataFrame({
                "fold": fold,
                "group": groups,
                "y": group_labels,
                "probability": group_probability,
            }))
            sample_fold_metrics.append(
                wb.metrics(group_labels, group_probability, 0.5)
            )
        pooled_images = pd.concat(image_parts, ignore_index=True)
        pooled_samples = pd.concat(sample_parts, ignore_index=True)
        if pooled_samples.group.duplicated().any():
            raise ValueError("sample-level out-of-fold groups are not unique")
        image_counts = images.groupby("group").path.size()
        pooled_samples["n_images"] = pooled_samples.group.map(image_counts)
        image_metrics = wb.metrics(
            pooled_images.y.to_numpy(), pooled_images.probability.to_numpy(), 0.5
        )
        image_ci = wb.bootstrap_metrics_ci(
            pooled_images.y.to_numpy(),
            pooled_images.probability.to_numpy(),
            n=2000,
            seed=42,
            groups=pooled_images.group.to_numpy(),
        )
        sample_metrics = wb.metrics(
            pooled_samples.y.to_numpy(), pooled_samples.probability.to_numpy(), 0.5
        )
        sample_ci = wb.bootstrap_metrics_ci(
            pooled_samples.y.to_numpy(),
            pooled_samples.probability.to_numpy(),
            n=2000,
            seed=42,
            groups=pooled_samples.group.to_numpy(),
        )
        for level, fold_metrics, metrics, intervals, image_count in (
            ("image", image_fold_metrics, image_metrics, image_ci, len(pooled_images)),
            ("sample", sample_fold_metrics, sample_metrics, sample_ci, int(pooled_samples.n_images.sum())),
        ):
            _add_metric_rows(
                output_rows, "model_vs_shortcuts", display, level, "pooled",
                fold_metrics, metrics, intervals, len(pooled_samples), image_count,
            )
        if method == DEEP_METHOD:
            sample_scores_by_method[method] = pooled_samples.merge(
                images.drop_duplicates("group")[
                    ["group", "camera", "flash_status", "iso"]
                ],
                on="group",
                how="left",
                validate="one_to_one",
            )

    deep_samples = sample_scores_by_method[DEEP_METHOD]
    subgroup_specs = []
    for camera, subset in deep_samples.groupby("camera", dropna=False):
        subgroup_specs.append((f"camera={camera}", subset))
    j600 = deep_samples[
        deep_samples.camera.astype(str).str.contains("SM-J600G", na=False)
    ]
    for flash_status, subset in j600.groupby("flash_status", dropna=False):
        subgroup_specs.append((f"SM-J600G flash={flash_status}", subset))
    iso = pd.to_numeric(j600.iso, errors="coerce")
    subgroup_specs.extend([
        ("SM-J600G ISO<=100", j600[iso <= 100]),
        ("SM-J600G ISO>100 or missing", j600[iso.isna() | (iso > 100)]),
    ])
    for scope, subset in subgroup_specs:
        labels, probability, groups = (
            subset.y.to_numpy(), subset.probability.to_numpy(),
            subset.group.astype(str).to_numpy(),
        )
        ci = wb.bootstrap_metrics_ci(
            labels, probability, n=2000, seed=42, groups=groups
        )
        metrics = wb.metrics(labels, probability)
        for metric in ("recall_fake", "specificity", "auc"):
            if metric == "recall_fake":
                mask = labels == 1
                estimate = float((probability[mask] >= 0.5).mean()) if mask.any() else np.nan
                low, high = ci.get(metric, (np.nan, np.nan))
                n = int(mask.sum())
            else:
                estimate = metrics[metric]
                low, high = ci.get(metric, (np.nan, np.nan))
                n = len(groups)
            output_rows.append({
                "section": "deep_subgroups",
                "method": "Deep model",
                "level": "sample",
                "scope": scope,
                "metric": METRIC_NAMES[metric],
                "fold_mean": np.nan,
                "fold_std": np.nan,
                "pooled_oof": estimate,
                "ci95_low": low,
                "ci95_high": high,
                "n_groups": len(groups),
                "n_images": int(subset.n_images.sum()),
                "n_samples": len(groups),
                "n_fake": int((labels == 1).sum()),
                "n_real": int((labels == 0).sum()),
                "seed": "",
            })

    shuffle_path = _required_file(shuffle_dir / "summary.csv")
    shuffle = pd.read_csv(shuffle_path)
    required_shuffle_columns = {
        "result", "shuffle_seed", "sample_auc", "sample_accuracy",
    }
    missing_shuffle = required_shuffle_columns - set(shuffle.columns)
    if missing_shuffle:
        raise ValueError(f"{shuffle_path} is missing columns: {sorted(missing_shuffle)}")
    real = shuffle[shuffle.result == "real labels"]
    if len(real) != 1:
        raise ValueError(f"{shuffle_path} must contain exactly one real-label result")
    shuffled = shuffle[shuffle.result == "group-label shuffle"]
    expected_seeds = {101, 202, 303}
    observed_seeds = set(pd.to_numeric(shuffled.shuffle_seed).dropna().astype(int))
    if observed_seeds != expected_seeds or len(shuffled) != 3:
        raise ValueError(
            f"{shuffle_path} must contain exactly shuffle seeds 101, 202, 303; "
            f"found {sorted(observed_seeds)}"
        )
    for _, row in pd.concat([real, shuffled]).iterrows():
        output_rows.append({
            "section": "real_vs_shuffled_labels",
            "method": row.result,
            "level": "sample",
            "scope": "fold0",
            "metric": "AUC",
            "fold_mean": float(row.sample_auc),
            "fold_std": np.nan,
            "pooled_oof": float(row.sample_auc),
            "ci95_low": np.nan,
            "ci95_high": np.nan,
            "n_groups": row.get("n_samples", np.nan),
            "n_images": row.get("n_images", np.nan),
            "n_samples": row.get("n_samples", np.nan),
            "n_fake": row.get("n_fake", np.nan),
            "n_real": row.get("n_real", np.nan),
            "seed": "" if pd.isna(row.shuffle_seed) else int(row.shuffle_seed),
        })
        output_rows.append({
            "section": "real_vs_shuffled_labels",
            "method": row.result,
            "level": "sample",
            "scope": "fold0",
            "metric": "Accuracy",
            "fold_mean": float(row.sample_accuracy),
            "fold_std": np.nan,
            "pooled_oof": float(row.sample_accuracy),
            "ci95_low": np.nan,
            "ci95_high": np.nan,
            "n_groups": row.get("n_samples", np.nan),
            "n_images": row.get("n_images", np.nan),
            "n_samples": row.get("n_samples", np.nan),
            "n_fake": row.get("n_fake", np.nan),
            "n_real": row.get("n_real", np.nan),
            "seed": "" if pd.isna(row.shuffle_seed) else int(row.shuffle_seed),
        })

    return pd.DataFrame(output_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv-dir", default="runs/cv_repro")
    parser.add_argument("--shuffle-dir", default="runs/audit/label_shuffle")
    parser.add_argument("--manifest", default="data/manifest.csv")
    parser.add_argument("--out", default="runs/audit/collected_phase2_results.csv")
    args = parser.parse_args()
    cv_dir = Path(args.cv_dir)
    shuffle_dir = Path(args.shuffle_dir)
    manifest_path = Path(args.manifest)
    output = Path(args.out)
    if not cv_dir.is_absolute():
        cv_dir = ROOT / cv_dir
    if not shuffle_dir.is_absolute():
        shuffle_dir = ROOT / shuffle_dir
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    if not output.is_absolute():
        output = ROOT / output
    table = collect(cv_dir, shuffle_dir, manifest_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    print(table.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"Wrote complete collected table: {output}")


if __name__ == "__main__":
    main()
