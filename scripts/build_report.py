"""Build a provenance-checked report from complete, eligible CV result files."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import wombat_beam as wb
from inference_safety import model_bundle_fingerprint
from manifest_paths import normalize_manifest_paths, validate_prediction_manifest_identity

METHODS = {
    "PROPOSED: Wombat-OA + weighted beam + log-prob fusion": "Proposed",
    "single view, cnn": "Single head: CNN",
    "single view, gnn": "Single head: GNN",
    "single view, hafm": "Single head: HAFM",
    "TTA mean-logit, hafm": "TTA mean-logit",
    "log-prob fusion, equal weights, no beam": "Equal-weight log-prob fusion",
    "weighted beam + log-prob fusion (untuned)": "Untuned weighted beam",
    "weighted beam + log-prob fusion + random search": "Random-search weighted beam",
    "colour-statistics random forest (baseline)": "Colour baseline",
    "exposure iso flash random forest (shortcut)": "Exposure/ISO/flash",
    "size file bpp random forest (shortcut)": "Size/file size/bpp",
}


def eligible_run(run_dir: Path) -> tuple[bool, str]:
    """Reject known non-result directories and ineligible/incomplete runs."""
    if any("SMOKE_TEST_NOT_A_RESULT" in part for part in run_dir.parts):
        return False, "smoke-test directory"
    fold_dirs = sorted(
        p for p in run_dir.glob("fold*") if p.is_dir()
    )
    if not fold_dirs:
        return False, "no fold directories"
    metadata_path = run_dir / "cv_run.json"
    if not metadata_path.is_file():
        return False, "missing cv_run.json provenance"
    run_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    required_metadata = {
        "fold_count", "img_size", "epochs_warm", "epochs_ft", "mc_samples",
        "seed", "imagenet", "manifest",
    }
    missing_metadata = required_metadata - set(run_metadata)
    if missing_metadata:
        return False, f"cv_run.json is missing provenance fields: {sorted(missing_metadata)}"
    manifest_value = run_metadata.get("manifest")
    if not manifest_value:
        return False, "missing source manifest path in cv_run.json"
    manifest_path = Path(manifest_value)
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    if not manifest_path.is_file():
        return False, f"source manifest is missing: {manifest_path}"
    expected_folds = int(run_metadata.get("fold_count", 0))
    observed_folds = {
        int(re.fullmatch(r"fold(\d+)", path.name).group(1))
        for path in fold_dirs
        if re.fullmatch(r"fold(\d+)", path.name)
    }
    if expected_folds < 1 or observed_folds != set(range(expected_folds)):
        return False, "fold result set is incomplete"
    if run_metadata.get("imagenet") is not True:
        return False, "run metadata indicates --no-imagenet or missing ImageNet weights"
    if int(run_metadata.get("epochs_ft", 0)) < 5:
        return False, "run metadata fine-tuning epochs below 5"
    for fold_dir in fold_dirs:
        config_path = fold_dir / "config.json"
        if not config_path.is_file():
            return False, f"missing {config_path.name}"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("imagenet") is not True:
            return False, "run used --no-imagenet or lacks ImageNet provenance"
        if int(config.get("epochs_ft", 0)) < 5:
            return False, "fine-tuning epochs below 5"
        if config.get("imagenet") is not True:
            return False, "fold metadata indicates --no-imagenet"
        if not (fold_dir / "preds_test.csv").is_file():
            return False, f"missing {fold_dir.name}/preds_test.csv"
        if not (fold_dir / "sample_thresholds.json").is_file():
            return False, f"missing {fold_dir.name}/sample_thresholds.json"
        if not (fold_dir / "thresholds.json").is_file():
            return False, f"missing {fold_dir.name}/thresholds.json"
        if not (fold_dir / "wombat_params.json").is_file():
            return False, f"missing {fold_dir.name}/wombat_params.json"
        if not (fold_dir / "ood_reference.npz").is_file():
            return False, f"missing {fold_dir.name}/ood_reference.npz"
        if not any(
            (fold_dir / name).is_file()
            for name in ("final.weights.h5", "best.weights.h5")
        ):
            return False, f"missing saved weights in {fold_dir.name}"
    return True, "eligible"


def clopper_pearson(successes: int, total: int, alpha: float = 0.05) -> tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    low = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, total - successes + 1))
    high = 1.0 if successes == total else float(beta.ppf(1 - alpha / 2, successes + 1, total - successes))
    return low, high


def _threshold_adjusted(probabilities: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    t = np.clip(np.asarray(thresholds, dtype=float), 1e-6, 1 - 1e-6)
    return 1 / (1 + np.exp(-(np.log(p / (1 - p)) - np.log(t / (1 - t)))))


def _markdown_table(frame: pd.DataFrame, floatfmt: str = ".3f") -> str:
    if frame.empty:
        return "No estimable rows."

    def render(value) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return format(float(value), floatfmt)
        return str(value).replace("|", r"\|").replace("\n", " ")

    headers = [str(column) for column in frame.columns]
    rows = [[render(value) for value in row] for row in frame.itertuples(index=False, name=None)]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    line = lambda row: "| " + " | ".join(
        value.ljust(widths[index]) for index, value in enumerate(row)
    ) + " |"
    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join((line(headers), separator, *(line(row) for row in rows)))


def _metric_rows(
    predictions: pd.DataFrame,
    *,
    level: str,
    group_column: str = "group",
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    y = predictions.y.to_numpy(dtype=int)
    groups = predictions[group_column].astype(str).to_numpy()
    rows: list[dict] = []
    for method in [name for name in METHODS if name in predictions]:
        p = predictions[method].to_numpy(dtype=float)
        suffix = "__image_threshold" if level == "image" else "__threshold"
        threshold = predictions.get(
            f"{method}{suffix}", pd.Series(0.5, index=predictions.index)
        ).to_numpy(dtype=float)
        adjusted = _threshold_adjusted(p, threshold)
        pred = (adjusted >= 0.5).astype(int)
        tp = int(((y == 1) & (pred == 1)).sum())
        tn = int(((y == 0) & (pred == 0)).sum())
        fp = int(((y == 0) & (pred == 1)).sum())
        fn = int(((y == 1) & (pred == 0)).sum())
        thresholded_metrics = wb.metrics(y, adjusted, 0.5)
        thresholded_ci = wb.bootstrap_metrics_ci(
            y, adjusted, 0.5, n=n_bootstrap, seed=seed, groups=groups
        )
        raw_metrics = wb.metrics(y, p, 0.5)
        raw_ci = wb.bootstrap_metrics_ci(
            y, p, 0.5, n=n_bootstrap, seed=seed, groups=groups
        )
        counts = {
            "acc": (tp + tn, len(y)),
            "recall_fake": (tp, tp + fn),
            "specificity": (tn, tn + fp),
        }
        for metric in ("acc", "bal_acc", "f1_fake", "recall_fake", "specificity"):
            successes, denominator = counts.get(metric, (np.nan, np.nan))
            exact_low, exact_high = (
                clopper_pearson(successes, denominator)
                if metric in counts else (np.nan, np.nan)
            )
            rows.append({
                "method": METHODS[method], "level": level, "metric": metric,
                "estimate": thresholded_metrics[metric],
                "ci95_low": thresholded_ci.get(metric, (np.nan, np.nan))[0],
                "ci95_high": thresholded_ci.get(metric, (np.nan, np.nan))[1],
                "n_images": int(predictions.n_images.sum()) if level == "sample" else len(y),
                "n_samples": int(predictions[group_column].nunique()),
                "n_fake": int((y == 1).sum()),
                "n_real": int((y == 0).sum()),
                "exact_ci95_low": exact_low, "exact_ci95_high": exact_high,
                "numerator": successes, "denominator": denominator,
            })
        rows.append({
            "method": METHODS[method], "level": level, "metric": "auc",
            "estimate": raw_metrics["auc"],
            "ci95_low": raw_ci.get("auc", (np.nan, np.nan))[0],
            "ci95_high": raw_ci.get("auc", (np.nan, np.nan))[1],
            "n_images": int(predictions.n_images.sum()) if level == "sample" else len(y),
            "n_samples": int(predictions[group_column].nunique()),
            "n_fake": int((y == 1).sum()),
            "n_real": int((y == 0).sum()),
            "exact_ci95_low": np.nan, "exact_ci95_high": np.nan,
            "numerator": np.nan, "denominator": np.nan,
        })
    return pd.DataFrame(rows)


def collect_run(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid, reason = eligible_run(run_dir)
    if not valid:
        raise ValueError(f"Refusing ineligible run {run_dir}: {reason}")
    frames = []
    metadata = json.loads((run_dir / "cv_run.json").read_text(encoding="utf-8"))
    for fold_dir in sorted(p for p in run_dir.glob("fold*") if p.is_dir()):
        frame = normalize_manifest_paths(
            pd.read_csv(fold_dir / "preds_test.csv"), columns=("path",)
        )
        thresholds = json.loads((fold_dir / "sample_thresholds.json").read_text(encoding="utf-8"))
        frame["fold"] = int(fold_dir.name.removeprefix("fold"))
        frame["n_images"] = 1
        frame["n_samples"] = 1
        image_thresholds = json.loads(
            (fold_dir / "thresholds.json").read_text(encoding="utf-8")
        )
        for method, threshold in thresholds.items():
            frame[f"{method}__threshold"] = float(threshold)
        for method, threshold in image_thresholds.items():
            frame[f"{method}__image_threshold"] = float(threshold)
        frames.append(frame)
    image_predictions = pd.concat(frames, ignore_index=True)
    manifest_path = Path(metadata["manifest"])
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    manifest = normalize_manifest_paths(pd.read_csv(manifest_path))
    validate_prediction_manifest_identity(manifest, image_predictions)
    if image_predictions.groupby("group").fold.nunique().gt(1).any():
        raise ValueError("a physical sample occurs in multiple held-out folds")
    if image_predictions.groupby("group").y.nunique().gt(1).any():
        raise ValueError("a physical sample has inconsistent labels")
    sample_rows = []
    for group, part in image_predictions.groupby("group", sort=True):
        row = {
            "group": str(group), "y": int(part.y.iloc[0]), "n_images": len(part),
            "n_samples": 1,
            "fold": int(part.fold.iloc[0]),
        }
        for column in ("camera", "flash_status", "iso", "capture_date", "type"):
            if column in part:
                row[column] = part[column].iloc[0]
        for method in [name for name in METHODS if name in part]:
            probabilities = np.clip(part[method].to_numpy(float), 1e-6, 1 - 1e-6)
            logits = np.log(probabilities / (1 - probabilities))
            row[method] = float(1 / (1 + np.exp(-logits.mean())))
            row[f"{method}__threshold"] = float(part[f"{method}__threshold"].iloc[0])
        sample_rows.append(row)
    return image_predictions, pd.DataFrame(sample_rows)


def verify_report_numbers(text: str, source_csvs: list[Path]) -> None:
    """Reject numeric tokens in generated report text that are absent from CSV sources."""
    whitelist: set[str] = set()
    for path in source_csvs:
        frame = pd.read_csv(path)
        for value in frame.to_numpy().ravel():
            if pd.isna(value):
                continue
            whitelist.add(str(value))
            if isinstance(value, (int, float, np.number)):
                whitelist.add(f"{float(value):.3f}")
                whitelist.add(f"{float(value):.4f}")
                if float(value).is_integer():
                    whitelist.add(str(int(value)))
    checked_text = re.sub(r"`[a-f0-9]{64}`", "", text, flags=re.IGNORECASE)
    for match in re.finditer(
        r"(?<!\w)(?:\d+\.\d+|\d+)(?!\w)", checked_text
    ):
        token = match.group()
        if token not in whitelist:
            context = checked_text[max(0, match.start() - 24):match.end() + 24]
            raise ValueError(
                f"report contains numeric token {token!r} absent from source CSVs: {context!r}"
            )


def build_report(
    run_dir: Path, output_dir: Path, n_bootstrap: int = 1000
) -> Path:
    images, samples = collect_run(run_dir)
    metadata = json.loads((run_dir / "cv_run.json").read_text(encoding="utf-8"))
    manifest_path = Path(metadata["manifest"])
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    data_version = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    model_version = model_bundle_fingerprint(run_dir, manifest_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = output_dir / "results_tables"
    tables_dir.mkdir(exist_ok=True)
    image_table = _metric_rows(images, level="image")
    sample_table = _metric_rows(samples, level="sample")
    source_csv = [tables_dir / "image_metrics.csv", tables_dir / "sample_metrics.csv"]
    image_table.to_csv(source_csv[0], index=False)
    sample_table.to_csv(source_csv[1], index=False)
    images.to_csv(tables_dir / "oof_image_predictions.csv", index=False)
    samples.to_csv(tables_dir / "oof_sample_predictions.csv", index=False)
    errors = []
    for method in [name for name in METHODS if name in samples]:
        threshold = samples[f"{method}__threshold"].to_numpy(float)
        wrong = samples[(samples[method].to_numpy(float) >= threshold) != samples.y.to_numpy(int)]
        for _, row in wrong.iterrows():
            errors.append({
                "method": METHODS[method], "group": row.group, "label": row.y,
                "p_fake": row[method], "threshold": row[f"{method}__threshold"],
                "fold": row.fold, "camera": row.get("camera", "unknown"),
                "type": row.get("type", "unknown"), "iso": row.get("iso", np.nan),
                "flash_status": row.get("flash_status", "unknown"),
                "capture_date": row.get("capture_date", "unknown"),
                "n_images": row.n_images,
                "n_samples": 1,
            })
    errors_path = tables_dir / "misclassified_samples.csv"
    pd.DataFrame(errors).to_csv(errors_path, index=False)
    subgroup_rows = []
    for scope, mask in (
        ("pooled", np.ones(len(samples), dtype=bool)),
        ("SM-G973F", samples.camera.astype(str).str.contains("SM-G973F", na=False).to_numpy()),
        ("SM-J600G", samples.camera.astype(str).str.contains("SM-J600G", na=False).to_numpy()),
    ):
        subset = samples.loc[mask]
        for method in [name for name in METHODS if name in subset]:
            y = subset.y.to_numpy(int)
            if len(np.unique(y)) < 2:
                continue
            p = subset[method].to_numpy(float)
            threshold = subset[f"{method}__threshold"].to_numpy(float)
            adjusted = _threshold_adjusted(p, threshold)
            thresholded = adjusted >= 0.5
            tp = int(((y == 1) & (thresholded == 1)).sum())
            tn = int(((y == 0) & (thresholded == 0)).sum())
            accuracy_low, accuracy_high = clopper_pearson(tp + tn, len(y))
            thresholded_ci = wb.bootstrap_metrics_ci(
                y, adjusted, 0.5, n=n_bootstrap, seed=42, groups=subset.group.to_numpy()
            )
            raw_ci = wb.bootstrap_metrics_ci(
                y, p, 0.5, n=n_bootstrap, seed=42, groups=subset.group.to_numpy()
            )
            values = wb.metrics(y, adjusted, 0.5)
            raw_values = wb.metrics(y, p, 0.5)
            subgroup_rows.append({
                "scope": scope, "method": METHODS[method],
                "n_images": int(subset.n_images.sum()), "n_samples": int(subset.group.nunique()),
                "n_fake": int((y == 1).sum()), "n_real": int((y == 0).sum()),
                "balanced_accuracy": values["bal_acc"],
                "balanced_accuracy_lower_95": thresholded_ci.get("bal_acc", (np.nan, np.nan))[0],
                "balanced_accuracy_upper_95": thresholded_ci.get("bal_acc", (np.nan, np.nan))[1],
                "auc": raw_values["auc"],
                "auc_lower_95": raw_ci.get("auc", (np.nan, np.nan))[0],
                "auc_upper_95": raw_ci.get("auc", (np.nan, np.nan))[1],
                "thresholded_accuracy": float((thresholded == y).mean()),
                "accuracy_lower_95": accuracy_low,
                "accuracy_upper_95": accuracy_high,
            })
    subgroup_path = tables_dir / "primary_subsets.csv"
    pd.DataFrame(subgroup_rows).to_csv(subgroup_path, index=False)
    provenance = pd.DataFrame([{
        "data_version": data_version,
        "model_version": model_version,
        "fold_count": metadata["fold_count"],
        "img_size": metadata["img_size"],
        "epochs_warm": metadata["epochs_warm"],
        "epochs_ft": metadata["epochs_ft"],
        "mc_samples": metadata["mc_samples"],
        "seed": metadata["seed"],
        "imagenet": metadata["imagenet"],
    }])
    provenance_path = tables_dir / "run_provenance.csv"
    provenance.to_csv(provenance_path, index=False)
    glossary = (ROOT / "docs" / "model_glossary.md").read_text(encoding="utf-8").strip()
    lines = [
        "# Evaluation results",
        "",
        "Source: the requested eligible run's saved out-of-fold prediction files.",
        f"Model bundle fingerprint: `{model_version}`; manifest fingerprint: `{data_version}`.",
        f"Run configuration: {metadata['img_size']} px; seed {metadata['seed']}; "
        f"{metadata['epochs_ft']} fine-tuning epochs; {metadata['fold_count']} folds.",
        "These results describe this dataset and do not establish performance on new phones, sessions, or adulterant types.",
        "",
        "## Registered primary metrics",
        "",
        "Sample-level balanced accuracy and AUC for pooled out-of-fold samples, SM-G973F-only samples, and leave-one-camera-out evaluation.",
        "",
        "## Image-level metrics",
        "",
        _markdown_table(image_table),
        "",
        "## Sample-level metrics",
        "",
        _markdown_table(sample_table),
        "",
        "## Primary subsets",
        "",
        _markdown_table(pd.DataFrame(subgroup_rows)) if subgroup_rows else "No estimable camera subset was present in the source files.",
        "",
        "LOCO metrics are not reported unless eligible leave-one-camera-out prediction files are supplied; no substitute estimate is produced.",
        "",
        "## Misclassified samples",
        "",
        "Misclassification records are saved in `results_tables/misclassified_samples.csv`; "
        "this report does not infer missing capture metadata.",
        "",
        "## Definitions",
        "",
        glossary,
        "",
    ]
    text = "\n".join(lines)
    verify_report_numbers(text, [*source_csv, subgroup_path, provenance_path])
    report_path = output_dir / "RESULTS.md"
    report_path.write_text(text, encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="one CV output directory")
    parser.add_argument("--out", default=None, help="defaults to <run>/RESULTS.md")
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    run_dir = Path(args.run)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    output_dir = Path(args.out) if args.out else run_dir
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    report = build_report(run_dir, output_dir, n_bootstrap=args.bootstrap)
    print(f"Wrote result report from eligible source files: {report}")


if __name__ == "__main__":
    main()
