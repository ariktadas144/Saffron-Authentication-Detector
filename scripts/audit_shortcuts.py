"""Audit capture metadata and grouped-CV shortcut baselines.

Run from the repository root:
    python scripts/audit_shortcuts.py --manifest data/manifest.csv
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ExifTags
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import color_baseline
import wombat_beam
from manifest_paths import resolve_manifest_path as _resolve_manifest_path

EXIF_NAMES = {name: tag for tag, name in ExifTags.TAGS.items()}
EXIF_FIELDS = (
    "Make", "Model", "Software", "DateTimeOriginal", "ISOSpeedRatings",
    "ExposureTime", "FNumber", "Flash", "BrightnessValue", "FocalLength",
    "WhiteBalance",
)
JPEG_LUMA_QTABLE = np.array([
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
])

BASELINE_FEATURES = {
    "exposure_iso_flash": [
        "iso", "exposure_time_s", "f_number", "brightness_value",
        "focal_length_mm", "flash_status",
    ],
    "flash_only": ["flash_status"],
    "size_file_bpp": [
        "width", "height", "pixel_count", "file_size_bytes", "bits_per_pixel",
    ],
    "colour_mean_std": [
        "r_mean", "g_mean", "b_mean", "r_std", "g_std", "b_std",
        "h_mean", "s_mean", "v_mean", "h_std", "s_std", "v_std",
    ],
    "camera_only": ["camera"],
}


def resolve_manifest_path(value: Any, root: Path = ROOT) -> Path:
    """Resolve manifest paths across POSIX and Windows separators."""
    return _resolve_manifest_path(value, root)


def _number(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        if isinstance(value, tuple) and len(value) == 2:
            denominator = float(value[1])
            return float(value[0]) / denominator if denominator else float("nan")
        return float(value)
    except (TypeError, ValueError, ZeroDivisionError, OverflowError):
        return float("nan")


def _text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _exif_mapping(image: Image.Image) -> dict[str, Any]:
    raw = image.getexif()
    exif_ifd_tag = EXIF_NAMES.get("ExifOffset")
    if exif_ifd_tag is not None and exif_ifd_tag in raw:
        try:
            raw.update(raw.get_ifd(exif_ifd_tag))
        except (AttributeError, KeyError, TypeError):
            pass
    return {
        name: raw.get(tag)
        for name, tag in EXIF_NAMES.items()
        if tag in raw and name in EXIF_FIELDS
    }


def estimate_jpeg_quality(image: Image.Image) -> float:
    """Estimate encoder quality from the JPEG luminance quantization table."""
    tables = getattr(image, "quantization", None)
    if not tables:
        return float("nan")
    table = tables.get(0)
    if not table or len(table) != len(JPEG_LUMA_QTABLE):
        return float("nan")
    observed = np.asarray(table, dtype=float)
    qualities = np.arange(1, 101)
    scales = np.where(qualities < 50, 5000 / qualities, 200 - 2 * qualities)
    expected = np.floor(
        (JPEG_LUMA_QTABLE[None, :] * scales[:, None] + 50) / 100
    ).clip(1, 255)
    errors = np.abs(expected - observed[None, :]).mean(axis=1)
    return float(qualities[int(errors.argmin())])


def read_image_audit(path: Path) -> dict[str, Any]:
    """Read selected EXIF tags and basic file/image metadata for one original."""
    try:
        display_path = path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        display_path = path.as_posix()
    result: dict[str, Any] = {
        "original_exists": path.is_file(),
        "original_path_resolved": display_path,
    }
    result.update({name: np.nan for name in EXIF_FIELDS})
    result.update({
        "width": np.nan,
        "height": np.nan,
        "file_size_bytes": np.nan,
        "jpeg_quality_estimate": np.nan,
        "metadata_error": "",
    })
    if not path.is_file():
        result["metadata_error"] = "original file missing"
        return result

    result["file_size_bytes"] = path.stat().st_size
    try:
        with Image.open(path) as image:
            result.update(_exif_mapping(image))
            result["width"], result["height"] = image.size
            if image.format == "JPEG":
                result["jpeg_quality_estimate"] = estimate_jpeg_quality(image)
    except (OSError, ValueError, SyntaxError) as exc:
        result["metadata_error"] = f"{type(exc).__name__}: {exc}"
        return result

    iso_value = result.get("ISOSpeedRatings")
    if isinstance(iso_value, (tuple, list)):
        iso_value = iso_value[0] if iso_value else None
    result["iso"] = _number(iso_value)
    result["exposure_time_s"] = _number(result.get("ExposureTime"))
    result["f_number"] = _number(result.get("FNumber"))
    result["brightness_value"] = _number(result.get("BrightnessValue"))
    result["focal_length_mm"] = _number(result.get("FocalLength"))

    flash = _number(result.get("Flash"))
    if np.isfinite(flash):
        result["flash_fired"] = int(int(flash) & 1)
        result["flash_status"] = "on" if result["flash_fired"] else "off"
    else:
        result["flash_fired"] = np.nan
        result["flash_status"] = "unknown"

    timestamp = _text(result.get("DateTimeOriginal"))
    result["capture_date"] = timestamp.split(" ", 1)[0] if timestamp else "unknown"
    make, model = _text(result.get("Make")), _text(result.get("Model"))
    camera = " ".join(value for value in (make, model) if value)
    result["camera"] = camera or "unknown"

    pixels = result["width"] * result["height"]
    result["pixel_count"] = pixels
    result["bits_per_pixel"] = (
        8 * result["file_size_bytes"] / pixels
        if pixels and result["file_size_bytes"]
        else np.nan
    )
    return result


def audit_manifest(manifest: pd.DataFrame, root: Path = ROOT) -> pd.DataFrame:
    """Join each manifest row with metadata from its original image path."""
    if "orig_path" not in manifest:
        raise ValueError("manifest must include orig_path for EXIF audit")
    records = [
        read_image_audit(resolve_manifest_path(row.orig_path, root))
        for row in manifest.itertuples(index=False)
    ]
    metadata = pd.DataFrame(records, index=manifest.index)
    return pd.concat([manifest.copy(), metadata], axis=1)


def _safe_crosstab(
    data: pd.DataFrame,
    index: list[str],
    columns: str = "class_name",
    count_groups: bool = False,
) -> pd.DataFrame:
    if count_groups:
        data = data.drop_duplicates(["group", *index])
    return pd.crosstab([data[name] for name in index], data[columns], dropna=False)


def add_feature_columns(audit: pd.DataFrame) -> pd.DataFrame:
    """Add normalized categorical and colour feature columns for baselines."""
    result = audit.copy()
    result["class_name"] = result["label"].map({0: "real", 1: "fake"})
    result["iso_bucket"] = pd.cut(
        pd.to_numeric(result["iso"], errors="coerce"),
        bins=[-np.inf, 100, 400, 800, np.inf],
        labels=["<=100", "101-400", "401-800", ">800"],
    ).astype(object).fillna("missing")
    result["flash_status"] = result["flash_status"].fillna("unknown")
    result["camera"] = result["camera"].fillna("unknown").replace("", "unknown")
    result["capture_date"] = (
        result["capture_date"].fillna("unknown").replace("", "unknown")
    )

    colour = color_baseline.features(result["path"].tolist())
    names = BASELINE_FEATURES["colour_mean_std"]
    result[names] = colour
    return result


def print_crosstabs(audit: pd.DataFrame) -> None:
    tables = [
        ("Camera x class", _safe_crosstab(audit, ["camera"])),
        (
            "Flash x camera x class",
            _safe_crosstab(audit, ["flash_status", "camera"]),
        ),
        (
            "Date x camera x class",
            _safe_crosstab(audit, ["capture_date", "camera"]),
        ),
        (
            "ISO bucket x camera x class",
            _safe_crosstab(audit, ["iso_bucket", "camera"]),
        ),
    ]
    for title, table in tables:
        print(f"\n=== {title} (image counts) ===")
        print(table.to_string())
        sample_table = _safe_crosstab(
            audit, table.index.names, count_groups=True
        )
        print(f"\n=== {title} (distinct sample counts) ===")
        print(sample_table.to_string())


def _classifier(feature_names: list[str], data: pd.DataFrame):
    categorical = [
        feature for feature in feature_names
        if not pd.api.types.is_numeric_dtype(data[feature])
    ]
    numeric = [feature for feature in feature_names if feature not in categorical]
    transformers = []
    if numeric:
        transformers.append((
            "numeric",
            make_pipeline(SimpleImputer(strategy="median")),
            numeric,
        ))
    if categorical:
        transformers.append((
            "categorical",
            make_pipeline(
                SimpleImputer(strategy="most_frequent"),
                OneHotEncoder(handle_unknown="ignore"),
            ),
            categorical,
        ))
    model = ColumnTransformer(transformers, remainder="drop")
    return make_pipeline(
        model,
        RandomForestClassifier(
            n_estimators=100,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=1,
        ),
    )


def _grouped_sample_metrics(
    rows: pd.DataFrame, probability: np.ndarray
) -> dict[str, float]:
    probs, labels = wombat_beam.aggregate_by_group(
        probability, rows["label"].to_numpy(), rows["group"].astype(str).to_numpy()
    )
    if np.unique(labels).size < 2:
        return {
            "auc": np.nan,
            "accuracy": np.nan,
            "balanced_accuracy": np.nan,
            "n_samples": len(labels),
        }
    predicted = probability_to_label(probs)
    return {
        "auc": float(roc_auc_score(labels, probs)),
        "accuracy": float(accuracy_score(labels, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "n_samples": int(len(labels)),
    }


def probability_to_label(probability: np.ndarray, threshold: float = 0.5):
    return (np.asarray(probability) >= threshold).astype(int)


def run_grouped_baselines(
    audit: pd.DataFrame,
    seeds: tuple[int, ...] = (42, 43, 44, 45, 46),
    max_splits: int = 5,
) -> pd.DataFrame:
    """Evaluate RF shortcut baselines with group-separated folds, pooled/by camera."""
    results: list[dict[str, Any]] = []
    scopes: list[tuple[str, pd.DataFrame]] = [("pooled", audit)]
    for camera, camera_data in audit.groupby("camera", sort=True):
        scopes.append((str(camera), camera_data.reset_index(drop=True)))

    for scope, scoped in scopes:
        scoped = scoped.reset_index(drop=True)
        y = scoped["label"].astype(int).to_numpy()
        groups = scoped["group"].astype(str).to_numpy()
        class_group_counts = scoped.groupby("label")["group"].nunique()
        if len(class_group_counts) != 2:
            print(f"Skipping scope={scope}: only one label present.")
            continue
        n_splits = min(max_splits, int(class_group_counts.min()))
        if n_splits < 2:
            print(f"Skipping scope={scope}: fewer than two groups per class.")
            continue

        for feature_set, feature_names in BASELINE_FEATURES.items():
            available = scoped[feature_names]
            usable = available.notna().any(axis=1)
            if feature_set == "exposure_iso_flash":
                numeric_features = [
                    name for name in feature_names
                    if name != "flash_status"
                ]
                usable = (
                    scoped["original_exists"].astype(bool)
                    & (
                        scoped[numeric_features].notna().any(axis=1)
                        | scoped["flash_status"].ne("unknown")
                    )
                )
            elif feature_set == "flash_only":
                usable = (
                    scoped["original_exists"].astype(bool)
                    & scoped["flash_status"].ne("unknown")
                )
            elif feature_set == "size_file_bpp":
                usable &= scoped["original_exists"].astype(bool)
            if not usable.all():
                omitted = int((~usable).sum())
                if feature_set not in ("exposure_iso_flash", "size_file_bpp"):
                    continue
                subset = scoped.loc[usable].reset_index(drop=True)
            else:
                subset = scoped

            labels = subset["label"].astype(int).to_numpy()
            subset_groups = subset["group"].astype(str).to_numpy()
            per_class_groups = subset.groupby("label")["group"].nunique()
            subset_splits = min(n_splits, int(per_class_groups.min()))
            if subset_splits < 2:
                continue
            if not usable.all():
                print(
                    f"Baseline {feature_set}, scope={scope}: omitted {omitted} "
                    "rows with all selected features missing."
                )

            for seed in seeds:
                splitter = StratifiedGroupKFold(
                    n_splits=subset_splits, shuffle=True, random_state=seed
                )
                for fold, (train_idx, test_idx) in enumerate(
                    splitter.split(subset, labels, subset_groups)
                ):
                    train, test = subset.iloc[train_idx], subset.iloc[test_idx]
                    if train["label"].nunique() < 2 or test["label"].nunique() < 2:
                        print(
                            f"Skipping one-class fold: scope={scope}, "
                            f"feature_set={feature_set}, seed={seed}, fold={fold}"
                        )
                        continue
                    model = _classifier(feature_names, subset)
                    model.set_params(
                        randomforestclassifier__random_state=seed * 100 + fold
                    )
                    model.fit(train[feature_names], train["label"].astype(int))
                    fake_column = list(model.classes_).index(1)
                    probability = model.predict_proba(test[feature_names])[:, fake_column]
                    metrics = _grouped_sample_metrics(test, probability)
                    results.append({
                        "scope": scope,
                        "feature_set": feature_set,
                        "seed": seed,
                        "fold": fold,
                        **metrics,
                    })

    raw = pd.DataFrame(results)
    if raw.empty:
        return raw
    summary = (
        raw.groupby(["scope", "feature_set"], as_index=False)
        .agg(
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
            runs=("fold", "count"),
            samples_per_run=("n_samples", "mean"),
        )
    )
    summary["seeds"] = ",".join(str(seed) for seed in seeds)
    summary["n_splits_max"] = max_splits
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/manifest.csv")
    parser.add_argument("--audit-out", default="data/audit_exif.csv")
    parser.add_argument("--summary-out", default="runs/audit/shortcut_baselines.csv")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    manifest_path = resolve_manifest_path(args.manifest)
    manifest = pd.read_csv(manifest_path)
    required = {"path", "label", "group", "orig_path"}
    missing = required - set(manifest.columns)
    if missing:
        parser.error(f"manifest is missing required columns: {sorted(missing)}")

    audit = audit_manifest(manifest)
    missing_originals = int((~audit["original_exists"]).sum())
    print(
        f"Manifest: {len(manifest)} rows, {manifest.group.nunique()} groups. "
        f"Original files missing: {missing_originals}/{len(manifest)}."
    )
    if missing_originals:
        print(
            "Missing originals remain explicitly marked; their EXIF and "
            "original-image metadata are not inferred from working copies."
        )

    audit = add_feature_columns(audit)
    audit_path = resolve_manifest_path(args.audit_out)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(audit_path, index=False)
    print(f"Wrote EXIF audit: {audit_path}")
    print_crosstabs(audit)

    summary = run_grouped_baselines(
        audit,
        seeds=tuple(args.seeds),
        max_splits=args.folds,
    )
    summary_path = resolve_manifest_path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    print(f"\nWrote grouped-CV shortcut baselines: {summary_path}")
    if not summary.empty:
        print(summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()
