"""Path normalization for manifests moved between Windows and POSIX hosts."""
from __future__ import annotations

import os
import ntpath
from pathlib import Path
from typing import Any

import pandas as pd


def normalize_manifest_path(value: Any) -> str:
    """Normalize Windows/POSIX separators without changing relative-path meaning."""
    path = os.fspath(value).strip()
    if not path:
        raise ValueError("manifest path cannot be empty")
    normalized = path.replace("\\", os.sep).replace("/", os.sep)
    drive, _ = ntpath.splitdrive(normalized)
    if drive and os.name != "nt":
        raise ValueError(
            f"Windows drive path {path!r} cannot be resolved on this host; "
            "use a path relative to the repository or data root"
        )
    return os.path.normpath(normalized)


def normalize_manifest_paths(manifest, columns=("path", "orig_path")):
    """Return a copy with supported manifest path columns normalized."""
    result = manifest.copy()
    for column in columns:
        if column in result:
            result[column] = result[column].map(normalize_manifest_path)
    return result


def resolve_manifest_path(value: Any, root: str | Path) -> Path:
    """Resolve a manifest path relative to root unless it is already absolute."""
    path = Path(normalize_manifest_path(value))
    return path if path.is_absolute() else Path(root) / path


def validate_prediction_manifest_identity(
    manifest: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    label_column: str = "label",
    prediction_label_column: str = "y",
) -> None:
    """Require one exact prediction per manifest image, group, label, and fold."""
    required_manifest = {"path", "group", label_column, "fold"}
    required_predictions = {"path", "group", prediction_label_column, "fold"}
    if missing := required_manifest - set(manifest.columns):
        raise ValueError(f"manifest is missing identity columns: {sorted(missing)}")
    if missing := required_predictions - set(predictions.columns):
        raise ValueError(f"predictions are missing identity columns: {sorted(missing)}")
    expected = normalize_manifest_paths(manifest, columns=("path",))
    actual = normalize_manifest_paths(predictions, columns=("path",))
    if expected.path.duplicated().any():
        raise ValueError("manifest contains duplicate image paths")
    if actual.path.duplicated().any():
        raise ValueError("predictions contain duplicate image paths")
    if (
        expected.group.isna().any()
        or expected.group.astype(str).str.strip().eq("").any()
        or actual.group.isna().any()
        or actual.group.astype(str).str.strip().eq("").any()
    ):
        raise ValueError("manifest and predictions must have non-empty sample groups")
    expected = expected[["path", "group", label_column, "fold"]].rename(
        columns={"group": "manifest_group", label_column: "manifest_label", "fold": "manifest_fold"}
    )
    actual = actual[["path", "group", prediction_label_column, "fold"]].rename(
        columns={
            "group": "prediction_group",
            prediction_label_column: "prediction_label",
            "fold": "prediction_fold",
        }
    )
    aligned = expected.merge(
        actual, on="path", how="outer", validate="one_to_one", indicator=True
    )
    if not aligned._merge.eq("both").all():
        raise ValueError("prediction image paths do not exactly match the manifest")
    if not aligned.manifest_group.astype(str).eq(
        aligned.prediction_group.astype(str)
    ).all():
        raise ValueError("prediction groups do not match the manifest")
    manifest_labels = pd.to_numeric(aligned.manifest_label, errors="coerce")
    prediction_labels = pd.to_numeric(aligned.prediction_label, errors="coerce")
    manifest_folds = pd.to_numeric(aligned.manifest_fold, errors="coerce")
    prediction_folds = pd.to_numeric(aligned.prediction_fold, errors="coerce")
    if (
        manifest_labels.isna().any()
        or prediction_labels.isna().any()
        or not manifest_labels.eq(prediction_labels).all()
    ):
        raise ValueError("prediction labels do not match the manifest")
    if (
        manifest_folds.isna().any()
        or prediction_folds.isna().any()
        or not manifest_folds.eq(prediction_folds).all()
    ):
        raise ValueError("prediction fold assignments do not match the manifest")
