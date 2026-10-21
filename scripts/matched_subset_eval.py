"""Evaluate saved out-of-fold predictions on capture-matched subsets."""
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

PROPOSED = "PROPOSED: Wombat-OA + weighted beam + log-prob fusion"
EXIF = "exposure iso flash random forest (shortcut)"


def load_oof(run_dir: Path, manifest_path: Path, audit_path: Path) -> pd.DataFrame:
    manifest = normalize_manifest_paths(pd.read_csv(manifest_path))
    audit = normalize_manifest_paths(pd.read_csv(audit_path), columns=("path",))
    if not manifest[["path", "group"]].merge(
        audit[["path", "group"]], on=["path", "group"], how="outer",
        validate="one_to_one", indicator=True,
    )._merge.eq("both").all():
        raise ValueError("audit image/group identity does not match the manifest")
    frames = []
    for fold in range(int(manifest.fold.max()) + 1):
        fold_dir = run_dir / f"fold{fold}"
        pred_path = fold_dir / "preds_test.csv"
        thresholds_path = fold_dir / "sample_thresholds.json"
        if not pred_path.is_file() or not thresholds_path.is_file():
            raise FileNotFoundError(f"fold {fold} OOF predictions or thresholds are missing")
        pred = normalize_manifest_paths(pd.read_csv(pred_path), columns=("path",))
        thresholds = json.loads(thresholds_path.read_text(encoding="utf-8"))
        if set((PROPOSED, EXIF)) - set(pred.columns):
            raise ValueError(f"{pred_path} lacks proposed or EXIF baseline predictions")
        pred["sample_threshold"] = float(thresholds[PROPOSED])
        pred["exif_threshold"] = float(thresholds[EXIF])
        pred["fold"] = fold
        frames.append(pred)
    result = pd.concat(frames, ignore_index=True)
    validate_prediction_manifest_identity(manifest, result)
    meta = manifest[["path", "type"]].merge(
        audit[["path", "camera", "flash_status", "iso", "capture_date"]],
        on="path", how="left", validate="one_to_one",
    )
    result = result.merge(meta, on="path", how="left", validate="one_to_one")
    if result.groupby("group").fold.nunique().gt(1).any():
        raise ValueError("an OOF sample is present in multiple held-out folds")
    return result


def _shift(p: np.ndarray, threshold: float) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    t = float(np.clip(threshold, 1e-6, 1 - 1e-6))
    return 1 / (1 + np.exp(-(np.log(p / (1 - p)) - np.log(t / (1 - t)))))


def sample_predictions(image_rows: pd.DataFrame) -> pd.DataFrame:
    data = image_rows.copy()
    # Threshold shifting must be fold-specific before aggregating each sample.
    data["proposed_calibrated"] = [
        _shift(np.asarray([p]), t)[0]
        for p, t in zip(data[PROPOSED], data.sample_threshold)
    ]
    data["exif_calibrated"] = [
        _shift(np.asarray([p]), t)[0]
        for p, t in zip(data[EXIF], data.exif_threshold)
    ]
    rows = []
    for group, part in data.groupby("group", sort=True):
        row = {
            "group": str(group), "y": int(part.y.iloc[0]),
            "camera": part.camera.iloc[0], "flash_status": part.flash_status.iloc[0],
            "iso": pd.to_numeric(part.iso, errors="coerce").median(),
            "capture_date": part.capture_date.iloc[0], "type": part.type.iloc[0],
            "n_images": len(part),
        }
        for method in ("proposed_calibrated", "exif_calibrated"):
            q = np.clip(part[method].to_numpy(float), 1e-6, 1 - 1e-6)
            row[method] = float(1 / (1 + np.exp(-np.log(q / (1 - q)).mean())))
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_subsets(samples: pd.DataFrame, n_bootstrap: int = 5000) -> pd.DataFrame:
    scopes: list[tuple[str, pd.DataFrame]] = []
    g973f = samples[samples.camera.astype(str).str.contains("SM-G973F", na=False)]
    iso = pd.to_numeric(g973f.iso, errors="coerce")
    scopes.append((
        "SM-G973F ISO 100/125/160, flash off",
        g973f[iso.isin([100, 125, 160]) & g973f.flash_status.eq("off")],
    ))
    scopes.extend(
        (f"camera={camera}", subset)
        for camera, subset in samples.groupby("camera", dropna=False)
    )
    scopes.extend(
        (f"capture_date={date}", subset)
        for date, subset in samples.groupby("capture_date", dropna=False)
    )
    rows = []
    for scope, subset in scopes:
        class_counts = subset.groupby("y").group.nunique().to_dict()
        for method, display in (
            ("proposed_calibrated", "Proposed"),
            ("exif_calibrated", "EXIF-only baseline"),
        ):
            row = {
                "scope": scope, "method": display,
                "n_samples": int(subset.group.nunique()),
                "n_images": int(subset.n_images.sum()),
                "n_fake": int(class_counts.get(1, 0)),
                "n_real": int(class_counts.get(0, 0)),
            }
            if len(class_counts) < 2:
                row.update(status="one class only; metrics not estimable", auc=np.nan, accuracy=np.nan)
                rows.append(row)
                continue
            y = subset.y.to_numpy(int)
            p = subset["proposed_calibrated" if method == "proposed_calibrated" else "exif_calibrated"].to_numpy(float)
            values = wb.metrics(y, p, 0.5)
            ci = wb.bootstrap_metrics_ci(
                y, p, 0.5, n=n_bootstrap, seed=42, groups=subset.group.to_numpy()
            )
            row.update(
                status="ok", auc=values["auc"], accuracy=values["acc"],
                auc_lower_95=ci.get("auc", (np.nan, np.nan))[0],
                auc_upper_95=ci.get("auc", (np.nan, np.nan))[1],
                accuracy_lower_95=ci.get("acc", (np.nan, np.nan))[0],
                accuracy_upper_95=ci.get("acc", (np.nan, np.nan))[1],
            )
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="runs/cv_repro")
    parser.add_argument("--manifest", default="data/manifest.csv")
    parser.add_argument("--audit", default="data/audit_exif.csv")
    parser.add_argument("--out", default="runs/audit/matched_subset_results.csv")
    args = parser.parse_args()
    paths = [Path(value) for value in (args.run, args.manifest, args.audit, args.out)]
    paths = [path if path.is_absolute() else ROOT / path for path in paths]
    rows = sample_predictions(load_oof(paths[0], paths[1], paths[2]))
    result = evaluate_subsets(rows)
    paths[3].parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(paths[3], index=False)
    print(result.to_string(index=False))
    print(f"Wrote {paths[3]}")


if __name__ == "__main__":
    main()
