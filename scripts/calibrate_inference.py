"""Build a versioned inference calibration file from complete OOF results."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import wombat_beam as wb
from inference_safety import (
    conformal_quantile, group_coverage_ci, model_bundle_fingerprint,
    prediction_set,
)
from manifest_paths import (
    normalize_manifest_paths, validate_prediction_manifest_identity,
)
from build_report import eligible_run

PROPOSED = "PROPOSED: Wombat-OA + weighted beam + log-prob fusion"


def collect_oof(run_dir: Path, manifest_path: Path, audit_path: Path):
    manifest = normalize_manifest_paths(pd.read_csv(manifest_path))
    audit = normalize_manifest_paths(pd.read_csv(audit_path), columns=("path",))
    identity = manifest[["path", "group"]].merge(
        audit[["path", "group"]], on="path", how="outer",
        suffixes=("_manifest", "_audit"), validate="one_to_one", indicator=True,
    )
    if not identity._merge.eq("both").all() or not identity.group_manifest.eq(identity.group_audit).all():
        raise ValueError("audit image/group identity does not match the manifest")
    parts = []
    uncertainties = []
    for fold in range(int(manifest.fold.max()) + 1):
        fold_dir = run_dir / f"fold{fold}"
        pred_path = fold_dir / "preds_test.csv"
        cache_path = fold_dir / "cache_test.npz"
        if not pred_path.is_file() or not cache_path.is_file():
            raise FileNotFoundError(f"fold {fold} is missing OOF predictions or cache")
        pred = pd.read_csv(pred_path)
        if PROPOSED not in pred:
            raise ValueError(f"{pred_path} lacks the proposed model predictions")
        pred["fold"] = fold
        parts.append(pred)
        cache = wb.Cache.load(cache_path)
        if cache.paths is None or not np.array_equal(cache.paths.astype(str), pred.path.astype(str).to_numpy()):
            raise ValueError(f"fold {fold} cache paths do not align with OOF predictions")
        uncertainties.extend(zip(cache.groups.astype(str), cache.U.mean(axis=(1, 2))))
    oof = pd.concat(parts, ignore_index=True)
    oof = normalize_manifest_paths(oof, columns=("path",))
    validate_prediction_manifest_identity(manifest, oof)
    if oof.groupby("group").fold.nunique().gt(1).any():
        raise ValueError("OOF groups occur in more than one test fold")
    metadata = manifest[["path", "label", "group"]].merge(
        audit[["path", "camera", "iso", "flash_status", "capture_date"]],
        on="path", how="left", validate="one_to_one",
    )
    oof = oof.merge(metadata, on=["path", "group"], how="left", validate="one_to_one")
    samples = []
    for group, rows in oof.groupby("group", sort=True):
        p = np.clip(rows[PROPOSED].to_numpy(float), 1e-6, 1 - 1e-6)
        samples.append({
            "group": str(group), "y": int(rows.y.iloc[0]),
            "p_fake": float(1 / (1 + np.exp(-np.log(p / (1 - p)).mean()))),
            "camera": rows.camera.iloc[0], "n_images": len(rows),
        })
    uncertainty_frame = pd.DataFrame(uncertainties, columns=["group", "mc_uncertainty"])
    uncertainty_by_group = uncertainty_frame.groupby("group").mc_uncertainty.mean()
    return pd.DataFrame(samples), metadata, uncertainty_by_group


def _support_config(metadata: pd.DataFrame) -> dict:
    group_meta = metadata.groupby(["camera", "label", "group"], as_index=False).agg(
        iso=("iso", "median"), flash=("flash_status", "first")
    )
    cameras: dict[str, dict] = {}
    for camera, part in group_meta.groupby("camera", dropna=False):
        key = str(camera) if pd.notna(camera) else "unknown"
        by_class = {}
        for label, rows in part.groupby("label"):
            iso = pd.to_numeric(rows.iso, errors="coerce").dropna()
            by_class[str(int(label))] = {
                "iso": {
                    "min": float(iso.min()) if len(iso) else None,
                    "max": float(iso.max()) if len(iso) else None,
                },
                "flash": sorted(rows.flash.fillna("unknown").astype(str).unique().tolist()),
            }
        cameras[key] = {
            "iso_by_class": {
                label: values["iso"] for label, values in by_class.items()
            },
            "flash_by_class": {
                label: values["flash"] for label, values in by_class.items()
            },
        }
    return {"cameras": cameras}


def calibration_outputs(
    samples: pd.DataFrame,
    metadata: pd.DataFrame,
    uncertainty,
    *,
    data_version: str,
    model_version: str,
) -> tuple[dict, pd.DataFrame]:
    y = samples.y.to_numpy(int)
    p = samples.p_fake.to_numpy(float)
    scores = np.where(y == 1, 1 - p, p)
    groups = samples.group.astype(str).to_numpy()
    quantiles = {}
    mondrian = {}
    rows = []
    for alpha in (0.05, 0.10):
        q = conformal_quantile(scores, alpha)
        key = f"{alpha:.2f}"
        quantiles[key] = q
        mondrian[key] = {
            str(label): conformal_quantile(scores[y == label], alpha)
            for label in (0, 1) if np.any(y == label)
        }
        for camera, subset in samples.groupby("camera", dropna=False):
            truth = subset.y.to_numpy(int)
            probs = subset.p_fake.to_numpy(float)
            include = np.asarray([
                truth_i in prediction_set(prob_i, q)
                for truth_i, prob_i in zip(truth, probs)
            ])
            singletons = np.asarray([
                len(prediction_set(prob_i, q)) == 1 for prob_i in probs
            ])
            ci = group_coverage_ci(include, subset.group, seed=42)
            rows.append({
                "calibration": "pooled", "alpha": alpha,
                "camera": camera, "n_samples": subset.group.nunique(),
                "n_images": int(subset.n_images.sum()),
                "coverage": float(include.mean()),
                "coverage_lower_95": ci[0], "coverage_upper_95": ci[1],
                "abstention_rate": float(1 - singletons.mean()),
            })
        for label in (0, 1):
            subset = samples[samples.y == label]
            q_class = mondrian[key].get(str(label))
            if q_class is None:
                continue
            for camera, camera_subset in subset.groupby("camera", dropna=False):
                truth = camera_subset.y.to_numpy(int)
                probs = camera_subset.p_fake.to_numpy(float)
                include = np.asarray([
                    truth_i in prediction_set(prob_i, q_class)
                    for truth_i, prob_i in zip(truth, probs)
                ])
                singleton = np.asarray([
                    len(prediction_set(prob_i, q_class)) == 1 for prob_i in probs
                ])
                ci = group_coverage_ci(include, camera_subset.group, seed=42)
                rows.append({
                    "calibration": f"mondrian_class_{label}", "alpha": alpha,
                    "camera": camera, "n_samples": camera_subset.group.nunique(),
                    "n_images": int(camera_subset.n_images.sum()),
                    "coverage": float(include.mean()),
                    "coverage_lower_95": ci[0], "coverage_upper_95": ci[1],
                    "abstention_rate": float(1 - singleton.mean()),
                })
    mc_by_sample = uncertainty.reindex(samples.group.astype(str)).to_numpy(dtype=float)
    finite_mc = mc_by_sample[np.isfinite(mc_by_sample)]
    report = pd.DataFrame(rows)
    config = {
        "version": "1.0",
        "data_version": data_version,
        "model_version": model_version,
        "method": "group-level out-of-fold conformal calibration",
        "calibration_unit": "one point per physical sample (manifest group)",
        "nonconformity": "1 - p(true class)",
        "quantile_index": "ceil((n+1)*(1-alpha)); if index exceeds n, quantile is infinity",
        "alpha_default": 0.05,
        "alpha_sensitivity": 0.10,
        "quantiles": quantiles,
        "mondrian_quantiles_report_only": mondrian,
        "max_mc_uncertainty": float(np.quantile(finite_mc, 0.99)) if len(finite_mc) else None,
        "capture_support": _support_config(metadata),
        "coverage_limitation": (
            "Empirical calibration coverage is measured on group-level OOF predictions. "
            "The conformal coverage guarantee applies only to data similar to the calibration data "
            "and does not cover new phones or lighting conditions."
        ),
    }
    return config, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="runs/cv_repro")
    parser.add_argument("--manifest", default="data/manifest.csv")
    parser.add_argument("--audit", default="data/audit_exif.csv")
    args = parser.parse_args()
    resolve = lambda value: Path(value) if Path(value).is_absolute() else ROOT / value
    run_dir = resolve(args.run)
    is_eligible, reason = eligible_run(run_dir)
    if not is_eligible:
        raise ValueError(f"refusing to calibrate from ineligible run {run_dir}: {reason}")
    manifest_path = resolve(args.manifest)
    model_version = model_bundle_fingerprint(run_dir, manifest_path)
    data_version = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    samples, metadata, uncertainty = collect_oof(
        run_dir, manifest_path, resolve(args.audit)
    )
    config, report = calibration_outputs(
        samples, metadata, uncertainty,
        data_version=data_version, model_version=model_version,
    )
    (run_dir / "inference_calibration.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    report_path = run_dir / "inference_calibration_report.csv"
    report.to_csv(report_path, index=False)
    print(report.to_string(index=False))
    print(f"Wrote calibration: {run_dir / 'inference_calibration.json'}")
    print(f"Wrote empirical coverage report: {report_path}")


if __name__ == "__main__":
    main()
