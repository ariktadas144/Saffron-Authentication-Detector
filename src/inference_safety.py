"""Group-aware calibration and abstention helpers for single-sample inference."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np


def model_bundle_fingerprint(run_dir: str | Path, manifest_path: str | Path) -> str:
    """Hash the data manifest and the exact decoder/model artifacts used at inference."""
    root = Path(run_dir)
    digest = hashlib.sha256()
    paths = [Path(manifest_path), root / "cv_run.json"]
    if not (root / "cv_run.json").is_file():
        raise FileNotFoundError(f"missing model run provenance: {root / 'cv_run.json'}")
    import json

    metadata = json.loads((root / "cv_run.json").read_text(encoding="utf-8"))
    fold_count = int(metadata.get("fold_count", 0))
    if fold_count < 1:
        raise ValueError("model run provenance has no valid fold_count")
    for fold in range(fold_count):
        fold_dir = root / f"fold{fold}"
        weights = next(
            (fold_dir / name for name in ("final.weights.h5", "best.weights.h5")
             if (fold_dir / name).is_file()),
            None,
        )
        if weights is None:
            raise FileNotFoundError(f"fold {fold} is missing saved model weights")
        paths.extend([
            fold_dir / "config.json", weights, fold_dir / "wombat_params.json",
            fold_dir / "thresholds.json", fold_dir / "sample_thresholds.json",
            fold_dir / "ood_reference.npz",
        ])
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"model bundle fingerprint input is missing: {path}")
        name = str(path.relative_to(root) if path.is_relative_to(root) else path.name)
        digest.update(name.encode("utf-8") + b"\0")
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def conformal_quantile(nonconformity: Any, alpha: float) -> float:
    """Finite-sample split-conformal order statistic for sample-level scores."""
    scores = np.asarray(nonconformity, dtype=float).reshape(-1)
    scores = scores[np.isfinite(scores)]
    if not len(scores):
        raise ValueError("conformal calibration requires at least one finite score")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between 0 and 1")
    k = math.ceil((len(scores) + 1) * (1 - alpha))
    if k > len(scores):
        return float("inf")
    return float(np.sort(scores)[k - 1])


def prediction_set(p_fake: float, quantile: float) -> tuple[int, ...]:
    """Return class labels included by the calibrated nonconformity threshold."""
    if not np.isfinite(p_fake) or not 0 <= p_fake <= 1:
        raise ValueError("p_fake must be a finite probability")
    return tuple(
        label for label, probability in ((0, 1 - p_fake), (1, p_fake))
        if 1 - probability <= quantile
    )


def assess_capture_support(
    metadata: dict[str, Any],
    support: dict[str, Any],
    class_candidate: int | None = None,
) -> str | None:
    """Return an abstention reason when capture metadata is unsupported."""
    camera = str(metadata.get("camera") or "unknown")
    camera_support = support.get("cameras", {}).get(camera)
    if camera_support is None:
        return f"camera model {camera!r} is outside calibration support"
    iso = metadata.get("iso")
    if iso is not None:
        ranges = camera_support.get("iso_by_class", {})
        class_ranges = (
            {str(class_candidate): ranges.get(str(class_candidate))}
            if class_candidate is not None else ranges
        )
        if ranges and not any(
            bounds is not None
            and bounds.get("min") is not None
            and bounds.get("max") is not None
            and bounds["min"] <= float(iso) <= bounds["max"]
            for bounds in class_ranges.values()
        ):
            return f"ISO {iso} is outside observed support for {camera}"
    flash = metadata.get("flash_status")
    if flash and flash != "unknown":
        observed = camera_support.get("flash_by_class", {})
        if observed and any(
            flash not in values for values in observed.values()
        ):
            return f"flash state {flash!r} was not observed for every class on {camera}"
    return None


def decide(
    p_fake: float,
    quantile: float,
    *,
    fold_probabilities: Any,
    fold_thresholds: Any,
    mc_uncertainty: float,
    max_mc_uncertainty: float,
    ood_distance: float | None,
    max_ood_distance: float | None,
    capture_reason: str | None,
    shot_count: int,
) -> dict[str, Any]:
    """Apply fail-closed inference checks and return the three-state verdict."""
    reasons: list[str] = []
    if shot_count < 3:
        reasons.append("fewer than 3 shots were supplied for this sample")
    fold_p = np.asarray(fold_probabilities, dtype=float).reshape(-1)
    fold_t = np.asarray(fold_thresholds, dtype=float).reshape(-1)
    if len(fold_p) == 0 or len(fold_p) != len(fold_t):
        reasons.append("fold probabilities or thresholds are unavailable")
    elif len(np.unique(fold_p >= fold_t)) > 1:
        reasons.append("fold models disagree on the verdict")
    if not np.isfinite(mc_uncertainty) or mc_uncertainty > max_mc_uncertainty:
        reasons.append("MC uncertainty exceeds calibrated support")
    if (
        ood_distance is None
        or max_ood_distance is None
        or not np.isfinite(ood_distance)
        or ood_distance > max_ood_distance
    ):
        reasons.append("embedding OOD distance is unavailable or exceeds support")
    if capture_reason:
        reasons.append(capture_reason)
    included = prediction_set(float(p_fake), float(quantile))
    if len(included) != 1:
        reasons.append("conformal prediction set is not a single class")
    label = (
        "FAKE" if included == (1,) else "REAL"
        if included == (0,) else "INCONCLUSIVE"
    )
    if reasons:
        label = "INCONCLUSIVE"
    return {
        "label": label,
        "prediction_set": included,
        "reasons": reasons,
    }


def group_coverage_ci(
    covered: Any,
    groups: Any,
    *,
    n_bootstrap: int = 5000,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap a coverage proportion by physical-sample group."""
    values = np.asarray(covered, dtype=float).reshape(-1)
    ids = np.asarray(groups).astype(str).reshape(-1)
    if len(values) != len(ids):
        raise ValueError("covered and groups must have the same length")
    unique = np.unique(ids)
    if not len(unique):
        raise ValueError("coverage interval requires at least one group")
    per_group = np.asarray([values[ids == group].mean() for group in unique])
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        estimates[index] = rng.choice(per_group, len(per_group), replace=True).mean()
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))
