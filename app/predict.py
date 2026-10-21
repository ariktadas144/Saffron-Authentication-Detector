"""
predict.py — single-image inference for the Saffron real-vs-fake pipeline.

Primary path  (deep model):
    Loads ConvNeXt-Tiny + GNN + HAFM weights + Wombat-OA beam params from
    runs/cv1/fold<k>/ and runs the full extract_cache → wombat decode pipeline
    on one image.  All saved folds are averaged (ensemble).

Fallback path (colour baseline):
    If no weights are found, uses RGB+HSV colour statistics + a Random-Forest
    trained from the manifest.  This is the ~0.94 AUC baseline described in the
    README and is always available without GPU.

Usage (from the project root):
    from app.predict import load_bundle, predict_pil
    bundle = load_bundle("runs/cv1", "data/manifest.csv")
    result = predict_pil(pil_image, bundle)
    # result = {"label": "FAKE", "confidence": 0.91, "method": "deep", ...}
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import ExifTags, Image

# ── make src/ importable ────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

COLOUR_FEATURE_NAMES = (
    "R mean", "G mean", "B mean", "R std", "G std", "B std",
    "H mean", "S mean", "V mean", "H std", "S std", "V std",
)

from inference_safety import (
    assess_capture_support, decide, model_bundle_fingerprint,
)


# ────────────────────────────────────────────────────────────────────────────
# Colour-statistics baseline (always available, no TF needed)
# ────────────────────────────────────────────────────────────────────────────
def _colour_features_pil(img: Image.Image) -> np.ndarray:
    """12-D colour feature vector: RGB mean/std + HSV mean/std on a 64×64 crop."""
    img = img.convert("RGB")
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2,
                    (w - s) // 2 + s, (h - s) // 2 + s)).resize(
                        (64, 64), Image.Resampling.BILINEAR)
    a = np.asarray(img, float) / 255.0
    v = np.asarray(img.convert("HSV"), float) / 255.0
    return np.array([*a.mean((0, 1)), *a.std((0, 1)),
                     *v.mean((0, 1)), *v.std((0, 1))], dtype=np.float32)


def _train_colour_baseline(manifest_path: str):
    """Train a Random-Forest on ALL manifest images (not CV-safe but fine for demo inference)."""
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier

    man = pd.read_csv(manifest_path)
    if not {"path", "label"}.issubset(man.columns) or man.empty:
        raise ValueError(f"Invalid colour-baseline manifest: {manifest_path}")
    if man["label"].nunique() != 2:
        raise ValueError(f"Colour-baseline manifest must contain both labels: {manifest_path}")

    manifest_dir = Path(manifest_path).resolve().parent
    image_paths = []
    for value in man.path:
        path = Path(str(value))
        if not path.is_absolute():
            relative_to_manifest = manifest_dir / path
            path = relative_to_manifest if relative_to_manifest.exists() else _ROOT / path
        image_paths.append(path)

    X = []
    for path in image_paths:
        with Image.open(path) as image:
            X.append(_colour_features_pil(image))
    X = np.asarray(X, dtype=np.float32)
    clf = RandomForestClassifier(300, min_samples_leaf=2,
                                 random_state=42, class_weight="balanced")
    clf.fit(X, man.label.to_numpy())
    return clf


# ────────────────────────────────────────────────────────────────────────────
# Bundle: everything needed to run inference
# ────────────────────────────────────────────────────────────────────────────
class InferenceBundle:
    """Holds either deep-model artefacts or a colour-baseline classifier."""

    def __init__(self):
        self.method: str = "colour"          # "deep" | "colour"
        self.folds: list[dict] = []          # deep: list of {M, cfg, wparams, thr}
        self.colour_clf = None               # colour: sklearn classifier
        self.img_size: int = 256
        self.mc_samples: int = 0
        self.fallback_reason: Optional[str] = None
        self.decoder_tuned: bool = False
        self.calibration: Optional[dict] = None
        self.run_metadata: Optional[dict] = None


def load_bundle(run_dir: str = "runs/cv1",
                manifest_path: str = "data/manifest.csv") -> InferenceBundle:
    """
    Attempt to load deep-model weights from run_dir/fold*/
    Falls back to colour baseline automatically.
    """
    bundle = InferenceBundle()
    run_dir = str(Path(run_dir).resolve())
    run_metadata_path = Path(run_dir) / "cv_run.json"
    if run_metadata_path.is_file():
        with run_metadata_path.open(encoding="utf-8") as stream:
            bundle.run_metadata = json.load(stream)
    calibration_path = Path(run_dir) / "inference_calibration.json"
    if calibration_path.is_file():
        with calibration_path.open(encoding="utf-8") as stream:
            bundle.calibration = json.load(stream)
        try:
            model_version = model_bundle_fingerprint(run_dir, manifest_path)
            data_version = hashlib.sha256(
                Path(manifest_path).resolve().read_bytes()
            ).hexdigest()
        except (FileNotFoundError, ValueError) as exc:
            bundle.calibration = None
            bundle.fallback_reason = f"Inference calibration cannot be verified: {exc}"
        else:
            if (
                bundle.calibration.get("model_version") != model_version
                or bundle.calibration.get("data_version") != data_version
            ):
                bundle.calibration = None
                bundle.fallback_reason = (
                    "Inference calibration does not match the current model bundle "
                    "and manifest"
                )
    else:
        bundle.fallback_reason = (
            f"Inference calibration is unavailable: {calibration_path}"
        )

    # ── Try loading deep-model folds ────────────────────────────────────────
    fold_dirs = sorted(
        d for d in (Path(run_dir).iterdir() if Path(run_dir).exists() else [])
        if d.is_dir() and d.name.startswith("fold")
    )
    loaded = 0
    loaded_fold_ids = []
    for fd in fold_dirs:
        wf = next((fd / n for n in ("final.weights.h5", "best.weights.h5")
                   if (fd / n).exists()), None)
        wp = fd / "wombat_params.json"
        th = fd / "thresholds.json"
        cfg_f = fd / "config.json"
        if not wf:
            continue
        if not cfg_f.exists():
            bundle.fallback_reason = (
                f"{fd.name} has model weights but is missing config.json"
            )
            continue

        import saffron_cnn_gnn as sg
        import wombat_beam as wb

        with cfg_f.open(encoding="utf-8") as stream:
            raw = json.load(stream)
        cfg = sg.Config(**{f.name: raw[f.name]
                           for f in dataclasses.fields(sg.Config)
                           if f.name in raw})
        cfg.imagenet = False
        M = sg.build_models(cfg)
        M["model"].load_weights(str(wf))
        decoder_tuned = wp.exists() and th.exists()
        if decoder_tuned:
            with wp.open(encoding="utf-8") as stream:
                wparams = json.load(stream)
            with th.open(encoding="utf-8") as stream:
                thresholds = json.load(stream)
        else:
            H, V = len(sg.HEADS), len(sg.VIEWS)
            wparams = wb.unpack(wb.default_params(H, V), H, V)
            thresholds = {}
            bundle.fallback_reason = (
                f"{fd.name} is missing "
                + " and ".join(
                    path.name for path in (wp, th) if not path.exists()
                )
                + "; using the untuned decoder and its 0.5 threshold"
            )
        ood_reference = None
        ood_path = fd / "ood_reference.npz"
        if ood_path.is_file():
            with np.load(ood_path, allow_pickle=False) as reference:
                ood_reference = {
                    "center": reference["center"].copy(),
                    "max_distance": float(reference["max_distance"]),
                }
        bundle.folds.append({
            "M": M, "cfg": cfg,
            "wparams": wparams,
            "thr": thresholds,
            "decoder_tuned": decoder_tuned,
            "ood_reference": ood_reference,
        })
        bundle.img_size = cfg.img_size
        bundle.mc_samples = cfg.mc_samples
        loaded += 1
        try:
            loaded_fold_ids.append(int(fd.name.removeprefix("fold")))
        except ValueError:
            bundle.fallback_reason = f"unexpected fold directory name: {fd.name}"

    expected_fold_count = (
        int(bundle.run_metadata.get("fold_count", 0))
        if bundle.run_metadata else 0
    )
    if (
        expected_fold_count < 1
        or sorted(loaded_fold_ids) != list(range(expected_fold_count))
        or loaded != expected_fold_count
    ):
        bundle.folds.clear()
        loaded = 0
        bundle.fallback_reason = (
            "A complete, provenance-recorded fold ensemble is required; "
            "partial or unversioned weights are not loaded"
        )

    if loaded > 0:
        bundle.method = "deep"
        bundle.decoder_tuned = all(
            fold["decoder_tuned"] for fold in bundle.folds
        )
        return bundle

    # ── Colour baseline fallback ─────────────────────────────────────────────
    manifest_path = str(Path(manifest_path).resolve())
    if Path(manifest_path).exists():
        bundle.colour_clf = _train_colour_baseline(manifest_path)
    elif bundle.fallback_reason is None:
        bundle.fallback_reason = f"Manifest not found: {manifest_path}"
    else:
        bundle.fallback_reason += f"; Manifest not found: {manifest_path}"
    # If even the manifest doesn't exist we return a bundle that classifies
    # purely from colour features with a threshold-free heuristic.
    return bundle


# ────────────────────────────────────────────────────────────────────────────
# Deep-model single-image inference
# ────────────────────────────────────────────────────────────────────────────
def _predict_deep(pil_img: Image.Image, bundle: InferenceBundle) -> dict:
    """Run the full TF extract_cache → Wombat decode pipeline on one PIL image."""
    import tensorflow as tf
    import saffron_cnn_gnn as sg
    import wombat_beam as wb

    # Save image to a temp file so _load_image can read it
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        pil_img.convert("RGB").save(tmp_path, "PNG")
        probs = []
        mc_uncertainties = []
        ood_ratios = []
        ood_available = True
        thresholds_by_fold = []
        for fold_data in bundle.folds:
            M, cfg = fold_data["M"], fold_data["cfg"]
            S = cfg.img_size

            image_tensor = _preprocess_path(tmp_path, S)
            ds = tf.data.Dataset.from_tensor_slices(
                (image_tensor, np.asarray([[0.0]], dtype=np.float32))
            ).batch(1)

            cache = sg.extract_cache(M, ds, cfg, groups=np.array(["img0"]))
            H, V = len(cache.head_names), len(cache.view_names)
            wp = fold_data["wparams"]
            params = dict(
                a=np.array(wp["a"]),
                tau=np.array(wp["tau"]),
                beta=wp["beta"], gamma=wp["gamma"], alpha=wp["alpha"],
                beam=int(wp["beam"]), depth=int(wp["depth"]),
                kappa=wp["kappa"], theta=wp["theta"],
            )
            p = wb.decode(cache, params)[0]  # scalar probability of FAKE
            probs.append(float(p))
            mc_uncertainties.append(float(np.mean(cache.U)))
            threshold_key = "PROPOSED: Wombat-OA + weighted beam + log-prob fusion"
            fold_threshold = fold_data["thr"].get(
                threshold_key, fold_data["wparams"].get("theta", 0.5)
            )
            thresholds_by_fold.append(float(fold_threshold))
            reference = fold_data.get("ood_reference")
            if reference is None:
                ood_available = False
            else:
                feature = M["trunk"](image_tensor, training=False)["cnn"].numpy()[0]
                distance = float(np.linalg.norm(feature - reference["center"]))
                cutoff = float(reference["max_distance"])
                ood_ratios.append(distance / max(cutoff, 1e-12))
    finally:
        os.unlink(tmp_path)

    mean_p = float(np.mean(probs))
    thr = float(np.mean(thresholds_by_fold))
    capture_metadata = _capture_metadata(pil_img)
    calibration = bundle.calibration or {}
    support = calibration.get("capture_support", {})
    candidate = int(mean_p >= thr)
    capture_reason = assess_capture_support(
        capture_metadata, support, class_candidate=candidate
    )
    if not support:
        capture_reason = "capture-support calibration is unavailable"
    quantile = calibration.get("quantiles", {}).get("0.05", float("nan"))
    decision = decide(
        mean_p, float(quantile),
        fold_probabilities=probs, fold_thresholds=thresholds_by_fold,
        mc_uncertainty=float(np.mean(mc_uncertainties)),
        max_mc_uncertainty=float(calibration.get("max_mc_uncertainty") or float("nan")),
        ood_distance=max(ood_ratios) if ood_available and ood_ratios else None,
        max_ood_distance=1.0 if ood_available and ood_ratios else None,
        capture_reason=capture_reason, shot_count=1,
    )
    colour_values = _colour_features_pil(pil_img)
    return {
        "label": decision["label"],
        "p_fake": mean_p,
        "threshold": thr,
        "method": "deep",
        "folds_used": len(probs),
        "per_fold_probs": probs,
        "details": {
            "threshold": thr,
            "per_fold_probs": probs,
            "per_fold_thresholds": thresholds_by_fold,
            "mc_uncertainty": float(np.mean(mc_uncertainties)),
            "ood_ratio": max(ood_ratios) if ood_available and ood_ratios else None,
            "capture_metadata": capture_metadata,
            "capture_reason": capture_reason,
            "prediction_set": list(decision["prediction_set"]),
            "abstention_reasons": decision["reasons"],
            "colour_features": dict(zip(
                COLOUR_FEATURE_NAMES, map(float, colour_values))),
            "img_size": bundle.img_size,
            "mc_samples": bundle.mc_samples,
            "decoder_tuned": all(
                fold["decoder_tuned"] for fold in bundle.folds
            ),
        },
    }


# ────────────────────────────────────────────────────────────────────────────
# Colour-baseline single-image inference
# ────────────────────────────────────────────────────────────────────────────
def _predict_colour(pil_img: Image.Image, bundle: InferenceBundle) -> dict:
    features = _colour_features_pil(pil_img)
    feat = features.reshape(1, -1)
    if bundle.colour_clf is not None:
        fake_column = list(bundle.colour_clf.classes_).index(1)
        p_fake = float(bundle.colour_clf.predict_proba(feat)[0, fake_column])
    else:
        # No manifest available: use a simple colour heuristic
        # Real saffron is deep red/orange; fakes tend to have different hue
        img = pil_img.convert("RGB")
        w, h = img.size
        s = min(w, h)
        img = img.crop(((w - s) // 2, (h - s) // 2,
                        (w - s) // 2 + s, (h - s) // 2 + s)).resize(
                            (64, 64), Image.Resampling.BILINEAR)
        arr = np.asarray(img, float) / 255.0
        # Heuristic fallback is only used when no training manifest is available.
        red_score = float(arr[:, :, 0].mean() - arr[:, :, 2].mean())
        p_fake = float(np.clip(0.5 - red_score, 0.0, 1.0))

    return {
        "label": "INCONCLUSIVE",
        "baseline_label": "FAKE" if p_fake >= 0.5 else "REAL",
        "p_fake": p_fake,
        "threshold": 0.5,
        "method": "colour",
        "folds_used": 0,
        "per_fold_probs": [],
        "details": {
            "threshold": 0.5,
            "colour_features": dict(zip(
                COLOUR_FEATURE_NAMES, map(float, features))),
            "baseline_trained": bundle.colour_clf is not None,
            "fallback_reason": bundle.fallback_reason,
            "abstention_reasons": [
                "colour baseline mode is not the validated primary model"
            ],
            "img_size": 64,
            "mc_samples": 0,
        },
    }


# ────────────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────────────
def predict_pil(pil_img: Image.Image,
                bundle: Optional[InferenceBundle] = None,
                run_dir: str = "runs/cv1",
                manifest_path: str = "data/manifest.csv") -> dict:
    """
    Classify a PIL image as REAL or FAKE saffron.

    Parameters
    ----------
    pil_img       : PIL.Image.Image
    bundle        : pre-loaded InferenceBundle (load once, call many times)
    run_dir       : path to the cross-validation output directory
    manifest_path : path to data/manifest.csv (for colour baseline training)

    Returns
    -------
    dict with keys: label, confidence, p_fake, threshold, method,
                    folds_used, per_fold_probs
    """
    if bundle is None:
        bundle = load_bundle(run_dir, manifest_path)
    if bundle.method == "deep":
        return _predict_deep(pil_img, bundle)
    return _predict_colour(pil_img, bundle)


def load_model(run_dir: str = "runs/cv1",
               manifest_path: str = "data/manifest.csv") -> InferenceBundle:
    """Load the best complete deep-model bundle, or the manifest RF baseline."""
    return load_bundle(run_dir, manifest_path)


def predict_image(pil_image: Image.Image,
                  model_bundle: InferenceBundle) -> dict:
    """Predict REAL/FAKE for one uploaded PIL image using a preloaded bundle."""
    return predict_pil(pil_image, bundle=model_bundle)


def _capture_metadata(image: Image.Image) -> dict:
    tags = {name: tag for tag, name in ExifTags.TAGS.items()}
    exif = image.getexif()
    make = str(exif.get(tags.get("Make"), "")).strip()
    model = str(exif.get(tags.get("Model"), "")).strip()
    camera = " ".join(value for value in (make, model) if value) or "unknown"
    iso = exif.get(tags.get("ISOSpeedRatings"))
    flash_raw = exif.get(tags.get("Flash"))
    flash = "unknown"
    if flash_raw is not None:
        try:
            flash = "on" if int(flash_raw) & 1 else "off"
        except (TypeError, ValueError):
            flash = "unknown"
    return {
        "camera": camera,
        "iso": float(iso) if iso is not None else None,
        "flash_status": flash,
    }


def _preprocess_path(path: str, image_size: int):
    """Use the exact centre-square crop and resize function used in training."""
    import tensorflow as tf
    import saffron_cnn_gnn as sg

    return sg._load_image(tf.constant(path), image_size, False)[None, ...]


def predict_images(
    images: list[Image.Image],
    model_bundle: InferenceBundle,
) -> dict:
    """Aggregate multiple uploaded shots as one physical sample and abstain safely."""
    if not images:
        raise ValueError("at least one image is required")
    results = [_predict_deep(image, model_bundle) for image in images] if model_bundle.method == "deep" else [
        _predict_colour(image, model_bundle) for image in images
    ]
    if model_bundle.method != "deep":
        result = results[0].copy()
        result["details"]["shot_count"] = len(images)
        return result

    per_fold = np.asarray(
        [result["details"]["per_fold_probs"] for result in results], dtype=float
    )
    if per_fold.ndim != 2 or per_fold.shape[1] != len(model_bundle.folds):
        raise ValueError("fold prediction dimensions do not match loaded models")
    fold_probabilities = per_fold.mean(axis=0)
    p_fake = float(fold_probabilities.mean())
    fold_thresholds = results[0]["details"]["per_fold_thresholds"]
    mc_uncertainty = float(np.mean([
        result["details"]["mc_uncertainty"] for result in results
    ]))
    ood_ratios = [result["details"]["ood_ratio"] for result in results]
    ood_available = all(value is not None for value in ood_ratios)
    ood_ratio = max(ood_ratios) if ood_available else None
    reasons = [
        result["details"]["capture_reason"]
        for result in results if result["details"].get("capture_reason")
    ]
    capture_reason = "; ".join(sorted(set(reasons))) or None
    calibration = model_bundle.calibration or {}
    quantile = calibration.get("quantiles", {}).get("0.05", float("nan"))
    decision = decide(
        p_fake, float(quantile),
        fold_probabilities=fold_probabilities,
        fold_thresholds=fold_thresholds,
        mc_uncertainty=mc_uncertainty,
        max_mc_uncertainty=float(calibration.get("max_mc_uncertainty") or float("nan")),
        ood_distance=ood_ratio, max_ood_distance=1.0 if ood_available else None,
        capture_reason=capture_reason, shot_count=len(images),
    )
    threshold = float(np.mean(fold_thresholds))
    return {
        "label": decision["label"], "p_fake": p_fake, "threshold": threshold,
        "method": "deep", "folds_used": len(fold_probabilities),
        "per_fold_probs": fold_probabilities.tolist(),
        "details": {
            **results[0]["details"],
            "shot_count": len(images),
            "mc_uncertainty": mc_uncertainty,
            "ood_ratio": ood_ratio,
            "prediction_set": list(decision["prediction_set"]),
            "abstention_reasons": decision["reasons"],
            "per_fold_probs": fold_probabilities.tolist(),
            "per_fold_thresholds": fold_thresholds,
            "colour_features": dict(zip(
                COLOUR_FEATURE_NAMES,
                map(float, _colour_features_pil(images[0])),
            )),
        },
    }
