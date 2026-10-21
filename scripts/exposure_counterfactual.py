"""Paired exposure-like counterfactuals on saved fold models; no retraining."""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import tensorflow as tf
import wombat_beam as wb
import saffron_cnn_gnn as sg
from manifest_paths import normalize_manifest_paths

PROPOSED = "PROPOSED: Wombat-OA + weighted beam + log-prob fusion"


def _decoder_params(raw: dict) -> dict:
    return {
        "a": np.asarray(raw["a"]), "tau": np.asarray(raw["tau"]),
        **{key: raw[key] for key in ("beta", "gamma", "alpha", "beam", "depth", "kappa", "theta")},
    }


def method_probabilities(cache: wb.Cache, params: dict, random_params: dict, restricted_params: dict | None = None) -> dict[str, np.ndarray]:
    hafm = cache.head_names[-1]
    values = {
        f"single view, {head}": wb.b_single(cache, head)
        for head in cache.head_names
    }
    values["late fusion (prob avg), identity view"] = wb.b_late_fusion_identity(cache)
    values["TTA mean-logit, hafm"] = wb.b_tta_mean_logit(cache, hafm)
    values["TTA top-k confidence (old), hafm"] = wb.b_tta_topk_conf(cache, hafm)
    values["log-prob fusion, equal weights, no beam"] = wb.b_logprob_fusion_no_beam(cache)
    h, v = cache.shape[2], cache.shape[1]
    untuned = wb.unpack(wb.default_params(h, v), h, v)
    values["weighted beam + log-prob fusion (untuned)"] = wb.decode(cache, untuned)
    values["weighted beam + log-prob fusion + random search"] = wb.decode(cache, random_params)
    values[PROPOSED] = wb.decode(cache, params)
    if restricted_params is not None:
        values["PROPOSED: restricted Wombat-OA decoder"] = wb.decode(cache, restricted_params)
    return values


def counterfactual(image: Image.Image, perturbation: str, seed: int = 0) -> Image.Image:
    """Apply an exposure/noise/glare transform to pixels without changing content."""
    x = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    rng = np.random.default_rng(seed)
    if perturbation.startswith("brightness_"):
        x *= float(perturbation.removeprefix("brightness_"))
    elif perturbation.startswith("gamma_"):
        x = np.power(np.clip(x, 0, 1), float(perturbation.removeprefix("gamma_")))
    elif perturbation.startswith("noise_"):
        sigma = float(perturbation.removeprefix("noise_"))
        x += rng.normal(0, sigma / 255.0, x.shape)
    elif perturbation == "flash_glare":
        h, w = x.shape[:2]
        yy, xx = np.mgrid[:h, :w]
        radius = max(min(h, w) * 0.23, 1)
        strength = np.exp(-((xx - w / 2) ** 2 + (yy - h / 2) ** 2) / (2 * radius**2))
        x = x * (1 - strength[..., None]) + strength[..., None]
    else:
        raise ValueError(f"unknown counterfactual: {perturbation}")
    return Image.fromarray(np.asarray(np.clip(x, 0, 1) * 255, dtype=np.uint8), "RGB")


def perturbation_names() -> list[str]:
    return [
        "brightness_0.6", "brightness_0.8", "brightness_1.2", "brightness_1.5",
        "gamma_0.7", "gamma_1.4", "noise_4", "noise_8", "noise_16",
        "flash_glare",
    ]


def _dataset(paths: list[str], labels: np.ndarray, size: int, batch: int):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels.astype("float32")))
    return ds.map(
        lambda path, label: (sg._load_image(path, size, False), label[None]),
        num_parallel_calls=1,
    ).batch(batch)


def evaluate_fold(
    manifest: pd.DataFrame,
    audit: pd.DataFrame,
    run_dir: Path,
    fold: int,
    batch_size: int = 8,
) -> list[dict]:
    fold_dir = run_dir / f"fold{fold}"
    weight = next(
        (fold_dir / name for name in ("final.weights.h5", "best.weights.h5")
         if (fold_dir / name).is_file()),
        None,
    )
    if weight is None:
        raise FileNotFoundError(f"fold {fold} has no saved model weights in {fold_dir}")
    for required in ("config.json", "wombat_params.json", "sample_thresholds.json"):
        if not (fold_dir / required).is_file():
            raise FileNotFoundError(f"fold {fold} is missing {required}")
    raw_config = json.loads((fold_dir / "config.json").read_text(encoding="utf-8"))
    config = sg.Config(**{
        field.name: raw_config[field.name]
        for field in dataclasses.fields(sg.Config) if field.name in raw_config
    })
    config.imagenet = False
    model = sg.build_models(config)
    model["model"].load_weights(str(weight))
    params_raw = json.loads((fold_dir / "wombat_params.json").read_text(encoding="utf-8"))
    params = _decoder_params(params_raw)
    random_path = fold_dir / "wombat_params_random.json"
    if not random_path.is_file():
        raise FileNotFoundError(
            f"fold {fold} is missing the random-search decoder parameters: {random_path}"
        )
    random_params = _decoder_params(json.loads(random_path.read_text(encoding="utf-8")))
    restricted_path = fold_dir / "wombat_params_restricted.json"
    restricted_params = (
        _decoder_params(json.loads(restricted_path.read_text(encoding="utf-8")))
        if restricted_path.is_file() else None
    )
    thresholds = json.loads((fold_dir / "sample_thresholds.json").read_text(encoding="utf-8"))
    test = manifest[manifest.fold == fold].reset_index(drop=True)
    test = test.merge(
        audit[["path", "camera", "flash_status", "iso"]],
        on="path", how="left", validate="one_to_one",
    )
    paths = test.path.astype(str).tolist()
    raw_cache = sg.extract_cache(
        model, _dataset(paths, test.label.to_numpy(), config.img_size, batch_size),
        config, groups=test.group.astype(str).to_numpy(), paths=paths,
    )
    baseline_methods = method_probabilities(
        raw_cache, params, random_params, restricted_params
    )
    groups = np.unique(raw_cache.groups)
    group_meta = test.groupby("group", sort=True).agg(
        camera=("camera", "first"), y=("label", "first"),
        n_images=("path", "size"),
    ).reindex(groups)

    results = []
    for index, name in enumerate(perturbation_names()):
        with tempfile.TemporaryDirectory(prefix="saffron_counterfactual_") as temp:
            perturbed_paths = []
            for row, source in enumerate(test.path):
                source_path = Path(str(source))
                if not source_path.is_absolute():
                    source_path = ROOT / source_path
                with Image.open(source_path) as image:
                    result = counterfactual(image, name, seed=1000 * fold + row + index)
                target = Path(temp) / f"{row:04d}.png"
                result.save(target, format="PNG")
                perturbed_paths.append(str(target))
            cache = sg.extract_cache(
                model, _dataset(perturbed_paths, test.label.to_numpy(), config.img_size, batch_size),
                config, groups=test.group.astype(str).to_numpy(),
            )
        perturbed_methods = method_probabilities(cache, params, random_params, restricted_params)
        for method, baseline in baseline_methods.items():
            if method not in thresholds:
                continue
            threshold = float(thresholds[method])
            baseline_sample, labels = wb.aggregate_by_group(
                baseline, raw_cache.y, raw_cache.groups
            )
            sample_probability, _ = wb.aggregate_by_group(
                perturbed_methods[method], cache.y, cache.groups
            )
            flip = (baseline_sample >= threshold) != (sample_probability >= threshold)
            for camera in group_meta.camera.drop_duplicates():
                mask = group_meta.camera.astype(str).to_numpy() == str(camera)
                for label in (0, 1):
                    selected = mask & (labels == label)
                    if not selected.any():
                        continue
                    results.append({
                        "fold": fold, "perturbation": name, "method": method,
                        "camera": camera, "label": label,
                        "n_samples": int(selected.sum()),
                        "n_images": int(group_meta.n_images.to_numpy()[selected].sum()),
                        "verdict_flip_rate": float(flip[selected].mean()),
                        "mean_delta_p_fake": float((sample_probability[selected] - baseline_sample[selected]).mean()),
                        "threshold": threshold,
                    })
        print(f"fold {fold}: completed {name}")
    tf.keras.backend.clear_session()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/manifest.csv")
    parser.add_argument("--audit", default="data/audit_exif.csv")
    parser.add_argument("--run", default="runs/cv_repro")
    parser.add_argument("--out", default="runs/cv_repro/exposure_counterfactual.csv")
    parser.add_argument("--folds", type=int, nargs="*", default=None)
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()
    resolve = lambda value: Path(value) if Path(value).is_absolute() else ROOT / value
    manifest = normalize_manifest_paths(pd.read_csv(resolve(args.manifest)))
    audit = pd.read_csv(resolve(args.audit))
    audit = normalize_manifest_paths(audit, columns=("path",))
    run_dir = resolve(args.run)
    folds = args.folds if args.folds is not None else sorted(manifest.fold.unique())
    rows = [row for fold in folds for row in evaluate_fold(manifest, audit, run_dir, int(fold), args.batch)]
    output = resolve(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"Wrote paired counterfactual results: {output}")


if __name__ == "__main__":
    main()
