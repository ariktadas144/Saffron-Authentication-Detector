"""Frozen ImageNet ConvNeXt features and group-shuffled logistic-head sanity checks."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import tensorflow as tf
import wombat_beam as wb
from label_shuffle_test import shuffle_partition_labels
from manifest_paths import normalize_manifest_paths
from saffron_cnn_gnn import _load_image

METRICS = ("auc", "acc", "bal_acc", "recall_fake", "specificity")


def extract_frozen_embeddings(
    manifest: pd.DataFrame,
    cache_path: Path,
    img_size: int = 256,
    batch_size: int = 4,
) -> np.ndarray:
    """Extract centre-crop embeddings, reusing cache only for matching paths and sizes."""
    manifest = normalize_manifest_paths(manifest)
    paths = manifest.path.astype(str).to_numpy()
    sizes = np.asarray([Path(path).stat().st_size for path in paths], dtype=np.int64)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as cache:
            if (
                np.array_equal(cache["paths"], paths)
                and np.array_equal(cache["sizes"], sizes)
                and int(cache["img_size"]) == img_size
            ):
                print(f"Using frozen embedding cache: {cache_path}")
                return cache["embeddings"]

    model = tf.keras.applications.ConvNeXtTiny(
        include_top=False,
        include_preprocessing=True,
        pooling="avg",
        weights="imagenet",
        input_shape=(img_size, img_size, 3),
    )
    path_ds = tf.data.Dataset.from_tensor_slices(paths)
    image_ds = path_ds.map(
        lambda path: _load_image(path, img_size, False),
        num_parallel_calls=1,
    ).batch(batch_size)
    batches = []
    for batch_index, batch in enumerate(image_ds):
        batches.append(model(batch, training=False).numpy())
        print(
            f"Frozen embeddings: {min((batch_index + 1) * batch_size, len(paths))}"
            f"/{len(paths)}",
            flush=True,
        )
    embeddings = np.concatenate(batches, axis=0).astype(np.float32)
    if len(embeddings) != len(manifest):
        raise RuntimeError("embedding count does not match manifest row count")
    np.savez_compressed(
        cache_path, paths=paths, sizes=sizes, img_size=np.asarray(img_size),
        embeddings=embeddings,
    )
    print(f"Wrote frozen embedding cache: {cache_path}")
    return embeddings


def _sample_predictions(
    labels: np.ndarray, probabilities: np.ndarray, groups: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scores, sample_labels = wb.aggregate_by_group(probabilities, labels, groups)
    sample_groups = np.unique(groups.astype(str))
    if len(sample_groups) != len(sample_labels):
        raise AssertionError("group aggregation returned an unexpected number of samples")
    return scores, sample_labels, sample_groups


def _metrics_and_ci(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    groups: np.ndarray,
    seed: int,
) -> tuple[dict[str, float], dict[str, tuple[float, float]]]:
    values = wb.metrics(labels, probabilities, threshold)
    intervals = wb.bootstrap_metrics_ci(
        labels, probabilities, threshold, n=2000, seed=seed, groups=groups
    )
    return values, intervals


def _shift_probability(probability: np.ndarray, threshold: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    target = np.clip(np.asarray(threshold, dtype=float), 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped)) - np.log(target / (1 - target))
    return 1 / (1 + np.exp(-logits))


def run_embedding_cv(
    manifest: pd.DataFrame,
    audit: pd.DataFrame,
    embeddings: np.ndarray,
    seed: int = 42,
    shuffle_seeds: tuple[int, ...] = (101, 202, 303),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return fold/pooled metrics, camera details, and group-label shuffle rows."""
    manifest = normalize_manifest_paths(manifest)
    audit = normalize_manifest_paths(audit, columns=("path",))
    identity = manifest[["path", "group"]].merge(
        audit[["path", "group"]], on="path", how="outer",
        suffixes=("_manifest", "_audit"), validate="one_to_one",
        indicator=True,
    )
    if (
        not identity._merge.eq("both").all()
        or not identity.group_manifest.astype(str).eq(
            identity.group_audit.astype(str)
        ).all()
    ):
        raise ValueError("audit image/group identity does not match the manifest")
    audit = manifest[["path"]].merge(
        audit, on="path", how="left", validate="one_to_one"
    )
    if len(embeddings) != len(manifest):
        raise ValueError("embedding row count does not match manifest")
    if (manifest.groupby("group").label.nunique() != 1).any():
        raise ValueError("each physical sample must have one label")

    n_folds = int(manifest.fold.max()) + 1
    fold_rows: list[dict] = []
    pooled_rows: list[pd.DataFrame] = []
    camera_rows: list[dict] = []
    for fold in range(n_folds):
        validation_fold = (fold + 1) % n_folds
        train_idx = np.flatnonzero(~manifest.fold.isin([fold, validation_fold]).to_numpy())
        val_idx = np.flatnonzero((manifest.fold == validation_fold).to_numpy())
        test_idx = np.flatnonzero((manifest.fold == fold).to_numpy())
        train = manifest.iloc[train_idx]
        val = manifest.iloc[val_idx]
        test = manifest.iloc[test_idx]
        if (
            set(train.group) & set(val.group)
            or set(train.group) & set(test.group)
            or set(val.group) & set(test.group)
        ):
            raise ValueError(f"group leakage in fold {fold}")

        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced", max_iter=2000, random_state=seed + fold
            ),
        )
        model.fit(embeddings[train_idx], train.label.astype(int))
        val_probability = model.predict_proba(embeddings[val_idx])[:, 1]
        test_probability = model.predict_proba(embeddings[test_idx])[:, 1]
        threshold = wb.select_group_threshold(
            val.label.to_numpy(),
            val_probability,
            val.group.astype(str).to_numpy(),
        )
        test_groups = test.group.astype(str).to_numpy()
        sample_probability, sample_labels, sample_groups = _sample_predictions(
            test.label.to_numpy(), test_probability, test_groups
        )
        metrics, intervals = _metrics_and_ci(
            sample_labels, sample_probability, threshold, sample_groups, seed + fold
        )
        fold_rows.append({
            "section": "frozen_embedding",
            "fold": fold,
            "level": "sample",
            "scope": "pooled",
            "threshold": threshold,
            "n_groups": len(sample_labels),
            "n_samples": len(sample_labels),
            "n_images": len(test),
            **metrics,
            **{
                f"{metric}_lower_95": intervals.get(metric, (np.nan, np.nan))[0]
                for metric in METRICS
            },
            **{
                f"{metric}_upper_95": intervals.get(metric, (np.nan, np.nan))[1]
                for metric in METRICS
            },
        })
        test_meta = audit.iloc[test_idx].copy()
        test_meta["y"] = test.label.to_numpy()
        test_meta["group"] = test.group.astype(str).to_numpy()
        test_meta["probability"] = test_probability
        test_meta["threshold"] = threshold
        for camera, subset in test_meta.groupby("camera", dropna=False):
            group_probability, group_labels, group_ids = _sample_predictions(
                subset.y.to_numpy(),
                subset.probability.to_numpy(),
                subset.group.to_numpy(),
            )
            camera_metrics, camera_ci = _metrics_and_ci(
                group_labels, group_probability, threshold, group_ids, seed + fold
            )
            camera_rows.append({
                "fold": fold,
                "camera": camera,
                "n_groups": len(group_labels),
                "n_samples": len(group_labels),
                "n_images": len(subset),
                **camera_metrics,
                **{
                    f"{metric}_lower_95": camera_ci.get(
                        metric, (np.nan, np.nan)
                    )[0]
                    for metric in METRICS
                },
                **{
                    f"{metric}_upper_95": camera_ci.get(
                        metric, (np.nan, np.nan)
                    )[1]
                    for metric in METRICS
                },
            })
        pooled_rows.append(pd.DataFrame({
            "fold": fold,
            "group": sample_groups,
            "y": sample_labels,
            "probability": sample_probability,
            "threshold": threshold,
            "camera": test_meta.groupby("group", sort=False).camera.first().reindex(
                sample_groups
            ).to_numpy(),
            "flash_status": test_meta.groupby(
                "group", sort=False
            ).flash_status.first().reindex(sample_groups).to_numpy(),
            "iso": test_meta.groupby("group", sort=False).iso.median().reindex(
                sample_groups
            ).to_numpy(),
        }))

    pooled = pd.concat(pooled_rows, ignore_index=True)
    pooled_prob = _shift_probability(
        pooled.probability.to_numpy(), pooled.threshold.to_numpy()
    )
    pooled_metrics, pooled_ci = _metrics_and_ci(
        pooled.y.to_numpy(), pooled_prob, 0.5, pooled.group.to_numpy(), seed
    )
    fold_frame = pd.DataFrame(fold_rows)
    fold_frame = pd.concat([
        fold_frame,
        pd.DataFrame([{
            "section": "frozen_embedding",
            "fold": "pooled_oof",
            "level": "sample",
            "scope": "pooled",
            "threshold": 0.5,
            "n_groups": len(pooled),
            "n_samples": len(pooled),
            "n_images": len(manifest),
            **pooled_metrics,
            **{
                f"{metric}_lower_95": pooled_ci.get(
                    metric, (np.nan, np.nan)
                )[0]
                for metric in METRICS
            },
            **{
                f"{metric}_upper_95": pooled_ci.get(
                    metric, (np.nan, np.nan)
                )[1]
                for metric in METRICS
            },
        }]),
    ], ignore_index=True)

    shuffle_rows: list[dict] = []
    for shuffle_seed in shuffle_seeds:
        for fold in range(n_folds):
            shuffled = shuffle_partition_labels(manifest, fold, shuffle_seed)
            validation_fold = (fold + 1) % n_folds
            train_idx = np.flatnonzero(
                ~shuffled.fold.isin([fold, validation_fold]).to_numpy()
            )
            val_idx = np.flatnonzero((shuffled.fold == validation_fold).to_numpy())
            test_idx = np.flatnonzero((shuffled.fold == fold).to_numpy())
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=seed + shuffle_seed + fold,
                ),
            )
            model.fit(
                embeddings[train_idx], shuffled.iloc[train_idx].label.astype(int)
            )
            val_probability = model.predict_proba(embeddings[val_idx])[:, 1]
            threshold = wb.select_group_threshold(
                shuffled.iloc[val_idx].label.to_numpy(),
                val_probability,
                shuffled.iloc[val_idx].group.astype(str).to_numpy(),
            )
            test = manifest.iloc[test_idx]
            test_probability = model.predict_proba(embeddings[test_idx])[:, 1]
            sample_probability, sample_labels, sample_groups = _sample_predictions(
                test.label.to_numpy(),
                test_probability,
                test.group.astype(str).to_numpy(),
            )
            sample_metrics, sample_ci = _metrics_and_ci(
                sample_labels,
                sample_probability,
                threshold,
                sample_groups,
                seed + shuffle_seed + fold,
            )
            shuffle_rows.append({
                "section": "group_label_shuffle",
                "shuffle_seed": shuffle_seed,
                "fold": fold,
                "n_groups": len(sample_labels),
                "n_samples": len(sample_labels),
                "n_images": len(test),
                **sample_metrics,
                **{
                    f"{metric}_lower_95": sample_ci.get(
                        metric, (np.nan, np.nan)
                    )[0]
                    for metric in METRICS
                },
                **{
                    f"{metric}_upper_95": sample_ci.get(
                        metric, (np.nan, np.nan)
                    )[1]
                    for metric in METRICS
                },
            })
    return fold_frame, pd.DataFrame(camera_rows), pd.DataFrame(shuffle_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/manifest.csv")
    parser.add_argument("--audit", default="data/audit_exif.csv")
    parser.add_argument("--cache", default="runs/audit/frozen_embeddings.npz")
    parser.add_argument("--out", default="runs/audit/frozen_embedding_cv.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--shuffle-seeds", nargs="+", type=int, default=[101, 202, 303])
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    audit_path = Path(args.audit)
    cache_path = Path(args.cache)
    output = Path(args.out)
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    if not audit_path.is_absolute():
        audit_path = ROOT / audit_path
    if not cache_path.is_absolute():
        cache_path = ROOT / cache_path
    if not output.is_absolute():
        output = ROOT / output
    manifest = normalize_manifest_paths(pd.read_csv(manifest_path))
    audit = pd.read_csv(audit_path)
    embeddings = extract_frozen_embeddings(manifest, cache_path, batch_size=args.batch_size)
    fold_table, camera_table, shuffle_table = run_embedding_cv(
        manifest,
        audit,
        embeddings,
        seed=args.seed,
        shuffle_seeds=tuple(args.shuffle_seeds),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fold_table.to_csv(output, index=False)
    camera_path = output.with_name(f"{output.stem}_camera.csv")
    shuffle_path = output.with_name(f"{output.stem}_shuffle.csv")
    camera_table.to_csv(camera_path, index=False)
    shuffle_table.to_csv(shuffle_path, index=False)
    print(f"Wrote frozen-fold results: {output}")
    print(f"Wrote per-camera results: {camera_path}")
    print(f"Wrote group-shuffle results: {shuffle_path}")
    print(f"SMOKE? No: frozen feature evaluation, not fine-tuned training.")
    print(fold_table.to_string(index=False))
    print(shuffle_table[["shuffle_seed", "fold", "auc"]].to_string(index=False))


if __name__ == "__main__":
    main()
