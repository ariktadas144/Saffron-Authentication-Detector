"""Group-label permutation null for the frozen-embedding logistic-head baseline."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from frozen_embedding_cv import extract_frozen_embeddings
from manifest_paths import normalize_manifest_paths
from wombat_beam import aggregate_by_group


def _oof_auc(embeddings: np.ndarray, manifest: pd.DataFrame, labels: np.ndarray, seed: int) -> float:
    folds = sorted(manifest.fold.unique())
    scores, truth, groups = [], [], []
    for fold in folds:
        validation_fold = (fold + 1) % (max(folds) + 1)
        train_mask = ~manifest.fold.isin([fold, validation_fold]).to_numpy()
        test_mask = (manifest.fold == fold).to_numpy()
        train_y = labels[train_mask]
        if np.unique(train_y).size < 2:
            raise ValueError(f"fold {fold} training partition has only one class")
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed + fold),
        )
        model.fit(embeddings[train_mask], train_y)
        probability = model.predict_proba(embeddings[test_mask])[:, 1]
        scores.extend(probability.tolist())
        truth.extend(labels[test_mask].tolist())
        groups.extend(manifest.loc[test_mask, "group"].astype(str).tolist())
    sample_p, sample_y = aggregate_by_group(scores, truth, groups)
    return float(roc_auc_score(sample_y, sample_p))


def permutation_null(
    embeddings: np.ndarray,
    manifest: pd.DataFrame,
    *,
    permutations: int = 200,
    seed: int = 42,
) -> tuple[float, np.ndarray, float]:
    if permutations < 1:
        raise ValueError("permutations must be positive")
    manifest = manifest.reset_index(drop=True)
    if len(embeddings) != len(manifest):
        raise ValueError("embedding rows must align with manifest")
    if manifest.groupby("group").label.nunique().gt(1).any():
        raise ValueError("group labels must be consistent")
    true_labels = manifest.label.to_numpy(int)
    real_auc = _oof_auc(embeddings, manifest, true_labels, seed)
    group_labels = manifest.groupby("group", sort=True).label.first()
    group_ids = group_labels.index.astype(str).to_numpy()
    observed = group_labels.to_numpy(int)
    rng = np.random.default_rng(seed)
    null = np.empty(permutations, dtype=float)
    for index in range(permutations):
        shuffled = rng.permutation(observed)
        lookup = dict(zip(group_ids, shuffled))
        labels = manifest.group.astype(str).map(lookup).to_numpy(int)
        null[index] = _oof_auc(embeddings, manifest, labels, seed + index + 1)
        if (index + 1) % 25 == 0:
            print(f"Completed {index + 1}/{permutations} group-label permutations", flush=True)
    p_value = float((1 + np.count_nonzero(null >= real_auc)) / (permutations + 1))
    return real_auc, null, p_value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/manifest.csv")
    parser.add_argument("--cache", default="runs/audit/frozen_embeddings.npz")
    parser.add_argument("--out", default="runs/audit/frozen_embedding_permutation_null.csv")
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    manifest_path = Path(args.manifest)
    cache_path = Path(args.cache)
    output_path = Path(args.out)
    for path in (manifest_path, cache_path, output_path):
        if not path.is_absolute():
            path = ROOT / path
        if path == manifest_path:
            manifest_path = path
        elif path == cache_path:
            cache_path = path
        else:
            output_path = path
    manifest = normalize_manifest_paths(pd.read_csv(manifest_path))
    embeddings = extract_frozen_embeddings(
        manifest, cache_path, batch_size=args.batch_size
    )
    real_auc, null, p_value = permutation_null(
        embeddings, manifest, permutations=args.permutations, seed=args.seed
    )
    rows = pd.DataFrame({
        "permutation": np.arange(1, len(null) + 1), "auc": null,
        "n_images": len(manifest), "n_samples": manifest.group.nunique(),
    })
    rows["real_oof_auc"] = real_auc
    rows["permutation_p_value"] = p_value
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output_path, index=False)
    print({
        "real_oof_auc": real_auc,
        "null_mean": float(null.mean()),
        "null_95_interval": [float(np.quantile(null, 0.025)), float(np.quantile(null, 0.975))],
        "permutation_p_value": p_value,
    })
    print(f"Wrote group-level permutation null: {output_path}")


if __name__ == "__main__":
    main()
