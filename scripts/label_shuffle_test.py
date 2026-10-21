"""Train fold models after shuffling labels at the physical-sample group level.

Example:
    python scripts/label_shuffle_test.py --fold 0 --seeds 101 202 303
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def shuffle_partition_labels(manifest: pd.DataFrame, fold: int, seed: int) -> pd.DataFrame:
    """Permute group labels separately within train and validation folds; preserve test labels."""
    result = manifest.copy()
    n_folds = int(manifest["fold"].max()) + 1
    validation_fold = (fold + 1) % n_folds
    rng = np.random.default_rng(seed)

    for partition_fold in sorted(set(manifest["fold"]) - {fold}):
        indices = np.flatnonzero(manifest["fold"].to_numpy() == partition_fold)
        part = manifest.iloc[indices]
        group_labels = part.groupby("group", sort=True)["label"].nunique()
        if (group_labels != 1).any():
            raise ValueError("manifest groups must have a single label before shuffling")
        groups = part.groupby("group", sort=True)["label"].first()
        shuffled = rng.permutation(groups.to_numpy())
        mapping = dict(zip(groups.index.astype(str), shuffled.astype(int)))
        result.loc[indices, "label"] = (
            result.loc[indices, "group"].astype(str).map(mapping).to_numpy()
        )

    test_mask = result["fold"] == fold
    if not np.array_equal(
        result.loc[test_mask, "label"].to_numpy(),
        manifest.loc[test_mask, "label"].to_numpy(),
    ):
        raise AssertionError("test labels changed during group-level shuffling")
    for group, rows in result.groupby("group"):
        if rows["label"].nunique() != 1:
            raise AssertionError(f"group {group!r} received inconsistent shuffled labels")
    for partition_fold in (fold, validation_fold):
        before = (
            manifest.loc[manifest.fold == partition_fold]
            .groupby("group")["label"].first().value_counts().sort_index()
        )
        after = (
            result.loc[result.fold == partition_fold]
            .groupby("group")["label"].first().value_counts().sort_index()
        )
        if not before.equals(after):
            raise AssertionError(
                f"group-level class proportions changed in fold {partition_fold}"
            )
    return result


def _read_result(
    out: Path, label: str, fold: int, seed: int | None = None
) -> dict:
    fold_dir = out / f"fold{fold}"
    metrics_path = fold_dir / "ablation_sample_level.csv"
    thresholds_path = fold_dir / "sample_thresholds.json"
    if not metrics_path.is_file() or not thresholds_path.is_file():
        raise FileNotFoundError(f"missing completed fold-{fold} results under {fold_dir}")
    predictions_path = fold_dir / "preds_test.csv"
    if not predictions_path.is_file():
        raise FileNotFoundError(f"missing saved test predictions: {predictions_path}")
    table = pd.read_csv(metrics_path, index_col=0)
    predictions = pd.read_csv(predictions_path)
    method = "PROPOSED: Wombat-OA + weighted beam + log-prob fusion"
    row = table.loc[method]
    return {
        "result": label,
        "shuffle_seed": seed,
        "sample_auc": float(row["auc"]),
        "sample_accuracy": float(row["acc"]),
        "sample_balanced_accuracy": float(row["bal_acc"]),
        "sample_threshold": json.loads(thresholds_path.read_text())[method],
        "n_images": len(predictions),
        "n_samples": int(predictions.group.nunique()),
        "n_fake": int(predictions.groupby("group").y.first().eq(1).sum()),
        "n_real": int(predictions.groupby("group").y.first().eq(0).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/manifest.csv")
    parser.add_argument("--out", default="runs/audit/label_shuffle")
    parser.add_argument("--real-out", default="runs/cv_repro")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seeds", nargs="+", type=int, default=[101, 202, 303])
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--epochs-warm", type=int, default=3)
    parser.add_argument("--epochs-ft", type=int, default=25)
    parser.add_argument("--mc-samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pop", type=int, default=20)
    parser.add_argument("--iters", type=int, default=25)
    parser.add_argument("--no-imagenet", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    manifest = pd.read_csv(manifest_path)
    if args.fold not in set(manifest.fold):
        parser.error(f"fold {args.fold} is not present in {manifest_path}")
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    real_out = Path(args.real_out)
    if not real_out.is_absolute():
        real_out = ROOT / real_out
    results = [_read_result(real_out, "real labels", args.fold)]
    runner = ROOT / "scripts" / "run_cv.py"
    for shuffle_seed in args.seeds:
        shuffled = shuffle_partition_labels(manifest, args.fold, shuffle_seed)
        shuffled_path = out / f"manifest_fold{args.fold}_seed{shuffle_seed}.csv"
        shuffled.to_csv(shuffled_path, index=False)
        run_out = out / f"seed{shuffle_seed}"
        fold_dir = run_out / f"fold{args.fold}"
        manifest_hash = hashlib.sha256(shuffled_path.read_bytes()).hexdigest()
        marker_path = fold_dir / "shuffle_manifest.sha256"
        completed = (
            (fold_dir / "ablation_sample_level.csv").is_file()
            and (fold_dir / "sample_thresholds.json").is_file()
        )
        if (
            not args.no_resume
            and completed
            and marker_path.is_file()
            and marker_path.read_text(encoding="utf-8").strip() == manifest_hash
        ):
            print(f"Resuming completed shuffle seed {shuffle_seed} from {run_out}")
            results.append(
                _read_result(run_out, "group-label shuffle", args.fold, shuffle_seed)
            )
            continue
        command = [
            sys.executable, str(runner),
            "--manifest", str(shuffled_path), "--out", str(run_out),
            "--folds", str(args.fold),
            "--img-size", str(args.img_size), "--batch", str(args.batch),
            "--epochs-warm", str(args.epochs_warm), "--epochs-ft", str(args.epochs_ft),
            "--mc-samples", str(args.mc_samples), "--seed", str(args.seed),
            "--pop", str(args.pop), "--iters", str(args.iters),
        ]
        if args.no_imagenet:
            command.append("--no-imagenet")
        if completed:
            command.append("--retrain")
        log_path = out / f"seed{shuffle_seed}.log"
        print("Running:", subprocess.list2cmdline(command), flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            subprocess.run(
                command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True
            )
        marker_path.write_text(manifest_hash + "\n", encoding="utf-8")
        results.append(
            _read_result(run_out, "group-label shuffle", args.fold, shuffle_seed)
        )
    summary = pd.DataFrame(results)
    summary.to_csv(out / "summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Wrote {out / 'summary.csv'}")


if __name__ == "__main__":
    main()
