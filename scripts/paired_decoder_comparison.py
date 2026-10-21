"""Paired group-bootstrap comparisons of decoder methods across training seeds."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from build_report import eligible_run
from wombat_beam import aggregate_by_group

PROPOSED = "PROPOSED: Wombat-OA + weighted beam + log-prob fusion"
COMPARATORS = {
    "best_validation_single_head": (
        "single view, cnn", "single view, gnn", "single view, hafm",
    ),
    "TTA mean-logit": ("TTA mean-logit, hafm",),
    "equal-weight log-prob fusion": ("log-prob fusion, equal weights, no beam",),
    "beam untuned": ("weighted beam + log-prob fusion (untuned)",),
    "beam random-search": ("weighted beam + log-prob fusion + random search",),
    "restricted decoder": ("PROPOSED: restricted Wombat-OA decoder",),
}


def _shift(p: np.ndarray, threshold: float) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    t = float(np.clip(threshold, 1e-6, 1 - 1e-6))
    return 1 / (1 + np.exp(-(np.log(p / (1 - p)) - np.log(t / (1 - t)))))


def run_sample_predictions(run_dir: Path) -> pd.DataFrame:
    valid, reason = eligible_run(run_dir)
    if not valid:
        raise ValueError(f"ineligible run {run_dir}: {reason}")
    all_rows = []
    for fold_dir in sorted(run_dir.glob("fold*")):
        fold = int(fold_dir.name.removeprefix("fold"))
        pred = pd.read_csv(fold_dir / "preds_test.csv")
        thresholds = json.loads((fold_dir / "sample_thresholds.json").read_text(encoding="utf-8"))
        for method, threshold in thresholds.items():
            if method in pred:
                pred[method] = _shift(pred[method].to_numpy(float), float(threshold))
        pred["fold"] = fold
        all_rows.append(pred)
    frame = pd.concat(all_rows, ignore_index=True)
    rows = []
    for group, part in frame.groupby("group", sort=True):
        row = {
            "group": str(group), "y": int(part.y.iloc[0]),
            "camera": part.camera.iloc[0] if "camera" in part else "unknown",
            "fold": int(part.fold.iloc[0]), "n_images": len(part),
        }
        for method in [column for column in part if column not in {
            "path", "group", "y", "fold", "camera", "flash_status", "iso",
            "sample_threshold", "n_images",
        } and not column.endswith("__threshold")]:
            p, labels = aggregate_by_group(
                part[method].to_numpy(float), part.y.to_numpy(int),
                part.group.astype(str).to_numpy(),
            )
            row[method] = float(p[0])
        rows.append(row)
    return pd.DataFrame(rows)


def choose_best_single_head(validation_dir: Path) -> str:
    samples = []
    for fold_dir in sorted(validation_dir.glob("fold*")):
        cache_path = fold_dir / "cache_val.npz"
        if not cache_path.is_file():
            raise FileNotFoundError(f"missing validation cache: {cache_path}")
        import wombat_beam as wb
        cache = wb.Cache.load(cache_path)
        values = {}
        for head in cache.head_names:
            p, y = aggregate_by_group(
                wb.b_single(cache, head), cache.y, cache.groups
            )
            values[f"single view, {head}"] = (
                balanced_accuracy_score(y, p >= 0.5)
            )
        samples.append(values)
    if not samples:
        raise FileNotFoundError(f"no validation caches under {validation_dir}")
    means = pd.DataFrame(samples).mean()
    return str(means.idxmax())


def paired_bootstrap(
    frames: list[pd.DataFrame],
    comparator: str,
    *,
    n_bootstrap: int = 5000,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    paired_frames = []
    expected_groups: set[str] | None = None
    image_counts: dict[str, int] | None = None
    for run_index, frame in enumerate(frames):
        chosen = choose_best_single_head(Path(frame.attrs["run_dir"])) if comparator == "best_validation_single_head" else None
        selected = chosen or comparator
        if PROPOSED not in frame or selected not in frame:
            raise ValueError(f"missing paired methods {PROPOSED!r} and {selected!r}")
        if frame.group.astype(str).duplicated().any():
            raise ValueError("each run must contain exactly one row per physical sample")
        groups_in_run = set(frame.group.astype(str))
        current_counts = dict(zip(
            frame.group.astype(str), pd.to_numeric(frame.n_images, errors="raise").astype(int)
        ))
        if expected_groups is None:
            expected_groups = groups_in_run
            image_counts = current_counts
        elif groups_in_run != expected_groups:
            raise ValueError(
                "seed runs do not contain identical held-out physical-sample groups"
            )
        elif current_counts != image_counts:
            raise ValueError("seed runs disagree on image counts for shared samples")
        paired_frames.append(frame[["group", "y", PROPOSED, selected]].rename(
            columns={PROPOSED: f"proposed_{run_index}", selected: f"compare_{run_index}"}
        ))
    merged = paired_frames[0]
    for other in paired_frames[1:]:
        merged = merged.merge(other, on=["group", "y"], validate="one_to_one")
    if len(merged) != len(expected_groups or ()):
        raise ValueError("seed runs disagree on labels for shared physical samples")
    groups = merged.group.to_numpy(str)
    unique = np.unique(groups)
    result_rows = []
    for metric in ("balanced_accuracy", "auc"):
        estimates = []
        for _ in range(n_bootstrap):
            draw = rng.choice(unique, size=len(unique), replace=True)
            indices = np.concatenate([np.flatnonzero(groups == group) for group in draw])
            run_diffs = []
            for index in range(len(frames)):
                proposed = merged[f"proposed_{index}"].to_numpy(float)[indices]
                compare = merged[f"compare_{index}"].to_numpy(float)[indices]
                labels = merged.y.to_numpy(int)[indices]
                if np.unique(labels).size < 2:
                    continue
                scorer = balanced_accuracy_score if metric == "balanced_accuracy" else roc_auc_score
                run_diffs.append(
                    scorer(labels, proposed >= 0.5) - scorer(labels, compare >= 0.5)
                    if metric == "balanced_accuracy"
                    else scorer(labels, proposed) - scorer(labels, compare)
                )
            estimates.append(float(np.mean(run_diffs)) if run_diffs else np.nan)
        result_rows.append({
            "comparator": comparator, "metric": metric,
            "difference_proposed_minus_comparator": float(np.nanmean(estimates)),
            "ci95_low": float(np.nanquantile(estimates, 0.025)),
            "ci95_high": float(np.nanquantile(estimates, 0.975)),
            "n_samples": int(len(unique)), "n_seeds": len(frames),
            "n_images": int(sum((image_counts or {}).values())),
            "interpretation": (
                "no evidence of benefit" if np.nanquantile(estimates, 0.025) <= 0 <= np.nanquantile(estimates, 0.975)
                else "paired interval excludes zero"
            ),
        })
    return pd.DataFrame(result_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True, help="at least three eligible seed run directories")
    parser.add_argument("--out", default="runs/audit/paired_decoder_comparisons.csv")
    parser.add_argument("--bootstrap", type=int, default=5000)
    args = parser.parse_args()
    if len(args.runs) < 3:
        parser.error("at least three training-seed run directories are required")
    run_dirs = [
        Path(value) if Path(value).is_absolute() else ROOT / value
        for value in args.runs
    ]
    frames = []
    for run_dir in run_dirs:
        frame = run_sample_predictions(run_dir)
        frame.attrs["run_dir"] = str(run_dir)
        frames.append(frame)
    outputs = []
    for label, candidates in COMPARATORS.items():
        if label == "best_validation_single_head":
            outputs.append(paired_bootstrap(
                frames, label, n_bootstrap=args.bootstrap, seed=42
            ))
        else:
            if any(candidate not in frames[0] for candidate in candidates):
                print(f"Skipping {label}: predictions are not present in every run.")
                continue
            outputs.append(paired_bootstrap(
                frames, candidates[0], n_bootstrap=args.bootstrap, seed=42
            ).assign(comparator=label))
    result = pd.concat(outputs, ignore_index=True)
    output = Path(args.out)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(result.to_string(index=False))
    print(f"Wrote paired comparisons: {output}")


if __name__ == "__main__":
    main()
