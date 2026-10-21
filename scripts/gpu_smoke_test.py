"""Run a tiny end-to-end pipeline check; its metrics are never research results."""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from label_shuffle_test import shuffle_partition_labels

SMOKE_LABEL = "SMOKE TEST, NOT A RESULT"


def run_logged(command: list[str], log_path: Path) -> None:
    print("Running:", subprocess.list2cmdline(command), flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False
        )
    if result.returncode:
        print(f"{SMOKE_LABEL}: command failed; log follows: {log_path}")
        print(log_path.read_text(encoding="utf-8", errors="replace"))
        raise SystemExit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/manifest.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--imagenet", action="store_true")
    parser.add_argument("--output-root", default="runs/audit")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    manifest = pd.read_csv(manifest_path)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = ROOT / args.output_root / f"SMOKE_TEST_NOT_A_RESULT_{stamp}"
    run_root.mkdir(parents=True, exist_ok=False)
    real_out = run_root / "real_fold0"

    run_command = [
        sys.executable, str(ROOT / "scripts" / "run_cv.py"),
        "--manifest", str(manifest_path),
        "--out", str(real_out),
        "--folds", "0",
        "--img-size", "64",
        "--batch", "16",
        "--epochs-warm", "1",
        "--epochs-ft", "1",
        "--mc-samples", "2",
        "--seed", str(args.seed),
        "--pop", "4",
        "--iters", "2",
    ]
    if not args.imagenet:
        run_command.append("--no-imagenet")
    run_logged(run_command, run_root / "real_fold0.log")

    required_real = [
        real_out / "fold0" / "cache_val.npz",
        real_out / "fold0" / "cache_test.npz",
        real_out / "fold0" / "ablation_test.csv",
        real_out / "fold0" / "ablation_sample_level.csv",
        real_out / "fold0" / "sample_thresholds.json",
    ]
    missing = [str(path) for path in required_real if not path.is_file()]
    if missing:
        raise RuntimeError(f"{SMOKE_LABEL}: fold pipeline omitted outputs: {missing}")

    shuffle_manifest = shuffle_partition_labels(manifest, fold=0, seed=101)
    shuffle_manifest_path = run_root / "SMOKE_TEST_NOT_A_RESULT_shuffle_manifest.csv"
    shuffle_manifest.to_csv(shuffle_manifest_path, index=False)
    shuffle_out = run_root / "shuffled_fold0"
    shuffle_command = [
        sys.executable, str(ROOT / "scripts" / "run_cv.py"),
        "--manifest", str(shuffle_manifest_path),
        "--out", str(shuffle_out),
        "--folds", "0",
        "--img-size", "64",
        "--batch", "16",
        "--epochs-warm", "1",
        "--epochs-ft", "1",
        "--mc-samples", "2",
        "--seed", str(args.seed),
        "--pop", "4",
        "--iters", "2",
    ]
    if not args.imagenet:
        shuffle_command.append("--no-imagenet")
    run_logged(shuffle_command, run_root / "shuffle_fold0.log")

    shuffle_table = shuffle_out / "fold0" / "ablation_sample_level.csv"
    if not shuffle_table.is_file():
        raise RuntimeError(f"{SMOKE_LABEL}: shuffled-label run did not write {shuffle_table}")

    summary = pd.read_csv(real_out / "fold0" / "ablation_sample_level.csv", index_col=0)
    summary.insert(0, "warning", SMOKE_LABEL)
    summary.to_csv(run_root / "SMOKE_TEST_NOT_A_RESULT_summary.csv")
    marker = run_root / "SMOKE_TEST_NOT_A_RESULT.txt"
    marker.write_text(
        f"{SMOKE_LABEL}\n"
        "Reduced image size, epochs, MC samples, and optimizer budget. "
        "These outputs only verify pipeline execution and must not be reported "
        "as model performance.\n",
        encoding="utf-8",
    )
    print(SMOKE_LABEL)
    print(summary.to_string())
    print(f"Smoke artifacts: {run_root}")


if __name__ == "__main__":
    main()
