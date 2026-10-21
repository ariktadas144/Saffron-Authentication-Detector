"""Summarize capture sessions and prepare group-safe leave-one-date-out folds."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from manifest_paths import normalize_manifest_paths


def session_summary(manifest: pd.DataFrame, audit: pd.DataFrame) -> pd.DataFrame:
    joined = manifest.merge(
        audit[["path", "camera", "capture_date", "iso"]],
        on="path", how="left", validate="one_to_one",
    )
    inconsistent = joined.groupby("group").agg(
        labels=("label", "nunique"),
        cameras=("camera", "nunique"),
        dates=("capture_date", "nunique"),
    )
    if inconsistent[["labels", "cameras", "dates"]].gt(1).any().any():
        raise ValueError("sample group spans inconsistent labels, cameras, or capture dates")
    joined["capture_date"] = joined.capture_date.fillna("unknown")
    joined["iso_bucket"] = pd.cut(
        pd.to_numeric(joined.iso, errors="coerce"),
        [-float("inf"), 100, 160, 250, float("inf")],
        labels=["<=100", "101-160", "161-250", ">250"],
        right=False,
    ).astype(object).fillna("missing")
    group_rows = joined.groupby("group", as_index=False).agg(
        label=("label", "first"), camera=("camera", "first"),
        capture_date=("capture_date", "first"), iso_bucket=("iso_bucket", "first"),
        n_images=("path", "size"),
    )
    if (group_rows.groupby("group").label.nunique() > 1).any():
        raise ValueError("physical sample has inconsistent labels")
    return group_rows


def make_date_folds(group_rows: pd.DataFrame) -> pd.DataFrame:
    """Assign dates as test-fold units; reject unsupported/ambiguous dates."""
    dates = sorted(date for date in group_rows.capture_date.unique() if date != "unknown")
    eligible_dates = {
        date for date, part in group_rows.groupby("capture_date")
        if date in dates and part.label.nunique() == 2
    }
    eligible_dates = sorted(eligible_dates)
    if len(eligible_dates) < 3:
        raise ValueError(
            "leave-one-date-out requires at least three known dates containing both classes"
        )
    date_to_fold = {date: fold for fold, date in enumerate(eligible_dates)}
    group_rows = group_rows[group_rows.capture_date.isin(eligible_dates)].copy()
    group_rows["fold"] = group_rows.capture_date.map(date_to_fold).astype(int)
    return group_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/manifest.csv")
    parser.add_argument("--audit", default="data/audit_exif.csv")
    parser.add_argument("--summary-out", default="runs/audit/session_group_counts.csv")
    parser.add_argument("--fold-manifest-out", default="data/manifest_leave_one_date_out.csv")
    args = parser.parse_args()
    manifest = normalize_manifest_paths(pd.read_csv(ROOT / args.manifest))
    audit = normalize_manifest_paths(
        pd.read_csv(ROOT / args.audit), columns=("path",)
    )
    group_rows = session_summary(manifest, audit)
    summary = pd.crosstab(
        [group_rows.camera, group_rows.capture_date, group_rows.iso_bucket],
        group_rows.label,
        dropna=False,
    )
    print("Group-level camera x date x ISO bucket x class:")
    print(summary.to_string())
    out = ROOT / args.summary_out
    out.parent.mkdir(parents=True, exist_ok=True)
    group_rows.to_csv(out, index=False)
    try:
        folded = make_date_folds(group_rows)
    except ValueError as exc:
        print(f"Leave-one-date-out not generated: {exc}")
        return
    fold_out = ROOT / args.fold_manifest_out
    fold_out.parent.mkdir(parents=True, exist_ok=True)
    controlled = manifest.merge(
        folded[["group", "fold"]], on="group", suffixes=("", "_date"),
        validate="many_to_one",
    )
    dropped_groups = manifest.loc[
        ~manifest.group.astype(str).isin(folded.group.astype(str)), "group"
    ].nunique()
    print(f"Excluded groups without an eligible mixed-class capture date: {dropped_groups}")
    controlled.drop(
        columns=["fold"]
    ).rename(columns={"fold_date": "fold"}).to_csv(fold_out, index=False)
    print(f"Wrote date-fold manifest: {fold_out}")


if __name__ == "__main__":
    main()
