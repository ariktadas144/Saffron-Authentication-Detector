"""Create G973F-only controlled manifests without altering the source manifest."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from manifest_paths import normalize_manifest_paths


def _group_summary(manifest: pd.DataFrame, label_name: str = "label") -> pd.DataFrame:
    for column in (label_name, "camera", "capture_date"):
        if manifest.groupby("group")[column].nunique(dropna=False).gt(1).any():
            raise ValueError(f"group metadata is inconsistent for {column!r}")
    groups = (
        manifest.groupby("group", as_index=False)
        .agg(
            label=(label_name, "first"),
            n_images=("path", "size"),
            iso=("iso", "median"),
            exposure=("exposure_time_s", "median"),
            brightness=("brightness_value", "median"),
            capture_date=("capture_date", "first"),
            camera=("camera", "first"),
        )
    )
    groups["iso_bucket"] = pd.cut(
        groups["iso"],
        bins=[-float("inf"), 100, 125, 160, float("inf")],
        labels=["<100", "100-124", "125-159", ">=160"],
        right=False,
    ).astype(object).fillna("missing")
    groups["brightness_bucket"] = pd.cut(
        groups["brightness"],
        bins=[-float("inf"), 5, 10, 15, float("inf")],
        labels=["<5", "5-10", "10-15", ">15"],
        right=False,
    ).astype(object).fillna("missing")
    groups["exposure_bucket"] = pd.cut(
        groups["exposure"],
        bins=[-float("inf"), 0.008, 0.010, 0.013, float("inf")],
        labels=["<=0.008", "0.008-0.010", "0.010-0.013", ">0.013"],
        right=False,
    ).astype(object).fillna("missing")
    return groups


def _print_group_breakdowns(groups: pd.DataFrame) -> None:
    print("\nG973F group summary (group=physical sample, not image):")
    print("\nCamera x class (groups):")
    print(pd.crosstab(groups["camera"], groups["label"], dropna=False).to_string())
    print("\nPer-class groups and images:")
    print(groups.groupby("label").agg(groups=("group", "nunique"), images=("n_images", "sum")).to_string())
    print("\nISO bucket x class (groups):")
    print(pd.crosstab(groups["iso_bucket"], groups["label"]).to_string())
    print("\nBrightness bucket x class (groups):")
    print(pd.crosstab(groups["brightness_bucket"], groups["label"]).to_string())
    print("\nCapture date x class (groups):")
    print(pd.crosstab(groups["capture_date"], groups["label"]).to_string())


def _matched_group_rows(groups: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
    """Coarsened group-level matching on ISO and exposure bins. Skip if insufficient groups remain."""
    keys = ["iso_bucket", "exposure_bucket"]
    counts = groups.groupby([*keys, label_col], dropna=False).size().unstack(label_col, fill_value=0)
    print("\nCoarsened ISO/exposure groups by label before matching:")
    print(counts.to_string())
    labels = sorted(groups[label_col].unique())
    if len(labels) != 2:
        print("Skipping matching: exactly two labels are required.")
        return groups.iloc[0:0].copy()
    selected: list[str] = []
    dropped_no_overlap = 0
    dropped_excess = 0
    for cell, subset in groups.groupby(keys, dropna=False, sort=True):
        per_class = {
            label: subset[subset[label_col] == label].sort_values("group")
            for label in labels
        }
        if any(part.empty for part in per_class.values()):
            dropped_no_overlap += len(subset)
            continue
        keep = min(len(part) for part in per_class.values())
        for part in per_class.values():
            selected.extend(part.group.iloc[:keep].astype(str).tolist())
            dropped_excess += len(part) - keep
    matched = groups[groups.group.astype(str).isin(selected)].copy()
    print(
        f"Matching dropped {dropped_no_overlap} groups without overlapping "
        f"ISO/exposure cells and {dropped_excess} excess groups to balance cells."
    )
    per_class_counts = matched.groupby(label_col).group.nunique()
    if len(per_class_counts) != 2 or per_class_counts.min() < 6:
        print(
            "Skipping matched manifest: coarsened matching leaves fewer than "
            "6 physical-sample groups in at least one class."
        )
        return groups.iloc[0:0].copy()
    return matched


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/manifest.csv")
    parser.add_argument("--audit", default="data/audit_exif.csv")
    parser.add_argument("--out", default="data/manifest_g973f.csv")
    parser.add_argument("--matched-out", default="data/manifest_g973f_matched.csv")
    args = parser.parse_args()
    manifest_path = Path(args.manifest)
    audit_path = Path(args.audit)
    out_path = Path(args.out)
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    if not audit_path.is_absolute():
        audit_path = ROOT / audit_path
    if not out_path.is_absolute():
        out_path = ROOT / out_path

    manifest = normalize_manifest_paths(pd.read_csv(manifest_path))
    if not audit_path.exists():
        raise FileNotFoundError(f"EXIF audit {audit_path} is required before building the G973F manifest")
    audit = normalize_manifest_paths(pd.read_csv(audit_path), columns=("path", "orig_path"))
    g973f = manifest.merge(
        audit[["path", "camera", "iso", "exposure_time_s", "brightness_value", "capture_date"]],
        on="path", how="left", validate="one_to_one",
    )
    g973f = g973f[g973f.camera.astype(str).str.contains("SM-G973F", na=False)].copy()
    if g973f.empty:
        raise RuntimeError("No SM-G973F rows are present; refusing to build a G973F manifest")
    g973f["label"] = g973f["label"].astype(int)
    groups = _group_summary(g973f, label_name="label")
    if groups.groupby("label").group.nunique().min() < 6:
        raise RuntimeError(
            "Refusing to continue: G973F classes do not each have at least 6 groups after filtering. "
            "Required for a stable control manifest."
        )
    _print_group_breakdowns(groups)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    g973f.to_csv(out_path, index=False)
    print(f"\nWrote G973F-only manifest to {out_path}")

    matched = _matched_group_rows(groups)
    if matched.empty:
        print("Skipping matched manifest because the optional group-level matching leaves too few groups or sparse cells.")
        return
    matched_path = Path(args.matched_out)
    if not matched_path.is_absolute():
        matched_path = ROOT / matched_path
    matched_frame = g973f[g973f.group.isin(matched["group"])].copy()
    if matched_frame.groupby("label").group.nunique().min() < 6:
        print("Skipping matched manifest: fewer than 6 groups remain in at least one class after coarsened matching.")
        return
    matched_frame.to_csv(matched_path, index=False)
    print(f"Wrote G973F matched manifest to {matched_path}")


if __name__ == "__main__":
    main()
