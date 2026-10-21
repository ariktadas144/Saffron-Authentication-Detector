"""Create a hash-locked manifest for a prospective, never-reused test set."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_LOG_COLUMNS = {
    "filename", "date", "phone", "sample_id", "label", "session",
    "ISO", "exposure", "flash", "background",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_capture_design(manifest: pd.DataFrame) -> None:
    group_metadata = manifest.groupby("group", as_index=False).agg(
        label_count=("label", "nunique"),
        phone_count=("camera", "nunique"),
        camera=("camera", "first"),
        label=("label", "first"),
    )
    if group_metadata[["label_count", "phone_count"]].gt(1).any().any():
        raise ValueError("each physical sample must have one label and one phone")
    if group_metadata.camera.isna().any() or group_metadata.camera.astype(str).str.strip().eq("").any():
        raise ValueError("each prospective sample must have a recorded phone model")
    phones = group_metadata.camera.nunique()
    if phones < 3:
        raise ValueError(
            f"prospective test requires at least 3 phone models; found {phones}"
        )
    counts = group_metadata.groupby(["camera", "label"]).size().unstack(fill_value=0)
    missing_classes = {0, 1} - set(counts.columns)
    if missing_classes:
        raise ValueError("prospective test must contain both classes on every phone")
    insufficient = counts.reindex(columns=[0, 1], fill_value=0) < 30
    if insufficient.any().any():
        raise ValueError(
            "prospective test requires at least 30 independent groups per class "
            "per phone; observed counts: "
            + counts.reindex(columns=[0, 1], fill_value=0).to_dict(orient="index").__repr__()
        )


def create_locked_manifest(image_dir: Path, capture_log: Path, output_dir: Path) -> Path:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"locked-test output directory is not empty: {output_dir}")
    log = pd.read_csv(capture_log)
    missing = REQUIRED_LOG_COLUMNS - set(log.columns)
    if missing:
        raise ValueError(f"capture log is missing columns: {sorted(missing)}")
    image_paths = [path.resolve() for path in image_dir.rglob("*") if path.is_file()]
    names = [path.name for path in image_paths]
    if len(names) != len(set(names)):
        raise ValueError("prospective image filenames must be unique within the image directory")
    by_name = {path.name: path for path in image_paths}
    rows = []
    for record in log.to_dict("records"):
        missing_fields = [
            column for column in REQUIRED_LOG_COLUMNS
            if pd.isna(record[column]) or not str(record[column]).strip()
        ]
        if missing_fields:
            raise ValueError(
                f"capture-log row is missing required values: {sorted(missing_fields)}"
            )
        filename = str(record["filename"])
        if filename not in by_name:
            raise FileNotFoundError(f"capture-log image does not exist: {filename}")
        path = by_name[filename]
        label = int(record["label"])
        if label not in (0, 1):
            raise ValueError(f"invalid label for {filename}: expected 0 or 1")
        rows.append({
            "path": str(path), "label": label,
            "group": str(record["sample_id"]),
            "camera": str(record["phone"]), "capture_date": str(record["date"]),
            "session": str(record["session"]), "iso": record["ISO"],
            "exposure": record["exposure"], "flash_status": str(record["flash"]),
            "background": str(record["background"]),
            "sha256": sha256_file(path),
        })
    manifest = pd.DataFrame(rows)
    if manifest.empty:
        raise ValueError("capture log contains no images")
    if manifest.groupby("group").label.nunique().gt(1).any():
        raise ValueError("each physical sample must have one consistent label")
    validate_capture_design(manifest)
    counts = manifest.groupby("group").size()
    if counts.lt(3).any():
        raise ValueError("each locked sample must contain at least three image shots")
    if manifest.path.duplicated().any():
        raise ValueError("each image file may appear only once in the capture log")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "locked_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    os.chmod(manifest_path, 0o444)
    lock_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    lock_path = output_dir / "LOCKED.sha256"
    lock_path.write_text(lock_digest + "\n", encoding="ascii")
    os.chmod(lock_path, 0o444)
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True)
    parser.add_argument("--capture-log", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    resolve = lambda value: Path(value).resolve()
    manifest = create_locked_manifest(
        resolve(args.images), resolve(args.capture_log), resolve(args.out)
    )
    print(f"Locked test manifest: {manifest}")
    print("Do not edit or re-evaluate this test set.")


if __name__ == "__main__":
    main()
