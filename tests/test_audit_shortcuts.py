from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.audit_shortcuts import (
    read_image_audit,
    resolve_manifest_path,
)
from src.manifest_paths import normalize_manifest_path


def test_manifest_windows_path_resolves_on_current_platform(tmp_path):
    target = tmp_path / "folder" / "original.jpg"
    target.parent.mkdir()
    target.touch()

    resolved = resolve_manifest_path(
        r"folder\original.jpg",
        root=tmp_path,
    )

    assert resolved == target


def test_shared_manifest_path_normalizer_converts_windows_separators():
    assert normalize_manifest_path(r"data\images\real\sample.jpg") == os.path.join(
        "data", "images", "real", "sample.jpg"
    )


def test_read_image_audit_reports_exif_and_file_facts(tmp_path):
    path = tmp_path / "capture.jpg"
    image = Image.new("RGB", (32, 24), (120, 40, 10))
    exif = image.getexif()
    exif[271] = "Example Make"
    exif[272] = "Example Model"
    exif[305] = "Example Editor"
    exif[34855] = 200
    exif[33434] = (1, 50)
    exif[33437] = (28, 10)
    exif[37385] = 1
    image.save(path, exif=exif, quality=90)

    result = read_image_audit(path)

    assert result["original_exists"] is True
    assert result["Make"] == "Example Make"
    assert result["Model"] == "Example Model"
    assert result["Software"] == "Example Editor"
    assert result["iso"] == 200
    assert result["exposure_time_s"] == 0.02
    assert result["f_number"] == 2.8
    assert result["flash_status"] == "on"
    assert result["width"] == 32
    assert result["height"] == 24
    assert result["file_size_bytes"] == path.stat().st_size
    assert result["jpeg_quality_estimate"] == 90
    assert result["original_path_resolved"] == path.as_posix()


def test_missing_original_is_explicit_and_not_inferred(tmp_path):
    result = read_image_audit(Path(tmp_path / "missing.jpg"))

    assert result["original_exists"] is False
    assert result["metadata_error"] == "original file missing"
    assert np.isnan(result["Model"])


def test_present_image_without_camera_exif_is_marked_unknown(tmp_path):
    path = tmp_path / "no-exif.jpg"
    Image.new("RGB", (16, 12), "red").save(path, quality=90)

    result = read_image_audit(path)

    assert result["original_exists"] is True
    assert result["camera"] == "unknown"
    assert result["capture_date"] == "unknown"
    assert np.isnan(result["iso"])
