from __future__ import annotations

import csv
import json

from PIL import Image

import saffron_cnn_gnn as sg
from app.predict import load_model, predict_image


def test_colour_baseline_loads_from_manifest_without_run_directory(tmp_path):
    data_dir = tmp_path / "images"
    data_dir.mkdir()
    manifest = tmp_path / "manifest.csv"
    rows = []

    for label, colour in ((0, (190, 35, 20)), (1, (30, 80, 180))):
        for index in range(3):
            image_path = data_dir / f"{label}_{index}.png"
            Image.new("RGB", (80, 40), colour).save(image_path)
            rows.append({"path": str(image_path), "label": label})

    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("path", "label"))
        writer.writeheader()
        writer.writerows(rows)

    bundle = load_model(str(tmp_path / "missing-runs"), str(manifest))
    result = predict_image(Image.new("RGB", (80, 40), (190, 35, 20)), bundle)

    assert bundle.method == "colour"
    assert bundle.colour_clf is not None
    assert result["label"] in {"REAL", "FAKE"}
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["method"] == "colour"
    assert result["details"]["baseline_trained"] is True
    assert set(result["details"]["colour_features"]) == {
        "R mean", "G mean", "B mean", "R std", "G std", "B std",
        "H mean", "S mean", "V mean", "H std", "S std", "V std",
    }


def test_colour_heuristic_is_available_without_manifest(tmp_path):
    bundle = load_model(
        str(tmp_path / "missing-runs"),
        str(tmp_path / "missing-manifest.csv"),
    )
    result = predict_image(Image.new("RGB", (30, 50), (180, 30, 20)), bundle)

    assert bundle.method == "colour"
    assert bundle.colour_clf is None
    assert result["method"] == "colour"
    assert result["details"]["baseline_trained"] is False
    assert "Manifest not found" in result["details"]["fallback_reason"]


def test_deep_weights_without_decoder_metadata_use_default_parameters(
    tmp_path, monkeypatch
):
    fold_dir = tmp_path / "fold0"
    fold_dir.mkdir()
    (fold_dir / "best.weights.h5").write_bytes(b"test weights")
    (fold_dir / "config.json").write_text(json.dumps({"img_size": 128}))

    class Model:
        def load_weights(self, path):
            assert path.endswith("best.weights.h5")

    monkeypatch.setattr(sg, "build_models", lambda cfg: {"model": Model()})

    bundle = load_model(str(tmp_path), str(tmp_path / "missing-manifest.csv"))

    assert bundle.method == "deep"
    assert bundle.decoder_tuned is False
    assert bundle.folds[0]["wparams"]["theta"] == 0.5
    assert bundle.folds[0]["thr"] == {}
    assert "untuned decoder" in bundle.fallback_reason
