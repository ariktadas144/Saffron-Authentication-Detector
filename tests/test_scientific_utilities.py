from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def test_report_rejects_smoke_runs_and_untraceable_numbers(tmp_path):
    from scripts.build_report import eligible_run, verify_report_numbers

    assert eligible_run(tmp_path / "SMOKE_TEST_NOT_A_RESULT_run")[0] is False
    run = tmp_path / "cv"
    fold = run / "fold0"
    fold.mkdir(parents=True)
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("path,label,group\nimage.png,0,g1\n")
    (run / "cv_run.json").write_text(json.dumps({
        "fold_count": 1, "img_size": 256, "epochs_warm": 3,
        "epochs_ft": 25, "mc_samples": 16, "seed": 42,
        "imagenet": True, "manifest": str(manifest),
    }))
    (fold / "config.json").write_text(json.dumps({"imagenet": True, "epochs_ft": 25}))
    (fold / "preds_test.csv").write_text("group,y\ns1,0\n")
    (fold / "sample_thresholds.json").write_text("{}")
    (fold / "thresholds.json").write_text("{}")
    (fold / "wombat_params.json").write_text("{}")
    (fold / "ood_reference.npz").write_text("placeholder")
    (fold / "final.weights.h5").write_text("placeholder")
    assert eligible_run(run)[0] is True
    run_meta = json.loads((run / "cv_run.json").read_text())
    run_meta["imagenet"] = False
    (run / "cv_run.json").write_text(json.dumps(run_meta))
    assert eligible_run(run) == (False, "run metadata indicates --no-imagenet or missing ImageNet weights")
    run_meta.update(imagenet=True, epochs_ft=4)
    (run / "cv_run.json").write_text(json.dumps(run_meta))
    assert eligible_run(run) == (False, "run metadata fine-tuning epochs below 5")

    csv = tmp_path / "source.csv"
    pd.DataFrame({"accuracy": [0.5], "n_samples": [4]}).to_csv(csv, index=False)
    verify_report_numbers("accuracy 0.500 from n=4", [csv])
    with pytest.raises(ValueError, match="numeric token"):
        verify_report_numbers("accuracy 0.700", [csv])


def test_matched_subset_reports_exif_baseline_on_same_groups():
    from scripts.matched_subset_eval import evaluate_subsets

    samples = pd.DataFrame({
        "group": ["a", "b", "c", "d"],
        "y": [0, 0, 1, 1],
        "camera": ["Samsung SM-G973F"] * 4,
        "flash_status": ["off"] * 4,
        "iso": [100, 125, 100, 160],
        "capture_date": ["2024-01-01"] * 4,
        "type": ["-", "-", "C", "C"],
        "n_images": [3, 3, 3, 3],
        "proposed_calibrated": [0.1, 0.2, 0.8, 0.9],
        "exif_calibrated": [0.2, 0.3, 0.7, 0.8],
    })
    result = evaluate_subsets(samples, n_bootstrap=20)
    matched = result[result.scope == "SM-G973F ISO 100/125/160, flash off"]
    assert set(matched.method) == {"Proposed", "EXIF-only baseline"}
    assert set(matched.n_samples) == {4}
    assert set(matched.n_fake) == {2}
    assert set(matched.n_real) == {2}


def test_counterfactual_transforms_are_deterministic_and_keep_size():
    from scripts.exposure_counterfactual import counterfactual

    image = Image.new("RGB", (20, 16), (40, 80, 120))
    dark = counterfactual(image, "brightness_0.6")
    noise_a = counterfactual(image, "noise_8", seed=13)
    noise_b = counterfactual(image, "noise_8", seed=13)
    assert dark.size == image.size
    assert np.asarray(dark).mean() < np.asarray(image).mean()
    assert np.array_equal(np.asarray(noise_a), np.asarray(noise_b))


def test_session_folds_are_group_level_and_refuse_too_few_dates():
    from scripts.session_analysis import make_date_folds

    groups = pd.DataFrame({
        "group": ["a", "b", "c", "d", "e", "f"],
        "label": [0, 1, 0, 1, 0, 1],
        "capture_date": ["d1", "d1", "d2", "d2", "d3", "d3"],
        "camera": ["phone"] * 6,
        "iso_bucket": ["100"] * 6,
    })
    result = make_date_folds(groups)
    assert result.groupby("group").fold.nunique().max() == 1
    assert result.groupby("fold").label.nunique().eq(2).all()
    assert result.fold.tolist() == [0, 0, 1, 1, 2, 2]
    with pytest.raises(ValueError, match="at least three"):
        make_date_folds(groups[groups.capture_date != "d3"])


def test_session_summary_refuses_groups_spanning_labels_or_dates():
    from scripts.session_analysis import session_summary

    manifest = pd.DataFrame({
        "path": ["a.png", "b.png"],
        "group": ["same-sample", "same-sample"],
        "label": [0, 1],
    })
    audit = pd.DataFrame({
        "path": ["a.png", "b.png"],
        "camera": ["phone", "phone"],
        "capture_date": ["d1", "d1"],
        "iso": [100, 100],
    })
    with pytest.raises(ValueError, match="inconsistent labels"):
        session_summary(manifest, audit)


def test_capture_invariant_luminance_normalization_preserves_rgb_ratios():
    import tensorflow as tf
    from saffron_cnn_gnn import normalize_luminance_preserve_chromaticity

    image = tf.ones((1, 4, 4, 3), dtype=tf.float32) * tf.constant([30.0, 60.0, 90.0])
    normalized = normalize_luminance_preserve_chromaticity(image).numpy()
    ratios_before = image.numpy()[0, 0, 0] / image.numpy()[0, 0, 0, 1]
    ratios_after = normalized[0, 0, 0] / normalized[0, 0, 0, 1]
    assert np.allclose(ratios_before, ratios_after, atol=1e-6)


def test_locked_test_set_hashes_and_exactly_once_guard(tmp_path):
    from scripts.evaluate_locked import claim_evaluation_once, verify_locked_manifest
    from scripts.lock_test_set import create_locked_manifest

    images = tmp_path / "images"
    images.mkdir()
    rows = []
    for phone_index in range(3):
        phone = f"TestPhone-{phone_index}"
        for label in (0, 1):
            for sample_index in range(30):
                group = f"{phone}-{label}-{sample_index}"
                for shot in range(3):
                    name = f"{group}-{shot}.png"
                    Image.new("RGB", (8, 8), (label * 100, shot, phone_index)).save(
                        images / name
                    )
                    rows.append({
                        "filename": name, "date": "2025-01-01", "phone": phone,
                        "sample_id": group, "label": label, "session": "s1",
                        "ISO": 100, "exposure": "1/60", "flash": "off",
                        "background": "white",
                    })
    log = tmp_path / "capture.csv"
    pd.DataFrame(rows).to_csv(log, index=False)
    lock_dir = tmp_path / "locked"
    create_locked_manifest(images, log, lock_dir)
    manifest = verify_locked_manifest(lock_dir)
    assert len(manifest) == 540
    assert manifest.group.nunique() == 180
    claim_evaluation_once(lock_dir)
    with pytest.raises(RuntimeError, match="already been claimed"):
        claim_evaluation_once(lock_dir)
    target = Path(manifest.path.iloc[0])
    target.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed or is missing"):
        verify_locked_manifest(lock_dir)


def test_locked_test_design_requires_minimum_per_phone_sample_groups():
    from scripts.lock_test_set import validate_capture_design

    manifest = pd.DataFrame({
        "group": ["a", "b", "c", "d", "e", "f"],
        "label": [0, 1, 0, 1, 0, 1],
        "camera": ["phone-a", "phone-a", "phone-b", "phone-b", "phone-c", "phone-c"],
    })
    with pytest.raises(ValueError, match="at least 30 independent groups"):
        validate_capture_design(manifest)


def test_inference_preprocessing_matches_training_center_crop_and_resize(tmp_path):
    import tensorflow as tf
    import saffron_cnn_gnn as sg
    from app.predict import _preprocess_path

    image_path = tmp_path / "wide.png"
    pixels = np.zeros((12, 20, 3), dtype=np.uint8)
    pixels[:, 4:16] = (120, 40, 20)
    Image.fromarray(pixels).save(image_path)
    training = sg._load_image(tf.constant(str(image_path)), 16, False)[None, ...]
    inference = _preprocess_path(str(image_path), 16)
    assert np.array_equal(training.numpy(), inference.numpy())


def test_group_conformal_reports_camera_coverage_and_sensitivity():
    from scripts.calibrate_inference import calibration_outputs

    samples = pd.DataFrame({
        "group": [f"g{i}" for i in range(8)],
        "y": [0, 0, 0, 0, 1, 1, 1, 1],
        "p_fake": [0.05, 0.1, 0.2, 0.3, 0.7, 0.8, 0.9, 0.95],
        "camera": ["phone-a"] * 4 + ["phone-b"] * 4,
        "n_images": [3] * 8,
    })
    metadata = pd.DataFrame({
        "group": samples.group, "camera": samples.camera,
        "label": samples.y, "iso": [100, 125, 160, 200] * 2,
        "flash_status": ["off"] * 8,
    })
    uncertainty = pd.Series([0.01] * 8, index=samples.group)
    config, report = calibration_outputs(
        samples, metadata, uncertainty,
        data_version="synthetic-data", model_version="synthetic-model",
    )
    assert config["alpha_default"] == 0.05
    assert config["alpha_sensitivity"] == 0.10
    assert set(config["quantiles"]) == {"0.05", "0.10"}
    assert set(config["mondrian_quantiles_report_only"]["0.05"]) == {"0", "1"}
    assert {"pooled", "mondrian_class_0", "mondrian_class_1"} <= set(report.calibration)
    assert set(report.camera) == {"phone-a", "phone-b"}
    assert config["max_mc_uncertainty"] == pytest.approx(0.01)
    assert config["data_version"] == "synthetic-data"
    assert config["model_version"] == "synthetic-model"


def test_bundle_fingerprint_changes_with_model_artifacts(tmp_path):
    from src.inference_safety import model_bundle_fingerprint

    run = tmp_path / "run"
    fold = run / "fold0"
    fold.mkdir(parents=True)
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("path,label,group\nx.jpg,0,g1\n")
    (run / "cv_run.json").write_text(json.dumps({"fold_count": 1}))
    for name, content in (
        ("config.json", "{}"),
        ("final.weights.h5", "weights"),
        ("wombat_params.json", "{}"),
        ("thresholds.json", "{}"),
        ("sample_thresholds.json", "{}"),
        ("ood_reference.npz", "reference"),
    ):
        (fold / name).write_text(content)
    original = model_bundle_fingerprint(run, manifest)
    (fold / "final.weights.h5").write_text("changed")
    assert model_bundle_fingerprint(run, manifest) != original


def test_manifest_identity_normalizes_paths_and_checks_group_label_fold():
    from src.manifest_paths import validate_prediction_manifest_identity

    manifest = pd.DataFrame({
        "path": [r"images\a.jpg"], "group": ["sample-1"],
        "label": [1], "fold": [2],
    })
    predictions = pd.DataFrame({
        "path": ["images/a.jpg"], "group": ["sample-1"],
        "y": [1], "fold": [2],
    })
    validate_prediction_manifest_identity(manifest, predictions)
    predictions.loc[0, "fold"] = 1
    with pytest.raises(ValueError, match="fold assignments"):
        validate_prediction_manifest_identity(manifest, predictions)


def test_paired_decoder_bootstrap_requires_same_groups_and_reports_differences():
    from scripts.paired_decoder_comparison import paired_bootstrap

    groups = [f"g{i}" for i in range(8)]
    labels = [0, 0, 0, 0, 1, 1, 1, 1]
    proposed = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]
    comparator = [0.6, 0.6, 0.6, 0.6, 0.4, 0.4, 0.4, 0.4]
    frames = [
        pd.DataFrame({
            "group": groups, "y": labels,
            "n_images": [2] * len(groups),
            "PROPOSED: Wombat-OA + weighted beam + log-prob fusion": proposed,
            "single view, cnn": comparator,
        })
        for _ in range(3)
    ]
    result = paired_bootstrap(frames, "single view, cnn", n_bootstrap=100, seed=4)
    assert set(result.metric) == {"balanced_accuracy", "auc"}
    assert (result.difference_proposed_minus_comparator > 0).all()
    frames[-1] = frames[-1].iloc[:-1]
    with pytest.raises(ValueError, match="identical held-out"):
        paired_bootstrap(frames, "single view, cnn", n_bootstrap=10)


def test_report_metric_rows_include_thresholded_group_bootstrap_intervals():
    from scripts.build_report import _metric_rows

    frame = pd.DataFrame({
        "group": ["a", "b", "c", "d"],
        "y": [0, 0, 1, 1],
        "n_images": [1, 1, 1, 1],
        "single view, cnn": [0.2, 0.4, 0.6, 0.8],
        "single view, cnn__threshold": [0.5] * 4,
        "single view, cnn__image_threshold": [0.5] * 4,
    })
    rows = _metric_rows(frame, level="sample", n_bootstrap=50)
    assert {"acc", "bal_acc", "f1_fake", "recall_fake", "specificity", "auc"} <= set(rows.metric)
    assert rows[rows.metric == "bal_acc"].ci95_low.notna().all()


def test_report_builder_emits_versioned_report_from_eligible_artifacts(tmp_path):
    from scripts.build_report import build_report

    run = tmp_path / "cv"
    fold = run / "fold0"
    fold.mkdir(parents=True)
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "path,label,group,fold\n"
        "a.png,0,a,0\nb.png,0,b,0\nc.png,1,c,0\nd.png,1,d,0\n"
    )
    metadata = {
        "manifest": str(manifest), "fold_count": 1, "img_size": 256,
        "epochs_warm": 3, "epochs_ft": 25, "mc_samples": 16,
        "seed": 42, "imagenet": True,
    }
    (run / "cv_run.json").write_text(json.dumps(metadata))
    (fold / "config.json").write_text(json.dumps({
        "imagenet": True, "epochs_ft": 25,
    }))
    (fold / "final.weights.h5").write_text("weights")
    (fold / "wombat_params.json").write_text("{}")
    (fold / "ood_reference.npz").write_text("reference")
    method = "PROPOSED: Wombat-OA + weighted beam + log-prob fusion"
    predictions = pd.DataFrame({
        "path": ["a.png", "b.png", "c.png", "d.png"],
        "group": ["a", "b", "c", "d"], "y": [0, 0, 1, 1],
        "camera": ["phone"] * 4, "flash_status": ["off"] * 4,
        "iso": [100] * 4, "capture_date": ["2025-01-01"] * 4,
        "type": ["-"] * 4, method: [0.1, 0.2, 0.8, 0.9],
    })
    predictions.to_csv(fold / "preds_test.csv", index=False)
    (fold / "thresholds.json").write_text(json.dumps({method: 0.5}))
    (fold / "sample_thresholds.json").write_text(json.dumps({method: 0.5}))

    report = build_report(run, tmp_path / "report", n_bootstrap=50)
    text = report.read_text(encoding="utf-8")
    assert "Model bundle fingerprint:" in text
    assert "manifest fingerprint:" in text
    assert "balanced_accuracy" in text
    assert (tmp_path / "report" / "results_tables" / "run_provenance.csv").is_file()
