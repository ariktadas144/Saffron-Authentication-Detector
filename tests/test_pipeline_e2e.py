"""End-to-end on a tiny synthetic dataset: check_dataset -> split_dataset -> train -> decode.  Slow (minutes on CPU)."""
import os, subprocess, sys
import pytest
pytest.importorskip("tensorflow")
import saffron_cnn_gnn as sg

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
run = lambda *a, cwd=ROOT: subprocess.run([sys.executable, *a], cwd=cwd, capture_output=True, text=True, env={**os.environ, "TF_CPP_MIN_LOG_LEVEL": "3"})


@pytest.mark.slow
def test_full_pipeline(tmp_path):
    raw, data, out = tmp_path / "raw", tmp_path / "data", tmp_path / "run"
    sg._make_fake_dataset(str(tmp_path / "pre"), n=40, size=96)           # make images, then re-pool them as "raw"
    for cls in sg.CLASS_NAMES:
        os.makedirs(raw / cls)
        for i, f in enumerate(os.listdir(tmp_path / "pre" / "train" / cls)):
            os.replace(tmp_path / "pre" / "train" / cls / f, raw / cls / f"lot{i // 2}_{f}")
    r = run("scripts/split_dataset.py", "--src", str(raw), "--dst", str(data), "--group-regex", r"^(lot\d+)_"); assert r.returncode == 0, r.stderr
    r = run("scripts/check_dataset.py", "--data", str(data)); assert r.returncode == 0 and "DATASET OK" in r.stdout, r.stdout + r.stderr
    r = run("scripts/train.py", "--data", str(data), "--out", str(out), "--img-size", "128", "--batch", "8", "--epochs-warm", "1",
            "--epochs-ft", "1", "--mc-samples", "4", "--no-imagenet"); assert r.returncode == 0, r.stderr[-2000:]
    assert (out / "cache_val.npz").exists() and (out / "cache_test.npz").exists() and (out / "final.weights.h5").exists()
    r = run("scripts/decode.py", "--out", str(out), "--pop", "6", "--iters", "3", "--seeds", "2"); assert r.returncode == 0, r.stderr[-2000:]
    assert (out / "ablation_test.csv").exists() and (out / "optimizer_comparison.png").exists() and "DONE" in r.stdout
