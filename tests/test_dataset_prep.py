"""prepare_dataset.py on a miniature copy of the real naming scheme (Training/Testing share samples, one duplicate photo)."""
import os, subprocess, sys, zipfile
import numpy as np, pandas as pd
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _img(path, seed):
    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 255, (40 + seed % 7, 50, 3), dtype=np.uint8)).save(path)


def test_prepare_dataset_is_leak_free(tmp_path):
    src = tmp_path / "src"; seed = 0
    for cls, ids, typ in [("Fake", range(1, 9), ""), ("Fake", range(9, 12), " M"), ("Real", range(1, 13), "")]:
        for split, shots in [("Training", (1, 2, 3)), ("Testing", (4, 5))]:
            d = src / split / cls; d.mkdir(parents=True, exist_ok=True)
            for i in ids:
                for s in shots:
                    seed += 1; _img(d / f"{i}{typ}_{s}.jpg", seed)
    # a repeated photo under a DIFFERENT sample id (as in the real data): Real 2 shot 1 == Real 7 shot 1
    (src / "Training/Real/7_1.jpg").write_bytes((src / "Training/Real/2_1.jpg").read_bytes())
    r = subprocess.run([sys.executable, "scripts/prepare_dataset.py", "--src", str(src), "--data", str(tmp_path / "data"), "--folds", "4"],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "[LEAK]" in r.stdout and "[DUP]" in r.stdout
    m = pd.read_csv(tmp_path / "data" / "manifest.csv")
    assert len(m) == 8 * 5 + 3 * 5 + 12 * 5 - 1                                    # one duplicate file dropped
    assert m.groupby("group").fold.nunique().max() == 1                            # a group never spans two folds
    assert m[m["sample"] == "real---2"].group.iloc[0] == m[m["sample"] == "real---7"].group.iloc[0]   # duplicate merged
    assert set(m.type) == {"-", "M"} and m.fold.nunique() == 4
    assert m.groupby("fold").label.mean().between(0.2, 0.8).all()                    # stratified


def test_prepare_dataset_expands_zip_wildcards(tmp_path):
    archives = tmp_path / "archives"; archives.mkdir()
    for split, pattern, first_id in [("Testing", "Test_dataset", 1), ("Training", "train_dataset", 3)]:
        source = tmp_path / split
        for cls in ("Fake", "Real"):
            directory = source / split / cls
            directory.mkdir(parents=True)
            for sample_id in range(first_id, first_id + 2):
                _img(directory / f"{sample_id}_1.jpg", sample_id)
        with zipfile.ZipFile(archives / f"{pattern}-sample.zip", "w") as archive:
            for path in source.rglob("*.jpg"):
                archive.write(path, path.relative_to(source))

    r = subprocess.run(
        [sys.executable, "scripts/prepare_dataset.py", "--src",
         str(archives / "Test_dataset-*.zip"), str(archives / "train_dataset-*.zip"),
         "--data", str(tmp_path / "data"), "--folds", "2"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert "found 8 images" in r.stdout
    assert (tmp_path / "data" / "manifest.csv").exists()


def test_sample_level_aggregation():
    import wombat_beam as wb
    p = np.array([.9, .8, .2, .1, .3, .7]); y = np.array([1, 1, 0, 0, 0, 0]); g = np.array(["a", "a", "b", "b", "c", "c"])
    ps, ys = wb.aggregate_by_group(p, y, g)
    assert ys.tolist() == [1, 0, 0] and ps[0] > 0.8 and ps[1] < 0.2 and abs(ps[2] - 0.5) < 1e-6


def test_label_shuffle_preserves_groups_and_each_fold_class_counts():
    import importlib.util
    script = os.path.join(ROOT, "scripts", "label_shuffle_test.py")
    spec = importlib.util.spec_from_file_location("label_shuffle_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    manifest = pd.DataFrame({
        "group": ["a", "a", "b", "c", "c", "d", "e", "e", "f"],
        "label": [0, 0, 1, 0, 0, 1, 0, 0, 1],
        "fold": [0, 0, 0, 1, 1, 1, 2, 2, 2],
    })
    shuffled = module.shuffle_partition_labels(manifest, fold=0, seed=9)
    assert shuffled.loc[shuffled.fold == 0, "label"].tolist() == [0, 0, 1]
    for fold in (1, 2):
        assert (
            shuffled.loc[shuffled.fold == fold].groupby("group").label.first()
            .value_counts().sort_index()
            .equals(
                manifest.loc[manifest.fold == fold].groupby("group").label.first()
                .value_counts().sort_index()
            )
        )
    assert (shuffled.groupby("group").label.nunique() == 1).all()
