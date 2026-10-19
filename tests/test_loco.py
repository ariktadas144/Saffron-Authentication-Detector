"""prepare_dataset.py on a miniature copy with two cameras -> LOCO manifests are group-safe and two-class."""
import os, subprocess, sys
import numpy as np, pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def test_loco_manifests_are_group_safe_and_two_class(tmp_path):
    rows = []
    for cam, n_real, n_fake in [("CAM-A", 12, 8), ("CAM-B", 8, 12)]:
        for k in range(n_real + n_fake):
            lab = int(k >= n_real)
            for s in range(3):
                rows.append(dict(
                    path=f"x_{cam}_{k}_{s}.jpg", label=lab,
                    cls="fake" if lab else "real",
                    type=("M" if k % 2 else "-") if lab else "-",
                    sample=f"{cam}{k}", group=f"{cam}-{k}", shot=s,
                    orig_split="train", fold=0, camera=cam))
    # one row with unknown camera (real-only, should be excluded)
    rows += [dict(path="u.jpg", label=0, cls="real", type="-", sample="u",
                  group="U", shot=1, orig_split="train", fold=0,
                  camera="unknown")]
    pd.DataFrame(rows).to_csv(tmp_path / "manifest.csv", index=False)

    r = subprocess.run(
        [sys.executable, "scripts/make_loco_manifests.py",
         "--manifest", str(tmp_path / "manifest.csv")],
        cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr

    for a, b in [("CAMA", "CAMB"), ("CAMB", "CAMA")]:
        m = pd.read_csv(tmp_path / f"manifest_loco_{a}_to_{b}.csv")
        # all three roles present, unknown camera excluded
        assert set(m.fold) == {0, 1, 2} and "U" not in set(m.group)
        # no group in two roles
        assert m.groupby("group").fold.nunique().max() == 1
        # test camera is the OTHER one, train+val is the ONE
        assert set(m[m.fold == 0].camera) == {f"CAM-{b[-1]}"}
        assert set(m[m.fold != 0].camera) == {f"CAM-{a[-1]}"}
        # both classes in test, val and train
        for f in (0, 1, 2):
            assert m[m.fold == f].label.nunique() == 2
