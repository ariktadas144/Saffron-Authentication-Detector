"""Colour-statistics baseline.  On this dataset, mean/std of RGB+HSV alone already separates real from fake very well
(group-CV AUC ~0.94 in our check), so ANY deep-learning claim must be compared against this row."""
import os
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier

from manifest_paths import normalize_manifest_path


def features(paths, cache=None):
    paths = [normalize_manifest_path(path) for path in paths]
    if cache and os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        if list(d["paths"]) == list(paths): return d["X"]
    X = []
    for p in paths:
        im = Image.open(p).convert("RGB"); w, h = im.size; s = min(w, h)
        im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).resize((64, 64))
        a = np.asarray(im, float) / 255; v = np.asarray(im.convert("HSV"), float) / 255
        X.append([*a.mean((0, 1)), *a.std((0, 1)), *v.mean((0, 1)), *v.std((0, 1))])
    X = np.array(X)
    if cache: np.savez(cache, paths=np.array(paths), X=X)
    return X


def fit_predict(Xtr, ytr, Xte, seed=0):
    m = RandomForestClassifier(300, min_samples_leaf=2, random_state=seed, class_weight="balanced").fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1]
