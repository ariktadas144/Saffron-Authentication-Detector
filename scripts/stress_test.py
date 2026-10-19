"""Stress test: re-evaluate the SAVED fold models on degraded test images
(no retraining, no retuning).

When every method scores 100% on clean images, nothing can be compared.
Degrading only the TEST images (blur, JPEG, noise, low resolution, grayscale,
hue shift, darkening) shows (a) what the network relies on (colour? fine
texture?) and (b) whether log-probability fusion, the weighted beam and the
Wombat-tuned parameters are more robust than a plain average.

   python scripts/stress_test.py --manifest data/manifest.csv --run runs/cv1
   python scripts/stress_test.py --manifest data/manifest.csv --run runs/cv1 --folds 0 1 --perturb clean blur jpeg

Needs per fold: config.json, final.weights.h5 (or best.weights.h5),
wombat_params.json, thresholds.json (all written by run_cv.py; weights are
only on the Colab disk unless you set SAVE_WEIGHTS_TO_DRIVE). Parameters tuned
on CLEAN validation data are reused unchanged.

The colour-statistics baseline is not re-evaluated here (it would need its own
feature extraction on degraded images).

Outputs: <run>/stress_results.csv and printed tables (sample level, one
decision per physical sample, pooled over folds).
"""
import _path  # noqa
import argparse, dataclasses, json, os
import numpy as np, pandas as pd
import tensorflow as tf
import saffron_cnn_gnn as sg
import wombat_beam as wb

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", default="data/manifest.csv")
ap.add_argument("--run", required=True)
ap.add_argument("--folds", type=int, nargs="*", default=None)
ap.add_argument("--perturb", nargs="*",
                default=["clean", "blur", "jpeg", "noise", "lowres", "gray",
                         "hue", "dark"])
ap.add_argument("--batch", type=int, default=8)
a = ap.parse_args()


# ---------------------------------------------------------------------------
# Perturbation functions (applied to batches of 0-255 float32 images)
# ---------------------------------------------------------------------------
def _gauss_kernel(sigma=2.0, size=9):
    x = np.arange(size) - size // 2
    k = np.exp(-(x ** 2) / (2 * sigma ** 2))
    k /= k.sum()
    k2 = np.outer(k, k).astype("float32")
    return tf.constant(np.tile(k2[:, :, None, None], (1, 1, 3, 1)))


GK = _gauss_kernel()

PERTURB = {
    "clean":  None,
    "blur":   lambda x: tf.nn.depthwise_conv2d(x, GK, [1, 1, 1, 1], "SAME"),
    "jpeg":   lambda x: tf.map_fn(
        lambda im: tf.image.adjust_jpeg_quality(im / 255.0, 20) * 255.0, x),
    "noise":  lambda x: tf.clip_by_value(
        x + tf.random.normal(tf.shape(x), 0.0, 12.0, seed=0), 0.0, 255.0),
    "lowres": lambda x: tf.image.resize(
        tf.image.resize(x, (x.shape[1] // 4, x.shape[2] // 4)),
        (x.shape[1], x.shape[2])),
    "gray":   lambda x: tf.tile(
        tf.image.rgb_to_grayscale(x), [1, 1, 1, 3]),
    "hue":    lambda x: tf.clip_by_value(
        tf.image.adjust_hue(x / 255.0, 0.08) * 255.0, 0.0, 255.0),
    "dark":   lambda x: x * 0.6,
}


# ---------------------------------------------------------------------------
# Dataset builder (deterministic, no augmentation)
# ---------------------------------------------------------------------------
def dataset(df, S, batch, fn):
    ds = tf.data.Dataset.from_tensor_slices(
        (df.path.values.astype(str), df.label.values.astype("float32")))
    ds = ds.map(lambda p, l: (sg._load_image(p, S, False), l[None]),
                num_parallel_calls=tf.data.AUTOTUNE).batch(batch)
    if fn is not None:
        ds = ds.map(lambda x, y: (fn(x), y))
    return ds


# ---------------------------------------------------------------------------
# Method builder (mirrors the ablation table rows)
# ---------------------------------------------------------------------------
def methods_for(cache, wparams):
    H = len(cache.head_names)
    V = len(cache.view_names)
    hafm = cache.head_names[-1]

    rows = {f"single view, {h}": (wb.b_single(cache, h), 0.5)
            for h in cache.head_names}
    rows["late fusion (prob avg), identity view"] = \
        (wb.b_late_fusion_identity(cache), 0.5)
    rows["TTA mean-logit, hafm"] = (wb.b_tta_mean_logit(cache, hafm), 0.5)
    rows["TTA top-k confidence (old), hafm"] = \
        (wb.b_tta_topk_conf(cache, hafm), 0.5)
    rows["log-prob fusion, equal weights, no beam"] = \
        (wb.b_logprob_fusion_no_beam(cache), 0.5)

    p0 = wb.unpack(wb.default_params(H, V), H, V)
    rows["weighted beam + log-prob fusion (untuned)"] = \
        (wb.decode(cache, p0), p0["theta"])

    pw = dict(
        a=np.array(wparams["a"]),
        tau=np.array(wparams["tau"]),
        beta=wparams["beta"],
        gamma=wparams["gamma"],
        alpha=wparams["alpha"],
        beam=int(wparams["beam"]),
        depth=int(wparams["depth"]),
        kappa=wparams["kappa"],
        theta=wparams["theta"],
    )
    rows["PROPOSED: Wombat-OA + weighted beam + log-prob fusion"] = \
        (wb.decode(cache, pw), pw["theta"])

    return rows


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
man = pd.read_csv(a.manifest)
K = int(man.fold.max()) + 1
folds = a.folds if a.folds is not None else list(range(K))

store = {}  # (perturb, method) -> list of (p_shifted, y, group)

for k in folds:
    fd = f"{a.run}/fold{k}"
    w = next((f"{fd}/{n}" for n in ("final.weights.h5", "best.weights.h5")
              if os.path.exists(f"{fd}/{n}")), None)
    if w is None or not os.path.exists(f"{fd}/wombat_params.json"):
        print(f"fold {k}: skipped (no saved weights / params in {fd})")
        continue

    raw = json.load(open(f"{fd}/config.json"))
    cfg = sg.Config(**{f.name: raw[f.name]
                       for f in dataclasses.fields(sg.Config) if f.name in raw})
    cfg.imagenet = False  # weights come from the file

    M = sg.build_models(cfg)
    M["model"].load_weights(w)
    wparams = json.load(open(f"{fd}/wombat_params.json"))
    thr = json.load(open(f"{fd}/thresholds.json"))

    test_df = man[man.fold == k].reset_index(drop=True)
    print(f"\nfold {k}: {len(test_df)} test images / "
          f"{test_df.group.nunique()} samples, weights {os.path.basename(w)}")

    for pert in a.perturb:
        cache = sg.extract_cache(
            M, dataset(test_df, cfg.img_size, a.batch, PERTURB[pert]),
            cfg, groups=test_df.group.values)
        for name, (p, t) in methods_for(cache, wparams).items():
            t = min(max(t, 1e-6), 1 - 1e-6)
            pc = np.clip(p, 1e-6, 1 - 1e-6)
            # shift so each method's own threshold becomes 0.5
            ps = 1 / (1 + np.exp(-(np.log(pc / (1 - pc))
                                   - np.log(t / (1 - t)))))
            store.setdefault((pert, name), []).append(
                (ps, cache.y, cache.groups))
        print(f"   {pert:7s} done")
    tf.keras.backend.clear_session()

if not store:
    raise SystemExit("nothing evaluated: no fold had saved weights")

# ---------------------------------------------------------------------------
# Aggregate and print tables
# ---------------------------------------------------------------------------
rows = []
for (pert, name), parts in store.items():
    p = np.concatenate([x[0] for x in parts])
    y = np.concatenate([x[1] for x in parts])
    g = np.concatenate([x[2] for x in parts])
    ps, ys = wb.aggregate_by_group(p, y, g)
    for level, (pp, yy) in {"image": (p, y), "sample": (ps, ys)}.items():
        r = wb.metrics(yy, pp, 0.5)
        rows.append(dict(
            perturbation=pert, method=name, level=level, n=len(yy),
            acc=r["acc"], f1_fake=r["f1_fake"], recall_fake=r["recall_fake"],
            specificity=r["specificity"], auc=r["auc"]))

R = pd.DataFrame(rows)
R.to_csv(f"{a.run}/stress_results.csv", index=False)

pd.set_option("display.width", 250, "display.max_columns", 30)
for level in ("sample", "image"):
    for metric in ("acc", "auc"):
        T = R[R.level == level].pivot(
            index="method", columns="perturbation", values=metric)
        T = T[[c for c in a.perturb if c in set(R.perturbation)]]
        print(f"\n=== {metric.upper()} by perturbation, {level.upper()} level, "
              f"pooled over folds {folds} (rows = methods) ===\n"
              f"{T.round(3).to_string()}")

print(f"\nsaved {a.run}/stress_results.csv")
