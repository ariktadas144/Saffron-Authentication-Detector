"""
saffron_cnn_gnn.py  --  TensorFlow/Keras 3 side of the saffron real-vs-fake pipeline.

Differences from the glaucoma notebook (all deliberate):
  * fundus-specific code removed (black-background mask, ungradability head with RANDOM labels)
  * COLOUR-PRESERVING augmentation only (dye colour is the main cue for fake saffron)
  * GNN attention fixed: softmax over each node's neighbours PER SAMPLE (old code normalised over the whole batch)
  * HAFM uses scaled dot-product cross-attention (old tanh-bounded logits could never become sharp)
  * three heads (cnn / gnn / hafm) trained jointly -> three "experts" for log-probability fusion
  * MC-Dropout is NOT baked into the layer; it is switched on only when sampling (training=True on the heads),
    so validation / checkpointing are deterministic and the expensive trunk runs once per view
  * data via tf.data + Keras preprocessing layers (ImageDataGenerator is deprecated and CPU-bound)

Folder layout expected:   root/{train,val,test}/{real,fake}/*.jpg      (fake = 1 = positive class)
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from manifest_paths import normalize_manifest_paths, normalize_manifest_path

CLASS_NAMES = ["real", "fake"]          # index 0 = real, index 1 = fake (positive)
HEADS = ["cnn", "gnn", "hafm"]
VIEWS = ["id", "hflip", "vflip", "rot90", "rot180", "rot270", "crop90", "crop80"]   # all colour-preserving


@dataclass
class Config:
    img_size: int = 256
    batch: int = 16
    epochs_warm: int = 3            # backbone frozen
    epochs_ft: int = 25             # everything unfrozen
    lr_warm: float = 1e-3
    lr_ft: float = 1e-4
    dropout: float = 0.3
    l2: float = 1e-4                # old notebook used L1=0.1 (loss started at 147) - far too strong
    gnn_dim: int = 128
    gnn_heads: int = 4
    gnn_layers: int = 2
    fusion_dim: int = 128
    mc_samples: int = 16
    imagenet: bool = True
    seed: int = 42
    capture_invariant: bool = False
    loss_weights: dict = field(default_factory=lambda: {"cnn": 0.5, "gnn": 0.5, "hafm": 1.0})


# =====================================================================================
# Data
# =====================================================================================
def _count(root, split, cls):
    d = os.path.join(root, split, cls)
    return len([f for f in os.listdir(d) if not f.startswith(".")])


def make_datasets(root, cfg: Config):
    S = cfg.img_size
    common = dict(class_names=CLASS_NAMES, label_mode="binary", image_size=(S, S), batch_size=cfg.batch)
    train_raw = keras.utils.image_dataset_from_directory(os.path.join(root, "train"), shuffle=True, seed=cfg.seed, **common)
    val_raw = keras.utils.image_dataset_from_directory(os.path.join(root, "val"), shuffle=False, **common)
    test_raw = keras.utils.image_dataset_from_directory(os.path.join(root, "test"), shuffle=False, **common)

    n_real, n_fake = _count(root, "train", "real"), _count(root, "train", "fake")
    n = n_real + n_fake
    w_real, w_fake = n / (2 * n_real), n / (2 * n_fake)             # balanced class weights
    print(f"train: real={n_real} fake={n_fake}  weights real={w_real:.2f} fake={w_fake:.2f}")

    # Colour-preserving augmentation. NO channel shift / hue / saturation / strong contrast changes.
    aug = keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.5, fill_mode="reflect"),            # +-180 deg: saffron has no canonical orientation
        layers.RandomZoom((-0.15, 0.15), fill_mode="reflect"),
        layers.RandomTranslation(0.05, 0.05, fill_mode="reflect"),
        layers.RandomBrightness(0.05, value_range=(0, 255)),        # mild lighting variation only
    ], name="augment")

    def _train_map(x, y):
        x = aug(x, training=True)
        x = x + tf.random.normal(tf.shape(x), 0.0, 4.0)             # sensor noise, std 4 on the 0-255 scale
        x = tf.clip_by_value(x, 0.0, 255.0)
        sw = tf.where(y[:, 0] > 0.5, w_fake, w_real)
        return x, {h: y for h in HEADS}, sw

    def _eval_map(x, y):
        return x, {h: y for h in HEADS}

    AUTO = tf.data.AUTOTUNE
    train = train_raw.map(_train_map, num_parallel_calls=AUTO).prefetch(AUTO)
    val = val_raw.map(_eval_map, num_parallel_calls=AUTO).prefetch(AUTO)
    return train, val, val_raw, test_raw          # *_raw yield (x, y) and are deterministic -> used for caching


# ---- manifest-based pipeline (sample-level splits, variable-size photos) -------------------------------------------------
def _load_image(path, S, train):
    """Decode a photo of ANY size/aspect -> (S,S,3) float32 in 0-255.
    Square crop (random position when training, centre otherwise) instead of squashing, so thread shapes are not
    distorted.  Training adds random down/up-resampling ('resolution jitter') so the net cannot use image sharpness /
    original resolution as a shortcut (real photos in this dataset are mostly larger than fake ones)."""
    x = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    x.set_shape([None, None, 3])
    h, w = tf.shape(x)[0], tf.shape(x)[1]
    sq = tf.minimum(h, w)
    if train:
        oy = tf.random.uniform([], 0, h - sq + 1, tf.int32); ox = tf.random.uniform([], 0, w - sq + 1, tf.int32)
    else:
        oy, ox = (h - sq) // 2, (w - sq) // 2
    x = tf.image.crop_to_bounding_box(x, oy, ox, sq, sq)
    x = tf.image.resize(tf.cast(x, tf.float32), (S, S), method="bilinear", antialias=True)
    if train:
        m = tf.cast(tf.cast(S, tf.float32) * tf.random.uniform([], 0.4, 1.0), tf.int32)
        x = tf.cond(tf.random.uniform([]) < 0.5,
                    lambda: tf.image.resize(tf.image.resize(x, (m, m), antialias=True), (S, S)), lambda: x)
    x = tf.clip_by_value(x, 0.0, 255.0)
    x.set_shape([S, S, 3])
    return x


def _build_aug():
    # Colour-preserving augmentation. NO channel shift / hue / saturation / strong contrast changes.
    return keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.5, fill_mode="reflect"),            # +-180 deg: saffron has no canonical orientation
        layers.RandomZoom((-0.15, 0.15), fill_mode="reflect"),
        layers.RandomTranslation(0.05, 0.05, fill_mode="reflect"),
        layers.RandomBrightness(0.05, value_range=(0, 255)),        # mild lighting variation only
    ], name="augment")


def normalize_luminance_preserve_chromaticity(images, target=127.5):
    """Scale RGB channels uniformly per image to normalize luminance, preserving ratios."""
    luminance = tf.image.rgb_to_grayscale(images)
    mean_luminance = tf.reduce_mean(luminance, axis=(1, 2, 3), keepdims=True)
    scale = tf.math.divide_no_nan(
        tf.cast(target, images.dtype),
        tf.maximum(mean_luminance, tf.cast(1.0, images.dtype)),
    )
    return tf.clip_by_value(images * scale, 0.0, 255.0)


def capture_invariant_augment(images):
    """Apply exposure/gamma perturbations to luminance and sensor noise to RGB."""
    x = normalize_luminance_preserve_chromaticity(images)
    luminance = tf.image.rgb_to_grayscale(x)
    shape = (tf.shape(x)[0], 1, 1, 1)
    exposure = tf.random.uniform(shape, 0.6, 1.5)
    gamma = tf.random.uniform(shape, 0.7, 1.4)
    shifted_luminance = 255.0 * tf.pow(
        tf.clip_by_value(luminance / 255.0 * exposure, 0.0, 1.0), gamma
    )
    luminance_scale = tf.math.divide_no_nan(
        shifted_luminance, tf.maximum(luminance, 1.0)
    )
    x = x * luminance_scale
    noise_std = tf.random.uniform(shape, 4.0, 16.0)
    return tf.clip_by_value(
        x + tf.random.normal(tf.shape(x), 0.0, noise_std), 0.0, 255.0
    )


def make_datasets_from_manifest(train_df, val_df, test_df, cfg: Config):
    """DataFrames need columns: path, label (1 = fake), group.  Returns the same 4 objects as make_datasets()."""
    train_df = normalize_manifest_paths(train_df, columns=("path",))
    val_df = normalize_manifest_paths(val_df, columns=("path",))
    test_df = normalize_manifest_paths(test_df, columns=("path",))
    S, AUTO = cfg.img_size, tf.data.AUTOTUNE
    n_fake = int(train_df.label.sum()); n_real = len(train_df) - n_fake
    w_real, w_fake = len(train_df) / (2 * n_real), len(train_df) / (2 * n_fake)
    print(f"train: real={n_real} fake={n_fake}  weights real={w_real:.2f} fake={w_fake:.2f}")
    aug = _build_aug()

    def raw(df, train):
        paths = np.asarray(
            [normalize_manifest_path(path) for path in df.path], dtype=str
        )
        ds = tf.data.Dataset.from_tensor_slices((paths, df.label.values.astype("float32")))
        if train: ds = ds.shuffle(len(df), seed=cfg.seed, reshuffle_each_iteration=True)
        return ds.map(lambda p, l: (_load_image(p, S, train), l[None]), num_parallel_calls=AUTO).batch(cfg.batch)

    def _train_map(x, y):
        x = aug(x, training=True)
        if cfg.capture_invariant:
            x = capture_invariant_augment(x)
        else:
            x = x + tf.random.normal(tf.shape(x), 0.0, 4.0)
        x = tf.clip_by_value(x, 0.0, 255.0)
        return x, {h: y for h in HEADS}, tf.where(y[:, 0] > 0.5, w_fake, w_real)

    train = raw(train_df, True).map(_train_map, num_parallel_calls=AUTO).prefetch(AUTO)
    val_raw, test_raw = raw(val_df, False).cache().prefetch(AUTO), raw(test_df, False).cache().prefetch(AUTO)
    val = val_raw.map(lambda x, y: (x, {h: y for h in HEADS}), num_parallel_calls=AUTO).prefetch(AUTO)
    return train, val, val_raw, test_raw


# =====================================================================================
# Layers
# =====================================================================================
@keras.saving.register_keras_serializable(package="saffron")
class GridGraphAttention(layers.Layer):
    """GAT-style multi-head attention on an HxW patch grid (8-neighbourhood + self loops).

    Message passing:  h_i' = sum_{j in N(i)} alpha_ij W h_j,   alpha_ij = softmax_j( LeakyReLU(a_src.Wh_i + a_dst.Wh_j) )
    The softmax runs over the neighbours j of each node i, independently for every sample and head.
    """

    def __init__(self, units, grid, heads=4, rate=0.1, **kw):
        super().__init__(**kw)
        assert units % heads == 0
        self.units, self.grid, self.heads, self.rate = units, tuple(grid), heads, rate
        self.d = units // heads
        self.W = layers.Dense(units, use_bias=False)
        self.skip = layers.Dense(units, use_bias=False)
        self.norm = layers.LayerNormalization(epsilon=1e-6)
        self.drop = layers.Dropout(rate)

    def build(self, input_shape):
        self.W.build(input_shape)
        self.skip.build(input_shape)
        self.norm.build((None, None, self.units))
        self.a_src = self.add_weight(name="a_src", shape=(self.heads, self.d), initializer="glorot_uniform")
        self.a_dst = self.add_weight(name="a_dst", shape=(self.heads, self.d), initializer="glorot_uniform")
        Hh, Ww = self.grid
        ys, xs = np.divmod(np.arange(Hh * Ww), Ww)
        adj = (np.abs(ys[:, None] - ys[None, :]) <= 1) & (np.abs(xs[:, None] - xs[None, :]) <= 1)
        self.adj_bias = np.where(adj, 0.0, -1e9).astype("float32")      # (N,N) additive mask
        super().build(input_shape)

    def call(self, x, training=None):
        Hh, Ww = self.grid
        N = Hh * Ww
        B = tf.shape(x)[0]
        h = tf.reshape(self.W(x), (B, N, self.heads, self.d))
        e_src = tf.einsum("bnhd,hd->bnh", h, self.a_src)
        e_dst = tf.einsum("bnhd,hd->bnh", h, self.a_dst)
        e = tf.nn.leaky_relu(e_src[:, :, None, :] + e_dst[:, None, :, :], 0.2)     # (B,N,N,heads)
        e = e + tf.constant(self.adj_bias)[None, :, :, None]
        att = self.drop(tf.nn.softmax(e, axis=2), training=training)               # over neighbours j
        out = tf.reshape(tf.einsum("bijh,bjhd->bihd", att, h), (B, N, self.units))
        return tf.nn.gelu(self.norm(out + self.skip(x)))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.units)

    def get_config(self):
        c = super().get_config(); c.update(units=self.units, grid=self.grid, heads=self.heads, rate=self.rate); return c


@keras.saving.register_keras_serializable(package="saffron")
class HybridAttentionFusion(layers.Layer):
    """HAFM: CNN global vector (query) attends over GNN node embeddings (keys/values), then a channel gate and a
    learnable scalar alpha blend the two streams.   Input: [cnn_vec (B,Cc), gnn_nodes (B,N,Cg)]."""

    def __init__(self, dim=128, **kw):
        super().__init__(**kw)
        self.dim = dim
        self.pc, self.pg = layers.Dense(dim), layers.Dense(dim)
        self.gate = layers.Dense(dim, activation="sigmoid")

    def build(self, input_shape):
        cs, gs = input_shape
        self.pc.build(cs); self.pg.build(gs); self.gate.build((None, self.dim))
        self.alpha_raw = self.add_weight(name="alpha_raw", shape=(), initializer="zeros")   # sigmoid(0)=0.5
        super().build(input_shape)

    def call(self, inputs):
        cnn_vec, nodes = inputs
        c, g = self.pc(cnn_vec), self.pg(nodes)                                            # (B,D), (B,N,D)
        attn = tf.nn.softmax(tf.einsum("bnd,bd->bn", g, c) / np.sqrt(self.dim), axis=1)    # (B,N)
        ctx = tf.einsum("bn,bnd->bd", attn, g)
        a = tf.sigmoid(self.alpha_raw)
        return a * c + (1.0 - a) * self.gate(c) * ctx, attn

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], self.dim), (input_shape[0][0], input_shape[1][1])

    def get_config(self):
        c = super().get_config(); c.update(dim=self.dim); return c


# =====================================================================================
# Model
# =====================================================================================
def _make_head(name, in_dim, cfg: Config):
    h = keras.Sequential([layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.L2(cfg.l2)),
                          layers.Dropout(cfg.dropout),            # MC-dropout source (stochastic only when training=True)
                          layers.Dense(1)], name=f"head_{name}")    # linear -> logit of P(fake)
    h.build((None, in_dim))
    return h


def build_models(cfg: Config):
    S = cfg.img_size
    inp = keras.Input((S, S, 3), name="image")                      # raw 0-255 pixels (ConvNeXt normalises internally)
    base = keras.applications.ConvNeXtTiny(include_top=False, include_preprocessing=True,
                                           weights="imagenet" if cfg.imagenet else None, input_shape=(S, S, 3))
    fmap = base(inp)
    h, w, c = fmap.shape[1:]
    f_cnn = layers.GlobalAveragePooling2D(name="gap")(fmap)
    g = layers.Reshape((h * w, c), name="to_nodes")(fmap)
    for i in range(cfg.gnn_layers):
        g = GridGraphAttention(cfg.gnn_dim, (h, w), heads=cfg.gnn_heads, name=f"gat_{i}")(g)
    f_gnn = layers.GlobalAveragePooling1D(name="gnn_pool")(g)
    f_hafm, attn = HybridAttentionFusion(cfg.fusion_dim, name="hafm")([f_cnn, g])

    heads = {"cnn": _make_head("cnn", c, cfg), "gnn": _make_head("gnn", cfg.gnn_dim, cfg),
             "hafm": _make_head("hafm", cfg.fusion_dim, cfg)}
    feats = {"cnn": f_cnn, "gnn": f_gnn, "hafm": f_hafm}
    model = keras.Model(inp, {k: heads[k](feats[k]) for k in HEADS}, name="saffron_cnn_gnn_hafm")
    trunk = keras.Model(inp, feats, name="trunk")                    # shares ALL layers with `model`
    attn_model = keras.Model(inp, attn, name="hafm_attention")       # for explainability maps (h x w grid)
    return dict(model=model, trunk=trunk, heads=heads, base=base, attn_model=attn_model, grid=(h, w))


def _compile(model, lr, cfg):
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss={k: keras.losses.BinaryCrossentropy(from_logits=True) for k in HEADS},
        loss_weights=cfg.loss_weights,
        metrics={k: [keras.metrics.BinaryAccuracy(threshold=0.0, name="acc"),
                     keras.metrics.AUC(from_logits=True, name="auc")] for k in HEADS})


def train(M, train_ds, val_ds, cfg: Config, ckpt="saffron_best.weights.h5"):
    model, base = M["model"], M["base"]
    base.trainable = False                                           # stage 1: heads / GNN / HAFM only
    _compile(model, cfg.lr_warm, cfg)
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=cfg.epochs_warm)
    base.trainable = True                                            # stage 2: fine-tune everything
    _compile(model, cfg.lr_ft, cfg)
    cbs = [keras.callbacks.ModelCheckpoint(ckpt, monitor="val_loss", save_best_only=True, save_weights_only=True),
           keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
           keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6)]
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=cfg.epochs_ft, callbacks=cbs)
    return h1, h2


# =====================================================================================
# Test-time views + MC-dropout cache  (feeds wombat_beam.Cache)
# =====================================================================================
def apply_view(x, name, S):
    if name == "id":     return x
    if name == "hflip":  return tf.image.flip_left_right(x)
    if name == "vflip":  return tf.image.flip_up_down(x)
    if name == "rot90":  return tf.image.rot90(x, 1)
    if name == "rot180": return tf.image.rot90(x, 2)
    if name == "rot270": return tf.image.rot90(x, 3)
    if name.startswith("crop"):
        return tf.image.resize(tf.image.central_crop(x, int(name[4:]) / 100.0), (S, S))
    raise ValueError(name)


def extract_cache(M, raw_ds, cfg: Config, views=VIEWS, T=None, groups=None, paths=None):
    """One pass over a *deterministic* (shuffle=False) dataset.
    For each image x view x head we store the MC-mean logit and MC std of the logit.
    Only the small heads are re-sampled T times (trunk runs once per view, deterministic)."""
    from wombat_beam import Cache
    T = T or cfg.mc_samples
    trunk, heads, S = M["trunk"], M["heads"], cfg.img_size
    Zs, Us, Ss, Ys = [], [], [], []
    for x, y in raw_ds:
        Zv, Uv, Ev = [], [], []
        for v in views:
            f = trunk(apply_view(x, v, S), training=False)
            per_head = []
            for hname in HEADS:
                s = tf.stack([heads[hname](f[hname], training=True)[:, 0] for _ in range(T)], 0)   # (T,B)
                per_head.append(s)
            zs = tf.stack(per_head, -1)                                                          # (T,B,H)
            Zv.append(tf.reduce_mean(zs, 0)); Uv.append(tf.math.reduce_std(zs, 0))
            Ev.append(tf.math.l2_normalize(f["cnn"], axis=-1))
        E = tf.stack(Ev, 1)                                                                      # (B,V,D)
        Ss.append(tf.clip_by_value(tf.einsum("bvd,bwd->bvw", E, E), 0.0, 1.0).numpy())
        Zs.append(tf.stack(Zv, 1).numpy()); Us.append(tf.stack(Uv, 1).numpy())
        Ys.append(y.numpy()[:, 0].astype(int))
    return Cache(np.concatenate(Zs), np.concatenate(Us), np.concatenate(Ss), np.concatenate(Ys),
                 list(views), list(HEADS),
                 None if groups is None else np.asarray(groups).astype(str),
                 None if paths is None else np.asarray(paths).astype(str))


def save_ood_reference(M, train_df, cfg: Config, out: str, percentile: float = 99.0):
    """Persist a per-fold CNN embedding centroid and train-distance cutoff."""
    paths = np.asarray(
        [normalize_manifest_path(value) for value in train_df.path], dtype=str
    )
    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.map(
        lambda path: _load_image(path, cfg.img_size, False),
        num_parallel_calls=1,
    ).batch(cfg.batch)
    batches = []
    for images in ds:
        features = M["trunk"](images, training=False)["cnn"]
        batches.append(features.numpy())
    embeddings = np.concatenate(batches, axis=0).astype(np.float32)
    center = embeddings.mean(axis=0)
    distances = np.linalg.norm(embeddings - center, axis=1)
    cutoff = float(np.percentile(distances, percentile))
    np.savez_compressed(
        os.path.join(out, "ood_reference.npz"),
        center=center, max_distance=np.asarray(cutoff),
        percentile=np.asarray(percentile), n_train=np.asarray(len(embeddings)),
    )


def run_fold(manifest, fold, n_folds, cfg: Config, out):
    """Outer test = fold k, validation = fold (k+1) % K, train = the rest.  Splits are by physical-sample group.
    Trains, then writes cache_val.npz / cache_test.npz (with sample ids) into `out`."""
    import json, dataclasses
    import pandas as pd
    os.makedirs(out, exist_ok=True)
    kv = (fold + 1) % n_folds
    test_df = manifest[manifest.fold == fold].reset_index(drop=True)
    val_df = manifest[manifest.fold == kv].reset_index(drop=True)
    train_df = manifest[~manifest.fold.isin([fold, kv])].reset_index(drop=True)
    assert not (set(train_df.group) & set(val_df.group)) and not (set(train_df.group) & set(test_df.group)) \
        and not (set(val_df.group) & set(test_df.group)), "sample leakage between splits"
    print(f"fold {fold}: train {len(train_df)} img / {train_df.group.nunique()} samples | "
          f"val {len(val_df)} / {val_df.group.nunique()} | test {len(test_df)} / {test_df.group.nunique()}")
    json.dump(dict(dataclasses.asdict(cfg), fold=fold, n_folds=n_folds), open(f"{out}/config.json", "w"), indent=1)
    keras.utils.set_random_seed(cfg.seed + fold)
    train_ds, val_ds, val_raw, test_raw = make_datasets_from_manifest(train_df, val_df, test_df, cfg)
    M = build_models(cfg)
    h1, h2 = train(M, train_ds, val_ds, cfg, ckpt=f"{out}/best.weights.h5")
    M["model"].save_weights(f"{out}/final.weights.h5")
    pd.DataFrame(h2.history).to_csv(f"{out}/history_finetune.csv", index=False)
    extract_cache(
        M, val_raw, cfg, groups=val_df.group.values, paths=val_df.path.values
    ).save(f"{out}/cache_val.npz")
    extract_cache(
        M, test_raw, cfg, groups=test_df.group.values, paths=test_df.path.values
    ).save(f"{out}/cache_test.npz")
    save_ood_reference(M, train_df, cfg, out)
    keras.backend.clear_session()


# =====================================================================================
# Smoke test:  python saffron_cnn_gnn.py --smoke
# =====================================================================================
def _make_fake_dataset(root, n=12, size=96, seed=0):
    from PIL import Image
    rng = np.random.default_rng(seed)
    for split in ("train", "val", "test"):
        for ci, cls in enumerate(CLASS_NAMES):
            d = os.path.join(root, split, cls); os.makedirs(d, exist_ok=True)
            for i in range(n):
                base = np.array([200, 40, 30]) if ci == 0 else np.array([220, 90, 20])
                img = np.clip(base + rng.normal(0, 25, (size, size, 3)), 0, 255).astype("uint8")
                Image.fromarray(img).save(os.path.join(d, f"{i}.jpg"))


if __name__ == "__main__":
    import sys, tempfile
    if "--smoke" in sys.argv:
        import wombat_beam as wb
        tmp = tempfile.mkdtemp(); _make_fake_dataset(tmp)
        cfg = Config(img_size=128, batch=8, epochs_warm=1, epochs_ft=1, imagenet=False, mc_samples=4)
        train_ds, val_ds, val_raw, test_raw = make_datasets(tmp, cfg)
        M = build_models(cfg); print("grid:", M["grid"]); M["model"].summary(line_length=100)
        train(M, train_ds, val_ds, cfg, ckpt=os.path.join(tmp, "best.weights.h5"))
        cv, ct = extract_cache(M, val_raw, cfg), extract_cache(M, test_raw, cfg)
        print("cache", cv.Z.shape, cv.U.shape, cv.Sim.shape, cv.y[:6])
        table, info = wb.evaluate_all(cv, ct, pop=6, iters=3)
        print(table[["acc", "f1_fake", "auc"]].round(3))
        print("SMOKE TEST OK")
