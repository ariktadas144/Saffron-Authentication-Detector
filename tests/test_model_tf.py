"""Model-level checks (needs TensorFlow; ~1 min on CPU, random init, no download)."""
import numpy as np
import pytest
pytest.importorskip("tensorflow")
import saffron_cnn_gnn as sg


@pytest.fixture(scope="module")
def M():
    return sg.build_models(sg.Config(img_size=128, imagenet=False))


@pytest.fixture(scope="module")
def x():
    return np.random.default_rng(0).uniform(0, 255, (4, 128, 128, 3)).astype("float32")


def test_outputs_and_grid(M, x):
    out = M["model"](x, training=False)
    assert set(out) == set(sg.HEADS) and all(out[k].shape == (4, 1) for k in sg.HEADS)
    assert M["grid"] == (4, 4)                     # 128 / 32


def test_gnn_has_no_cross_sample_leakage(M, x):
    a, b = M["model"](x, training=False), M["model"](x[:1], training=False)
    assert all(np.allclose(a[k][0], b[k][0], atol=1e-4) for k in sg.HEADS)


def test_trunk_plus_heads_equals_model(M, x):
    f, a = M["trunk"](x, training=False), M["model"](x, training=False)
    assert all(np.allclose(M["heads"][k](f[k], training=False).numpy(), a[k].numpy(), atol=1e-4) for k in sg.HEADS)


def test_mc_dropout_only_when_training_true(M, x):
    f = M["trunk"](x, training=False); h = M["heads"]["hafm"]
    assert not np.allclose(h(f["hafm"], training=True).numpy(), h(f["hafm"], training=True).numpy())
    assert np.allclose(h(f["hafm"], training=False).numpy(), h(f["hafm"], training=False).numpy())


def test_hafm_attention_is_a_distribution(M, x):
    att = M["attn_model"](x).numpy()
    assert np.allclose(att.sum(1), 1.0, atol=1e-4) and (att >= 0).all()


def test_weights_roundtrip(M, x, tmp_path):
    p = str(tmp_path / "w.weights.h5"); M["model"].save_weights(p)
    M2 = sg.build_models(sg.Config(img_size=128, imagenet=False)); M2["model"].load_weights(p)
    a, b = M["model"](x, training=False), M2["model"](x, training=False)
    assert all(np.allclose(a[k], b[k], atol=1e-5) for k in sg.HEADS)
