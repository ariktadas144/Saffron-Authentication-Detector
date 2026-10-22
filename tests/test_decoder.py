"""Fast tests (seconds, no TensorFlow): optimizer, beam search, decoder, metrics."""
import itertools
import numpy as np
import pytest
import wombat_beam as wb


def test_wombat_minimises_sphere_and_respects_bounds():
    f = lambda x: float(np.sum((x - 0.3) ** 2))
    lb, ub = -np.ones(6), np.ones(6) * 2
    x, fx, hist = wb.WombatOptimizer(f, lb, ub, pop_size=20, iters=40, seed=0).run()
    assert fx < 1e-2 and np.all(x >= lb) and np.all(x <= ub)
    assert np.all(np.diff(hist) <= 1e-12), "best fitness must never get worse (greedy acceptance)"


def test_beam_matches_exhaustive_search_when_beam_is_wide():
    rng = np.random.default_rng(0)
    N, V = 4, 5
    ell = -rng.random((N, V)); w = rng.uniform(.2, 1, (N, V))
    E = rng.normal(size=(N, V, 6)); E /= np.linalg.norm(E, axis=-1, keepdims=True)
    Sim = np.clip(np.einsum("nvd,nwd->nvw", E, E), 0, 1)
    gamma, alpha = 0.7, 0.6
    for L in (1, 2, 3, 5):
        S = wb._beam_one_class(ell, w, Sim, B=50, L=L, gamma=gamma, alpha=alpha)
        for n in range(N):
            best = {}
            for perm in itertools.permutations(range(V), L):
                s = ws = 0.0
                for i, v in enumerate(perm):
                    s += w[n, v] * ell[n, v] - gamma * max([Sim[n, v, u] for u in perm[:i]], default=0.0); ws += w[n, v]
                best[frozenset(perm)] = max(best.get(frozenset(perm), -1e9), s / ws ** alpha)
            assert np.allclose(np.sort(list(best.values()))[::-1], np.sort(S[n][np.isfinite(S[n])])[::-1])


def test_decode_outputs_valid_probabilities_and_beats_chance():
    c = wb.Cache.synthetic(N=400, seed=5, sep=0.8)
    N, V, H = c.shape
    p = wb.unpack(wb.default_params(H, V), H, V)
    prob = wb.decode(c, p)
    assert prob.shape == (N,) and np.all((prob >= 0) & (prob <= 1)) and np.isfinite(prob).all()
    assert wb.metrics(c.y, prob)["auc"] > 0.9


def test_wombat_tuning_improves_over_default_on_validation():
    val = wb.Cache.synthetic(N=400, seed=7, sep=0.35)
    N, V, H = val.shape
    fit = wb.make_fitness(val)
    x, f, _ = wb.tune(val, "wombat", pop=12, iters=8, seed=0)
    assert f <= fit(wb.default_params(H, V)) + 1e-9


def test_cache_roundtrip(tmp_path):
    c = wb.Cache.synthetic(N=50, seed=1)
    c.paths = np.array([f"image-{i}.jpg" for i in range(len(c.y))])
    c.save(tmp_path / "c.npz")
    d = wb.Cache.load(tmp_path / "c.npz")
    assert np.allclose(c.Z, d.Z) and d.head_names == c.head_names and d.view_names == c.view_names
    assert np.array_equal(c.paths, d.paths)


def test_metrics_perfect_and_worst():
    y = np.array([0, 0, 1, 1])
    assert wb.metrics(y, np.array([.1, .2, .8, .9]))["acc"] == 1.0
    assert wb.metrics(y, np.array([.9, .8, .2, .1]))["acc"] == 0.0


def test_aggregate_by_group_rejects_inconsistent_labels():
    with pytest.raises(ValueError, match="group 'sample-1' has inconsistent labels"):
        wb.aggregate_by_group(
            np.array([.2, .8]), np.array([0, 1]), np.array(["sample-1", "sample-1"])
        )


def test_bootstrap_ci_resamples_complete_groups():
    groups = np.array(["a", "a", "a", "b", "c", "c", "d"])
    y = np.array([0, 0, 0, 0, 1, 1, 1])
    p = np.array([.1, .2, .3, .9, .8, .7, .2])
    n, seed = 200, 13
    actual = wb.bootstrap_ci(y, p, key="acc", n=n, seed=seed, groups=groups)

    rng = np.random.default_rng(seed)
    group_rows = [np.flatnonzero(groups == group) for group in np.unique(groups)]
    scores = []
    for _ in range(n):
        draws = rng.integers(0, len(group_rows), len(group_rows))
        rows = np.concatenate([group_rows[index] for index in draws])
        scores.append(wb.metrics(y[rows], p[rows])["acc"])
    expected = (np.percentile(scores, 2.5), np.percentile(scores, 97.5))
    assert actual == expected


def test_sample_threshold_uses_validation_group_aggregation():
    groups = np.array(["real", "real", "fake", "fake"])
    y = np.array([0, 0, 1, 1])
    # Image threshold 0.5 misses one real shot; group fusion separates both samples.
    probabilities = np.array([.1, .9, .7, .8])
    threshold = wb.select_group_threshold(y, probabilities, groups)
    p_group, y_group = wb.aggregate_by_group(probabilities, y, groups)
    assert threshold > .5
    assert wb.metrics(y_group, p_group, threshold)["acc"] == 1.0


def test_evaluate_all_returns_group_bootstrap_intervals(monkeypatch):
    val = wb.Cache.synthetic(N=48, V=3, H=3, seed=21, sep=.5)
    test = wb.Cache.synthetic(N=48, V=3, H=3, seed=22, sep=.5)

    def fixed_tune(cache, method, pop, iters, seed):
        _, lower, upper = wb.param_spec(cache.shape[2], cache.shape[1])
        return (lower + upper) / 2, 0.0, np.array([0.0])

    monkeypatch.setattr(wb, "tune", fixed_tune)
    table, info = wb.evaluate_all(val, test, pop=1, iters=0, ci_samples=30)
    method = "PROPOSED: Wombat-OA + weighted beam + log-prob fusion"
    assert method in info["image_ci"] and method in info["sample_ci"]
    assert method in info["sample_thr"]
    assert set(info["image_ci"][method]) == set(table.columns)
    assert set(info["sample_ci"][method]) == set(table.columns)
    assert isinstance(info["params_random"], dict)
    assert {"a", "tau", "beta", "gamma", "alpha", "beam", "depth", "kappa", "theta"} <= set(
        info["params_random"]
    )
