"""
wombat_beam.py  --  pure NumPy/SciPy/sklearn (no TensorFlow needed)

Contents
--------
1. WombatOptimizer      : Wombat Optimization Algorithm (Benmamoun et al., Mathematics 2024, 12(7):1059),
                          Eq.(2),(4)-(8): foraging phase (exploration) + tunnel-escape phase (exploitation).
                          Named "Wombat-OA" here to avoid confusion with the Whale OA (also abbreviated WOA).
2. RandomSearch         : equal-budget baseline optimizer (reviewers WILL ask for this).
3. Cache                : cached per-(image, view, head) logits / MC-uncertainty / view-similarity.
4. decode()             : weighted, diversity-aware beam search over test-time views
                          + log-probability (log-opinion-pool) fusion of heads and views.
5. metrics + baselines  : everything needed for the ablation table.

Notation
--------
N images, V test-time views, H heads (cnn, gnn, hafm).  Positive class = 1 = FAKE (adulterated).
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from scipy.special import expit, logsumexp
from sklearn.metrics import roc_auc_score


# =====================================================================================
# 1. Wombat Optimization Algorithm
# =====================================================================================
class WombatOptimizer:
    """Minimises `fitness(x)` over the box [lb, ub].

    Phase 1 (foraging / exploration): each wombat moves toward a randomly chosen *better* wombat
        x' = x + r * (x_forage - I * x),   I in {1,2},  r ~ U(0,1)           (paper Eq.5)
    Phase 2 (escape to tunnel / exploitation): shrinking local random step
        x' = x + (1 - 2r) * (ub - lb) / t,  t = iteration counter            (paper Eq.7)
    Each phase uses greedy replacement (Eq.6, Eq.8).
    """

    def __init__(self, fitness, lb, ub, pop_size=20, iters=30, seed=0, verbose=False):
        self.f = fitness
        self.lb = np.asarray(lb, float)
        self.ub = np.asarray(ub, float)
        self.N, self.T = int(pop_size), int(iters)
        self.rng = np.random.default_rng(seed)
        self.verbose = verbose
        self.n_evals = 0

    def _eval(self, x):
        self.n_evals += 1
        return float(self.f(x))

    def run(self):
        rng, lb, ub, m = self.rng, self.lb, self.ub, self.lb.size
        X = lb + rng.random((self.N, m)) * (ub - lb)              # Eq.(2)
        F = np.array([self._eval(x) for x in X])
        hist = [F.min()]
        for t in range(1, self.T + 1):
            for i in range(self.N):
                # ---- Phase 1: foraging (exploration) ----
                better = np.where(F < F[i])[0]                    # Eq.(4): candidate forage positions
                if better.size:
                    k = rng.choice(better)
                    I = rng.integers(1, 3, size=m)                # randomly 1 or 2
                    r = rng.random(m)
                    xn = np.clip(X[i] + r * (X[k] - I * X[i]), lb, ub)   # Eq.(5)
                    fn = self._eval(xn)
                    if fn <= F[i]:                                # Eq.(6)
                        X[i], F[i] = xn, fn
                # ---- Phase 2: escape to tunnel (exploitation) ----
                r = rng.random(m)
                xn = np.clip(X[i] + (1 - 2 * r) * (ub - lb) / t, lb, ub)  # Eq.(7)
                fn = self._eval(xn)
                if fn <= F[i]:                                    # Eq.(8)
                    X[i], F[i] = xn, fn
            hist.append(F.min())
            if self.verbose:
                print(f"[Wombat-OA] iter {t:3d}/{self.T}  best={F.min():.5f}  evals={self.n_evals}")
        b = int(np.argmin(F))
        return X[b].copy(), float(F[b]), np.array(hist)


class RandomSearch:
    """Same interface, same evaluation budget -> fair optimizer baseline."""

    def __init__(self, fitness, lb, ub, budget=1000, seed=0):
        self.f, self.lb, self.ub = fitness, np.asarray(lb, float), np.asarray(ub, float)
        self.budget, self.rng = int(budget), np.random.default_rng(seed)

    def run(self):
        best_x, best_f, hist = None, np.inf, []
        for _ in range(self.budget):
            x = self.lb + self.rng.random(self.lb.size) * (self.ub - self.lb)
            fx = float(self.f(x))
            if fx < best_f:
                best_x, best_f = x, fx
            hist.append(best_f)
        return best_x, best_f, np.array(hist)


# =====================================================================================
# 2. Cache of model outputs (computed once with TF, reused by every optimizer evaluation)
# =====================================================================================
@dataclass
class Cache:
    Z: np.ndarray          # (N,V,H) MC-mean logits (fake-vs-real), per view & head
    U: np.ndarray          # (N,V,H) MC std of logits  (epistemic uncertainty)
    Sim: np.ndarray        # (N,V,V) cosine similarity between view embeddings, clipped to [0,1]
    y: np.ndarray          # (N,) 1 = fake
    view_names: list
    head_names: list
    groups: np.ndarray | None = None   # (N,) physical-sample id of each image (for sample-level aggregation)

    @property
    def shape(self):
        return self.Z.shape

    def save(self, path):
        extra = {} if self.groups is None else {"groups": np.asarray(self.groups).astype(str)}
        np.savez_compressed(path, Z=self.Z, U=self.U, Sim=self.Sim, y=self.y,
                            view_names=np.array(self.view_names), head_names=np.array(self.head_names), **extra)

    @staticmethod
    def load(path):
        d = np.load(path, allow_pickle=False)
        g = d["groups"] if "groups" in d.files else None
        return Cache(d["Z"], d["U"], d["Sim"], d["y"], list(d["view_names"]), list(d["head_names"]), g)

    @staticmethod
    def synthetic(N=600, V=8, H=3, seed=0, sep=1.2):
        """Toy cache for unit-testing the decoder/optimizer without any model."""
        rng = np.random.default_rng(seed)
        groups = np.arange(N) // 3                                  # 3 "shots" per physical sample
        y = rng.integers(0, 2, groups.max() + 1)[groups]
        s = 2 * y - 1
        view_noise = rng.uniform(0.6, 2.2, V)                    # some views are less reliable
        head_noise = rng.uniform(0.8, 1.6, H)
        Z = np.zeros((N, V, H)); U = np.zeros((N, V, H))
        for v in range(V):
            for h in range(H):
                sd = view_noise[v] * head_noise[h]
                Z[:, v, h] = sep * s + rng.normal(0, sd, N)
                U[:, v, h] = np.abs(rng.normal(0.15 * sd, 0.05, N)) + 0.02
        E = rng.normal(size=(N, V, 16)) + rng.normal(size=(N, 1, 16))   # shared component -> correlated views
        E /= np.linalg.norm(E, axis=-1, keepdims=True)
        Sim = np.clip(np.einsum("nvd,nwd->nvw", E, E), 0, 1)
        return Cache(Z, U, Sim, y, [f"v{i}" for i in range(V)], ["cnn", "gnn", "hafm"][:H], groups.astype(str))


# =====================================================================================
# 3. Weighted beam decoding + log-probability fusion
# =====================================================================================
def param_spec(H, V):
    """Names and box bounds of the decision vector optimised by Wombat-OA."""
    spec = [(f"a_{i}", 0.0, 1.0) for i in range(H)]            # head weights   (log-opinion-pool weights)
    spec += [(f"tau_{i}", 0.5, 3.0) for i in range(H)]         # head temperatures (calibration)
    spec += [("beta", 0.0, 4.0),                               # view reliability: w = exp(-beta * uncertainty)
             ("gamma", 0.0, 2.0),                              # diversity (redundancy) penalty in beam
             ("alpha", 0.0, 1.0),                              # length-normalisation exponent
             ("beam", 1.0, 8.0),                               # beam width B       (rounded)
             ("depth", 1.0, float(V)),                         # decoding depth L   (rounded)
             ("kappa", 0.05, 2.0),                             # sharpness of final fused posterior
             ("theta", 0.2, 0.8)]                              # decision threshold on p(fake)
    names = [s[0] for s in spec]
    return names, np.array([s[1] for s in spec]), np.array([s[2] for s in spec])


def unpack(x, H, V):
    names, _, _ = param_spec(H, V)
    d = dict(zip(names, np.asarray(x, float)))
    a = np.array([d[f"a_{i}"] for i in range(H)]) + 1e-6
    return dict(a=a / a.sum(),
                tau=np.array([d[f"tau_{i}"] for i in range(H)]),
                beta=d["beta"], gamma=d["gamma"], alpha=d["alpha"],
                beam=int(round(d["beam"])), depth=int(np.clip(round(d["depth"]), 1, V)),
                kappa=d["kappa"], theta=d["theta"])


def default_params(H, V):
    """Hand-set, un-tuned configuration (ablation row 'beam, no optimiser')."""
    names, lb, ub = param_spec(H, V)
    d = {f"a_{i}": 1.0 / H for i in range(H)}
    d.update({f"tau_{i}": 1.0 for i in range(H)})
    d.update(beta=1.0, gamma=0.5, alpha=1.0, beam=4, depth=min(V, 5), kappa=1.0, theta=0.5)
    return np.array([d[n] for n in names])


def _softplus(x):
    return np.logaddexp(0.0, x)


def _beam_one_class(ell, w, Sim, B, L, gamma, alpha):
    """Vectorised (over images) beam search for ONE class hypothesis.

    ell : (N,V) log q_v(c)   -- fused per-view log-probability of this class (<= 0)
    w   : (N,V) view weights in (0,1]
    Sim : (N,V,V) non-negative view-similarity
    A hypothesis is an ordered set of views.  Extending hypothesis h by view v adds
          w_v * ell_v  -  gamma * max_{u in h} Sim(v,u)
    (first view has no redundancy penalty).  Beams are ranked by length-normalised score
          S / (sum_w)^alpha ,  duplicates (same view-set reached by different orders) are merged.
    Returns the (N,k) length-normalised scores of the final beam (-inf = dead slot).
    """
    N, V = ell.shape
    gain = w * ell
    score = np.zeros((N, 1)); wsum = np.zeros((N, 1))
    used = np.zeros((N, 1, V), bool)
    mask = np.zeros((N, 1), np.int64)
    maxsim = np.zeros((N, 1, V))
    bits = (1 << np.arange(V)).astype(np.int64)
    rows = np.arange(N)[:, None]
    final = None
    for _ in range(L):
        Bc = score.shape[1]
        cand = score[:, :, None] + gain[:, None, :] - gamma * maxsim          # (N,Bc,V)
        cand_w = wsum[:, :, None] + w[:, None, :]
        cand = np.where(used, -np.inf, cand)
        norm = cand / np.power(np.maximum(cand_w, 1e-9), alpha)
        keys = mask[:, :, None] | bits[None, None, :]
        M = Bc * V
        norm_f, keys_f = norm.reshape(N, M), keys.reshape(N, M)
        # --- merge duplicates: sort by (view-set key, -score), keep first of each key ---
        order = np.lexsort((-norm_f, keys_f), axis=1)
        ks = np.take_along_axis(keys_f, order, 1)
        ns = np.take_along_axis(norm_f, order, 1)
        first = np.ones_like(ks, bool)
        first[:, 1:] = ks[:, 1:] != ks[:, :-1]
        ns = np.where(first, ns, -np.inf)
        k = min(B, M)
        top = np.argsort(-ns, axis=1)[:, :k]
        sel = np.take_along_axis(order, top, 1)                               # flat (parent*V+view)
        sel_ns = np.take_along_axis(ns, top, 1)
        parent, v = sel // V, sel % V
        score = cand.reshape(N, M)[rows, sel]
        wsum = cand_w.reshape(N, M)[rows, sel]
        used = used[rows, parent].copy()
        np.put_along_axis(used, v[:, :, None], True, axis=2)
        maxsim = np.maximum(maxsim[rows, parent], Sim[rows, v])               # (N,k,V)
        mask = keys_f[rows, sel]
        score = np.where(np.isfinite(sel_ns), score, -np.inf)
        final = sel_ns
    return final


def decode(cache: Cache, p: dict) -> np.ndarray:
    """Returns p(fake) for every image.

    1. Head fusion in log-domain (log-opinion pool), per view:
          l_c(v) = sum_h a_h * log sigma_c(z_vh / tau_h),  then renormalise over c.
    2. View reliability weight  w_v = exp(-beta * sum_h a_h * std_vh / tau_h)   (MC-dropout uncertainty).
    3. Class-conditional weighted beam search over view sets (diversity-penalised).
    4. Log-prob fusion over the final beam:  F_c = logmeanexp_h S_norm(h);  p(fake) = sigmoid(kappa (F_1 - F_0)).
    """
    Z, U, Sim = cache.Z, cache.U, cache.Sim
    N, V, H = Z.shape
    a, tau = p["a"], p["tau"]
    zt = Z / tau                                                       # (N,V,H)
    lp1 = -_softplus(-zt)                                              # log sigma(z/tau)
    lp0 = -_softplus(zt)
    l1 = (lp1 * a).sum(-1); l0 = (lp0 * a).sum(-1)
    lse = np.logaddexp(l0, l1)
    l1, l0 = l1 - lse, l0 - lse                                        # proper per-view distribution
    unc = ((U / tau) * a).sum(-1)
    w = np.exp(-p["beta"] * unc)
    F = []
    for ell in (l0, l1):
        S = _beam_one_class(ell, w, Sim, p["beam"], p["depth"], p["gamma"], p["alpha"])
        cnt = np.isfinite(S).sum(1)
        F.append(logsumexp(S, axis=1) - np.log(np.maximum(cnt, 1)))
    return expit(p["kappa"] * (F[1] - F[0]))


# =====================================================================================
# 4. Metrics
# =====================================================================================
def ece_score(p, y, bins=10):
    idx = np.minimum((p * bins).astype(int), bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def metrics(y, p, thr=0.5):
    y = np.asarray(y).astype(int); pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1); prec = tp / max(tp + fp, 1)
    f1 = 2 * prec * sens / max(prec + sens, 1e-12)
    pc = np.clip(p, 1e-7, 1 - 1e-7)
    return dict(acc=(tp + tn) / len(y), bal_acc=(sens + spec) / 2, precision=prec,
                recall_fake=sens, specificity=spec, f1_fake=f1,
                auc=float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
                ece=ece_score(p, y), nll=float(-(y * np.log(pc) + (1 - y) * np.log(1 - pc)).mean()),
                brier=float(((p - y) ** 2).mean()), fn=fn, fp=fp)


def _macro_f1_fast(y, pred):
    f = []
    for c in (0, 1):
        tp = ((pred == c) & (y == c)).sum(); fp = ((pred == c) & (y != c)).sum(); fn = ((pred != c) & (y == c)).sum()
        f.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(f))


def aggregate_by_group(p, y, groups, eps=1e-6):
    """Sample-level decision: fuse all photos (shots) of one physical sample by averaging their logits
    (= log-probability fusion across shots).  Returns (p_sample, y_sample)."""
    g = np.asarray(groups); u, inv = np.unique(g, return_inverse=True)
    lg = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
    m = np.bincount(inv, weights=lg) / np.bincount(inv)
    ys = np.zeros(len(u)); ys[inv] = y
    return expit(m), ys.astype(int)


def bootstrap_ci(y, p, thr=0.5, key="acc", n=1000, seed=0):
    rng = np.random.default_rng(seed); y = np.asarray(y); vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        vals.append(metrics(y[i], p[i], thr)[key])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# =====================================================================================
# 5. Fitness + tuning
# =====================================================================================
def make_fitness(cache: Cache, lam_ece=0.5, lam_fn=0.25):
    """Minimise  (1 - macroF1) + lam_ece*ECE + lam_fn*FNR   on the VALIDATION cache.
    FNR (fakes accepted as real) is penalised extra because that is the costly error in authentication."""
    N, V, H = cache.shape
    y = cache.y.astype(int)

    def fit(x):
        p = unpack(x, H, V)
        prob = decode(cache, p)
        pred = (prob >= p["theta"]).astype(int)
        fnr = ((pred == 0) & (y == 1)).sum() / max((y == 1).sum(), 1)
        return (1 - _macro_f1_fast(y, pred)) + lam_ece * ece_score(prob, y) + lam_fn * fnr
    return fit


def tune(cache_val: Cache, method="wombat", pop=20, iters=25, seed=0, verbose=False, **fit_kw):
    N, V, H = cache_val.shape
    names, lb, ub = param_spec(H, V)
    fit = make_fitness(cache_val, **fit_kw)
    if method == "wombat":
        opt = WombatOptimizer(fit, lb, ub, pop_size=pop, iters=iters, seed=seed, verbose=verbose)
        x, f, hist = opt.run()
    elif method == "random":
        x, f, hist = RandomSearch(fit, lb, ub, budget=pop + 2 * pop * iters, seed=seed).run()
    else:
        raise ValueError(method)
    return x, f, hist


# =====================================================================================
# 6. Baselines for the ablation table (all on cached outputs -> identical inputs everywhere)
# =====================================================================================
def b_single(c: Cache, head, view=0):
    return expit(c.Z[:, view, c.head_names.index(head)])


def b_late_fusion_identity(c: Cache):                      # prob-average of heads, identity view only
    return expit(c.Z[:, 0, :]).mean(-1)


def b_tta_mean_logit(c: Cache, head):                      # plain TTA
    return expit(c.Z[:, :, c.head_names.index(head)].mean(1))


def b_tta_topk_conf(c: Cache, head, k=4, temp=2.0):        # the old notebook's "TTA-Beam" (top-k by confidence)
    p = expit(c.Z[:, :, c.head_names.index(head)])
    conf = np.maximum(p, 1 - p)
    idx = np.argsort(-conf, axis=1)[:, :k]
    pk, ck = np.take_along_axis(p, idx, 1), np.take_along_axis(conf, idx, 1)
    w = np.exp(ck * temp); w /= w.sum(1, keepdims=True)
    return (w * pk).sum(1)


def b_logprob_fusion_no_beam(c: Cache):                    # equal-weight log-prob fusion over ALL heads and views
    return expit(c.Z.mean((1, 2)))


def evaluate_all(val: Cache, test: Cache, pop=20, iters=25, seed=0, extra_rows=None):
    """Ablation table on TEST. Optimiser/threshold use VAL only."""
    import pandas as pd
    N, V, H = test.shape
    rows = {}
    for h in test.head_names:
        rows[f"single view, {h}"] = (b_single(test, h), 0.5)
    rows["late fusion (prob avg), identity view"] = (b_late_fusion_identity(test), 0.5)
    rows["TTA mean-logit, hafm"] = (b_tta_mean_logit(test, test.head_names[-1]), 0.5)
    rows["TTA top-k confidence (old), hafm"] = (b_tta_topk_conf(test, test.head_names[-1]), 0.5)
    rows["log-prob fusion, equal weights, no beam"] = (b_logprob_fusion_no_beam(test), 0.5)
    pd0 = unpack(default_params(H, V), H, V)
    rows["weighted beam + log-prob fusion (untuned)"] = (decode(test, pd0), pd0["theta"])
    xr, fr, _ = tune(val, "random", pop, iters, seed)
    pr = unpack(xr, H, V); rows["weighted beam + log-prob fusion + random search"] = (decode(test, pr), pr["theta"])
    xw, fw, hist = tune(val, "wombat", pop, iters, seed)
    pw = unpack(xw, H, V); rows["PROPOSED: Wombat-OA + weighted beam + log-prob fusion"] = (decode(test, pw), pw["theta"])
    if extra_rows:                                       # e.g. {"colour-statistics random forest": (p, 0.5)}
        rows.update(extra_rows)
    table = pd.DataFrame({k: metrics(test.y, p, t) for k, (p, t) in rows.items()}).T
    info = dict(params=pw, fitness=fw, history=hist, params_random=pr, probs={k: p for k, (p, _) in rows.items()},
                thr={k: t for k, (_, t) in rows.items()})
    if test.groups is not None:                          # one decision per physical sample (all shots fused)
        agg = {k: aggregate_by_group(p, test.y, test.groups) for k, (p, _) in rows.items()}
        info["sample_table"] = pd.DataFrame({k: metrics(ys, ps, rows[k][1]) for k, (ps, ys) in agg.items()}).T
    return table, info


# =====================================================================================
# 7. Optimiser comparison over many seeds (needed to claim "Wombat-OA is better than X")
# =====================================================================================
def compare_optimizers(val: Cache, seeds=range(10), pop=20, iters=25, **fit_kw):
    """Runs Wombat-OA and equal-budget Random Search with the same seeds on the validation cache.
    Returns best-fitness arrays, convergence curves and a one-sided Wilcoxon signed-rank p-value
    (H1: Wombat-OA reaches lower fitness than random search)."""
    from scipy.stats import wilcoxon
    fw, fr, hw, hr = [], [], [], []
    for s in seeds:
        _, f1, h1 = tune(val, "wombat", pop, iters, s, **fit_kw)
        _, f2, h2 = tune(val, "random", pop, iters, s, **fit_kw)
        fw.append(f1); fr.append(f2); hw.append(h1); hr.append(h2)
    fw, fr = np.array(fw), np.array(fr)
    try:
        p = float(wilcoxon(fw, fr, alternative="less").pvalue)
    except ValueError:                                   # all differences zero
        p = float("nan")
    return dict(wombat=fw, random=fr, hist_wombat=hw, hist_random=hr, wilcoxon_p=p)


# =====================================================================================
# Self-test on synthetic data
# =====================================================================================
if __name__ == "__main__":
    import time
    val, test = Cache.synthetic(seed=1), Cache.synthetic(seed=2)
    t0 = time.time()
    table, info = evaluate_all(val, test, pop=12, iters=10, seed=0)
    import pandas as pd
    pd.set_option("display.width", 200)
    print(table[["acc", "f1_fake", "recall_fake", "specificity", "auc", "ece"]].round(4))
    print("tuned params:", {k: (np.round(v, 3) if hasattr(v, 'shape') else v) for k, v in info["params"].items()})
    print(f"elapsed {time.time() - t0:.1f}s")
