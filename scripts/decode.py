"""Wombat-OA + weighted beam decoding on the cached outputs (no TensorFlow / GPU needed).
   python scripts/decode.py --out runs/run1
"""
import _path  # noqa
import argparse, json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import wombat_beam as wb

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="runs/run1"); ap.add_argument("--pop", type=int, default=20)
ap.add_argument("--iters", type=int, default=25); ap.add_argument("--seeds", type=int, default=10)
a = ap.parse_args()
val, test = wb.Cache.load(f"{a.out}/cache_val.npz"), wb.Cache.load(f"{a.out}/cache_test.npz")
N, V, H = test.shape
print(f"val {val.shape}  test {test.shape}  fake fraction test={test.y.mean():.2f}")

table, info = wb.evaluate_all(val, test, pop=a.pop, iters=a.iters, seed=0)
pd.set_option("display.width", 220, "display.max_colwidth", 70)
cols = ["acc", "bal_acc", "f1_fake", "recall_fake", "specificity", "auc", "ece", "nll", "fn", "fp"]
print(table[cols].round(4)); table.to_csv(f"{a.out}/ablation_test.csv")
json.dump({k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in info["params"].items()},
          open(f"{a.out}/wombat_params.json", "w"), indent=1)

cmp = wb.compare_optimizers(val, seeds=range(a.seeds), pop=a.pop, iters=a.iters)
print("\nbest val fitness  Wombat-OA %.4f+-%.4f | Random %.4f+-%.4f | Wilcoxon p=%.4g" %
      (cmp["wombat"].mean(), cmp["wombat"].std(), cmp["random"].mean(), cmp["random"].std(), cmp["wilcoxon_p"]))
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
for h in cmp["hist_wombat"]: ax[0].plot(np.linspace(0, 1, len(h)), h, c="tab:green", alpha=.35)
for h in cmp["hist_random"]: ax[0].plot(np.linspace(0, 1, len(h)), h, c="tab:gray", alpha=.35)
ax[0].set_title("convergence (green=Wombat-OA, gray=random)"); ax[0].set_xlabel("fraction of budget"); ax[0].set_ylabel("best val fitness")
ax[1].boxplot([cmp["wombat"], cmp["random"]], tick_labels=["Wombat-OA", "Random"]); ax[1].set_title(f"final fitness, {a.seeds} seeds")
plt.tight_layout(); plt.savefig(f"{a.out}/optimizer_comparison.png", dpi=200)

P = info["params"]; p_prop = wb.decode(test, P); p_base = wb.b_tta_mean_logit(test, "hafm")
print("\nbootstrap 95% CI on TEST")
for name, p, thr in [("proposed", p_prop, P["theta"]), ("TTA mean-logit", p_base, 0.5)]:
    for k in ("acc", "f1_fake", "auc"):
        print(f"  {name:16s}{k:8s}{wb.metrics(test.y, p, thr)[k]:.4f}  CI {tuple(round(x, 4) for x in wb.bootstrap_ci(test.y, p, thr, k, groups=test.groups))}")
print("DONE ->", a.out)
