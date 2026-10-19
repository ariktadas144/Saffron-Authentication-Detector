"""Validate the dataset BEFORE training: structure, readable images, class balance, size stats, cross-split duplicates."""
import _path  # noqa
import argparse, os, hashlib, collections, sys
from PIL import Image

ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data"); a = ap.parse_args()
EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
ok, seen, dups = True, {}, []
counts = {}
for s in ("train", "val", "test"):
    for c in ("real", "fake"):
        d = os.path.join(a.data, s, c)
        if not os.path.isdir(d):
            print(f"[MISSING] {d}"); ok = False; continue
        files = [f for f in os.listdir(d) if f.lower().endswith(EXT)]
        counts[(s, c)] = len(files); bad = 0; sizes = collections.Counter()
        for f in files:
            p = os.path.join(d, f)
            try:
                with Image.open(p) as im:
                    im.verify()
                with Image.open(p) as im:
                    sizes[im.size] += 1
            except Exception as e:
                bad += 1; print(f"[CORRUPT] {p}: {e}")
            h = hashlib.md5(open(p, "rb").read()).hexdigest()
            if h in seen and seen[h][0] != s: dups.append((seen[h], (s, c, f)))
            seen.setdefault(h, (s, c, f))
        top = ", ".join(f"{w}x{h}:{n}" for (w, h), n in sizes.most_common(2))
        print(f"{s:5s} {c:5s} n={len(files):5d} corrupt={bad}  common sizes: {top}")
        ok &= (len(files) > 0 and bad == 0)
print("\nsplit sizes:", {s: sum(counts.get((s, c), 0) for c in ("real", "fake")) for s in ("train", "val", "test")})
for s in ("train", "val", "test"):
    r, f = counts.get((s, "real"), 0), counts.get((s, "fake"), 0)
    if r and f and max(r, f) / min(r, f) > 1.5: print(f"[WARN] {s} is imbalanced (real={r}, fake={f}) -> class weights handle train; use balanced metrics")
if dups: print(f"[FAIL] {len(dups)} exact duplicate(s) across splits (leakage), e.g. {dups[0]}"); ok = False
else: print("no exact cross-split duplicates")
print("\nDATASET OK" if ok else "\nDATASET HAS PROBLEMS"); sys.exit(0 if ok else 1)
