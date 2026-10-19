"""Build a leakage-free manifest + sample-level K-fold splits from your Training/Testing zips (or folders).

Your data looks like   Training/{Fake,Real}/<id>[ <TYPE>]_<shot>.jpg   and   Testing/{Fake,Real}/...
Several photos (shots) belong to the SAME physical sample, and the same samples appear in both Training and Testing.
So the original Training/Testing split leaks.  This script
  1. extracts the zips / reads the folders,
  2. parses the filenames   '12_3.jpg' -> sample 12, shot 3 ;  '17 C_2.jpg' -> sample 17, type C, shot 2,
  3. finds exact duplicate files and MERGES the samples that share a photo into one group (dropping the repeat copy),
  4. makes stratified GROUP K-fold splits (all shots of a sample, plus merged duplicates, always in the same fold),
  5. writes data/manifest.csv   (columns: path,label,cls,type,sample,group,shot,orig_split,fold)  and prints a report.

   python scripts/prepare_dataset.py --src data/Test_dataset-....zip data/train_dataset-....zip --folds 5
"""
import _path  # noqa
import argparse, glob, os, re, zipfile, hashlib, collections, sys
import numpy as np, pandas as pd
from PIL import Image, ImageOps
from sklearn.model_selection import StratifiedGroupKFold

ap = argparse.ArgumentParser()
ap.add_argument("--src", nargs="+", required=True, help="zip files and/or folders containing .../{Fake,Real}/*.jpg")
ap.add_argument("--data", default="data"); ap.add_argument("--folds", type=int, default=5); ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--max-side", type=int, default=768, help="working copies: shorter side is capped at this many pixels (never upscaled)")
a = ap.parse_args()
sources = []
for src in a.src:
    if glob.has_magic(src):
        matches = sorted(glob.glob(src))
        if not matches: ap.error(f"--src pattern matched no files: {src}")
        sources.extend(matches)
    else:
        sources.append(src)
a.src = sources
EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
RX = re.compile(r"^(\d+)(?:\s*([A-Za-z]+))?_(\d+)\.\w+$")

# ---- 1. collect files -----------------------------------------------------------------------------------------------
roots = []
for s in a.src:
    if s.lower().endswith(".zip"):
        dst = os.path.join(a.data, "extracted", os.path.splitext(os.path.basename(s))[0])
        if not os.path.isdir(dst):
            print("extracting", s, "->", dst); zipfile.ZipFile(s).extractall(dst)
        roots.append(dst)
    else:
        roots.append(s)
rows = []
for root in roots:
    for dp, _, fs in os.walk(root):
        cls = os.path.basename(dp).lower()
        if cls not in ("fake", "real"): continue
        parent_name = os.path.basename(os.path.dirname(dp)).lower()          # e.g. "Training" / "Testing"
        split = "test" if "test" in parent_name else "train" if "train" in parent_name else "unknown"
        for f in sorted(fs):
            if not f.lower().endswith(EXT) or f.startswith("."): continue
            m = RX.match(f)
            sid, typ, shot = (m[1], (m[2] or "-").upper(), int(m[3])) if m else (os.path.splitext(f)[0], "-", 0)
            rows.append(dict(path=os.path.join(dp, f), label=int(cls == "fake"), cls=cls, type=typ,
                             sample=f"{cls}-{typ}-{sid}", shot=shot, orig_split=split))
df = pd.DataFrame(rows)
if df.empty: sys.exit("no images found: expected folders named Fake / Real")
print(f"found {len(df)} images: {df.groupby('cls').size().to_dict()}  | physical samples: {df['sample'].nunique()}")

# ---- 2. report the leakage of the ORIGINAL split ---------------------------------------------------------------------
tr, te = set(df[df.orig_split == "train"]["sample"]), set(df[df.orig_split == "test"]["sample"])
if tr and te:
    print(f"[LEAK] original split: {len(tr & te)} of {len(te)} test samples ALSO appear in train (different shots of the same sample)")

# ---- 3. exact duplicates -> merge groups, drop repeat copies -----------------------------------------------------------
df["md5"] = [hashlib.md5(open(p, "rb").read()).hexdigest() for p in df.path]
parent = {s: s for s in df["sample"]}
def find(x):
    while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
    return x
first = {}
for s, h in zip(df["sample"], df.md5):
    if h in first: parent[find(s)] = find(first[h])
    else: first[h] = s
n_dup = int(df.duplicated("md5").sum())
if n_dup: print(f"[DUP] {n_dup} exact duplicate file(s) found -> merged their samples into one group and dropped the repeat copy")
df = df.drop_duplicates("md5").copy()
roots_ = {s: find(s) for s in df["sample"]}; names = {r: f"G{i:03d}" for i, r in enumerate(sorted(set(roots_.values())))}
df["group"] = df["sample"].map(lambda s: names[roots_[s]])
bad = df.groupby("group").label.nunique(); bad = bad[bad > 1]
if len(bad): sys.exit(f"[ERROR] groups mixing real and fake (check duplicates): {list(bad.index)}")
print(f"{df.group.nunique()} groups (physical samples after merging) | per class: {df.groupby('cls').group.nunique().to_dict()}")

# ---- 4. stratified group K-fold -----------------------------------------------------------------------------------------
df = df.reset_index(drop=True); df["fold"] = -1
for k, (_, te_idx) in enumerate(StratifiedGroupKFold(a.folds, shuffle=True, random_state=a.seed).split(df, df.label, df.group)):
    df.loc[te_idx, "fold"] = k
assert (df.fold >= 0).all() and df.groupby("group").fold.nunique().max() == 1
rep = df.groupby("fold").agg(images=("path", "size"), groups=("group", "nunique"), fake_frac=("label", "mean"))
rep["types(fake)"] = [dict(collections.Counter(df[(df.fold == k) & (df.label == 1)].type)) for k in rep.index]
print("\nfolds (each fold is the TEST set once; the next fold is VAL; the rest TRAIN):\n", rep.round(2).to_string())

# ---- 5. working copies: EXIF-rotated, shorter side capped (3024x3024 originals are slow and memory hungry) ----------------
img_dir = os.path.join(a.data, "images"); new_paths = []
for _, r in df.iterrows():
    dst_dir = os.path.join(img_dir, r.cls); os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, f"{r.group}__{os.path.splitext(os.path.basename(r.path))[0]}.jpg")
    if not os.path.exists(dst):
        with Image.open(r.path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB"); w, h = im.size; sc = a.max_side / min(w, h)
            if sc < 1: im = im.resize((round(w * sc), round(h * sc)), Image.LANCZOS)
            im.save(dst, quality=95)
    new_paths.append(dst)
df["orig_path"] = df.path; df["path"] = new_paths
print(f"wrote {len(df)} working copies to {img_dir} (shorter side <= {a.max_side}px)")

df = df.drop(columns=["md5"]); out = os.path.join(a.data, "manifest.csv"); df.to_csv(out, index=False)
print(f"\nwrote {out}  ({len(df)} images)")
if df.groupby("fold").group.nunique().min() < 8:
    print("[NOTE] fewer than 8 groups in a fold: metrics per fold will be very noisy -> report mean +- std over folds")
