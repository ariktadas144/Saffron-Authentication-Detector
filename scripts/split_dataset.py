"""Split  data/raw/{real,fake}/*  ->  data/{train,val,test}/{real,fake}/*   (stratified).

Leakage warning: if several photos come from the SAME physical saffron sample / lot / session, they must land in the
same split.  Encode the sample id in the file name and pass --group-regex with ONE capture group, e.g.
    --group-regex "^(lot\\d+)_"         for names like  lot17_img003.jpg
Without --group-regex every image is treated as its own group (fine only if images are truly independent).
"""
import _path  # noqa
import argparse, os, re, shutil, random, collections

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="data/raw"); ap.add_argument("--dst", default="data")
ap.add_argument("--val", type=float, default=0.15); ap.add_argument("--test", type=float, default=0.15)
ap.add_argument("--group-regex", default=None); ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--move", action="store_true", help="move instead of copy")
a = ap.parse_args()
rng = random.Random(a.seed)
EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
for cls in ("real", "fake"):
    d = os.path.join(a.src, cls)
    files = sorted(f for f in os.listdir(d) if f.lower().endswith(EXT))
    groups = collections.defaultdict(list)
    for f in files:
        m = re.search(a.group_regex, f) if a.group_regex else None
        groups[m.group(1) if m else f].append(f)
    keys = list(groups); rng.shuffle(keys)
    n = len(files); tgt = {"test": a.test * n, "val": a.val * n}; cnt = collections.Counter(); out = {}
    for k in keys:                                        # fill test, then val, remainder -> train
        s = "test" if cnt["test"] < tgt["test"] else "val" if cnt["val"] < tgt["val"] else "train"
        out[k] = s; cnt[s] += len(groups[k])
    for k, s in out.items():
        os.makedirs(os.path.join(a.dst, s, cls), exist_ok=True)
        for f in groups[k]:
            (shutil.move if a.move else shutil.copy2)(os.path.join(d, f), os.path.join(a.dst, s, cls, f))
    print(f"{cls}: {n} images, {len(groups)} groups ->", dict(cnt))
