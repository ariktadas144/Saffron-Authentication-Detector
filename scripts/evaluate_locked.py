"""Evaluate a locked prospective set once and HMAC-sign the resulting report."""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import secrets
from pathlib import Path
import sys

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.lock_test_set import sha256_file


def verify_locked_manifest(directory: Path) -> pd.DataFrame:
    manifest_path = directory / "locked_manifest.csv"
    digest_path = directory / "LOCKED.sha256"
    if not manifest_path.is_file() or not digest_path.is_file():
        raise FileNotFoundError(f"locked manifest or digest missing in {directory}")
    expected = digest_path.read_text(encoding="ascii").strip()
    actual = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if not hmac.compare_digest(expected, actual):
        raise ValueError("locked manifest changed after creation")
    manifest = pd.read_csv(manifest_path)
    for row in manifest.itertuples(index=False):
        path = Path(row.path)
        if not path.is_file() or sha256_file(path) != row.sha256:
            raise ValueError(f"locked image changed or is missing: {path}")
    return manifest


def claim_evaluation_once(directory: Path) -> Path:
    marker = directory / "EVALUATION_CLAIMED"
    try:
        descriptor = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError("this locked test set has already been claimed for evaluation") from exc
    with os.fdopen(descriptor, "w", encoding="ascii") as stream:
        stream.write(secrets.token_hex(16))
    return marker


def evaluate_once(directory: Path, run_dir: Path, output_path: Path, signing_key: bytes) -> Path:
    manifest = verify_locked_manifest(directory)
    claim_evaluation_once(directory)
    from app.predict import load_model, predict_images

    bundle = load_model(str(run_dir), str(ROOT / "data" / "manifest.csv"))
    rows = []
    for group, subset in manifest.groupby("group", sort=True):
        images = []
        for path in subset.path:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        result = predict_images(images, bundle)
        rows.append({
            "group": str(group), "label": int(subset.label.iloc[0]),
            "camera": str(subset.camera.iloc[0]),
            "n_images": len(subset), "n_samples": 1,
            "prediction": result["label"], "p_fake": result["p_fake"],
            "method": result["method"],
            "abstention_reasons": json.dumps(result["details"].get("abstention_reasons", [])),
        })
    frame = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path = output_path.with_suffix(".csv")
    frame.to_csv(report_path, index=False)
    signature = hmac.new(signing_key, report_path.read_bytes(), hashlib.sha256).hexdigest()
    output_path.with_suffix(".csv.hmac").write_text(signature + "\n", encoding="ascii")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--locked-dir", required=True)
    parser.add_argument("--run", default="runs/cv1")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    key = os.environ.get("SAFFRON_REPORT_HMAC_KEY")
    if not key:
        raise RuntimeError("set SAFFRON_REPORT_HMAC_KEY to sign the locked evaluation report")
    resolve = lambda value: Path(value) if Path(value).is_absolute() else ROOT / value
    path = evaluate_once(
        resolve(args.locked_dir), resolve(args.run), resolve(args.out), key.encode("utf-8")
    )
    print(f"Evaluated locked set once; report: {path}")
    print(f"HMAC signature: {path.with_suffix(path.suffix + '.hmac')}")


if __name__ == "__main__":
    main()
