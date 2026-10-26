"""Split MVTec AD bottle once (seed 42), SHA-256 every image, write splits/*.csv and splits/manifest.json.

Run from the project root: python -m src.data [--force]
Every run checks folders, counts and 900x900 size and hashes all images. A re-run loads the saved splits and
verifies them against the manifest and the files on disk; only --force re-splits. Test images are only sized
(header read) and hashed here; load_split("test") raises unless allow_test=True, which only src/eval.py passes.
"""
import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "mvtec" / "bottle"
SPLITS = ROOT / "splits"
MANIFEST = SPLITS / "manifest.json"
SEED = 42
IMG = 224
N_VAL, N_CALIB = 21, 50   # val = 10% of 209 train/good; calib is drawn from fit
EXPECTED = {"train": {"good": 209},
            "test": {"good": 20, "broken_large": 20, "broken_small": 22, "contamination": 21}}
NAMES = ("fit", "val", "calib", "test")
COLUMNS = ["path", "label", "defect_type", "sha256"]
CLASSES = ["good", "broken_large", "broken_small", "contamination"]


def sha256_of(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def list_images(part):
    """Rows for every image under train/ or test/, sorted by path, after checking folders, counts and size."""
    expected = EXPECTED[part]
    folders = sorted(d.name for d in (DATA / part).iterdir())
    assert folders == sorted(expected), f"{part}/ holds {folders}, expected {sorted(expected)}"
    rows = []
    for defect in sorted(expected):
        paths = sorted((DATA / part / defect).glob("*.png"))
        assert len(paths) == expected[defect], f"{part}/{defect}: {len(paths)} images, expected {expected[defect]}"
        for p in paths:
            rel = p.relative_to(ROOT).as_posix()
            with Image.open(p) as img:   # matching counts alone do not prove a third-party mirror holds the originals
                assert img.size == (900, 900), f"{rel}: {img.size}, expected 900x900"
            rows.append({"path": rel, "label": int(defect != "good"), "defect_type": defect, "sha256": sha256_of(p)})
    return pd.DataFrame(rows, columns=COLUMNS)


def split(train, test):
    """val = 21 of train/good, fit = the other 188, calib = 50 of fit; test is the official test folder as is."""
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(train))   # train is sorted by path, so the draw never depends on listing order
    val, fit = train.iloc[order[:N_VAL]], train.iloc[order[N_VAL:]]
    calib = fit.iloc[rng.choice(len(fit), N_CALIB, replace=False)]
    return {name: df.sort_values("path", ignore_index=True) for name, df in zip(NAMES, (fit, val, calib, test))}


def write(splits):
    """Write the CSVs with fixed newlines so --force rebuilds identical bytes, then the manifest last."""
    SPLITS.mkdir(exist_ok=True)
    for name, df in splits.items():
        df.to_csv(SPLITS / f"{name}.csv", index=False, lineterminator="\n")
    manifest = {f"{name}.csv": sha256_of(SPLITS / f"{name}.csv") for name in NAMES}
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")   # LF: freeze.py hashes it
    return manifest


def load_saved(images):
    """Saved splits, after checking each CSV against the manifest and fit + val + test against the images on disk."""
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for name in NAMES:
        assert sha256_of(SPLITS / f"{name}.csv") == manifest[f"{name}.csv"], f"splits/{name}.csv changed since the manifest"
    splits = {name: pd.read_csv(SPLITS / f"{name}.csv") for name in NAMES}
    saved = pd.concat([splits["fit"], splits["val"], splits["test"]])
    assert set(zip(saved["path"], saved["sha256"])) == set(zip(images["path"], images["sha256"])), \
        "fit + val + test no longer match the images on disk (path or sha256)"
    return splits, manifest


def load_split(name, allow_test=False):
    """The only way later code reads a split; test.csv needs allow_test=True, which only src/eval.py passes."""
    if name == "test" and not allow_test:
        raise PermissionError("test.csv is locked: only src/eval.py may open it (allow_test=True)")
    return pd.read_csv(SPLITS / f"{name}.csv")


def load_images(df):
    """(n, 3, 224, 224) float32 in [0, 1]: RGB, resized with no crop so rim defects stay in frame, not normalized."""
    out = torch.empty(len(df), 3, IMG, IMG)
    for i, p in enumerate(df["path"]):
        with Image.open(ROOT / p) as img:
            arr = np.array(img.convert("RGB").resize((IMG, IMG), Image.Resampling.BILINEAR))
        out[i] = torch.from_numpy(arr).permute(2, 0, 1).float() / 255
    return out


def report(splits, manifest):
    """Counts per split and class, exact-duplicate counts, manifest hashes. Duplicates are reported, not asserted."""
    counts = pd.DataFrame({n: df["defect_type"].value_counts().reindex(CLASSES, fill_value=0)
                           for n, df in splits.items()}).T
    counts["defect"] = counts[CLASSES[1:]].sum(axis=1)
    counts["total"] = counts["good"] + counts["defect"]
    print(f"\nimages per split and class (seed {SEED})\n{counts.to_string()}")
    sha = {n: set(df["sha256"]) for n, df in splits.items()}
    dups = [(f"within {n}", len(df) - len(sha[n])) for n, df in splits.items()]
    dups += [(f"{a} x {b}", len(sha[a] & sha[b])) for a, b in combinations(("fit", "val", "test"), 2)]
    dups += [("calib not in fit", len(sha["calib"] - sha["fit"])),
             ("calib x val", len(sha["calib"] & sha["val"])),
             ("calib x test", len(sha["calib"] & sha["test"]))]
    print("\nexact duplicates by sha256 (expected 0; calib is drawn from fit by design)")
    for label, n in dups:
        print(f"  {label:<18}{n}")
    print("\nsplits/manifest.json")
    for name, h in manifest.items():
        print(f"  {name:<10}{h}")


def self_check():
    """The test lock fires and the shared loader returns the documented shape, dtype and range."""
    try:
        load_split("test")
        raise AssertionError("load_split('test') did not raise PermissionError")
    except PermissionError:
        pass
    x = load_images(load_split("val").head(1))
    lo, hi = x.min().item(), x.max().item()
    assert x.shape == (1, 3, IMG, IMG) and x.dtype == torch.float32 and 0 <= lo and hi <= 1, (x.shape, x.dtype, lo, hi)
    print(f"\nself-check OK: load_split('test') raises PermissionError; "
          f"load_images gives {tuple(x.shape)} float32, range [{lo:.3f}, {hi:.3f}]")


def main(force=False):
    train, test = list_images("train"), list_images("test")
    print(f"checked {len(train) + len(test)} images: folders and counts match MVTec AD bottle, all 900x900, sha256 computed")
    if force or not MANIFEST.exists():
        old = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else None
        manifest = write(splits := split(train, test))
        print("--force: re-split and rewrote splits/" if force else "first run: split and wrote splits/")
        if old:
            print(f"rebuilt CSVs byte-identical to the previous ones: {old == manifest}")
    else:
        splits, manifest = load_saved(pd.concat([train, test]))
        print("splits already on disk: verified against manifest and images, not re-split (use --force to rebuild)")
    report(splits, manifest)
    self_check()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="re-split and rewrite splits/ (byte-identical)")
    main(parser.parse_args().force)
