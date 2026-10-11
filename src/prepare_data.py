# -*- coding: utf-8 -*-
"""Data preparation from scratch: Hugging Face `kitrofimov/cbsc` (Car Body Style Classification, MIT per dataset card)
-> cleaned, de-duplicated, labelled image set used by every model in this repo.

Steps (all deterministic, seed 42):
  1. download the raw dataset (skipped if already present)
  2. drop the classes that are not part of the project's body-type chart (limousine, bus)
  3. convert every image to RGB JPEG (quality 90, long side <= 1024)
  4. remove near-duplicates with a perceptual hash (pHash, Hamming distance <= 6) BEFORE any split, so that copies of
     the same photo can never fall on both sides of a train/test split
  5. keep 10 'support' images per class (few-shot reference pool; 5 if a class has < 25 images), the rest are 'eval'
  6. write data/dataset.csv (+ data/dataset_stats.json)

Usage:  python src/prepare_data.py [--data-dir data] [--raw-dir path/to/an/already-downloaded/cbsc]
"""
import os, sys, json, csv, random, shutil, argparse, collections
from PIL import Image
import numpy as np
from scipy.fft import dct

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DROP = {"limousine", "bus"}
# CBSC class -> project body-type group (one-to-one for the 8 classes kept)
GROUP_OF = {"hatchback": "small", "sedan": "sedan", "sports": "sporty_closed", "cabriolet": "open_top",
            "suv": "suv_family", "pickup": "light_truck", "van": "van_family", "truck": "heavy_truck"}
EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
HASH_THRESHOLD, SEED, MAX_SIDE, JPEG_Q = 6, 42, 1024, 90


def phash(im, hash_size=8, factor=4):
    """Perceptual hash written out by hand: grayscale -> 32x32 -> 2-D DCT-II -> keep the 8x8 lowest frequencies ->
    bit = coefficient above the median.  Visually similar images get hashes with a small Hamming distance."""
    s = hash_size * factor
    a = np.asarray(im.convert("L").resize((s, s), Image.LANCZOS), dtype=np.float64)
    low = dct(dct(a, axis=0), axis=1)[:hash_size, :hash_size]
    return low > np.median(low)


def fetch(raw):
    if os.path.isdir(os.path.join(raw, "sedan")):
        return
    from huggingface_hub import snapshot_download
    snapshot_download("kitrofimov/cbsc", repo_type="dataset", local_dir=raw)


def main(data_dir, raw_dir=None):
    raw = raw_dir or os.path.join(data_dir, "cbsc_raw")
    fetch(raw)
    files = []                                                 # deterministic order: sorted folder, sorted file
    for d in sorted(os.listdir(raw), key=str.lower):
        dp = os.path.join(raw, d)
        if os.path.isdir(dp) and not d.startswith("."):
            files += [(d.lower(), os.path.join(dp, f)) for f in sorted(os.listdir(dp)) if f.lower().endswith(EXT)]
    found = collections.Counter(c for c, _ in files)

    kept, dropped, unreadable, dup_pairs, hashes = [], collections.Counter(), 0, [], []
    for idx, (cls, path) in enumerate(files):
        if cls in DROP:
            dropped[cls] += 1; continue
        try:
            im = Image.open(path).convert("RGB")
        except Exception:
            unreadable += 1; continue
        h = phash(im)
        twin = next((name for hh, name in hashes if np.count_nonzero(h != hh) <= HASH_THRESHOLD), None)
        if twin is not None:
            dup_pairs.append((cls, os.path.basename(path), twin)); continue
        hashes.append((h, (cls, os.path.basename(path))))
        kept.append({"cls": cls, "idx": idx, "im": im})

    img_dir = os.path.join(data_dir, "images")
    if os.path.exists(img_dir):
        shutil.rmtree(img_dir)
    rng, rows = random.Random(SEED), []
    for cls in sorted(GROUP_OF):
        items = [k for k in kept if k["cls"] == cls]
        support = set(rng.sample(range(len(items)), 10 if len(items) >= 25 else 5))
        os.makedirs(os.path.join(img_dir, cls))
        for i, k in enumerate(items):
            im = k["im"]
            if max(im.size) > MAX_SIDE:
                s = MAX_SIDE / max(im.size); im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
            fn = os.path.join(img_dir, cls, f"{i + 1:03d}.jpg"); im.save(fn, "JPEG", quality=JPEG_Q)
            rows.append({"path": os.path.relpath(fn, ROOT).replace("\\", "/"), "class": cls, "coarse_group": GROUP_OF[cls],
                         "cbsc_index": k["idx"], "width": im.width, "height": im.height,
                         "split_role": "support" if i in support else "eval"})
    with open(os.path.join(data_dir, "dataset.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, ["path", "class", "coarse_group", "cbsc_index", "width", "height", "split_role"])
        w.writeheader(); w.writerows(rows)
    stats = {"source": "hf:kitrofimov/cbsc", "files_found": len(files), "original_counts": dict(found),
             "dropped_classes": dict(dropped), "unreadable": unreadable, "duplicates_removed": len(dup_pairs),
             "total_kept": len(rows), "per_class": {c: sum(r["class"] == c for r in rows) for c in sorted(GROUP_OF)}}
    json.dump(stats, open(os.path.join(data_dir, "dataset_stats.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--data-dir", default=os.path.join(ROOT, "data")); ap.add_argument("--raw-dir", default=None)
    a = ap.parse_args()
    main(os.path.abspath(a.data_dir), os.path.abspath(a.raw_dir) if a.raw_dir else None)
