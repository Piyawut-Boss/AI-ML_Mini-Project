# -*- coding: utf-8 -*-
"""Shared paths and dataset loading. Every model is evaluated on the same images and the same outer folds."""
import os, sys
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")
FIG = os.path.join(RESULTS, "figures")
PROBS = os.path.join(RESULTS, "probs")
for _d in (RESULTS, FIG, PROBS):
    os.makedirs(_d, exist_ok=True)

from mlkit import stratified_kfold
from vehicle_lib.classes import COARSE_NAMES            # the 8 body-type groups = the 8 CBSC classes kept
GROUPS = list(COARSE_NAMES)
SEED, K_OUT = 42, 5


def load_dataset():
    """data/dataset.csv -> DataFrame with absolute image path `p` (re-based to this checkout) and label index `y`."""
    d = pd.read_csv(os.path.join(DATA, "dataset.csv"), encoding="utf-8-sig")
    d["p"] = [os.path.join(DATA, "images", c, os.path.basename(p.replace("\\", "/"))) for c, p in zip(d["class"], d["path"])]
    missing = [p for p in d.p if not os.path.exists(p)]
    assert not missing, f"{len(missing)} image files missing, e.g. {missing[:2]} (run src/prepare_data.py)"
    d["y"] = d.coarse_group.map({g: i for i, g in enumerate(GROUPS)}).astype(int)
    return d


def outer_splits(y):
    """The single outer 5-fold stratified split shared by all models (hand-written, deterministic)."""
    return stratified_kfold(np.asarray(y), K_OUT, SEED)
