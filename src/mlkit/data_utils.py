# -*- coding: utf-8 -*-
"""Hand-written data utilities (NumPy only): stratified k-fold, stratified hold-out, feature standardiser."""
import numpy as np


def stratified_kfold(y, k, seed):
    """Return a list of k (train_idx, test_idx) pairs. Every class is shuffled once and dealt round-robin into the k folds,
    so each fold keeps (almost exactly) the class proportions of the whole data set."""
    y = np.asarray(y); rng = np.random.default_rng(seed)
    fold_of = np.empty(len(y), dtype=int)
    offset = 0                                        # continue the round-robin across classes -> fold sizes stay balanced
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        fold_of[idx] = (np.arange(len(idx)) + offset) % k
        offset += len(idx)
    return [(np.where(fold_of != f)[0], np.where(fold_of == f)[0]) for f in range(k)]


def stratified_holdout(y, frac, seed):
    """Split indices into (rest, held_out) with `frac` of every class held out."""
    y = np.asarray(y); rng = np.random.default_rng(seed); held = []
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        held += list(idx[:max(1, int(round(frac * len(idx))))])
    held = np.array(sorted(held)); rest = np.setdiff1d(np.arange(len(y)), held)
    return rest, held


class Standardizer:
    """z = (x - mean) / std, statistics learned on the training part only (prevents leakage)."""

    def fit(self, X):
        self.mean_ = X.mean(0)
        std = X.std(0)
        self.std_ = np.where(std < 1e-8, 1.0, std)    # constant features are left unscaled
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)
