# -*- coding: utf-8 -*-
"""Hand-written random hyper-parameter search with inner cross-validation (course week 14: random search, nested CV)."""
import numpy as np
from .data_utils import stratified_kfold
from .metrics import macro_f1


def loguniform(rng, lo, hi):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def sample_params(rng):
    """One random configuration of the softmax-regression head."""
    return {"lr": loguniform(rng, 1e-3, 3e-1), "l2": loguniform(rng, 1e-4, 1e1),
            "epochs": int(rng.choice([50, 100, 200, 400])), "batch_size": rng.choice([0, 64]).item() or None}


def random_search(fit_predict, X, y, n_iter, k_inner, seed):
    """fit_predict(Xtr, ytr, Xva, params) -> probabilities of Xva.  Returns (best_params, best_score, all_trials).
    Score = mean macro-F1 over the inner folds; the outer test fold is never touched."""
    rng = np.random.default_rng(seed)
    folds = stratified_kfold(y, k_inner, seed)
    trials = []
    for _ in range(n_iter):
        p = sample_params(rng)
        scores = [macro_f1(y[va], fit_predict(X[tr], y[tr], X[va], p).argmax(1), 8) for tr, va in folds]
        trials.append({"params": p, "inner_macro_f1": float(np.mean(scores))})
    best = max(trials, key=lambda t: t["inner_macro_f1"])
    return best["params"], best["inner_macro_f1"], trials
