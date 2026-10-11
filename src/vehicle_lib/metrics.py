# -*- coding: utf-8 -*-
"""Metrics: top-k accuracy, macro-F1, per-class P/R/F1, confusion matrix, coarse-8, bootstrap CI."""
import numpy as np
from .classes import CLASSES, COARSE_NAMES, COARSE_OF_IDX


def confusion(y, pred, n=16):
    cm = np.zeros((n, n), int)
    np.add.at(cm, (y, pred), 1)
    return cm


def prf(cm):
    tp = np.diag(cm).astype(float)
    p = tp / np.clip(cm.sum(0), 1, None)
    r = tp / np.clip(cm.sum(1), 1, None)
    f = np.where(p + r > 0, 2 * p * r / np.clip(p + r, 1e-12, None), 0.0)
    return p, r, f


def topk_acc(y, probs, k=3):
    top = np.argsort(-probs, 1)[:, :k]
    return float((top == y[:, None]).any(1).mean())


def bootstrap_ci(correct, n_boot=1000, seed=42):
    rng = np.random.RandomState(seed)
    n = len(correct)
    idx = rng.randint(0, n, size=(n_boot, n))
    accs = correct[idx].mean(1)
    return float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))


def evaluate(y, probs):
    """y: int labels [N]; probs [N,16]. Returns dict + confusion matrix + per-class arrays."""
    y = np.asarray(y)
    pred = probs.argmax(1)
    cm = confusion(y, pred, 16)
    p, r, f = prf(cm)
    present = cm.sum(1) > 0                      # macro over classes that occur in the eval set
    correct = (pred == y).astype(float)
    lo, hi = bootstrap_ci(correct)
    yc, pc = COARSE_OF_IDX[y], COARSE_OF_IDX[pred]
    ccm = confusion(yc, pc, 8)
    _, _, cf = prf(ccm)
    cpresent = ccm.sum(1) > 0
    clo, chi = bootstrap_ci((yc == pc).astype(float))
    res = dict(n=len(y), acc=float(correct.mean()), ci_lo=lo, ci_hi=hi,
               top3=topk_acc(y, probs, 3), macro_f1=float(f[present].mean()),
               coarse_acc=float((yc == pc).mean()), coarse_macro_f1=float(cf[cpresent].mean()),
               coarse_ci_lo=clo, coarse_ci_hi=chi)
    return res, cm, dict(precision=p, recall=r, f1=f, support=cm.sum(1))


def evaluate_coarse(y, p8):
    """Primary evaluation: y in 0..7 (COARSE_NAMES), p8 [N,8]."""
    y = np.asarray(y)
    pred = p8.argmax(1)
    cm = confusion(y, pred, 8)
    p, r, f = prf(cm)
    present = cm.sum(1) > 0
    correct = (pred == y).astype(float)
    lo, hi = bootstrap_ci(correct)
    res = dict(n=len(y), acc=float(correct.mean()), ci_lo=lo, ci_hi=hi, top2=topk_acc(y, p8, 2),
               macro_f1=float(f[present].mean()))
    return res, cm, dict(precision=p, recall=r, f1=f, support=cm.sum(1))
