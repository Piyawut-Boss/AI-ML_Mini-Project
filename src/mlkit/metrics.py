# -*- coding: utf-8 -*-
"""Hand-written evaluation metrics (NumPy only): accuracy, confusion matrix, precision/recall/F1, top-k,
Brier score, expected calibration error, bootstrap confidence intervals (course week 14)."""
import numpy as np


def confusion_matrix(y, pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(cm, (np.asarray(y), np.asarray(pred)), 1)          # rows = true class, columns = predicted class
    return cm


def prf(y, pred, n_classes):
    """Per-class precision, recall, F1 and support."""
    cm = confusion_matrix(y, pred, n_classes)
    tp = np.diag(cm).astype(float)
    precision = np.divide(tp, cm.sum(0), out=np.zeros(n_classes), where=cm.sum(0) > 0)
    recall = np.divide(tp, cm.sum(1), out=np.zeros(n_classes), where=cm.sum(1) > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros(n_classes), where=denom > 0)
    return precision, recall, f1, cm.sum(1)


def macro_f1(y, pred, n_classes):
    return float(prf(y, pred, n_classes)[2].mean())


def top_k_accuracy(y, P, k):
    top = np.argsort(-P, axis=1)[:, :k]
    return float((top == np.asarray(y)[:, None]).any(1).mean())


def brier(P, y):
    return float(((P - np.eye(P.shape[1])[y]) ** 2).sum(1).mean())


def ece(P, y, bins=10):
    """Expected calibration error with equal-width confidence bins (an addition to the course's Brier score)."""
    conf, correct = P.max(1), (P.argmax(1) == y).astype(float)
    edges, total = np.linspace(0, 1, bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def bootstrap_ci(values, B=1000, seed=42):
    """95% percentile bootstrap interval of the mean of per-image 0/1 outcomes (or any per-image score)."""
    v = np.asarray(values, float); rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), (B, len(v)))].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_bootstrap_diff(correct_a, correct_b, B=1000, seed=42):
    """Bootstrap of accuracy(A) - accuracy(B) on the SAME images (paired)."""
    a, b = np.asarray(correct_a, float), np.asarray(correct_b, float)
    idx = np.random.default_rng(seed).integers(0, len(a), (B, len(a)))
    d = a[idx].mean(1) - b[idx].mean(1)
    return {"mean": float(d.mean()), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}


def summarize(y, P, group_names):
    """All headline metrics of probability matrix P [N, C] against labels y, as a JSON-ready dict."""
    y = np.asarray(y); C = P.shape[1]; pred = P.argmax(1); correct = (pred == y).astype(float)
    precision, recall, f1, support = prf(y, pred, C)
    return {"n": int(len(y)), "accuracy": float(correct.mean()), "ci95": list(bootstrap_ci(correct)),
            "macro_f1": float(f1.mean()), "top2": top_k_accuracy(y, P, 2),
            "brier": brier(P, y), "ece_10bins": ece(P, y), "mean_confidence": float(P.max(1).mean()),
            "per_class": {g: {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]),
                              "n": int(support[i])} for i, g in enumerate(group_names)},
            "confusion_matrix": confusion_matrix(y, pred, C).tolist()}
