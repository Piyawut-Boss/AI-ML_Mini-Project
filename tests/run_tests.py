# -*- coding: utf-8 -*-
"""Checks of the hand-written components (plain asserts, no test framework):  python tests/run_tests.py
scikit-learn is used ONLY here, as an independent reference to validate our own implementations."""
import os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from mlkit import (SoftmaxRegression, loss_and_grad, stratified_kfold, stratified_holdout, Standardizer,
                   confusion_matrix, prf, macro_f1, top_k_accuracy, brier, bootstrap_ci)
from mlkit.softmax_regression import softmax

PASSED = []


def check(name):
    def deco(fn):
        fn(); PASSED.append(name); print("ok  ", name); return fn
    return deco


def blobs(n_per=60, d=20, k=4, sep=3.0, seed=0):
    r = np.random.default_rng(seed); centers = r.normal(0, sep, (k, d))
    X = np.concatenate([centers[c] + r.normal(0, 1, (n_per, d)) for c in range(k)]); y = np.repeat(np.arange(k), n_per)
    return X, y


@check("softmax rows sum to 1 and are stable for huge logits")
def _():
    P = softmax(np.array([[1000.0, 1001.0, 999.0], [-1000.0, 0.0, 1000.0]]))
    assert np.allclose(P.sum(1), 1) and np.isfinite(P).all()


@check("analytic gradient equals finite differences (weights and bias)")
def _():
    r = np.random.default_rng(1); X = r.normal(size=(30, 6)); y = r.integers(0, 4, 30)
    W = r.normal(size=(6, 4)); b = r.normal(size=4); l2 = 0.3
    _, gW, gb = loss_and_grad(W, b, X, y, l2); h = 1e-6
    for (i, j) in [(0, 0), (2, 3), (5, 1)]:
        Wp, Wm = W.copy(), W.copy(); Wp[i, j] += h; Wm[i, j] -= h
        num = (loss_and_grad(Wp, b, X, y, l2)[0] - loss_and_grad(Wm, b, X, y, l2)[0]) / (2 * h)
        assert abs(num - gW[i, j]) < 1e-6, (i, j, num, gW[i, j])
    for j in range(4):
        bp, bm = b.copy(), b.copy(); bp[j] += h; bm[j] -= h
        num = (loss_and_grad(W, bp, X, y, l2)[0] - loss_and_grad(W, bm, X, y, l2)[0]) / (2 * h)
        assert abs(num - gb[j]) < 1e-6


@check("training loss decreases and separable data is fitted perfectly")
def _():
    X, y = blobs(); X = Standardizer().fit_transform(X)
    m = SoftmaxRegression(4, lr=0.05, l2=1e-3, epochs=150, seed=0).fit(X, y)
    assert m.history[-1] < m.history[0] * 0.2 and (m.predict(X) == y).mean() == 1.0


@check("mini-batch Adam also converges")
def _():
    X, y = blobs(seed=3); X = Standardizer().fit_transform(X)
    m = SoftmaxRegression(4, lr=0.02, l2=1e-3, epochs=60, batch_size=32, seed=0).fit(X, y)
    assert (m.predict(X) == y).mean() > 0.98


@check("agrees with scikit-learn LogisticRegression on held-out data")
def _():
    from sklearn.linear_model import LogisticRegression
    X, y = blobs(n_per=100, sep=1.0, seed=5); X = Standardizer().fit_transform(X)
    idx = np.random.default_rng(0).permutation(len(y)); tr, te = idx[:300], idx[300:]
    ours = SoftmaxRegression(4, lr=0.05, l2=1e-2, epochs=400, seed=0).fit(X[tr], y[tr]).predict(X[te])
    ref = LogisticRegression(C=1.0 / (1e-2 * len(tr)), max_iter=5000).fit(X[tr], y[tr]).predict(X[te])
    assert (ours == ref).mean() > 0.95, (ours == ref).mean()          # same objective -> (almost) same decisions


@check("stratified k-fold: disjoint, complete, class proportions kept")
def _():
    y = np.repeat(np.arange(8), [50, 84, 51, 34, 93, 46, 47, 39]); folds = stratified_kfold(y, 5, 42)
    seen = np.concatenate([te for _, te in folds]); assert sorted(seen) == list(range(len(y)))
    for tr, te in folds:
        assert not set(tr) & set(te)
        for c in range(8):
            assert abs((y[te] == c).sum() - (y == c).sum() / 5) <= 1.0
    assert all(np.array_equal(a[1], b[1]) for a, b in zip(folds, stratified_kfold(y, 5, 42)))    # deterministic


@check("stratified hold-out keeps every class represented")
def _():
    y = np.repeat(np.arange(4), [10, 20, 30, 40]); rest, held = stratified_holdout(y, 0.2, 0)
    assert sorted(np.concatenate([rest, held])) == list(range(100)) and set(y[held]) == {0, 1, 2, 3}


@check("standardiser uses training statistics only and survives constant columns")
def _():
    Xtr = np.array([[1.0, 5.0], [3.0, 5.0], [5.0, 5.0]]); s = Standardizer().fit(Xtr)
    Z = s.transform(np.array([[3.0, 5.0], [7.0, 5.0]]))
    assert abs(Z[0, 0]) < 1e-12 and np.isfinite(Z).all() and Z[1, 1] == 0.0


@check("metrics agree with scikit-learn")
def _():
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix as sk_cm, precision_score, recall_score
    r = np.random.default_rng(7); y = r.integers(0, 5, 400); P = r.dirichlet(np.ones(5), 400); pred = P.argmax(1)
    assert np.array_equal(confusion_matrix(y, pred, 5), sk_cm(y, pred, labels=range(5)))
    p, rc, f1, _ = prf(y, pred, 5)
    assert np.allclose(p, precision_score(y, pred, average=None)) and np.allclose(rc, recall_score(y, pred, average=None))
    assert abs(macro_f1(y, pred, 5) - f1_score(y, pred, average="macro")) < 1e-12
    assert abs((pred == y).mean() - accuracy_score(y, pred)) < 1e-12
    assert top_k_accuracy(y, P, 5) == 1.0 and 0.0 <= brier(P, y) <= 2.0
    lo, hi = bootstrap_ci((pred == y).astype(float)); assert lo <= (pred == y).mean() <= hi


@check("pHash: same image -> 0 bits apart, resized copy -> close, unrelated image -> far; equals imagehash")
def _():
    from PIL import Image
    import prepare_data as pdata
    yy, xx = np.mgrid[0:96, 0:96]
    art = np.stack([xx * 2.5, yy * 2.5, (xx + yy) * 1.2], -1); art[20:60, 30:80] = 255     # smooth picture with a bright block
    a = Image.fromarray(art.astype(np.uint8)); small = a.resize((48, 48)).resize((96, 96))
    other = Image.fromarray(np.stack([yy[::-1] * 2.5, xx[::-1] * 2.5, np.abs(xx - yy) * 2.5], -1).astype(np.uint8))
    ha, hs, ho = pdata.phash(a), pdata.phash(small), pdata.phash(other)
    assert np.count_nonzero(ha != pdata.phash(a)) == 0
    assert np.count_nonzero(ha != hs) <= pdata.HASH_THRESHOLD < np.count_nonzero(ha != ho)
    try:
        import imagehash                                # optional, used only as an independent reference
        assert np.array_equal(ha, imagehash.phash(a).hash) and np.array_equal(ho, imagehash.phash(other).hash)
    except ImportError:
        pass


print(f"\n{len(PASSED)} checks passed")
