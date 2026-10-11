# -*- coding: utf-8 -*-
"""Multinomial logistic (softmax) regression trained from scratch with NumPy.

forward   : z = X W + b ,  P = softmax(z)
loss      : mean cross-entropy  +  (l2 / 2) * ||W||^2            (bias is not penalised)
gradient  : dL/dW = X^T (P - Y) / m + l2 * W ,   dL/db = sum(P - Y) / m      (derived by hand, checked in tests/)
optimiser : Adam (Kingma & Ba) written out below, full-batch or mini-batch
This is the 'new head on a frozen encoder' of course weeks 9-10: only these weights are learned."""
import numpy as np


def softmax(z):
    z = z - z.max(1, keepdims=True)                   # numerical stability
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def loss_and_grad(W, b, X, y, l2):
    """Cross-entropy + L2 and its analytic gradient on the batch (X, y)."""
    m = len(y)
    P = softmax(X @ W + b)
    loss = -np.log(P[np.arange(m), y] + 1e-12).mean() + 0.5 * l2 * float((W * W).sum())
    G = P.copy(); G[np.arange(m), y] -= 1.0; G /= m   # (P - Y) / m
    return loss, X.T @ G + l2 * W, G.sum(0)


class SoftmaxRegression:
    def __init__(self, n_classes, lr=0.05, l2=1e-2, epochs=200, batch_size=None, seed=0,
                 beta1=0.9, beta2=0.999, eps=1e-8):
        self.n_classes, self.lr, self.l2, self.epochs, self.batch_size = n_classes, lr, l2, epochs, batch_size
        self.seed, self.beta1, self.beta2, self.eps = seed, beta1, beta2, eps

    def fit(self, X, y):
        n, d = X.shape
        rng = np.random.default_rng(self.seed)
        self.W = rng.normal(0.0, 0.01, (d, self.n_classes)); self.b = np.zeros(self.n_classes)
        mW, vW = np.zeros_like(self.W), np.zeros_like(self.W)          # Adam moment estimates
        mb, vb = np.zeros_like(self.b), np.zeros_like(self.b)
        t, self.history = 0, []
        for _ in range(self.epochs):
            order = rng.permutation(n) if self.batch_size else np.arange(n)
            step = self.batch_size or n
            for s in range(0, n, step):
                idx = order[s:s + step]
                _, gW, gb = loss_and_grad(self.W, self.b, X[idx], y[idx], self.l2)
                t += 1
                mW = self.beta1 * mW + (1 - self.beta1) * gW; vW = self.beta2 * vW + (1 - self.beta2) * gW * gW
                mb = self.beta1 * mb + (1 - self.beta1) * gb; vb = self.beta2 * vb + (1 - self.beta2) * gb * gb
                c1, c2 = 1 - self.beta1 ** t, 1 - self.beta2 ** t      # bias correction
                self.W -= self.lr * (mW / c1) / (np.sqrt(vW / c2) + self.eps)
                self.b -= self.lr * (mb / c1) / (np.sqrt(vb / c2) + self.eps)
            self.history.append(loss_and_grad(self.W, self.b, X, y, self.l2)[0])
        return self

    def predict_proba(self, X):
        return softmax(X @ self.W + self.b)

    def predict(self, X):
        return self.predict_proba(X).argmax(1)
