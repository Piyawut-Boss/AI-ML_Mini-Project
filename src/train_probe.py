# -*- coding: utf-8 -*-
"""Supervised training of a softmax-regression head on frozen SigLIP ViT-B/16 image embeddings.

Everything that LEARNS here is hand-written (src/mlkit): softmax regression + Adam, standardiser, stratified k-fold,
random search, metrics. The only pretrained part is the frozen SigLIP image encoder (open_clip, webli weights), which
turns each photo into a 768-d embedding.

Protocol (course week 14, nested cross-validation): an OUTER 5-fold stratified loop estimates generalisation; inside each
outer training part an INNER 5-fold random search (20 candidates over learning rate, L2 strength, epochs, batch size)
picks the hyper-parameters, so no image is ever predicted by a model that saw it or whose hyper-parameters were chosen with it.
The same 444 images / the same outer folds are used for the dummy baseline and the zero-shot SigLIP system, enabling a paired comparison.

Outputs: results/train_probe.json, results/models/probe_softmax.npz, results/probs/trained_oof.npy, results/figures/tp_*.png
"""
import json, os, time, platform
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import ROOT, RESULTS, FIG, PROBS, GROUPS, SEED, K_OUT, load_dataset, outer_splits
from mlkit import (SoftmaxRegression, Standardizer, stratified_kfold, random_search, summarize,
                   bootstrap_ci, paired_bootstrap_diff, macro_f1)
from vehicle_lib import build                       # zero-shot SigLIP system (prompt ensemble + cosine similarity + softmax)
from vehicle_lib.models import get_extractor        # frozen pretrained SigLIP image encoder (disk-cached embeddings)

METHOD, K_IN, N_ITER, C = "clip_zeroshot_ensemble_siglip_b16", 5, 20, 8
os.makedirs(os.path.join(RESULTS, "models"), exist_ok=True)

d = load_dataset(); y = d.y.values; n = len(y)
counts = np.bincount(y, minlength=C)
print(f"images={n} classes={dict(zip(GROUPS, counts.tolist()))}")
t0 = time.time()
X = get_extractor("siglip_b16")(list(d.p)); X = X / np.linalg.norm(X, axis=1, keepdims=True)
Pz = build(METHOD).predict_proba_coarse(list(d.p))
print(f"embeddings {X.shape} + zero-shot probabilities {Pz.shape} in {time.time() - t0:.0f}s")


def fit_predict(Xtr, ytr, Xte, p, seed=0):
    """Standardise with TRAIN statistics only -> train softmax regression -> probabilities of Xte."""
    sc = Standardizer().fit(Xtr)
    m = SoftmaxRegression(C, lr=p["lr"], l2=p["l2"], epochs=p["epochs"], batch_size=p["batch_size"], seed=seed).fit(sc.transform(Xtr), ytr)
    return m.predict_proba(sc.transform(Xte))


# ---------------- nested cross-validation ----------------
oof = np.zeros((n, C)); folds_log = []; t0 = time.time()
for k, (tr, te) in enumerate(outer_splits(y)):
    best, inner_f1, trials = random_search(fit_predict, X[tr], y[tr], N_ITER, K_IN, seed=SEED + k)
    oof[te] = fit_predict(X[tr], y[tr], X[te], best, seed=SEED + k)
    folds_log.append({"fold": k, "n_train": int(len(tr)), "n_test": int(len(te)), "best_params": best,
                      "inner_best_macro_f1": inner_f1, "outer_test_acc": float((oof[te].argmax(1) == y[te]).mean()),
                      "trials": trials})
    print(f"fold {k}: acc={folds_log[-1]['outer_test_acc']:.3f}  inner F1={inner_f1:.3f}  params={best}")
nested_seconds = time.time() - t0

trained = summarize(y, oof, GROUPS)
zero_shot = summarize(y, Pz, GROUPS)
dummy_P = np.zeros((n, C)); dummy_P[:, counts.argmax()] = 1.0                      # always answer the majority class
dummy = summarize(y, dummy_P, GROUPS)
paired = paired_bootstrap_diff(oof.argmax(1) == y, Pz.argmax(1) == y)

# threshold -> coverage (answer only when confident, the rest goes to a human)
conf, hit = oof.max(1), (oof.argmax(1) == y)
threshold_table = [{"threshold": t, "coverage": float((conf >= t).mean()),
                    "accuracy_answered": float(hit[conf >= t].mean()) if (conf >= t).any() else None}
                   for t in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]]

# ---------------- final deployable model on all images (hyper-parameters from inner CV only) ----------------
t0 = time.time()
best, cv_f1, _ = random_search(fit_predict, X, y, N_ITER, K_IN, seed=SEED)
sc = Standardizer().fit(X)
final = SoftmaxRegression(C, lr=best["lr"], l2=best["l2"], epochs=best["epochs"], batch_size=best["batch_size"], seed=SEED).fit(sc.transform(X), y)
final_seconds = time.time() - t0
np.savez(os.path.join(RESULTS, "models", "probe_softmax.npz"), W=final.W, b=final.b, mean=sc.mean_, std=sc.std_, groups=np.array(GROUPS))
np.save(os.path.join(PROBS, "trained_oof.npy"), oof)

# ---------------- learning curve with FIXED, untuned hyper-parameters ----------------
fixed = {"lr": 0.05, "l2": 0.1, "epochs": 200, "batch_size": None}
rng = np.random.default_rng(SEED); curve = []
for per in [3, 5, 10, 15, 20, 30]:
    a, z = [], []
    for _ in range(30):
        tr = np.concatenate([rng.choice(np.where(y == c)[0], size=min(per, int((y == c).sum()) - 8), replace=False) for c in range(C)])
        te = np.setdiff1d(np.arange(n), tr)
        a.append((fit_predict(X[tr], y[tr], X[te], fixed).argmax(1) == y[te]).mean()); z.append((Pz[te].argmax(1) == y[te]).mean())
    curve.append({"train_per_class": per, "n_train": int(len(tr)), "acc_mean": float(np.mean(a)), "acc_sd": float(np.std(a)),
                  "zero_shot_same_test_mean": float(np.mean(z))})

res = {
    "n_images": int(n), "class_counts": dict(zip(GROUPS, counts.tolist())),
    "protocol": {"outer_folds": K_OUT, "inner_folds": K_IN, "random_search_n_iter": N_ITER, "scoring_inner": "macro-F1",
                 "search_space": {"lr": "loguniform(1e-3, 3e-1)", "l2": "loguniform(1e-4, 1e1)", "epochs": [50, 100, 200, 400], "batch_size": [None, 64]},
                 "model": "StandardScaler + softmax regression (cross-entropy + L2), Adam optimiser, all implemented in src/mlkit (NumPy)",
                 "features": "SigLIP ViT-B/16 (open_clip, webli) frozen image embeddings, 768-d, L2-normalised", "seed": SEED,
                 "fits_nested": K_OUT * (N_ITER * K_IN + 1), "fits_final": N_ITER * K_IN + 1, "nested_cv_seconds": nested_seconds},
    "folds": [{k: v for k, v in f.items() if k != "trials"} for f in folds_log],
    "trained_oof": trained, "zero_shot_same_images": zero_shot, "dummy_majority": dummy,
    "baseline": {"random_chance": 1 / C, "dummy_majority_accuracy": dummy["accuracy"], "dummy_majority_macro_f1": dummy["macro_f1"],
                 "majority_class": GROUPS[int(counts.argmax())]},
    "paired_diff_trained_minus_zeroshot_acc": paired,
    "confusion_matrix": trained["confusion_matrix"],
    "calibration_trained": {"brier": trained["brier"], "ece_10bins": trained["ece_10bins"], "mean_confidence": trained["mean_confidence"]},
    "threshold_coverage_trained": threshold_table,
    "final_model": {"best_params": best, "cv_macro_f1_inner": cv_f1, "fit_seconds_incl_search": final_seconds,
                    "train_accuracy_on_its_own_training_data": float((final.predict(sc.transform(X)) == y).mean()),
                    "final_training_loss": float(final.history[-1]), "loss_first_epoch": float(final.history[0])},
    "learning_curve_fixed_hparams": {"hyperparameters": fixed, "points": curve},
    "env": {"numpy": np.__version__, "python": platform.python_version()},
}
json.dump(res, open(os.path.join(RESULTS, "train_probe.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)

# ---------------- figures ----------------
cm = np.array(trained["confusion_matrix"]); cmn = cm / cm.sum(1, keepdims=True)
fig, ax = plt.subplots(figsize=(5.4, 4.6)); ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
for i in range(C):
    for j in range(C):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7, color="white" if cmn[i, j] > .5 else "black")
ax.set_xticks(range(C)); ax.set_yticks(range(C)); ax.set_xticklabels(GROUPS, rotation=45, ha="right", fontsize=7); ax.set_yticklabels(GROUPS, fontsize=7)
ax.set_xlabel("Predicted group"); ax.set_ylabel("True group"); ax.set_title("Out-of-fold confusion matrix (softmax-regression head, n=%d)" % n, fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "tp_confusion.png"), dpi=200); plt.close(fig)
fig, ax = plt.subplots(figsize=(5.2, 3.8)); xs = [c["train_per_class"] for c in curve]
ax.errorbar(xs, [c["acc_mean"] for c in curve], yerr=[c["acc_sd"] for c in curve], fmt="o-", color="#1d4f91", capsize=3, label="Softmax-regression head (trained, fixed hyper-parameters)")
ax.plot(xs, [c["zero_shot_same_test_mean"] for c in curve], "s--", color="#d62728", label="Zero-shot (no training)")
ax.set_xlabel("Labelled training images per class"); ax.set_ylabel("Accuracy on remaining images (8 classes)")
ax.set_title("Learning curve (30 random splits)", fontsize=9); ax.grid(alpha=.3); ax.legend(fontsize=6.5)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "tp_learning_curve.png"), dpi=200); plt.close(fig)

print(json.dumps({"trained_acc": trained["accuracy"], "ci95": trained["ci95"], "macro_f1": trained["macro_f1"], "top2": trained["top2"],
                  "zero_shot_acc": zero_shot["accuracy"], "dummy_acc": dummy["accuracy"], "paired_diff": paired,
                  "brier": trained["brier"], "ece": trained["ece_10bins"], "mean_conf": trained["mean_confidence"],
                  "final": res["final_model"], "fits_nested": res["protocol"]["fits_nested"], "nested_seconds": round(nested_seconds, 1)}, ensure_ascii=False, indent=1))
print("curve:", [(c["train_per_class"], round(c["acc_mean"], 3)) for c in curve])
