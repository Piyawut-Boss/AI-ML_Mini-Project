# -*- coding: utf-8 -*-
"""mlkit: the machine-learning parts of this project written from scratch with NumPy
(softmax regression + Adam, stratified k-fold, standardiser, random search, evaluation metrics)."""
from .data_utils import stratified_kfold, stratified_holdout, Standardizer
from .softmax_regression import SoftmaxRegression, softmax, loss_and_grad
from .search import random_search, sample_params
from .metrics import (confusion_matrix, prf, macro_f1, top_k_accuracy, brier, ece, bootstrap_ci,
                      paired_bootstrap_diff, summarize)
