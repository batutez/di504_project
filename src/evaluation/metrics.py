"""
Util functions for evaluation
bootstrap confidence intervals for weighted F1 and macro AUROC.
"""
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

def bootstrap_metric_ci(y_true, y_pred_or_proba, metric_fn, n_iterations=1000,
                         confidence_level=0.95, seed=42):
    
    y_true = np.asarray(y_true)
    y_pred_or_proba = np.asarray(y_pred_or_proba)

    rng = np.random.RandomState(seed)
    n = len(y_true)
    scores = []

    for _ in range(n_iterations):
        idx = rng.randint(0, n, n)  # resample n indices WITH replacement
        try:
            score = metric_fn(y_true[idx], y_pred_or_proba[idx])
            scores.append(score)
        except ValueError:
            # some resamples can have misisng classes. This block skips them without raising error.
            continue

    # computed on the FULL data, not the mean of the bootstrap scores —
    # the CI brackets this point estimate, it isn't derived from it
    point_estimate = metric_fn(y_true, y_pred_or_proba)

    lower = np.percentile(scores, (1 - confidence_level) / 2 * 100)
    upper = np.percentile(scores, (1 + confidence_level) / 2 * 100)

    return point_estimate, (lower, upper)


def weighted_f1_with_ci(y_true, y_pred, n_iterations=1000, seed=42):
    # This is a wrapper function, calculates weighted F1 as the metric being bootstrapped
    metric_fn = lambda yt, yp: f1_score(yt, yp, average='weighted')
    return bootstrap_metric_ci(y_true, y_pred, metric_fn, n_iterations=n_iterations, seed=seed)


def macro_auroc_with_ci(y_true, y_proba, n_iterations=1000, seed=42):
    # This is a thin wrapper, calculates macro one vs rest (ovr) AUROC as the metric being bootstrapped

    metric_fn = lambda y_true, y_score: roc_auc_score(y_true, y_score, average='macro', multi_class='ovr')
    return bootstrap_metric_ci(y_true, y_proba, metric_fn, n_iterations=n_iterations, seed=seed)