"""Fast DeLong test for pairwise ROC-AUC comparison.
Reference: Sun & Xu (2014) IEEE Signal Processing Letters, 21(11).
"""
import numpy as np
from scipy import stats


def _midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N)
    T2[J] = T + 1
    return T2


def _fast_delong(preds_sorted, m):
    """preds_sorted: (k, n) array; first m columns positives, rest negatives."""
    k, N = preds_sorted.shape
    n = N - m
    tx = np.empty((k, m)); ty = np.empty((k, n)); tz = np.empty((k, N))
    for r in range(k):
        tx[r] = _midrank(preds_sorted[r, :m])
        ty[r] = _midrank(preds_sorted[r, m:])
        tz[r] = _midrank(preds_sorted[r])
    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    if k > 1:
        sx = np.cov(v01)
        sy = np.cov(v10)
    else:
        sx = np.array([[np.var(v01, ddof=1)]])
        sy = np.array([[np.var(v10, ddof=1)]])
    cov = sx / m + sy / n
    return aucs, cov


def delong_pairwise(y_true, scores_dict):
    """
    y_true: (n,) binary
    scores_dict: {name: (n,) probability array}
    Returns (pairwise_dict, auc_by_model_dict).
    """
    y_true = np.asarray(y_true).astype(int)
    order = np.argsort(-y_true, kind='stable')  # positives first
    m = int(y_true.sum())
    names = list(scores_dict.keys())
    P = np.array([np.asarray(scores_dict[n])[order] for n in names])
    aucs, cov = _fast_delong(P, m)

    results = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            var = cov[i, i] + cov[j, j] - 2 * cov[i, j]
            se = np.sqrt(max(var, 0.0))
            if se > 0:
                z = (aucs[i] - aucs[j]) / se
                p = 2 * (1 - stats.norm.cdf(abs(z)))
            else:
                z, p = 0.0, 1.0
            results[f"{names[i]} vs {names[j]}"] = {
                'auc_a': float(aucs[i]),
                'auc_b': float(aucs[j]),
                'delta': float(aucs[i] - aucs[j]),
                'se':    float(se),
                'z':     float(z),
                'p_value': float(p),
                'significant_at_0.05': bool(p < 0.05),
            }
    return results, {names[i]: float(aucs[i]) for i in range(len(names))}
