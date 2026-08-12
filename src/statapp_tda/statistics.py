"""Two-sample tests for topological differences between cell types.

Both tests operate on *tumor-level mean landscape vectors*, not on individual
ROIs. The reason is pseudoreplication: ROIs from the same tumor are not
independent, so the independent unit is the tumor (N = 16 here).

Because every tumor contributes ROIs of all three cell types, the two samples
being compared are **paired by tumor**. The exchangeable null is therefore the
within-tumor label swap, not a free permutation of tumor labels.
"""

import numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics.pairwise import rbf_kernel

from .config import RANDOM_SEED


# ─────────────────────────────────────────────────────────────────────────────
# Tumor-level aggregation
# ─────────────────────────────────────────────────────────────────────────────

def tumor_mean_landscapes(X, y, tumors, celltype):
    """Return one mean landscape vector per tumor for a given cell type.

    Returns
    -------
    X_tumor : array of shape (n_tumors, n_features)
        Mean landscape vector for each tumor.
    tumor_ids : array
        Tumor IDs in the same order as rows of X_tumor.
    """
    mask = y == celltype
    X_ct = X[mask]
    t_ct = tumors[mask]

    tumor_ids = np.unique(t_ct)
    X_tumor = np.vstack([
        X_ct[t_ct == tid].mean(axis=0)
        for tid in tumor_ids
    ])

    return X_tumor, tumor_ids


def paired_tumor_mean_landscapes(X, y, tumors, ct1, ct2):
    """Build paired tumor-level mean landscapes for two cell types.

    The output matrices X1 and X2 are aligned by tumor: row i of X1 and row i
    of X2 correspond to the same tumor.
    """
    X1_all, tumors1 = tumor_mean_landscapes(X, y, tumors, ct1)
    X2_all, tumors2 = tumor_mean_landscapes(X, y, tumors, ct2)

    map1 = {tid: X1_all[i] for i, tid in enumerate(tumors1)}
    map2 = {tid: X2_all[i] for i, tid in enumerate(tumors2)}

    common_tumors = np.array(sorted(set(map1.keys()) & set(map2.keys())))

    X1 = np.vstack([map1[tid] for tid in common_tumors])
    X2 = np.vstack([map2[tid] for tid in common_tumors])

    return X1, X2, common_tumors


# ─────────────────────────────────────────────────────────────────────────────
# Test 1 — equality of MEAN topological signatures
# ─────────────────────────────────────────────────────────────────────────────

def paired_mean_landscape_permutation_test(X1, X2, n_permutations=10000, seed=RANDOM_SEED):
    """Paired within-tumor sign-flip permutation test on mean landscape vectors.

    H0: the two cell types have the same tumor-level mean landscape
        distribution, so their labels are exchangeable within each tumor.

    Test statistic: L2 distance between the two group mean landscape vectors.

    X1, X2 must have shape (n_tumors, n_features) and must be aligned by tumor.
    """
    rng = np.random.default_rng(seed)

    X1 = np.asarray(X1)
    X2 = np.asarray(X2)

    if X1.shape != X2.shape:
        raise ValueError("X1 and X2 must have the same shape and be aligned by tumor.")

    observed = np.linalg.norm(X1.mean(axis=0) - X2.mean(axis=0))

    # Paired differences. Swapping labels within a tumor is equivalent to
    # multiplying the tumor-level difference by +1 or -1.
    D = X1 - X2
    n_tumors = D.shape[0]

    # Vectorised sign-flip permutations
    signs = rng.choice([-1, 1], size=(n_permutations, n_tumors))
    permuted_mean_diffs = (signs[:, :, None] * D[None, :, :]).mean(axis=1)
    permuted_stats = np.linalg.norm(permuted_mean_diffs, axis=1)

    # +1 smoothing avoids zero p-values
    p_value = (1 + np.sum(permuted_stats >= observed)) / (n_permutations + 1)

    return observed, p_value, permuted_stats


# ─────────────────────────────────────────────────────────────────────────────
# Test 2 — equality of the full DISTRIBUTION of topological signatures
# ─────────────────────────────────────────────────────────────────────────────

def mmd2_rbf(X, Y, gamma=None):
    """Squared Maximum Mean Discrepancy with an RBF kernel."""
    XX = rbf_kernel(X, X, gamma=gamma)
    YY = rbf_kernel(Y, Y, gamma=gamma)
    XY = rbf_kernel(X, Y, gamma=gamma)
    return XX.mean() + YY.mean() - 2 * XY.mean()


def paired_within_tumor_mmd_test(X_all, y_all, t_all, ct_a, ct_b,
                                 n_permutations=10000, seed=RANDOM_SEED):
    """Paired MMD test on tumor-level mean landscapes for two cell types.

    For each tumor that has at least one ROI of each type, we compute the mean
    landscape of type A and the mean landscape of type B. The test statistic is
    MMD^2 (RBF kernel, default gamma) between the two paired sets. The null
    distribution is built by independently flipping the A/B label of each tumor
    with probability 1/2.

    Note on a bug this replaces: an earlier version permuted tumor LABELS
    across cell types, but every tumor contributes ROIs to *all* cell types, so
    every permutation assigned all tumors to group 1 and left group 2 empty,
    yielding an artificial p-value of exactly 0.0000. The correct exchangeable
    null for paired tumor-level data is the within-tumor swap implemented here.
    """
    rng = np.random.default_rng(seed)

    # Tumors present in BOTH cell types
    t_a = np.unique(t_all[y_all == ct_a])
    t_b = np.unique(t_all[y_all == ct_b])
    common_tumors = np.intersect1d(t_a, t_b)

    A = np.vstack([
        X_all[(y_all == ct_a) & (t_all == tid)].mean(axis=0)
        for tid in common_tumors
    ])
    B = np.vstack([
        X_all[(y_all == ct_b) & (t_all == tid)].mean(axis=0)
        for tid in common_tumors
    ])

    obs = mmd2_rbf(A, B)
    n_tumors = len(common_tumors)

    count = 0
    for _ in range(n_permutations):
        flips = rng.integers(0, 2, size=n_tumors).astype(bool)
        A_perm = np.where(flips[:, None], B, A)
        B_perm = np.where(flips[:, None], A, B)
        if mmd2_rbf(A_perm, B_perm) >= obs:
            count += 1

    # +1 smoothing so the p-value is never exactly 0
    return obs, (count + 1) / (n_permutations + 1), n_tumors


def ks_test_tumor_level(X1_tumor_means, X2_tumor_means):
    """Two-sample KS test on the L2 norm of tumor-level mean landscapes."""
    norm1 = np.linalg.norm(X1_tumor_means, axis=1)
    norm2 = np.linalg.norm(X2_tumor_means, axis=1)
    return ks_2samp(norm1, norm2)


def bonferroni_alpha(n_comparisons, alpha=0.05):
    """Bonferroni-corrected significance level."""
    return alpha / n_comparisons
