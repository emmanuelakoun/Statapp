import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from scipy.stats import ks_2samp

# ==========================================
# 1. MMD Grouped Permutation Test (ROI-level)
# ==========================================
def mmd2_rbf(X, Y, gamma=None):
    """Compute squared MMD using RBF kernel."""
    XX = rbf_kernel(X, X, gamma=gamma)
    YY = rbf_kernel(Y, Y, gamma=gamma)
    XY = rbf_kernel(X, Y, gamma=gamma)
    return XX.mean() + YY.mean() - 2 * XY.mean()

def grouped_mmd_permutation_test(X1, groups1, X2, groups2, n_permutations=1000, seed=42):
    """
    MMD permutation test for two samples (X1, X2) with grouped observations (tumors).
    - X1, X2: Landscape features for each ROI (e.g., shape (n_rois, 800))
    - groups1, groups2: Array of tumor IDs for each ROI
    """
    rng = np.random.default_rng(seed)
    obs_mmd = mmd2_rbf(X1, X2)
    
    # Pool all data
    X_pooled = np.vstack([X1, X2])
    groups_pooled = np.concatenate([groups1, groups2])
    unique_groups = np.unique(groups_pooled)
    
    # Number of tumors in group 1
    n_groups1 = len(np.unique(groups1))
    
    count = 0
    for _ in range(n_permutations):
        # Permute the TUMOR labels (Grouped Permutation)
        permuted_groups = rng.permutation(unique_groups)
        groups_in_1 = permuted_groups[:n_groups1]
        
        mask1 = np.isin(groups_pooled, groups_in_1)
        mask2 = ~mask1
        
        X1_perm = X_pooled[mask1]
        X2_perm = X_pooled[mask2]
        
        if len(X1_perm) == 0 or len(X2_perm) == 0:
            continue
            
        perm_mmd = mmd2_rbf(X1_perm, X2_perm)
        if perm_mmd >= obs_mmd:
            count += 1
            
    return obs_mmd, count / n_permutations


# ==========================================
# 2. Kolmogorov-Smirnov Test (Tumor-level)
# ==========================================
def ks_test_tumor_level(X1_tumor_means, X2_tumor_means):
    """
    KS test on the L2 norm of the tumor-level mean landscapes.
    - X1_tumor_means: Mean landscape vectors per tumor (e.g., shape (16, 800))
    """
    norm1 = np.linalg.norm(X1_tumor_means, axis=1)
    norm2 = np.linalg.norm(X2_tumor_means, axis=1)
    
    # Perform Kolmogorov-Smirnov test on the 1D distributions
    stat, p_value = ks_2samp(norm1, norm2)
    return stat, p_value

# ==========================================
# Exemple d'exécution à copier dans main.ipynb
# ==========================================
'''
# Assure-toi d'avoir X_features_h1 (matrice numpy) et y_all, groups_all
mask_cd8 = (y_all == 'CD8')
mask_foxp3 = (y_all == 'FoxP3')

X_cd8 = X_features_h1[mask_cd8]
groups_cd8 = groups_all[mask_cd8]

X_foxp3 = X_features_h1[mask_foxp3]
groups_foxp3 = groups_all[mask_foxp3]

# Test MMD sur les distributions complètes (attention, 1000 permutations peut prendre 1-2 minutes)
obs_mmd, p_mmd = grouped_mmd_permutation_test(X_cd8, groups_cd8, X_foxp3, groups_foxp3, n_permutations=1000)
print(f"MMD Grouped Permutation p-value (CD8 vs FoxP3): {p_mmd}")

# Pour le test KS, tu peux utiliser les listes de moyennes par tumeur générées pour le test existant
# stat_ks, p_ks = ks_test_tumor_level(tumor_means_cd8, tumor_means_foxp3)
# print(f"KS test p-value: {p_ks}")
'''
