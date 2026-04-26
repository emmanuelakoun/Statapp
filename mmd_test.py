import numpy as np
import pickle
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel

def mmd2_linear(X, Y):
    XX = np.dot(X, X.T)
    YY = np.dot(Y, Y.T)
    XY = np.dot(X, Y.T)
    return XX.mean() + YY.mean() - 2 * XY.mean()

# Dummy grouped permutation test to test the logic
def group_permutation_mmd(X1, X2, groups1, groups2, n_permutations=1000):
    pass
