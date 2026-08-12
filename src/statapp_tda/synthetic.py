"""Synthetic point clouds with known topology, used to validate the pipeline.

If persistent homology cannot recover the ground-truth H0/H1 features of a
circle, a torus or a Klein bottle, there is no reason to trust it on cell
coordinates. Each generator has a clean and a noisy variant so that the
stability of the diagrams under perturbation can be inspected.
"""

import numpy as np
import tadasets

from .config import RANDOM_SEED


# ─────────────────────────────────────────────────────────────────────────────
# Circles (1-spheres) in R^2
# ─────────────────────────────────────────────────────────────────────────────

def spheres_clean_and_noisy(n=1000, noise=0.2, seed=4242):
    """A clean circle and a noisy circle, both sampled in R^2.

    Ground truth: exactly one prominent H0 feature (one connected component)
    and one prominent H1 feature (the circular void).
    """
    np.random.seed(seed)
    data_clean = tadasets.dsphere(d=1, n=n, noise=0.0)
    data_noisy = tadasets.dsphere(d=1, n=n, noise=noise)
    return data_clean, data_noisy


# ─────────────────────────────────────────────────────────────────────────────
# Torus in R^3
# ─────────────────────────────────────────────────────────────────────────────

def torus_point_cloud(n=1000, noise=0.0, seed=4052003):
    """Sample a torus in R^3. Ground truth: one H0, two H1 loops, one H2 void."""
    np.random.seed(seed)
    return tadasets.torus(n=n, noise=noise)


# ─────────────────────────────────────────────────────────────────────────────
# Klein bottle in R^4
# ─────────────────────────────────────────────────────────────────────────────

def klein_bottle_r4(n=200, R=2.0, r=1.0, rng=None):
    """Sample a Klein bottle immersed in R^4.

    Parameters
    ----------
    R, r : float
        Large and small radii of the parametrisation.
    """
    rng = np.random.default_rng(RANDOM_SEED) if rng is None else rng

    u = rng.uniform(low=0, high=2 * np.pi, size=n)
    v = rng.uniform(low=0, high=2 * np.pi, size=n)

    x = (R + r * np.cos(u)) * np.cos(v)
    y = (R + r * np.cos(u)) * np.sin(v)
    z = r * np.sin(u) * np.cos(v / 2)
    w = r * np.sin(u) * np.sin(v / 2)

    return np.column_stack([x, y, z, w])


def klein_bottle_r4_noisy(n=200, sigma=1.0, rng=None):
    """Klein bottle in R^4 with additive isotropic Gaussian noise."""
    rng = np.random.default_rng(RANDOM_SEED) if rng is None else rng

    points = klein_bottle_r4(n=n, rng=rng)
    mean = np.zeros(4)
    cov = np.eye(4) * (sigma ** 2)
    noise = rng.multivariate_normal(mean, cov, size=n)
    return points + noise


def project_r4_to_r3(X):
    """Smooth deterministic projection from R^4 to R^3 for visualisation.

    Combines the z and w coordinates to preserve structure.
    """
    x = X[:, 0]
    y = X[:, 1]
    z = X[:, 2] + X[:, 3]
    return np.column_stack([x, y, z])


# ─────────────────────────────────────────────────────────────────────────────
# Pure noise control
# ─────────────────────────────────────────────────────────────────────────────

def uniform_bounding_box_noise(reference, n_points=1000, rng=None):
    """Uniform points in the bounding box of ``reference``.

    This is the "no structure" control: any persistent feature found here is an
    artefact of sampling, not of topology.
    """
    rng = np.random.default_rng(RANDOM_SEED) if rng is None else rng
    reference = np.asarray(reference)
    low = reference.min(axis=0)
    high = reference.max(axis=0)
    return rng.uniform(low=low, high=high, size=(n_points, reference.shape[1]))
