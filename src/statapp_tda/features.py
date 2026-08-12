"""Coordinate-based geometric baseline features.

These 15 statistics summarise the relative geometry of a point cloud without
any topological machinery. They are the non-TDA baseline of the ablation
study: spread and anisotropy (4), nearest-neighbour distances (5), pairwise
distances (5) and convex hull area (1).
"""

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist
from sklearn.neighbors import NearestNeighbors

COORD_FEATURE_NAMES = [
    "std_x", "std_y", "range_x", "range_y",
    "nn_mean", "nn_std", "nn_min", "nn_max", "nn_median",
    "pw_mean", "pw_std", "pw_median", "pw_q25", "pw_q75",
    "convex_hull_area",
]


def normalize_point_cloud(pts):
    """Center a point cloud and scale it to unit RMS radius."""
    pts = np.asarray(pts, dtype=float)
    if len(pts) == 0:
        return pts.copy()
    centered = pts - pts.mean(axis=0)
    scale = max(np.sqrt(np.mean(np.sum(centered ** 2, axis=1))), 1e-12)
    return centered / scale


def extract_coordinate_features(pts):
    """15 spatial statistics from a normalized 2D point cloud."""
    if len(pts) < 4:
        return np.zeros(15)

    x, y = pts[:, 0], pts[:, 1]
    nn_dists = NearestNeighbors(n_neighbors=2).fit(pts).kneighbors(pts)[0][:, 1]
    pw = pdist(pts)

    try:
        area = ConvexHull(pts).volume
    except Exception:
        area = 0.0

    return np.array([
        np.std(x), np.std(y), np.ptp(x), np.ptp(y),
        np.mean(nn_dists), np.std(nn_dists), np.min(nn_dists),
        np.max(nn_dists), np.median(nn_dists),
        np.mean(pw), np.std(pw), np.median(pw),
        np.percentile(pw, 25), np.percentile(pw, 75),
        area,
    ])


def coordinate_feature_matrix(roi_records):
    """Stack the 15 coordinate features of every ROI into an (n_rois, 15) matrix."""
    return np.vstack([
        extract_coordinate_features(normalize_point_cloud(r["points"]))
        for r in roi_records
    ])
