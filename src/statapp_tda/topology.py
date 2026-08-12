"""Persistent homology on 2D cell point clouds.

All diagrams are computed from an Alpha complex filtration (gudhi). Alpha
complex filtration values are *squared* radii, so a square root is applied to
recover birth/death coordinates on the Euclidean scale.
"""

import numpy as np
import gudhi
from gudhi.representations import Landscape

from .config import CELLTYPES, LANDSCAPE_STEPS, NUM_LANDSCAPES


def alpha_persistence_diagrams(points):
    """Compute the H0 and H1 persistence diagrams of a 2D point cloud.

    Returns
    -------
    [dgm_h0, dgm_h1] : list of (n, 2) arrays of birth/death pairs, on the
        Euclidean scale (square root of the Alpha filtration values).
    """
    alpha = gudhi.AlphaComplex(points=points)
    simplex_tree = alpha.create_simplex_tree()
    simplex_tree.compute_persistence()
    dgm_h0 = np.sqrt(simplex_tree.persistence_intervals_in_dimension(0))
    dgm_h1 = np.sqrt(simplex_tree.persistence_intervals_in_dimension(1))
    return [dgm_h0, dgm_h1]


def compute_diagrams_by_type(regions, celltypes=CELLTYPES, verbose=True):
    """Compute persistence diagrams for every ROI, grouped by cell type."""
    diagrams_by_type = {}
    for ct in celltypes:
        diagrams_by_type[ct] = []
        for i, region in enumerate(regions[ct]):
            diagrams_by_type[ct].append(alpha_persistence_diagrams(region["points"]))
            if verbose and (i + 1) % 100 == 0:
                print(f"  {ct}: {i + 1}/{len(regions[ct])} done")
        if verbose:
            print(f"{ct}: {len(diagrams_by_type[ct])} diagrams computed")
    return diagrams_by_type


def finite_h0(dgm):
    """Remove the single infinite H0 point before vectorisation."""
    return dgm[0][np.isfinite(dgm[0][:, 1])]


def build_roi_records(regions_by_type, diagrams_lookup, celltypes=CELLTYPES):
    """Join point clouds, metadata and diagrams into one flat list of records.

    Each record has keys ``celltype``, ``tumor_id``, ``filename``, ``points``,
    ``diagram_h0`` (finite part only) and ``diagram_h1``.
    """
    records = []
    for ct in celltypes:
        for region, dgm in zip(regions_by_type[ct], diagrams_lookup[ct]):
            records.append({
                "celltype": ct,
                "tumor_id": region["tumor_id"],
                "filename": region["filename"],
                "points": region["points"],
                "diagram_h0": np.array(finite_h0(dgm)),
                "diagram_h1": np.array(dgm[1]),
            })
    return records


def records_to_arrays(roi_records):
    """Split ROI records into the arrays consumed by the evaluation routines.

    Returns
    -------
    X_h0, X_h1 : object arrays of persistence diagrams
    y : array of cell-type labels
    groups : array of tumor ids (the cross-validation grouping variable)
    """
    X_h0 = np.array([r["diagram_h0"] for r in roi_records], dtype=object)
    X_h1 = np.array([r["diagram_h1"] for r in roi_records], dtype=object)
    y = np.array([r["celltype"] for r in roi_records])
    groups = np.array([r["tumor_id"] for r in roi_records])
    return X_h0, X_h1, y, groups


def make_landscape(num_landscapes=NUM_LANDSCAPES, resolution=LANDSCAPE_STEPS):
    """Build a persistence-landscape vectoriser with the project settings."""
    return Landscape(num_landscapes=num_landscapes, resolution=resolution)
