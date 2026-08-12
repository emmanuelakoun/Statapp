"""Loading of the two datasets from Vipond et al. (2021)."""

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CELLTYPES, LHR_DIR, MIN_CELLS_PER_ROI, RANDOM_SEED, ROI_DIR


def load_large_hypoxic_regions(directory=LHR_DIR):
    """Load the two large hypoxic regions (LHR1, LHR2).

    Each row is a cell described by its coordinates (x, y), its immune cell
    type and binary markers for hypoxia (CAIX+), tumor cells (PanCK+) and
    necrotic tissue.

    Returns
    -------
    (lhr1, lhr2) : tuple of pandas.DataFrame
    """
    directory = Path(directory)
    lhr1 = pd.read_csv(directory / "large_hypoxic_region_1.csv")
    lhr2 = pd.read_csv(directory / "large_hypoxic_region_2.csv")
    return lhr1, lhr2


def extract_tumor_id(filepath):
    """Extract the tumor identifier from a ROI filename.

    ``T_I_ROI_12_locations_CD8.csv`` -> ``T_I``
    """
    return Path(filepath).name.split("_ROI_")[0]


def load_roi_dataset(base_path=ROI_DIR, celltypes=CELLTYPES,
                     min_cells=MIN_CELLS_PER_ROI):
    """Load the 1.5 mm x 1.5 mm regions of interest, one point cloud per file.

    ROIs with fewer than ``min_cells`` cells are discarded.

    Returns
    -------
    regions : dict[str, list[dict]]
        Maps a cell type to a list of records with keys ``points`` (an
        (n, 2) array of coordinates), ``tumor_id`` and ``filename``.
    """
    base_path = str(base_path)
    regions = {}
    for ct in celltypes:
        folder = os.path.join(base_path, ct)
        files = sorted(glob.glob(os.path.join(folder, "*.csv")))
        regions[ct] = []
        for f in files:
            df = pd.read_csv(f)
            if len(df) >= min_cells:
                regions[ct].append({
                    "points": df[["x", "y"]].values,
                    "tumor_id": extract_tumor_id(f),
                    "filename": Path(f).name,
                })
    return regions


def summarise_regions(regions):
    """Return a one-line summary per cell type: number of ROIs and of tumors."""
    lines = []
    for ct, records in regions.items():
        tumor_ids = sorted({r["tumor_id"] for r in records})
        lines.append(f"{ct}: {len(records)} regions loaded across {len(tumor_ids)} tumors")
    return "\n".join(lines)


def subsample_points(pts, max_points, seed_offset=0, seed=RANDOM_SEED):
    """Subsample a point cloud reproducibly when it exceeds ``max_points``."""
    if len(pts) <= max_points:
        return pts.copy()
    rng_local = np.random.default_rng(seed + seed_offset)
    idx = rng_local.choice(len(pts), max_points, replace=False)
    return pts[idx]
