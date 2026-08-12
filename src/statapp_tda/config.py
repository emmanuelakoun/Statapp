"""Project-wide constants and filesystem paths.

Every value here is shared by the notebooks and by the analysis modules, so
that the pipeline is reproducible from any working directory.
"""

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
# src/statapp_tda/config.py -> src/statapp_tda -> src -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
LHR_DIR = DATA_DIR / "large_hypoxic_regions"
ROI_DIR = DATA_DIR / "roi_1_5mm"
FIGURES_DIR = PROJECT_ROOT / "figures"

# ── Reproducibility ──────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ── Data ─────────────────────────────────────────────────────────────────────
CELLTYPES = ["CD8", "FoxP3", "CD68"]

# ROIs with fewer cells than this are discarded: such sparse point clouds do
# not produce meaningful topological features.
MIN_CELLS_PER_ROI = 20

# The exploratory analysis of the large hypoxic regions is run on the first
# MAX_LHR_POINTS cells of each (cell type, region) pair, for tractability.
MAX_LHR_POINTS = 1500

# ── Vectorisation ────────────────────────────────────────────────────────────
NUM_LANDSCAPES = 10   # number of landscape depths kept
LANDSCAPE_STEPS = 80  # resolution of the landscape discretisation grid

# ── Cross-validation ─────────────────────────────────────────────────────────
N_SPLITS = 10  # folds of the tumor-level StratifiedGroupKFold
