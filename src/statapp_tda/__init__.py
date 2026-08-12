"""Topological classification of immune cell spatial patterns.

ENSAE Paris x Max Planck Institute of Biochemistry, Applied Statistics
Project 2025-2026.

The package is organised around the stages of the pipeline:

- :mod:`config`     project constants and paths
- :mod:`data`       loading the Vipond et al. (2021) datasets
- :mod:`synthetic`  point clouds with known topology, used for validation
- :mod:`topology`   Alpha-complex persistent homology and landscape vectorisation
- :mod:`features`   the coordinate-based geometric baseline
- :mod:`evaluation` tumor-level cross-validated classification
- :mod:`statistics` paired two-sample tests on tumor-level landscapes
- :mod:`plotting`   every figure
"""

from . import (
    config,
    data,
    evaluation,
    features,
    plotting,
    statistics,
    synthetic,
    topology,
)

__all__ = [
    "config", "data", "evaluation", "features",
    "plotting", "statistics", "synthetic", "topology",
]
