"""Figures for the exploratory analysis and the results.

Everything that draws is collected here so the notebooks stay readable: they
call one function per figure and keep only the narrative.
"""

import matplotlib.pyplot as plt
import numpy as np
import persim
import seaborn as sns
from persim import PersistenceImager, PersLandscapeApprox
from persim.landscapes import plot_landscape_simple
from sklearn.metrics import ConfusionMatrixDisplay

from .config import CELLTYPES, LANDSCAPE_STEPS

# Display order used for the large hypoxic regions (differs from the loading
# order CELLTYPES, which drives the ROI pipeline).
LHR_DISPLAY_ORDER = ["CD8", "CD68", "FoxP3"]

CELLTYPE_COLORS = {"CD8": "#3b528b", "CD68": "#21918c", "FoxP3": "#5ec962"}


# ─────────────────────────────────────────────────────────────────────────────
# Exploratory analysis of the large hypoxic regions
# ─────────────────────────────────────────────────────────────────────────────

def plot_celltype_distribution(lhr1, lhr2):
    """Cell-type counts in each large hypoxic region."""
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Distribution of cell types in the tumor region',
                 fontsize=16, fontweight='bold')

    for ax, df, label in [(axes[0], lhr1, 'LHR1'), (axes[1], lhr2, 'LHR2')]:
        sns.countplot(data=df, x='Celltype', hue='Celltype', palette='viridis',
                      legend=False, ax=ax)
        ax.set_title(label, fontsize=14)
        ax.set_xlabel('Cell type', fontsize=12)
        ax.set_ylabel('Total number of cells', fontsize=12)

    plt.tight_layout()
    return fig


def plot_hypoxia_by_celltype(lhr1, lhr2):
    """Hypoxic status (CAIX+) broken down by cell type."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('State of hypoxia (CAIX+) according to cell type',
                 fontsize=16, fontweight='bold')

    for ax, df, label in [(axes[0], lhr1, 'LHR1'), (axes[1], lhr2, 'LHR2')]:
        sns.countplot(data=df, x='Celltype', hue='CAIX+', palette='Set2', ax=ax)
        ax.set_title(label, fontsize=14)
        ax.set_xlabel('Cell type', fontsize=12)
        ax.set_ylabel('Total number of cells', fontsize=12)
        ax.legend(title='Hypoxia marker (CAIX+)',
                  labels=['Normoxia (0)', 'Hypoxia (1)'])

    plt.tight_layout()
    return fig


def add_tissue_structure(df):
    """Label each cell as necrotic tissue, tumor cell (PanCK+) or stroma/immune."""
    df = df.copy()
    conditions = [
        (df['Necrosis'] == 1),
        (df['PanCK+'] == 1),
    ]
    choices = ['Necrotic Tissue', 'Tumor Cells (PanCK+)']
    df['Tissue_Structure'] = np.select(conditions, choices, default='Stroma / Immune')
    return df


def plot_spatial_architecture(lhr1, lhr2):
    """Three spatial maps per region: immune infiltration, hypoxia, architecture."""
    df1 = add_tissue_structure(lhr1)
    df2 = add_tissue_structure(lhr2)

    fig, axes = plt.subplots(2, 3, figsize=(24, 18))
    fig.suptitle('Spatial Distribution of Cells within the Tumor Microenvironment',
                 fontsize=22, fontweight='bold')

    for row, (df, label) in enumerate([(df1, 'LHR1'), (df2, 'LHR2')]):
        sns.scatterplot(
            data=df, x='x', y='y', hue='Celltype',
            palette=CELLTYPE_COLORS,
            s=15, alpha=0.8, edgecolor=None, ax=axes[row, 0]
        )
        axes[row, 0].set_title(f'1. Immune Infiltration (Cell Types) — {label}', fontsize=16)
        axes[row, 0].set_xlabel('X coordinate (µm)', fontsize=12)
        axes[row, 0].set_ylabel('Y coordinate (µm)', fontsize=12)
        axes[row, 0].legend(title='Cell Type', markerscale=2)

        df_sorted_hypoxia = df.sort_values(by='CAIX+')
        sns.scatterplot(
            data=df_sorted_hypoxia, x='x', y='y', hue='CAIX+',
            palette={0: '#e0e0e0', 1: '#d62728'},
            s=15, alpha=0.8, edgecolor=None, ax=axes[row, 1]
        )
        axes[row, 1].set_title(f'2. Hypoxic Regions (CAIX+) — {label}', fontsize=16)
        axes[row, 1].set_xlabel('X coordinate (µm)', fontsize=12)
        axes[row, 1].set_ylabel('')
        axes[row, 1].legend(title='Hypoxia Status',
                            labels=['Normoxia (0)', 'Hypoxia (1)'], markerscale=2)

        df_sorted_structure = df.sort_values(by='Tissue_Structure', ascending=False)
        sns.scatterplot(
            data=df_sorted_structure, x='x', y='y', hue='Tissue_Structure',
            palette={'Stroma / Immune': '#e0e0e0',
                     'Tumor Cells (PanCK+)': '#ff7f0e',
                     'Necrotic Tissue': '#000000'},
            s=15, alpha=0.8, edgecolor=None, ax=axes[row, 2]
        )
        axes[row, 2].set_title(f'3. Tumor Architecture & Necrosis — {label}', fontsize=16)
        axes[row, 2].set_xlabel('X coordinate (µm)', fontsize=12)
        axes[row, 2].set_ylabel('')
        axes[row, 2].legend(title='Tissue Component', markerscale=2)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def plot_persistence_diagrams_grid(all_diagrams, celltypes=LHR_DISPLAY_ORDER,
                                   labels=('LHR1', 'LHR2')):
    """Persistence diagrams (H0 and H1) for each (cell type, region) pair."""
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    fig.suptitle('Persistence Diagrams', fontsize=20, fontweight='bold')

    for row, label in enumerate(labels):
        for col, celltype in enumerate(celltypes):
            ax = axes[row, col]
            persim.plot_diagrams(all_diagrams[(celltype, label)], show=False,
                                 title=f"{celltype} — {label}",
                                 labels=['$H_0$', '$H_1$'], ax=ax)
            ax.set_xlabel("Birth", fontsize=12)
            ax.set_ylabel("Death", fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def plot_persistence_images_grid(all_diagrams, celltypes=LHR_DISPLAY_ORDER,
                                 labels=('LHR1', 'LHR2')):
    """H1 persistence images on a shared birth/persistence range."""
    diagram_list = [all_diagrams[(ct, lb)] for lb in labels for ct in celltypes]

    all_births = [np.max(d[1][:, 0]) for d in diagram_list if d[1].size > 0]
    all_pers = [np.max(d[1][:, 1] - d[1][:, 0]) for d in diagram_list if d[1].size > 0]
    global_max_birth = max(all_births)
    global_max_pers = max(all_pers)

    pimgr = PersistenceImager(
        pixel_size=global_max_birth / 50,
        birth_range=(0, global_max_birth * 1.1),
        pers_range=(0, global_max_pers * 1.1),
    )

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Persistence Images', fontsize=20, fontweight='bold')

    for row, label in enumerate(labels):
        for col, celltype in enumerate(celltypes):
            ax = axes[row, col]
            img = pimgr.transform(all_diagrams[(celltype, label)][1])
            pimgr.plot_image(img, ax=ax)
            ax.set_title(f"{celltype} — {label}", fontsize=14)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def plot_landscapes_grid(all_diagrams, celltypes=LHR_DISPLAY_ORDER,
                         labels=('LHR1', 'LHR2'), depth=10):
    """H1 persistence landscapes for each (cell type, region) pair."""
    fig, axes = plt.subplots(2, 3, figsize=(24, 14))
    fig.suptitle('Persistence Landscapes', fontsize=20, fontweight='bold')

    for row, label in enumerate(labels):
        for col, celltype in enumerate(celltypes):
            ax = axes[row, col]
            pla = PersLandscapeApprox(dgms=all_diagrams[(celltype, label)], hom_deg=1)
            plot_landscape_simple(pla, depth_range=range(depth), ax=ax)
            ax.set_title(f"{celltype} — {label}", fontsize=14)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Results
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrices(matrices, celltypes=CELLTYPES):
    """Row-normalised out-of-fold confusion matrices, one panel per classifier."""
    fig, axes = plt.subplots(1, len(matrices), figsize=(6 * len(matrices), 5))
    fig.suptitle('Confusion matrices — 3-way classification on $H_0 + H_1$ landscapes',
                 fontsize=15, fontweight='bold')

    for ax, (name, cm) in zip(np.atleast_1d(axes), matrices.items()):
        disp = ConfusionMatrixDisplay(cm, display_labels=list(celltypes))
        disp.plot(cmap='Blues', ax=ax, colorbar=False, values_format='.2f')
        ax.set_title(name, fontsize=13)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def plot_mean_landscapes(tumor_means, celltypes=CELLTYPES,
                         landscape_steps=LANDSCAPE_STEPS, n_cols=5):
    """Mean H1 landscape per cell type, with a +/- 1 std band across tumors."""
    n_features = next(iter(tumor_means.values())).shape[1]
    n_landscapes = n_features // landscape_steps
    n_rows = int(np.ceil(n_landscapes / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.5 * n_cols, 3.5 * n_rows),
        squeeze=False,
        sharex=True,
    )

    fig.suptitle(
        "Mean $H_1$ persistence landscapes by cell type (± 1 std across tumors)",
        fontsize=16,
        fontweight="bold",
    )

    xs = np.linspace(0, 1, landscape_steps)

    for k in range(n_rows * n_cols):
        ax = axes[k // n_cols, k % n_cols]

        if k >= n_landscapes:
            ax.set_visible(False)
            continue

        sl = slice(k * landscape_steps, (k + 1) * landscape_steps)

        for ct in celltypes:
            X_ct = tumor_means[ct]
            mean_ls = X_ct[:, sl].mean(axis=0)
            std_ls = X_ct[:, sl].std(axis=0)

            ax.plot(xs, mean_ls, label=ct)
            ax.fill_between(xs, mean_ls - std_ls, mean_ls + std_ls, alpha=0.2)

        ax.set_title(f"Landscape depth {k + 1}")

        if k // n_cols == n_rows - 1:
            ax.set_xlabel("Filtration parameter (normalized)")

        if k % n_cols == 0:
            ax.set_ylabel("Landscape value")

        if k == 0:
            ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig
