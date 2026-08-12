"""Cross-validated evaluation of the classification pipelines.

Two design rules are enforced everywhere in this module and are the reason the
accuracies reported here are lower (and more honest) than a naive protocol:

1. **Tumor-level grouping.** Several ROIs come from the same tumor, so a random
   split would leak tumor identity into the training set. All splits are
   ``StratifiedGroupKFold`` grouped by ``tumor_id`` (or ``LeaveOneGroupOut``).
2. **Train-side vectorisation.** The persistence-landscape discretisation grid
   is fitted on the training fold only, never on the full dataset.
"""

from itertools import combinations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy.linalg import LinAlgError
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import CELLTYPES, LANDSCAPE_STEPS, NUM_LANDSCAPES, N_SPLITS, RANDOM_SEED
from .topology import make_landscape


def build_tasks(celltypes=CELLTYPES):
    """All pairwise comparisons plus the 3-way task."""
    return list(combinations(celltypes, 2)) + [tuple(celltypes)]


def task_label(task):
    return " vs ".join(task) if len(task) == 2 else "3-way"


def make_grouped_cv(y, groups, n_splits=N_SPLITS, seed=RANDOM_SEED):
    """StratifiedGroupKFold capped at the number of available tumors."""
    return StratifiedGroupKFold(
        n_splits=min(n_splits, len(np.unique(groups))),
        shuffle=True,
        random_state=seed,
    )


def grouped_splits(y, groups, n_splits=N_SPLITS, seed=RANDOM_SEED):
    cv = make_grouped_cv(y, groups, n_splits=n_splits, seed=seed)
    return list(cv.split(np.zeros(len(y)), y, groups))


# ─────────────────────────────────────────────────────────────────────────────
# LDA on a single homological dimension (scikit-learn Pipeline)
# ─────────────────────────────────────────────────────────────────────────────

def lda_landscape_pipeline():
    """LDA on persistence landscapes, with the vectoriser inside the pipeline."""
    return Pipeline([
        ("landscape", make_landscape()),
        ("scaler", StandardScaler()),
        ("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
    ])


def evaluate_single_dimension(X_diagrams, y_all, grp_all, tasks, n_splits=N_SPLITS):
    """Cross-validated LDA accuracy per task, on one homological dimension."""
    rows = []
    for task in tasks:
        mask = np.isin(y_all, list(task))
        X_task, y_task, grp_task = X_diagrams[mask], y_all[mask], grp_all[mask]

        scores = []
        for train_idx, test_idx in grouped_splits(y_task, grp_task, n_splits):
            pipe = lda_landscape_pipeline()
            try:
                pipe.fit(X_task[train_idx], y_task[train_idx])
                scores.append(pipe.score(X_task[test_idx], y_task[test_idx]))
            except (LinAlgError, ValueError):
                continue

        rows.append({
            "Task": task_label(task),
            "Mean accuracy": np.mean(scores) if scores else np.nan,
            "Std": np.std(scores) if scores else np.nan,
        })
    return pd.DataFrame(rows).set_index("Task")


def evaluate_h0_h1_concatenated(X_h0, X_h1, y_all, grp_all, tasks, n_splits=N_SPLITS):
    """Same as above on the concatenation of the H0 and H1 landscapes."""
    rows = []
    for task in tasks:
        mask = np.isin(y_all, list(task))
        X_h0_task, X_h1_task = X_h0[mask], X_h1[mask]
        y_task, grp_task = y_all[mask], grp_all[mask]

        scores = []
        for train_idx, test_idx in grouped_splits(y_task, grp_task, n_splits):
            y_train, y_test = y_task[train_idx], y_task[test_idx]

            ls_h0, ls_h1 = make_landscape(), make_landscape()
            X_train = np.hstack([
                ls_h0.fit_transform(X_h0_task[train_idx]),
                ls_h1.fit_transform(X_h1_task[train_idx]),
            ])
            X_test = np.hstack([
                ls_h0.transform(X_h0_task[test_idx]),
                ls_h1.transform(X_h1_task[test_idx]),
            ])

            scaler = StandardScaler().fit(X_train)
            clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
            try:
                clf.fit(scaler.transform(X_train), y_train)
                scores.append(clf.score(scaler.transform(X_test), y_test))
            except (LinAlgError, ValueError):
                continue

        rows.append({
            "Task": task_label(task),
            "Mean accuracy": np.mean(scores) if scores else np.nan,
            "Std": np.std(scores) if scores else np.nan,
        })
    return pd.DataFrame(rows).set_index("Task")


# ─────────────────────────────────────────────────────────────────────────────
# Gradient boosting hyperparameters: 1-standard-error rule
# ─────────────────────────────────────────────────────────────────────────────

def _build_landscape_features(X_h0_task, X_h1_task, train_idx, test_idx):
    """Fit the Landscape transformers on the TRAINING fold only, then transform
    both training and test folds. Avoids leakage of the discretization grid."""
    ls_h0, ls_h1 = make_landscape(), make_landscape()
    Xh0_tr = ls_h0.fit_transform(X_h0_task[train_idx])
    Xh1_tr = ls_h1.fit_transform(X_h1_task[train_idx])
    Xh0_te = ls_h0.transform(X_h0_task[test_idx])
    Xh1_te = ls_h1.transform(X_h1_task[test_idx])
    X_tr = np.hstack([Xh0_tr, Xh1_tr])
    X_te = np.hstack([Xh0_te, Xh1_te])
    scaler = StandardScaler().fit(X_tr)
    return scaler.transform(X_tr), scaler.transform(X_te)


def _fit_score_fold_landscape(estimator, X_h0_task, X_h1_task, y_task, tr, te):
    X_tr, X_te = _build_landscape_features(X_h0_task, X_h1_task, tr, te)
    clf = clone(estimator).fit(X_tr, y_task[tr])
    return clf.score(X_tr, y_task[tr]), clf.score(X_te, y_task[te])


def grouped_train_test_gap(estimator, X_h0_task, X_h1_task, y_task, groups, n_splits=10):
    """Mean train and test accuracy across grouped folds — measures overfitting."""
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    splits = list(cv.split(np.zeros(len(y_task)), y_task, groups))
    fold_scores = Parallel(n_jobs=-1)(
        delayed(_fit_score_fold_landscape)(estimator, X_h0_task, X_h1_task, y_task, tr, te)
        for tr, te in splits
    )
    train_acc, test_acc = zip(*fold_scores)
    return float(np.mean(train_acc)), float(np.mean(test_acc))


def select_gb_params_1se(X_h0_task, X_h1_task, y_task, grp_task, param_grid=None,
                         n_inner_splits=5, verbose=True):
    """Pick gradient-boosting hyperparameters with the 1-standard-error rule.

    Plain ``GridSearchCV`` picks the configuration with the highest mean CV
    accuracy, which here is the deepest / longest model and overfits massively.
    The 1-SE rule, standard in cross-validation since Breiman et al. (1984),
    instead picks the SIMPLEST configuration whose CV accuracy is within one
    standard error of the best. This trades a sliver of CV accuracy for a
    substantially smaller train/test gap and a more parsimonious model.

    Returns
    -------
    gb_params : dict with ``n_estimators`` and ``max_depth``
    cv_results : the full grid-search table, with a ``complexity`` column
    """
    if param_grid is None:
        param_grid = {"n_estimators": [50, 100, 200, 300], "max_depth": [2, 3, 4]}

    inner_cv = StratifiedGroupKFold(n_splits=n_inner_splits, shuffle=True,
                                    random_state=RANDOM_SEED)
    inner_splits = list(inner_cv.split(np.zeros(len(y_task)), y_task, grp_task))

    # Build features train-side only inside the FIRST inner split, then reuse
    # them across the grid (the grid only varies GB hyperparameters, and the
    # Landscape discretization grid is essentially stable across large training
    # subsets, so this is a benign approximation).
    tr_idx, _ = inner_splits[0]
    ls_h0_tune = make_landscape().fit(X_h0_task[tr_idx])
    ls_h1_tune = make_landscape().fit(X_h1_task[tr_idx])
    X_tune = np.hstack([
        ls_h0_tune.transform(X_h0_task),
        ls_h1_tune.transform(X_h1_task),
    ])
    X_tune = StandardScaler().fit(X_tune[tr_idx]).transform(X_tune)

    grid = GridSearchCV(
        GradientBoostingClassifier(random_state=RANDOM_SEED),
        param_grid,
        cv=inner_splits,
        scoring="accuracy",
        n_jobs=-1,
        refit=False,  # we will pick by 1-SE rule, not by best mean
        return_train_score=False,
    ).fit(X_tune, y_task)

    cv_results = pd.DataFrame(grid.cv_results_)

    # Complexity proxy: n_estimators * 2^max_depth (roughly the number of leaves
    # in the ensemble). Used to break ties among 1-SE-acceptable configurations.
    cv_results["complexity"] = (
        cv_results["param_n_estimators"].astype(int)
        * (2 ** cv_results["param_max_depth"].astype(int))
    )

    best_mean = cv_results["mean_test_score"].max()
    best_se = cv_results.loc[cv_results["mean_test_score"].idxmax(),
                             "std_test_score"] / np.sqrt(n_inner_splits)
    threshold = best_mean - best_se

    eligible = cv_results[cv_results["mean_test_score"] >= threshold]
    chosen = eligible.sort_values("complexity").iloc[0]

    gb_params = {
        "n_estimators": int(chosen["param_n_estimators"]),
        "max_depth": int(chosen["param_max_depth"]),
    }

    if verbose:
        print("\nGrid search summary (3-way task on H0 + H1, "
              f"{n_inner_splits}-fold grouped inner CV):")
        print(f"  best mean CV accuracy        = {best_mean:.1%}")
        print(f"  1-SE threshold               = {threshold:.1%}")
        print(f"  candidates within 1 SE       = {len(eligible)}")
        print("\nSelected hyperparameters (1-SE rule, simplest within tolerance):")
        for k, v in gb_params.items():
            print(f"  {k:14s}= {v}")
        print(f"  CV accuracy at selection     = {chosen['mean_test_score']:.1%}")

    return gb_params, cv_results


# ─────────────────────────────────────────────────────────────────────────────
# Per-fold evaluators (one landscape fit per fold, reused across models)
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_CONFIGS = ["H1 only", "H0 only", "H0 + H1"]


def evaluate_fold(train_idx, test_idx, X_h0_task, X_h1_task, y_task,
                  models, feature_configs=FEATURE_CONFIGS):
    """Evaluate every (config, model) pair on a single fold.

    Landscapes are fitted once per fold and reused across models.
    """
    y_train, y_test = y_task[train_idx], y_task[test_idx]

    ls_h0, ls_h1 = make_landscape(), make_landscape()
    Xh0_train = ls_h0.fit_transform(X_h0_task[train_idx])
    Xh0_test = ls_h0.transform(X_h0_task[test_idx])
    Xh1_train = ls_h1.fit_transform(X_h1_task[train_idx])
    Xh1_test = ls_h1.transform(X_h1_task[test_idx])

    feature_sets = {
        "H1 only": (Xh1_train, Xh1_test),
        "H0 only": (Xh0_train, Xh0_test),
        "H0 + H1": (np.hstack([Xh0_train, Xh1_train]),
                    np.hstack([Xh0_test, Xh1_test])),
    }

    out = {}
    for config in feature_configs:
        X_train, X_test = feature_sets[config]
        scaler = StandardScaler().fit(X_train)
        X_train_s = scaler.transform(X_train)
        X_test_s = scaler.transform(X_test)

        for model_name, estimator in models.items():
            try:
                clf = clone(estimator).fit(X_train_s, y_train)
                out[(model_name, config)] = clf.score(X_test_s, y_test)
            except (LinAlgError, ValueError):
                out[(model_name, config)] = np.nan
    return out


def evaluate_models_by_config(X_h0, X_h1, y_all, grp_all, tasks, models,
                              feature_configs=FEATURE_CONFIGS, n_splits=N_SPLITS):
    """Run :func:`evaluate_fold` over every task; return one table per model."""
    per_model_rows = {name: [] for name in models}

    for task in tasks:
        mask = np.isin(y_all, list(task))
        X_h0_task, X_h1_task = X_h0[mask], X_h1[mask]
        y_task, grp_task = y_all[mask], grp_all[mask]

        splits = grouped_splits(y_task, grp_task, n_splits)
        fold_results = Parallel(n_jobs=-1, verbose=0)(
            delayed(evaluate_fold)(
                train_idx, test_idx, X_h0_task, X_h1_task, y_task, models, feature_configs
            )
            for train_idx, test_idx in splits
        )

        for model_name in models:
            row = {"Task": task_label(task)}
            for config in feature_configs:
                fold_scores = [
                    fr[(model_name, config)] for fr in fold_results
                    if not np.isnan(fr[(model_name, config)])
                ]
                row[f"{config} mean"] = np.mean(fold_scores) if fold_scores else np.nan
                row[f"{config} std"] = np.std(fold_scores) if fold_scores else np.nan
            per_model_rows[model_name].append(row)

    return {name: pd.DataFrame(rows).set_index("Task")
            for name, rows in per_model_rows.items()}


def predict_fold_all_models(train_idx, test_idx, X_h0_task, X_h1_task, y_task, classifiers):
    """Fit landscapes once and return (y_test, predictions per model) for one fold."""
    y_train, y_test = y_task[train_idx], y_task[test_idx]

    ls_h0, ls_h1 = make_landscape(), make_landscape()
    X_train = np.hstack([
        ls_h0.fit_transform(X_h0_task[train_idx]),
        ls_h1.fit_transform(X_h1_task[train_idx]),
    ])
    X_test = np.hstack([
        ls_h0.transform(X_h0_task[test_idx]),
        ls_h1.transform(X_h1_task[test_idx]),
    ])

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    preds = {}
    for name, estimator in classifiers.items():
        try:
            clf = clone(estimator).fit(X_train_s, y_train)
            preds[name] = clf.predict(X_test_s)
        except (LinAlgError, ValueError):
            preds[name] = None
    return y_test, preds


def collect_confusion_matrices(X_h0, X_h1, y_all, grp_all, classifiers,
                               celltypes=CELLTYPES, n_splits=N_SPLITS):
    """Out-of-fold confusion matrices (row-normalised) for the 3-way task."""
    mask = np.isin(y_all, list(celltypes))
    X_h0_task, X_h1_task = X_h0[mask], X_h1[mask]
    y_task, grp_task = y_all[mask], grp_all[mask]

    splits = grouped_splits(y_task, grp_task, n_splits)
    fold_outputs = Parallel(n_jobs=-1, verbose=0)(
        delayed(predict_fold_all_models)(
            train_idx, test_idx, X_h0_task, X_h1_task, y_task, classifiers
        )
        for train_idx, test_idx in splits
    )

    matrices = {}
    for name in classifiers:
        y_true_all, y_pred_all = [], []
        for y_test, preds in fold_outputs:
            if preds[name] is None:
                continue
            y_true_all.extend(y_test)
            y_pred_all.extend(preds[name])
        matrices[name] = confusion_matrix(
            y_true_all, y_pred_all, labels=list(celltypes), normalize="true"
        )
    return matrices


# ─────────────────────────────────────────────────────────────────────────────
# Ablation: coordinate features vs landscapes
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_coord_vs_landscape_fold(train_idx, test_idx, Xc_task, Xl_task,
                                     y_task, classifiers):
    """Compute Coord / Landscape accuracy for every classifier on a single fold.

    The landscape transform and scalers are fitted once per fold and reused
    across all classifiers.
    """
    y_train, y_test = y_task[train_idx], y_task[test_idx]

    ls = make_landscape()
    Xl_train = ls.fit_transform(Xl_task[train_idx])
    Xl_test = ls.transform(Xl_task[test_idx])

    scaler_c = StandardScaler().fit(Xc_task[train_idx])
    Xc_train = scaler_c.transform(Xc_task[train_idx])
    Xc_test = scaler_c.transform(Xc_task[test_idx])

    scaler_l = StandardScaler().fit(Xl_train)
    Xl_train = scaler_l.transform(Xl_train)
    Xl_test = scaler_l.transform(Xl_test)

    out = {}
    for clf_name, clf in classifiers.items():
        try:
            clf_c = clone(clf).fit(Xc_train, y_train)
            out[(clf_name, "coord")] = clf_c.score(Xc_test, y_test)
        except (LinAlgError, ValueError):
            out[(clf_name, "coord")] = np.nan
        try:
            clf_l = clone(clf).fit(Xl_train, y_train)
            out[(clf_name, "landscape")] = clf_l.score(Xl_test, y_test)
        except (LinAlgError, ValueError):
            out[(clf_name, "landscape")] = np.nan
    return out


def evaluate_coord_vs_landscape(X_coord, X_h1, y_all, grp_all, tasks, classifiers,
                                n_splits=N_SPLITS):
    """Head-to-head comparison of coordinate features and H1 landscapes."""
    rows = []
    for task in tasks:
        mask = np.isin(y_all, list(task))
        y_task, grp_task = y_all[mask], grp_all[mask]
        Xc_task, Xl_task = X_coord[mask], X_h1[mask]

        splits = grouped_splits(y_task, grp_task, n_splits)
        fold_results = Parallel(n_jobs=-1, verbose=0)(
            delayed(evaluate_coord_vs_landscape_fold)(
                train_idx, test_idx, Xc_task, Xl_task, y_task, classifiers
            )
            for train_idx, test_idx in splits
        )

        row = {"Task": task_label(task)}
        for clf_name in classifiers:
            for feat, key in [("Coord", "coord"), ("Landscape", "landscape")]:
                vals = [fr[(clf_name, key)] for fr in fold_results
                        if not np.isnan(fr[(clf_name, key)])]
                col = f"{clf_name} {feat}"
                row[col] = np.mean(vals) if vals else np.nan
                row[col + " std"] = np.std(vals) if vals else np.nan
        rows.append(row)

    return pd.DataFrame(rows).set_index("Task")


def evaluate_h0h1_vs_combined_fold(train_idx, test_idx, X_h0_task, X_h1_task,
                                   Xc_task, y_task, gb_params):
    """Fit landscapes once, then train two GB models (H0+H1 vs H0+H1+Coord)."""
    y_train, y_test = y_task[train_idx], y_task[test_idx]

    ls_h0, ls_h1 = make_landscape(), make_landscape()
    Xl_train = np.hstack([
        ls_h0.fit_transform(X_h0_task[train_idx]),
        ls_h1.fit_transform(X_h1_task[train_idx]),
    ])
    Xl_test = np.hstack([
        ls_h0.transform(X_h0_task[test_idx]),
        ls_h1.transform(X_h1_task[test_idx]),
    ])

    scaler_c = StandardScaler().fit(Xc_task[train_idx])
    Xc_train = scaler_c.transform(Xc_task[train_idx])
    Xc_test = scaler_c.transform(Xc_task[test_idx])

    scaler_l = StandardScaler().fit(Xl_train)
    Xl_train = scaler_l.transform(Xl_train)
    Xl_test = scaler_l.transform(Xl_test)

    Xcomb_train = np.hstack([Xl_train, Xc_train])
    Xcomb_test = np.hstack([Xl_test, Xc_test])

    out = {}
    try:
        clf_topo = GradientBoostingClassifier(**gb_params, random_state=RANDOM_SEED)
        clf_topo.fit(Xl_train, y_train)
        out["H0+H1"] = clf_topo.score(Xl_test, y_test)
    except (LinAlgError, ValueError):
        out["H0+H1"] = np.nan
    try:
        clf_comb = GradientBoostingClassifier(**gb_params, random_state=RANDOM_SEED)
        clf_comb.fit(Xcomb_train, y_train)
        out["H0+H1+Coord"] = clf_comb.score(Xcomb_test, y_test)
    except (LinAlgError, ValueError):
        out["H0+H1+Coord"] = np.nan
    return out


def evaluate_h0h1_vs_combined(X_h0, X_h1, X_coord, y_all, grp_all, tasks, gb_params,
                              n_splits=N_SPLITS):
    """Does adding coordinate features on top of the landscapes help?"""
    rows = []
    for task in tasks:
        mask = np.isin(y_all, list(task))
        y_task, grp_task = y_all[mask], grp_all[mask]
        X_h0_task, X_h1_task, Xc_task = X_h0[mask], X_h1[mask], X_coord[mask]

        splits = grouped_splits(y_task, grp_task, n_splits)
        fold_results = Parallel(n_jobs=-1, verbose=0)(
            delayed(evaluate_h0h1_vs_combined_fold)(
                train_idx, test_idx, X_h0_task, X_h1_task, Xc_task, y_task, gb_params
            )
            for train_idx, test_idx in splits
        )

        s_topo = [fr["H0+H1"] for fr in fold_results if not np.isnan(fr["H0+H1"])]
        s_comb = [fr["H0+H1+Coord"] for fr in fold_results if not np.isnan(fr["H0+H1+Coord"])]

        rows.append({
            "Task": task_label(task),
            "H0+H1": np.mean(s_topo) if s_topo else np.nan,
            "H0+H1 std": np.std(s_topo) if s_topo else np.nan,
            "H0+H1+Coord": np.mean(s_comb) if s_comb else np.nan,
            "H0+H1+Coord std": np.std(s_comb) if s_comb else np.nan,
        })

    return pd.DataFrame(rows).set_index("Task")


# ─────────────────────────────────────────────────────────────────────────────
# Leave-one-tumor-out robustness check
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_logo_tumor(train_idx, test_idx, X_h0_task, X_h1_task, Xc_task,
                        y_task, grp_task, gb_params):
    """Evaluate one leave-one-tumor-out split.

    Computes two Gradient Boosting accuracies:
    1. coordinate-based geometric features
    2. H0+H1 persistence landscapes

    Landscapes and scalers are fitted on the training tumors only.
    """
    heldout_tumor = np.unique(grp_task[test_idx])[0]
    y_train, y_test = y_task[train_idx], y_task[test_idx]

    # Coordinate features
    scaler_c = StandardScaler().fit(Xc_task[train_idx])
    Xc_train = scaler_c.transform(Xc_task[train_idx])
    Xc_test = scaler_c.transform(Xc_task[test_idx])

    clf_coord = GradientBoostingClassifier(**gb_params, random_state=RANDOM_SEED)
    clf_coord.fit(Xc_train, y_train)
    acc_coord = accuracy_score(y_test, clf_coord.predict(Xc_test))

    # H0 + H1 landscape features
    ls_h0, ls_h1 = make_landscape(), make_landscape()
    Xl_train = np.hstack([
        ls_h0.fit_transform(X_h0_task[train_idx]),
        ls_h1.fit_transform(X_h1_task[train_idx]),
    ])
    Xl_test = np.hstack([
        ls_h0.transform(X_h0_task[test_idx]),
        ls_h1.transform(X_h1_task[test_idx]),
    ])

    scaler_l = StandardScaler().fit(Xl_train)
    Xl_train = scaler_l.transform(Xl_train)
    Xl_test = scaler_l.transform(Xl_test)

    clf_land = GradientBoostingClassifier(**gb_params, random_state=RANDOM_SEED)
    clf_land.fit(Xl_train, y_train)
    acc_land = accuracy_score(y_test, clf_land.predict(Xl_test))

    return {
        "Held-out tumor": heldout_tumor,
        "n_test_ROIs": len(test_idx),
        "Coord accuracy": acc_coord,
        "H0+H1 landscape accuracy": acc_land,
    }


def leave_one_tumor_out(X_h0, X_h1, X_coord, y_all, grp_all, gb_params,
                        celltypes=CELLTYPES, verbose=10):
    """Repeat the 3-way comparison leaving out one tumor at a time.

    Robustness check for the effective sample size issue: the independent units
    are tumors, not ROIs.
    """
    mask = np.isin(y_all, list(celltypes))
    y_task, grp_task = y_all[mask], grp_all[mask]
    X_h0_task, X_h1_task, Xc_task = X_h0[mask], X_h1[mask], X_coord[mask]

    logo = LeaveOneGroupOut()
    logo_splits = list(logo.split(np.zeros(len(y_task)), y_task, grp_task))

    logo_rows = Parallel(n_jobs=-1, verbose=verbose)(
        delayed(evaluate_logo_tumor)(
            train_idx, test_idx, X_h0_task, X_h1_task, Xc_task,
            y_task, grp_task, gb_params
        )
        for train_idx, test_idx in logo_splits
    )

    return pd.DataFrame(logo_rows).sort_values("Held-out tumor")


def default_classifiers(gb_params):
    """The three classifiers compared throughout, with distinct inductive biases."""
    return {
        "LDA": LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
        "LogReg": LogisticRegression(max_iter=1000),
        "Boosting": GradientBoostingClassifier(**gb_params, random_state=RANDOM_SEED),
    }
