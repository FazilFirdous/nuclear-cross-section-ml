"""
Cross-validation strategies for nuclear cross-section prediction.

Standard k-fold cross-validation is inadequate because nearby isotopes on the
nuclear chart are correlated. The leave-one-isotope-out (LOIO) strategy evaluates
generalization to truly unseen nuclei, which matches the scientific goal.
"""

import logging
from typing import Callable, Dict, Iterator, List, Optional, Tuple, Type

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)


class LeaveOneIsotopeOutCV:
    """
    Leave-one-isotope-out cross-validation.

    In each fold, all data points for one isotope (Z, A) are held out as the
    test set, and the model is trained on all remaining isotopes. This is the
    most scientifically meaningful validation for the extrapolation problem.

    Parameters
    ----------
    n_splits : int or None
        Number of folds (number of isotopes to cycle through). If None,
        uses all isotopes (true LOIO-CV).
    shuffle : bool
        Whether to shuffle isotopes before splitting.
    random_state : int or None
        Random seed for reproducibility.
    min_test_points : int
        Minimum data points required for a held-out isotope to be included.
    """

    def __init__(
        self,
        n_splits: Optional[int] = None,
        shuffle: bool = True,
        random_state: Optional[int] = 42,
        min_test_points: int = 3,
    ):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.min_test_points = min_test_points

    def split(
        self, df: pd.DataFrame
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test index splits by isotope.

        Parameters
        ----------
        df : pd.DataFrame
            Dataset with "Z" and "A" columns.

        Yields
        ------
        (train_indices, test_indices) as numpy integer arrays.
        """
        isotope_labels = df[["Z", "A"]].apply(
            lambda r: (int(r["Z"]), int(r["A"])), axis=1
        ).values

        unique_isotopes = list(set(isotope_labels))

        if self.shuffle:
            rng = np.random.RandomState(self.random_state)
            rng.shuffle(unique_isotopes)

        if self.n_splits is not None:
            unique_isotopes = unique_isotopes[:self.n_splits]

        for isotope in unique_isotopes:
            test_mask = np.array([label == isotope for label in isotope_labels])
            n_test = test_mask.sum()
            if n_test < self.min_test_points:
                continue
            train_indices = np.where(~test_mask)[0]
            test_indices = np.where(test_mask)[0]
            yield train_indices, test_indices

    def get_n_splits(self, df: pd.DataFrame) -> int:
        """Return the number of folds that will be generated."""
        unique = df[["Z", "A"]].drop_duplicates()
        if self.n_splits is not None:
            return min(self.n_splits, len(unique))
        return len(unique)


def run_kfold_cv(
    model_class: Type,
    model_kwargs: Dict,
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    random_state: int = 42,
    fit_kwargs: Optional[Dict] = None,
) -> Dict[str, np.ndarray]:
    """
    Run standard k-fold cross-validation for quick benchmarking.

    Parameters
    ----------
    model_class : type
        Model class with fit(X, y) and predict(X) interface.
    model_kwargs : dict
        Keyword arguments passed to model_class constructor.
    X, y : np.ndarray
        Features and targets.
    n_folds : int
        Number of CV folds.
    random_state : int
        Random seed.
    fit_kwargs : dict, optional
        Extra keyword arguments for model.fit().

    Returns
    -------
    dict with per-fold metrics and aggregated statistics.
    """
    from .metrics import compute_metrics

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    fit_kwargs = fit_kwargs or {}

    fold_metrics = []
    oof_preds = np.zeros(len(y))

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = model_class(**model_kwargs)
        model.fit(X_train, y_train, **fit_kwargs)

        val_pred = model.predict(X_val)
        oof_preds[val_idx] = val_pred

        metrics = compute_metrics(y_val, val_pred)
        fold_metrics.append(metrics)
        logger.info("Fold %d/%d: RMSE=%.4f, R2=%.4f", fold_idx + 1, n_folds, metrics["rmse"], metrics["r2"])

    overall_metrics = compute_metrics(y, oof_preds)
    logger.info(
        "Overall OOF: RMSE=%.4f, R2=%.4f", overall_metrics["rmse"], overall_metrics["r2"]
    )

    return {
        "fold_metrics": fold_metrics,
        "oof_metrics": overall_metrics,
        "oof_predictions": oof_preds,
    }


def run_loio_cv(
    model_class: Type,
    model_kwargs: Dict,
    df: pd.DataFrame,
    feature_columns: List[str],
    target_column: str,
    n_isotopes: Optional[int] = None,
    feature_engineer=None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Run leave-one-isotope-out cross-validation.

    For each isotope in turn, trains on all other isotopes and predicts for
    the held-out isotope. Returns per-isotope predictions and metrics.

    Parameters
    ----------
    model_class : type
        Model class.
    model_kwargs : dict
        Model constructor arguments.
    df : pd.DataFrame
        Full dataset with Z, A, feature_columns, and target_column.
    feature_columns : list of str
        Input feature names.
    target_column : str
        Target column name.
    n_isotopes : int or None
        If set, limit to the first n_isotopes folds.
    feature_engineer : FeatureEngineer or None
        If provided, re-fit on training split and transform each fold.
    random_state : int

    Returns
    -------
    pd.DataFrame
        Per-data-point predictions with columns:
        Z, A, energy_eV, y_true, y_pred, fold_isotope.
    """
    from .metrics import compute_metrics

    cv = LeaveOneIsotopeOutCV(
        n_splits=n_isotopes,
        shuffle=True,
        random_state=random_state,
    )

    all_results = []
    total_folds = cv.get_n_splits(df)
    logger.info("Running LOIO-CV with %d folds", total_folds)

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(df)):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        Z_test = int(test_df["Z"].iloc[0])
        A_test = int(test_df["A"].iloc[0])

        if feature_engineer is not None:
            X_train = feature_engineer.fit_transform(train_df)
            X_test_arr = feature_engineer.transform(test_df)
        else:
            X_train = train_df[feature_columns].values.astype(np.float32)
            X_test_arr = test_df[feature_columns].values.astype(np.float32)

        y_train = train_df[target_column].values
        y_test = test_df[target_column].values

        model = model_class(**model_kwargs)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test_arr)

        metrics = compute_metrics(y_test, y_pred)

        for i, (pred, true) in enumerate(zip(y_pred, y_test)):
            row = {
                "Z": Z_test,
                "A": A_test,
                "fold_isotope": f"Z{Z_test}A{A_test}",
                "y_true": true,
                "y_pred": pred,
                "rmse_fold": metrics["rmse"],
                "r2_fold": metrics["r2"],
            }
            if "energy_eV" in test_df.columns:
                row["energy_eV"] = test_df["energy_eV"].iloc[i]
            all_results.append(row)

        if fold_idx % 20 == 0:
            logger.info(
                "LOIO fold %d/%d: Z=%d A=%d, RMSE=%.3f",
                fold_idx + 1, total_folds, Z_test, A_test, metrics["rmse"],
            )

    return pd.DataFrame(all_results)
