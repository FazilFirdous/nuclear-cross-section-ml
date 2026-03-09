"""
Random forest model for cross-section prediction.

Uses ensemble variance across trees as the primary uncertainty estimate.
"""

import logging
import pickle
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

logger = logging.getLogger(__name__)

DEFAULT_PARAMS = {
    "n_estimators": 500,
    "max_depth": 20,
    "min_samples_leaf": 5,
    "max_features": "sqrt",
    "bootstrap": True,
    "oob_score": True,
    "n_jobs": -1,
    "random_state": 42,
}


class RandomForestCrossSectionModel:
    """
    Random forest model for nuclear cross-section prediction.

    Uncertainty is estimated from the variance of individual tree predictions
    in the ensemble (jackknife-after-bootstrap estimator).

    Parameters
    ----------
    params : dict, optional
        scikit-learn RandomForestRegressor parameters.
    feature_names : list of str, optional
        Names of input features.
    """

    def __init__(
        self,
        params: Optional[Dict] = None,
        feature_names: Optional[List[str]] = None,
    ):
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.feature_names = feature_names
        self._model: Optional[RandomForestRegressor] = None
        self._fitted = False

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "RandomForestCrossSectionModel":
        """
        Fit the random forest model.

        Parameters
        ----------
        X_train, y_train : np.ndarray
            Training data.
        X_val, y_val : np.ndarray, optional
            Validation data (used only for logging, not for model selection).

        Returns
        -------
        self
        """
        logger.info(
            "Training Random Forest on %d samples, %d features",
            len(X_train), X_train.shape[1],
        )
        self._model = RandomForestRegressor(**self.params)
        self._model.fit(X_train, y_train)
        self._fitted = True

        train_rmse = np.sqrt(mean_squared_error(y_train, self._model.predict(X_train)))
        logger.info("OOB score: %.4f", self._model.oob_score_)
        logger.info("Training RMSE: %.4f log10(barn)", train_rmse)

        if X_val is not None:
            val_rmse = np.sqrt(mean_squared_error(y_val, self._model.predict(X_val)))
            logger.info("Validation RMSE: %.4f log10(barn)", val_rmse)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return mean prediction across all trees."""
        if not self._fitted:
            raise RuntimeError("Call fit() before predict().")
        return self._model.predict(X)

    def predict_with_uncertainty(
        self, X: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Return predictions with tree-ensemble uncertainty estimates.

        The standard deviation across individual tree predictions provides
        an empirical estimate of epistemic uncertainty.

        Parameters
        ----------
        X : np.ndarray

        Returns
        -------
        dict with keys: mean, std, lower_68, upper_68, lower_95, upper_95.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before predict_with_uncertainty().")

        # Collect predictions from all trees
        tree_preds = np.array([tree.predict(X) for tree in self._model.estimators_])
        # tree_preds shape: (n_estimators, n_samples)

        mean_pred = tree_preds.mean(axis=0)
        std_pred = tree_preds.std(axis=0)

        return {
            "mean": mean_pred,
            "std": std_pred,
            "lower_68": mean_pred - std_pred,
            "upper_68": mean_pred + std_pred,
            "lower_95": mean_pred - 2.0 * std_pred,
            "upper_95": mean_pred + 2.0 * std_pred,
            "q05": np.percentile(tree_preds, 5, axis=0),
            "q16": np.percentile(tree_preds, 16, axis=0),
            "median": np.median(tree_preds, axis=0),
            "q84": np.percentile(tree_preds, 84, axis=0),
            "q95": np.percentile(tree_preds, 95, axis=0),
        }

    def feature_importance(self) -> pd.DataFrame:
        """
        Return mean decrease in impurity (MDI) feature importances.

        Also includes the standard deviation across trees as a reliability
        indicator; features with high std are unreliable.

        Returns
        -------
        pd.DataFrame with columns: feature, importance_mean, importance_std.
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted.")

        importances = self._model.feature_importances_
        std = np.std(
            [tree.feature_importances_ for tree in self._model.estimators_], axis=0
        )
        features = self.feature_names or [f"f{i}" for i in range(len(importances))]

        df = pd.DataFrame({
            "feature": features,
            "importance_mean": importances,
            "importance_std": std,
        }).sort_values("importance_mean", ascending=False)

        return df.reset_index(drop=True)

    def evaluate(
        self, X_test: np.ndarray, y_test: np.ndarray
    ) -> Dict[str, float]:
        """Compute evaluation metrics on a test set."""
        preds = self.predict_with_uncertainty(X_test)
        y_pred = preds["mean"]

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = np.mean(np.abs(y_test - y_pred))
        r2 = r2_score(y_test, y_pred)

        coverage_68 = np.mean(
            (y_test >= preds["lower_68"]) & (y_test <= preds["upper_68"])
        )
        coverage_95 = np.mean(
            (y_test >= preds["lower_95"]) & (y_test <= preds["upper_95"])
        )

        return {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "picp_68": coverage_68,
            "picp_95": coverage_95,
            "mpiw_68": np.mean(preds["upper_68"] - preds["lower_68"]),
            "mpiw_95": np.mean(preds["upper_95"] - preds["lower_95"]),
            "oob_r2": self._model.oob_score_,
        }

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("Saved Random Forest model to %s", path)

    @classmethod
    def load(cls, path: str) -> "RandomForestCrossSectionModel":
        with open(path, "rb") as f:
            return pickle.load(f)
