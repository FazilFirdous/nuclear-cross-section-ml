"""
XGBoost gradient boosted trees model for cross-section prediction.

Uses quantile regression within XGBoost to provide prediction intervals
alongside point estimates.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score

logger = logging.getLogger(__name__)

DEFAULT_PARAMS = {
    "n_estimators": 1000,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "min_child_weight": 5,
    "tree_method": "hist",
    "random_state": 42,
}

QUANTILE_ALPHAS = [0.05, 0.16, 0.50, 0.84, 0.95]


class XGBoostCrossSectionModel:
    """
    Gradient boosted tree model for nuclear cross-section prediction.

    Trains a central (median) XGBoost regressor plus quantile regressors
    at the 5th, 16th, 84th, and 95th percentiles to provide 68% and 90%
    prediction intervals.

    Parameters
    ----------
    params : dict, optional
        XGBoost hyperparameters. Defaults to DEFAULT_PARAMS.
    feature_names : list of str, optional
        Names of input features for interpretability.
    early_stopping_rounds : int
        Patience for early stopping on validation set.
    """

    def __init__(
        self,
        params: Optional[Dict] = None,
        feature_names: Optional[List[str]] = None,
        early_stopping_rounds: int = 50,
    ):
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.feature_names = feature_names
        self.early_stopping_rounds = early_stopping_rounds

        self._median_model: Optional[xgb.XGBRegressor] = None
        self._quantile_models: Dict[float, xgb.XGBRegressor] = {}
        self._fitted = False

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "XGBoostCrossSectionModel":
        """
        Fit the median and quantile XGBoost models.

        Parameters
        ----------
        X_train, y_train : np.ndarray
            Training features and targets.
        X_val, y_val : np.ndarray, optional
            Validation set for early stopping.

        Returns
        -------
        self
        """
        # Train median (point estimate) model using squared error
        logger.info(
            "Training XGBoost median model on %d samples, %d features",
            len(X_train), X_train.shape[1],
        )
        median_params = {**self.params, "objective": "reg:squarederror"}
        self._median_model = xgb.XGBRegressor(**median_params)

        fit_kwargs = {}
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            fit_kwargs["early_stopping_rounds"] = self.early_stopping_rounds
            fit_kwargs["verbose"] = False

        self._median_model.fit(X_train, y_train, **fit_kwargs)

        # Train quantile models for uncertainty quantification
        for alpha in QUANTILE_ALPHAS:
            logger.info("Training quantile model for alpha=%.2f", alpha)
            q_params = {
                **self.params,
                "objective": "reg:quantileerror",
                "quantile_alpha": alpha,
                "n_estimators": self.params.get("n_estimators", 500),
            }
            q_model = xgb.XGBRegressor(**q_params)
            q_model.fit(X_train, y_train, **fit_kwargs)
            self._quantile_models[alpha] = q_model

        self._fitted = True
        train_rmse = np.sqrt(mean_squared_error(y_train, self._median_model.predict(X_train)))
        logger.info("Training RMSE (median model): %.4f log10(barn)", train_rmse)

        if X_val is not None:
            val_rmse = np.sqrt(mean_squared_error(y_val, self._median_model.predict(X_val)))
            logger.info("Validation RMSE: %.4f log10(barn)", val_rmse)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return point predictions (median estimate).

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)

        Returns
        -------
        np.ndarray of shape (n_samples,) with predicted log10(cross-section/barn).
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before predict().")
        return self._median_model.predict(X)

    def predict_with_uncertainty(
        self, X: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Return point predictions with quantile-based uncertainty intervals.

        Parameters
        ----------
        X : np.ndarray

        Returns
        -------
        dict with keys:
            "median": point prediction
            "q05", "q16", "q84", "q95": quantile predictions
            "sigma_68": half-width of 68% interval (q84-q16)/2
            "sigma_90": half-width of 90% interval (q95-q05)/2
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before predict_with_uncertainty().")

        result = {"median": self._median_model.predict(X)}
        for alpha, model in self._quantile_models.items():
            key = f"q{int(alpha * 100):02d}"
            result[key] = model.predict(X)

        if "q16" in result and "q84" in result:
            result["sigma_68"] = (result["q84"] - result["q16"]) / 2.0
        if "q05" in result and "q95" in result:
            result["sigma_90"] = (result["q95"] - result["q05"]) / 2.0

        return result

    def feature_importance(self) -> pd.DataFrame:
        """
        Return feature importance from the median model.

        Returns
        -------
        pd.DataFrame with columns: feature, importance_gain, importance_cover.
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted.")

        gain = self._median_model.get_booster().get_score(importance_type="gain")
        cover = self._median_model.get_booster().get_score(importance_type="cover")

        features = self.feature_names or [f"f{i}" for i in range(len(gain))]
        rows = []
        for feat in features:
            rows.append({
                "feature": feat,
                "importance_gain": gain.get(feat, 0.0),
                "importance_cover": cover.get(feat, 0.0),
            })

        df = pd.DataFrame(rows).sort_values("importance_gain", ascending=False)
        return df.reset_index(drop=True)

    def save(self, path: str) -> None:
        """Save model to disk using pickle."""
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("Saved XGBoost model to %s", path)

    @classmethod
    def load(cls, path: str) -> "XGBoostCrossSectionModel":
        """Load model from disk."""
        with open(path, "rb") as f:
            return pickle.load(f)

    def evaluate(
        self, X_test: np.ndarray, y_test: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics on a test set.

        Returns
        -------
        dict with keys: rmse, mae, r2, picp_68, picp_95, mpiw_68, mpiw_95.
        """
        preds = self.predict_with_uncertainty(X_test)
        y_pred = preds["median"]

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = np.mean(np.abs(y_test - y_pred))
        r2 = r2_score(y_test, y_pred)

        metrics = {"rmse": rmse, "mae": mae, "r2": r2}

        if "q16" in preds and "q84" in preds:
            coverage_68 = np.mean(
                (y_test >= preds["q16"]) & (y_test <= preds["q84"])
            )
            width_68 = np.mean(preds["q84"] - preds["q16"])
            metrics["picp_68"] = coverage_68
            metrics["mpiw_68"] = width_68

        if "q05" in preds and "q95" in preds:
            coverage_90 = np.mean(
                (y_test >= preds["q05"]) & (y_test <= preds["q95"])
            )
            width_90 = np.mean(preds["q95"] - preds["q05"])
            metrics["picp_90"] = coverage_90
            metrics["mpiw_90"] = width_90

        return metrics
