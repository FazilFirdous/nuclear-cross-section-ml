"""
Ensemble model combining XGBoost, Random Forest, and Neural Network predictions.

Implements a stacking ensemble where a meta-learner (linear regression) combines
base model predictions, and uncertainty is propagated from base models.
"""

import logging
import pickle
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

from .neural_network import NeuralNetworkCrossSectionModel
from .random_forest_model import RandomForestCrossSectionModel
from .xgboost_model import XGBoostCrossSectionModel

logger = logging.getLogger(__name__)

BaseModel = Union[
    XGBoostCrossSectionModel,
    RandomForestCrossSectionModel,
    NeuralNetworkCrossSectionModel,
]


class EnsembleCrossSectionModel:
    """
    Stacking ensemble of XGBoost, Random Forest, and Neural Network models.

    A Ridge regression meta-learner is trained on out-of-fold base model
    predictions to combine them optimally. Uncertainty from each base model
    is propagated through the ensemble.

    Parameters
    ----------
    models : list of fitted base models
        Pre-trained base models to combine.
    model_names : list of str
        Labels for each model (for logging and output).
    meta_alpha : float
        Ridge regression regularization for the meta-learner.
    """

    def __init__(
        self,
        models: Optional[List[BaseModel]] = None,
        model_names: Optional[List[str]] = None,
        meta_alpha: float = 1.0,
    ):
        self.models = models or []
        self.model_names = model_names or [f"model_{i}" for i in range(len(self.models))]
        self.meta_alpha = meta_alpha
        self._meta_learner: Optional[Ridge] = None
        self._fitted = False

    def add_model(self, model: BaseModel, name: str) -> None:
        """Add a base model to the ensemble."""
        self.models.append(model)
        self.model_names.append(name)

    def fit_meta_learner(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> "EnsembleCrossSectionModel":
        """
        Fit the meta-learner on validation set base model predictions.

        Parameters
        ----------
        X_val : np.ndarray
            Validation features (not used to train base models).
        y_val : np.ndarray
            Validation targets.

        Returns
        -------
        self
        """
        base_preds = self._get_base_predictions(X_val)
        meta_X = np.stack(base_preds, axis=1)  # (n_val, n_models)

        self._meta_learner = Ridge(alpha=self.meta_alpha)
        self._meta_learner.fit(meta_X, y_val)
        self._fitted = True

        meta_pred = self._meta_learner.predict(meta_X)
        val_rmse = np.sqrt(mean_squared_error(y_val, meta_pred))
        logger.info(
            "Ensemble meta-learner fitted. Val RMSE=%.4f. Weights: %s",
            val_rmse,
            dict(zip(self.model_names, self._meta_learner.coef_.round(3))),
        )
        return self

    def _get_base_predictions(self, X: np.ndarray) -> List[np.ndarray]:
        """Collect point predictions from each base model."""
        preds = []
        for model, name in zip(self.models, self.model_names):
            try:
                p = model.predict(X)
                preds.append(p)
            except Exception as exc:
                logger.warning("Base model %s prediction failed: %s", name, exc)
                preds.append(np.zeros(len(X)))
        return preds

    def _get_base_uncertainties(self, X: np.ndarray) -> List[np.ndarray]:
        """Collect standard deviation estimates from each base model."""
        stds = []
        for model, name in zip(self.models, self.model_names):
            try:
                if hasattr(model, "predict_with_uncertainty"):
                    result = model.predict_with_uncertainty(X)
                    std = result.get("std", result.get("sigma_68", np.zeros(len(X))))
                else:
                    std = np.zeros(len(X))
                stds.append(std)
            except Exception as exc:
                logger.warning("Base model %s uncertainty failed: %s", name, exc)
                stds.append(np.zeros(len(X)))
        return stds

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return ensemble point prediction."""
        if not self._fitted:
            # Fall back to simple average if meta-learner not fitted
            base_preds = self._get_base_predictions(X)
            return np.stack(base_preds, axis=1).mean(axis=1)

        base_preds = self._get_base_predictions(X)
        meta_X = np.stack(base_preds, axis=1)
        return self._meta_learner.predict(meta_X)

    def predict_with_uncertainty(
        self, X: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Ensemble prediction with propagated uncertainty.

        Total uncertainty is estimated as the quadrature sum of:
        1. Epistemic uncertainty from base model spread (disagreement between models)
        2. Aleatoric uncertainty averaged from base model internal estimates

        Parameters
        ----------
        X : np.ndarray

        Returns
        -------
        dict with mean, std_total, std_epistemic, std_aleatoric, lower/upper bounds.
        """
        base_preds = self._get_base_predictions(X)
        base_stds = self._get_base_uncertainties(X)

        preds_array = np.stack(base_preds, axis=1)  # (n, n_models)
        stds_array = np.stack(base_stds, axis=1)     # (n, n_models)

        if self._fitted and self._meta_learner is not None:
            weights = np.abs(self._meta_learner.coef_)
            weights = weights / weights.sum()
            mean = np.average(preds_array, axis=1, weights=weights)
        else:
            mean = preds_array.mean(axis=1)

        # Epistemic uncertainty: disagreement between models
        std_epistemic = preds_array.std(axis=1)

        # Aleatoric uncertainty: mean of base model internal estimates
        std_aleatoric = stds_array.mean(axis=1)

        # Total uncertainty (in quadrature)
        std_total = np.sqrt(std_epistemic ** 2 + std_aleatoric ** 2)

        return {
            "mean": mean,
            "std_total": std_total,
            "std_epistemic": std_epistemic,
            "std_aleatoric": std_aleatoric,
            "lower_68": mean - std_total,
            "upper_68": mean + std_total,
            "lower_95": mean - 2 * std_total,
            "upper_95": mean + 2 * std_total,
            "base_predictions": {
                name: pred for name, pred in zip(self.model_names, base_preds)
            },
        }

    def evaluate(
        self, X_test: np.ndarray, y_test: np.ndarray
    ) -> Dict[str, float]:
        """Compute evaluation metrics for the ensemble."""
        preds = self.predict_with_uncertainty(X_test)
        y_pred = preds["mean"]

        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(np.mean(np.abs(y_test - y_pred)))
        r2 = float(r2_score(y_test, y_pred))

        coverage_68 = float(np.mean(
            (y_test >= preds["lower_68"]) & (y_test <= preds["upper_68"])
        ))
        coverage_95 = float(np.mean(
            (y_test >= preds["lower_95"]) & (y_test <= preds["upper_95"])
        ))

        metrics = {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "picp_68": coverage_68,
            "picp_95": coverage_95,
            "mean_epistemic_std": float(np.mean(preds["std_epistemic"])),
            "mean_aleatoric_std": float(np.mean(preds["std_aleatoric"])),
        }

        # Per-base-model metrics
        for name, base_pred in preds["base_predictions"].items():
            metrics[f"rmse_{name}"] = float(
                np.sqrt(mean_squared_error(y_test, base_pred))
            )

        return metrics

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("Saved ensemble to %s", path)

    @classmethod
    def load(cls, path: str) -> "EnsembleCrossSectionModel":
        with open(path, "rb") as f:
            return pickle.load(f)
