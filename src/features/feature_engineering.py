"""
Feature engineering for nuclear cross-section prediction.

Constructs the nuclear structure feature vector from raw nuclear properties
and prepares it for ML model ingestion, including imputation and scaling.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

logger = logging.getLogger(__name__)

# Ordered list of feature columns fed to ML models.
# This ordering must be consistent between training and inference.
FEATURE_COLUMNS = [
    "Z",
    "N",
    "A",
    "Z_even",
    "N_even",
    "isospin_asym",
    "surface_area",
    "binding_energy_per_A_MeV",
    "Sn_MeV",
    "S2n_MeV",
    "Sp_MeV",
    "S2p_MeV",
    "delta_n",
    "delta_p",
    "spin",
    "parity",
    "beta2",
    "beta4",
    "dist_to_magic_Z",
    "dist_to_magic_N",
    "Q_value_MeV",
    "log10_energy_eV",
]


class FeatureEngineer:
    """
    Preprocesses raw nuclear property DataFrames into model-ready feature matrices.

    Handles:
    - Column selection and ordering
    - Missing value imputation (KNN imputation using nearby isotopes on nuclear chart)
    - Robust scaling to reduce sensitivity to outliers

    Parameters
    ----------
    impute_n_neighbors : int
        Number of neighbors for KNN imputation of missing nuclear properties.
    scale_features : bool
        Whether to apply RobustScaler. Tree-based models do not require scaling,
        but neural networks benefit from it.
    """

    def __init__(
        self,
        impute_n_neighbors: int = 5,
        scale_features: bool = True,
    ):
        self.impute_n_neighbors = impute_n_neighbors
        self.scale_features = scale_features
        self._imputer: Optional[KNNImputer] = None
        self._scaler: Optional[RobustScaler] = None
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> "FeatureEngineer":
        """
        Fit imputer and scaler on training data.

        Parameters
        ----------
        df : pd.DataFrame
            Training dataset containing FEATURE_COLUMNS.

        Returns
        -------
        self
        """
        X = self._select_features(df)
        self._imputer = KNNImputer(
            n_neighbors=self.impute_n_neighbors,
            weights="distance",
        )
        X_imputed = self._imputer.fit_transform(X)

        if self.scale_features:
            self._scaler = RobustScaler()
            self._scaler.fit(X_imputed)

        self._fitted = True
        logger.info(
            "FeatureEngineer fitted on %d samples, %d features. "
            "Missing rate before imputation: %.1f%%.",
            len(df),
            len(FEATURE_COLUMNS),
            100.0 * np.isnan(X).mean(),
        )
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform a DataFrame into a scaled, imputed feature matrix.

        Parameters
        ----------
        df : pd.DataFrame
            Dataset containing FEATURE_COLUMNS.

        Returns
        -------
        np.ndarray of shape (n_samples, n_features).
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")

        X = self._select_features(df)
        X_imputed = self._imputer.transform(X)

        if self.scale_features and self._scaler is not None:
            return self._scaler.transform(X_imputed)
        return X_imputed

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit on df and return transformed array."""
        return self.fit(df).transform(df)

    def _select_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Select and order FEATURE_COLUMNS from df.

        Missing columns are filled with NaN so downstream imputation handles them.
        """
        result = np.full((len(df), len(FEATURE_COLUMNS)), np.nan, dtype=np.float64)
        for i, col in enumerate(FEATURE_COLUMNS):
            if col in df.columns:
                result[:, i] = pd.to_numeric(df[col], errors="coerce").values
        return result

    def get_feature_names(self) -> List[str]:
        """Return the ordered list of feature names."""
        return list(FEATURE_COLUMNS)

    def missing_rate_report(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute per-feature missing value rates for diagnostics.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        pd.Series indexed by feature name with missing fractions.
        """
        X = self._select_features(df)
        return pd.Series(
            np.isnan(X).mean(axis=0),
            index=FEATURE_COLUMNS,
        ).sort_values(ascending=False)


def add_physics_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add physics-motivated interaction features to a feature DataFrame.

    These second-order features capture known correlations in nuclear physics:
    - Proximity to double-magic nuclei (Z and N both near magic numbers)
    - Wigner energy term for N=Z nuclei
    - Shell gap products

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame containing at least Z, N, A columns.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with additional interaction feature columns appended.
    """
    df = df.copy()

    # Double-magic proximity
    if "dist_to_magic_Z" in df.columns and "dist_to_magic_N" in df.columns:
        df["double_magic_proximity"] = np.sqrt(
            df["dist_to_magic_Z"] ** 2 + df["dist_to_magic_N"] ** 2
        )

    # Wigner energy indicator: enhanced binding near N=Z
    df["wigner_nz"] = np.exp(-((df["N"] - df["Z"]) / df["A"]) ** 2 / 0.01)

    # Deformation magnitude
    if "beta2" in df.columns and "beta4" in df.columns:
        df["deformation_magnitude"] = np.sqrt(
            df["beta2"].fillna(0) ** 2 + df["beta4"].fillna(0) ** 2
        )

    # Shell energy scale: Z*(Z-1)/A^(1/3) captures Coulomb energy
    df["coulomb_term"] = df["Z"] * (df["Z"] - 1) / (df["A"] ** (1.0 / 3.0))

    # Asymmetry energy: (N-Z)^2 / A
    df["asymmetry_energy"] = (df["N"] - df["Z"]) ** 2 / df["A"]

    return df
