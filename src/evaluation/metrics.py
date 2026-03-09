"""
Evaluation metrics for cross-section prediction.

Implements standard regression metrics plus calibration metrics specific
to uncertainty quantification in nuclear data prediction.
"""

from typing import Dict, Optional

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_lower: Optional[np.ndarray] = None,
    y_upper: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute regression and interval coverage metrics.

    All inputs are in log10(cross-section/barn) space.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth log10 cross-sections.
    y_pred : np.ndarray
        Predicted log10 cross-sections (point estimate).
    y_lower, y_upper : np.ndarray, optional
        Lower and upper bounds of prediction interval.

    Returns
    -------
    dict with keys: rmse, mae, r2, mbe, median_abs_error,
                    picp (if intervals provided), mpiw (if intervals provided).
    """
    residuals = y_pred - y_true

    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "mbe": float(np.mean(residuals)),           # mean bias error
        "median_abs_error": float(np.median(np.abs(residuals))),
        "fraction_within_factor2": float(np.mean(np.abs(residuals) < np.log10(2))),
        "fraction_within_factor10": float(np.mean(np.abs(residuals) < 1.0)),
    }

    if y_lower is not None and y_upper is not None:
        coverage = np.mean((y_true >= y_lower) & (y_true <= y_upper))
        width = np.mean(y_upper - y_lower)
        metrics["picp"] = float(coverage)
        metrics["mpiw"] = float(width)
        # Winkler score: penalizes wide intervals and non-coverage
        alpha = 1 - coverage  # empirical alpha
        metrics["winkler_score"] = float(_winkler_score(y_true, y_lower, y_upper, alpha=0.68))

    return metrics


def _winkler_score(
    y_true: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
    alpha: float = 0.32,
) -> float:
    """
    Compute the Winkler score for prediction interval quality.

    Lower is better. The score penalizes wide intervals but awards coverage.
    """
    width = y_upper - y_lower
    below = (y_true < y_lower).astype(float)
    above = (y_true > y_upper).astype(float)
    penalty = (2 / alpha) * (below * (y_lower - y_true) + above * (y_true - y_upper))
    return float(np.mean(width + penalty))


def calibration_error(
    y_true: np.ndarray,
    y_pred_mean: np.ndarray,
    y_pred_std: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, np.ndarray]:
    """
    Compute calibration error for Gaussian uncertainty estimates.

    A well-calibrated model satisfies: for expected coverage p,
    the fraction of test points within p-sigma should equal p.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth values.
    y_pred_mean : np.ndarray
        Predicted mean values.
    y_pred_std : np.ndarray
        Predicted standard deviations.
    n_bins : int
        Number of coverage levels to evaluate.

    Returns
    -------
    dict with keys: expected_coverage, observed_coverage, ece (expected calibration error).
    """
    from scipy import stats

    expected_coverages = np.linspace(0.01, 0.99, n_bins)
    observed_coverages = np.zeros(n_bins)

    z_scores = (y_true - y_pred_mean) / (y_pred_std + 1e-8)

    for i, p in enumerate(expected_coverages):
        # z-score for p-coverage under standard normal
        z_critical = stats.norm.ppf((1 + p) / 2)
        observed_coverages[i] = np.mean(np.abs(z_scores) <= z_critical)

    ece = float(np.mean(np.abs(expected_coverages - observed_coverages)))

    return {
        "expected_coverage": expected_coverages,
        "observed_coverage": observed_coverages,
        "ece": ece,
    }


def mass_region_breakdown(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    A_values: np.ndarray,
    regions: Optional[Dict] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics broken down by nuclear mass region.

    Parameters
    ----------
    y_true, y_pred : np.ndarray
        Ground truth and predictions.
    A_values : np.ndarray
        Mass numbers for each data point.
    regions : dict, optional
        Mass region definitions: {name: (A_min, A_max)}.

    Returns
    -------
    dict mapping region name to metrics dict.
    """
    if regions is None:
        regions = {
            "light (A<40)": (1, 39),
            "medium (40<=A<=100)": (40, 100),
            "heavy (100<A<=209)": (101, 209),
            "actinide (A>209)": (210, 300),
        }

    results = {}
    for region_name, (a_min, a_max) in regions.items():
        mask = (A_values >= a_min) & (A_values <= a_max)
        if mask.sum() < 5:
            results[region_name] = {"n_samples": int(mask.sum())}
            continue
        results[region_name] = {
            "n_samples": int(mask.sum()),
            **compute_metrics(y_true[mask], y_pred[mask]),
        }

    return results


def compute_factor_of_x_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    factors: Optional[list] = None,
) -> Dict[str, float]:
    """
    Compute the fraction of predictions within a factor X of the true value.

    This is the standard metric in nuclear data evaluation: "within a factor of 2"
    means the predicted cross-section is between true/2 and 2*true.

    Parameters
    ----------
    y_true, y_pred : np.ndarray
        Log10 cross-sections.
    factors : list of float
        Factors X to evaluate. Defaults to [1.5, 2, 3, 5, 10].

    Returns
    -------
    dict mapping "within_factor_X" to fraction of predictions within that factor.
    """
    if factors is None:
        factors = [1.5, 2.0, 3.0, 5.0, 10.0]

    residuals = np.abs(y_pred - y_true)  # in log10 space
    result = {}
    for x in factors:
        threshold = np.log10(x)
        result[f"within_factor_{x:.1f}"] = float(np.mean(residuals < threshold))

    return result
