"""
Publication-quality plots for nuclear cross-section prediction results.

All figures are designed for submission to physical review journals:
- 300 DPI minimum
- Colorblind-accessible palettes (ColorBrewer)
- No chart junk (Tufte principles)
- All axes labeled with units
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "text.usetex": False,
})

logger = logging.getLogger(__name__)

# ColorBrewer qualitative palette (colorblind-friendly)
PALETTE = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e"]


def plot_predicted_vs_experimental(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_lower: Optional[np.ndarray] = None,
    y_upper: Optional[np.ndarray] = None,
    A_values: Optional[np.ndarray] = None,
    title: str = "",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot predicted vs experimental log10 cross-sections (parity plot).

    Points are optionally colored by mass number. Prediction intervals
    are shown as error bars. Factor-of-2 and factor-of-10 bands are shaded.

    Parameters
    ----------
    y_true : np.ndarray
        Experimental log10(sigma/barn).
    y_pred : np.ndarray
        Predicted log10(sigma/barn).
    y_lower, y_upper : np.ndarray, optional
        Prediction interval bounds.
    A_values : np.ndarray, optional
        Mass numbers for coloring.
    title : str
        Plot title.
    output_path : str, optional
        If provided, save figure to this path.

    Returns
    -------
    matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    lim_min = min(y_true.min(), y_pred.min()) - 0.5
    lim_max = max(y_true.max(), y_pred.max()) + 0.5

    # Factor-of-2 and factor-of-10 bands
    x_band = np.linspace(lim_min, lim_max, 200)
    ax.fill_between(
        x_band, x_band - np.log10(10), x_band + np.log10(10),
        alpha=0.08, color="gray", label="Factor of 10"
    )
    ax.fill_between(
        x_band, x_band - np.log10(2), x_band + np.log10(2),
        alpha=0.15, color="steelblue", label="Factor of 2"
    )

    # 1:1 line
    ax.plot(x_band, x_band, "k-", linewidth=1.0, zorder=3)

    # Data points
    if y_lower is not None and y_upper is not None:
        err_lo = y_pred - y_lower
        err_hi = y_upper - y_pred
        ax.errorbar(
            y_true, y_pred,
            yerr=[err_lo, err_hi],
            fmt="o", markersize=3, alpha=0.5, color=PALETTE[0],
            elinewidth=0.5, capsize=1, zorder=4, label="ML prediction"
        )
    elif A_values is not None:
        scatter = ax.scatter(
            y_true, y_pred, c=A_values, cmap="viridis",
            s=10, alpha=0.6, zorder=4
        )
        cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Mass number A")
    else:
        ax.scatter(
            y_true, y_pred, s=8, alpha=0.5, color=PALETTE[0], zorder=4,
            label="ML prediction"
        )

    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_xlabel("Experimental log$_{10}$($\\sigma$ / barn)")
    ax.set_ylabel("Predicted log$_{10}$($\\sigma$ / barn)")
    ax.legend(loc="upper left", framealpha=0.9, edgecolor="none")

    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2)
    ax.text(
        0.97, 0.05,
        f"RMSE = {rmse:.3f}\n$R^2$ = {r2:.3f}",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=10, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    if title:
        ax.set_title(title)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path)
        logger.info("Saved predicted vs experimental plot to %s", output_path)

    return fig


def plot_nuclear_chart_heatmap(
    Z_values: np.ndarray,
    N_values: np.ndarray,
    values: np.ndarray,
    value_label: str = "log$_{10}$($\\sigma$ / barn)",
    title: str = "",
    log_scale: bool = False,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot a heatmap on the (N, Z) nuclear chart.

    Useful for visualizing predictions, uncertainties, or residuals
    across the chart of nuclides.

    Parameters
    ----------
    Z_values, N_values : np.ndarray
        Proton and neutron numbers.
    values : np.ndarray
        Values to display.
    value_label : str
        Colorbar label.
    title : str
        Plot title.
    log_scale : bool
        If True, use logarithmic color scale.
    output_path : str, optional
        Save path.

    Returns
    -------
    matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    # Magic numbers for reference lines
    magic = [2, 8, 20, 28, 50, 82, 126]

    scatter_kwargs = {
        "c": values,
        "cmap": "RdYlGn_r" if "residual" in value_label.lower() else "viridis",
        "s": 15,
        "marker": "s",
    }
    if log_scale and values.min() > 0:
        scatter_kwargs["norm"] = LogNorm(vmin=values.min(), vmax=values.max())
    else:
        vmax = max(abs(values.min()), abs(values.max()))
        if "residual" in value_label.lower():
            scatter_kwargs["vmin"] = -vmax
            scatter_kwargs["vmax"] = vmax

    sc = ax.scatter(N_values, Z_values, **scatter_kwargs)
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(value_label)

    # Magic number lines
    for m in magic:
        ax.axvline(m, color="gray", linewidth=0.3, alpha=0.6, linestyle=":")
        ax.axhline(m, color="gray", linewidth=0.3, alpha=0.6, linestyle=":")

    ax.set_xlabel("Neutron number N")
    ax.set_ylabel("Proton number Z")
    if title:
        ax.set_title(title)

    # Label magic numbers
    for m in magic:
        ax.text(m + 0.5, ax.get_ylim()[0] + 0.5, str(m), fontsize=7, color="gray")
        ax.text(ax.get_xlim()[0] + 0.5, m + 0.2, str(m), fontsize=7, color="gray")

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path)
        logger.info("Saved nuclear chart heatmap to %s", output_path)

    return fig


def plot_cross_section_energy_dependence(
    energies_eV: np.ndarray,
    sigma_pred: np.ndarray,
    sigma_lower: Optional[np.ndarray] = None,
    sigma_upper: Optional[np.ndarray] = None,
    sigma_exfor: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    sigma_endf: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    sigma_talys: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    isotope_label: str = "",
    reaction: str = "(n,gamma)",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot energy-dependent cross-section with comparison to theory and experiment.

    Parameters
    ----------
    energies_eV : np.ndarray
        Incident neutron energies in eV.
    sigma_pred : np.ndarray
        ML-predicted cross-sections (in barn, linear scale).
    sigma_lower, sigma_upper : np.ndarray, optional
        Prediction interval bounds.
    sigma_exfor : tuple (energies, cross_sections), optional
        EXFOR experimental data points.
    sigma_endf : tuple (energies, cross_sections), optional
        ENDF/B evaluated curve.
    sigma_talys : tuple (energies, cross_sections), optional
        TALYS Hauser-Feshbach calculation.
    isotope_label : str
        Nucleus label, e.g., "^{197}Au".
    reaction : str
        Reaction label for axis title.
    output_path : str, optional
        Save path.

    Returns
    -------
    matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # ML prediction
    ax.loglog(energies_eV, sigma_pred, color=PALETTE[0], linewidth=2.0,
              label="ML (this work)", zorder=5)

    if sigma_lower is not None and sigma_upper is not None:
        ax.fill_between(
            energies_eV, sigma_lower, sigma_upper,
            alpha=0.25, color=PALETTE[0], label="68% prediction interval"
        )

    if sigma_endf is not None:
        ax.loglog(sigma_endf[0], sigma_endf[1], color="black", linewidth=1.5,
                  linestyle="-", label="ENDF/B-VIII.0", zorder=4)

    if sigma_talys is not None:
        ax.loglog(sigma_talys[0], sigma_talys[1], color=PALETTE[1], linewidth=1.5,
                  linestyle="--", label="TALYS (Hauser-Feshbach)", zorder=3)

    if sigma_exfor is not None:
        ax.scatter(
            sigma_exfor[0], sigma_exfor[1],
            marker="o", s=20, color="black", zorder=6,
            label="EXFOR experimental", alpha=0.7
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Incident neutron energy (eV)")
    ax.set_ylabel("Cross-section (barn)")
    ax.legend(framealpha=0.9, edgecolor="none")
    ax.grid(True, which="both", alpha=0.3, linewidth=0.5)

    title_str = f"{isotope_label} {reaction}" if isotope_label else reaction
    ax.set_title(title_str)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path)

    return fig


def plot_uncertainty_calibration(
    expected_coverages: np.ndarray,
    observed_coverages: np.ndarray,
    model_labels: Optional[List[str]] = None,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot uncertainty calibration (reliability diagram).

    A perfectly calibrated model falls on the diagonal. Points above the
    diagonal indicate overconfident intervals; below indicates underconfident.

    Parameters
    ----------
    expected_coverages : np.ndarray or list of np.ndarray
        Expected coverage probabilities (x-axis).
    observed_coverages : np.ndarray or list of np.ndarray
        Observed coverage fractions (y-axis).
    model_labels : list of str, optional
        Labels for multiple models on the same plot.
    output_path : str, optional

    Returns
    -------
    matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(5, 5))

    # Diagonal reference
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.0, label="Perfect calibration")
    ax.fill_between([0, 1], [0, 1], [1, 1], alpha=0.05, color="red", label="Overconfident region")
    ax.fill_between([0, 1], [0, 0], [0, 1], alpha=0.05, color="blue", label="Underconfident region")

    # Handle single or multiple model comparison
    if isinstance(expected_coverages, np.ndarray) and expected_coverages.ndim == 1:
        expected_list = [expected_coverages]
        observed_list = [observed_coverages]
        labels = model_labels or ["Model"]
    else:
        expected_list = expected_coverages
        observed_list = observed_coverages
        labels = model_labels or [f"Model {i}" for i in range(len(expected_list))]

    for exp, obs, label, color in zip(expected_list, observed_list, labels, PALETTE):
        ax.plot(exp, obs, "o-", color=color, linewidth=1.5, markersize=5, label=label)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Expected coverage probability")
    ax.set_ylabel("Observed coverage fraction")
    ax.legend(framealpha=0.9, edgecolor="none", loc="upper left")
    ax.set_aspect("equal")

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path)

    return fig


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 15,
    title: str = "Feature Importance",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Horizontal bar chart of feature importances.

    Parameters
    ----------
    importance_df : pd.DataFrame
        DataFrame with "feature" and "importance_mean" columns.
        Optional "importance_std" column for error bars.
    top_n : int
        Show only top N features.
    title, output_path : str

    Returns
    -------
    matplotlib Figure.
    """
    df = importance_df.head(top_n).copy()
    df = df.sort_values("importance_mean", ascending=True)

    fig, ax = plt.subplots(figsize=(7, max(4, top_n * 0.35)))

    xerr = df["importance_std"].values if "importance_std" in df.columns else None
    ax.barh(
        df["feature"], df["importance_mean"],
        xerr=xerr, color=PALETTE[0], alpha=0.8, edgecolor="none",
        error_kw={"elinewidth": 1.0, "capsize": 3, "ecolor": "gray"},
    )

    ax.set_xlabel("Feature importance")
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path)

    return fig


def plot_mass_region_accuracy(
    region_metrics: Dict[str, Dict],
    metric: str = "rmse",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Bar chart showing prediction accuracy by mass region.

    Parameters
    ----------
    region_metrics : dict
        Output of evaluation.metrics.mass_region_breakdown().
    metric : str
        Metric to plot ("rmse", "r2", "mae", etc.).
    output_path : str, optional

    Returns
    -------
    matplotlib Figure.
    """
    regions = [r for r, m in region_metrics.items() if metric in m]
    values = [region_metrics[r][metric] for r in regions]
    n_samples = [region_metrics[r].get("n_samples", 0) for r in regions]

    fig, ax = plt.subplots(figsize=(7, 4))

    bars = ax.bar(regions, values, color=PALETTE[:len(regions)], alpha=0.85, edgecolor="none")

    # Annotate with sample counts
    for bar, n in zip(bars, n_samples):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005 * max(values),
            f"n={n}",
            ha="center", va="bottom", fontsize=9,
        )

    metric_labels = {
        "rmse": "RMSE (log$_{10}$ barn)",
        "mae": "MAE (log$_{10}$ barn)",
        "r2": "$R^2$",
        "fraction_within_factor2": "Fraction within factor 2",
    }
    ax.set_ylabel(metric_labels.get(metric, metric))
    ax.set_title(f"Prediction accuracy by mass region ({metric.upper()})")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xticks(rotation=20, ha="right")

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path)

    return fig
