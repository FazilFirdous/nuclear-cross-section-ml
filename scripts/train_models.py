#!/usr/bin/env python3
"""
Train ML models for nuclear cross-section prediction.

Trains XGBoost, Random Forest, and Neural Network models on a processed
dataset, evaluates each model on the held-out test set, and saves trained
model artifacts.

Usage:
    python scripts/train_models.py --dataset data/processed/n-gamma_dataset.h5
    python scripts/train_models.py --dataset data/processed/n-gamma_dataset.h5 --model xgboost
    python scripts/train_models.py --dataset data/processed/n-gamma_dataset.h5 --run-loio
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.dataset_builder import DatasetBuilder, FEATURE_COLUMNS
from evaluation.metrics import (
    compute_metrics,
    mass_region_breakdown,
    compute_factor_of_x_accuracy,
    calibration_error,
)
from features.feature_engineering import FeatureEngineer
from models.ensemble import EnsembleCrossSectionModel
from models.neural_network import NeuralNetworkCrossSectionModel
from models.random_forest_model import RandomForestCrossSectionModel
from models.xgboost_model import XGBoostCrossSectionModel
from visualization.plots import (
    plot_predicted_vs_experimental,
    plot_feature_importance,
    plot_mass_region_accuracy,
    plot_uncertainty_calibration,
)
from loguru import logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train nuclear cross-section prediction models."
    )
    parser.add_argument(
        "--dataset", required=True,
        help="Path to HDF5 dataset file (output of build_dataset.py)"
    )
    parser.add_argument(
        "--model", default="all",
        choices=["xgboost", "random_forest", "neural_network", "ensemble", "all"],
        help="Which model(s) to train"
    )
    parser.add_argument(
        "--output", default="results/models",
        help="Directory to save trained models and metrics"
    )
    parser.add_argument(
        "--figures-dir", default="results/figures",
        help="Directory to save evaluation figures"
    )
    parser.add_argument(
        "--config", default="config/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--scale-features", action="store_true",
        help="Apply RobustScaler to features (required for neural network)"
    )
    parser.add_argument(
        "--run-loio", action="store_true",
        help="Run leave-one-isotope-out cross-validation (slow)"
    )
    parser.add_argument(
        "--loio-n-isotopes", type=int, default=50,
        help="Number of isotopes for LOIO-CV (default: 50)"
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def train_xgboost(X_train, y_train, X_val, y_val, config, feature_names):
    logger.info("Training XGBoost model...")
    params = config.get("models", {}).get("xgboost", {})
    model = XGBoostCrossSectionModel(
        params={k: v for k, v in params.items() if k != "quantile_alphas"},
        feature_names=feature_names,
        early_stopping_rounds=params.get("early_stopping_rounds", 50),
    )
    model.fit(X_train, y_train, X_val, y_val)
    return model


def train_random_forest(X_train, y_train, X_val, y_val, config, feature_names):
    logger.info("Training Random Forest model...")
    params = config.get("models", {}).get("random_forest", {})
    model = RandomForestCrossSectionModel(params=params, feature_names=feature_names)
    model.fit(X_train, y_train, X_val, y_val)
    return model


def train_neural_network(X_train, y_train, X_val, y_val, config, feature_names):
    logger.info("Training Neural Network model...")
    nn_config = config.get("models", {}).get("neural_network", {})
    model = NeuralNetworkCrossSectionModel(
        n_features=X_train.shape[1],
        hidden_dims=tuple(nn_config.get("hidden_dims", [256, 256, 128, 64])),
        dropout_rate=nn_config.get("dropout_rate", 0.2),
        learning_rate=nn_config.get("learning_rate", 1e-3),
        weight_decay=nn_config.get("weight_decay", 1e-4),
        batch_size=nn_config.get("batch_size", 256),
        max_epochs=nn_config.get("max_epochs", 200),
        patience=nn_config.get("patience", 20),
        mc_samples=nn_config.get("mc_dropout_samples", 100),
        feature_names=feature_names,
    )
    model.fit(X_train, y_train, X_val, y_val)
    return model


def evaluate_and_save(
    model,
    model_name: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
    A_test: np.ndarray,
    output_dir: Path,
    figures_dir: Path,
):
    """Evaluate model, save metrics and figures."""
    logger.info("Evaluating %s on test set...", model_name)

    preds = model.predict_with_uncertainty(X_test)
    y_pred_key = "mean" if "mean" in preds else "median"
    y_pred = preds[y_pred_key]

    # Point metrics
    metrics = compute_metrics(y_test, y_pred)
    factor_metrics = compute_factor_of_x_accuracy(y_test, y_pred)
    metrics.update(factor_metrics)

    # Calibration
    std_key = "std_total" if "std_total" in preds else "std"
    if std_key in preds:
        cal = calibration_error(y_test, y_pred, preds[std_key])
        metrics["ece"] = cal["ece"]

    # Mass region breakdown
    region_metrics = mass_region_breakdown(y_test, y_pred, A_test)

    # Log results
    logger.info("%s Test Results:", model_name)
    for k, v in sorted(metrics.items()):
        logger.info("  %s: %.4f", k, v)

    # Save metrics JSON
    metrics_path = output_dir / f"{model_name}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({"overall": metrics, "by_mass_region": region_metrics}, f, indent=2)

    # Figures
    lower_key = "lower_68" if "lower_68" in preds else None
    upper_key = "upper_68" if "upper_68" in preds else None

    fig = plot_predicted_vs_experimental(
        y_test, y_pred,
        y_lower=preds.get(lower_key) if lower_key else None,
        y_upper=preds.get(upper_key) if upper_key else None,
        A_values=A_test,
        title=f"{model_name}: Predicted vs Experimental",
        output_path=str(figures_dir / f"{model_name}_pred_vs_exp.pdf"),
    )

    # Feature importance
    if hasattr(model, "feature_importance"):
        try:
            imp_df = model.feature_importance()
            plot_feature_importance(
                imp_df,
                title=f"{model_name}: Feature Importance",
                output_path=str(figures_dir / f"{model_name}_feature_importance.pdf"),
            )
        except Exception as exc:
            logger.warning("Could not plot feature importance: %s", exc)

    # Mass region accuracy
    plot_mass_region_accuracy(
        region_metrics,
        metric="rmse",
        output_path=str(figures_dir / f"{model_name}_mass_region_rmse.pdf"),
    )

    # Calibration plot
    if "ece" in metrics:
        try:
            cal = calibration_error(y_test, y_pred, preds[std_key])
            plot_uncertainty_calibration(
                cal["expected_coverage"],
                cal["observed_coverage"],
                model_labels=[model_name],
                output_path=str(figures_dir / f"{model_name}_calibration.pdf"),
            )
        except Exception as exc:
            logger.warning("Could not plot calibration: %s", exc)

    return metrics


def main():
    args = parse_args()

    log_level = "DEBUG" if args.verbose else "INFO"
    logger.remove()
    logger.add(sys.stderr, level=log_level, colorize=True)

    output_dir = Path(args.output)
    figures_dir = Path(args.figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    config = {}
    if Path(args.config).exists():
        config = load_config(args.config)

    # Load dataset
    logger.info("Loading dataset from %s", args.dataset)
    splits = DatasetBuilder.load_from_hdf5(args.dataset)

    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    X_test, y_test = splits["test"]

    logger.info(
        "Dataset loaded: train=%d, val=%d, test=%d samples",
        len(X_train), len(X_val), len(X_test),
    )

    # Feature scaling (for neural network)
    engineer = FeatureEngineer(scale_features=args.scale_features)
    # Note: HDF5 already has imputed data; just scale
    if args.scale_features:
        logger.info("Applying RobustScaler to features")
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

    # Load Z, A for mass-region analysis (reload from HDF5)
    import h5py
    A_test = np.ones(len(X_test), dtype=int)
    with h5py.File(args.dataset) as f:
        if "test" in f and "A" in f["test"]:
            A_test = f["test"]["A"][:]

    feature_names = FEATURE_COLUMNS

    # Train models
    trained_models = {}
    all_metrics = {}

    models_to_train = (
        ["xgboost", "random_forest", "neural_network"]
        if args.model == "all"
        else [args.model]
    )

    for model_name in models_to_train:
        start_time = time.time()
        try:
            if model_name == "xgboost":
                model = train_xgboost(X_train, y_train, X_val, y_val, config, feature_names)
            elif model_name == "random_forest":
                model = train_random_forest(X_train, y_train, X_val, y_val, config, feature_names)
            elif model_name == "neural_network":
                model = train_neural_network(X_train, y_train, X_val, y_val, config, feature_names)
            else:
                logger.warning("Unknown model: %s", model_name)
                continue

            elapsed = time.time() - start_time
            logger.info("%s training completed in %.1fs", model_name, elapsed)

            # Save model
            model_path = output_dir / f"{model_name}.pkl"
            model.save(str(model_path))

            # Evaluate
            metrics = evaluate_and_save(
                model, model_name, X_test, y_test, A_test, output_dir, figures_dir
            )
            trained_models[model_name] = model
            all_metrics[model_name] = metrics

        except Exception as exc:
            logger.error("Failed to train/evaluate %s: %s", model_name, exc, exc_info=True)

    # Ensemble
    if args.model in ("ensemble", "all") and len(trained_models) >= 2:
        logger.info("Building ensemble from %d base models...", len(trained_models))
        ensemble = EnsembleCrossSectionModel(
            models=list(trained_models.values()),
            model_names=list(trained_models.keys()),
        )
        ensemble.fit_meta_learner(X_val, y_val)
        ensemble.save(str(output_dir / "ensemble.pkl"))

        ens_metrics = evaluate_and_save(
            ensemble, "ensemble", X_test, y_test, A_test, output_dir, figures_dir
        )
        all_metrics["ensemble"] = ens_metrics

    # Save comparison table
    comparison_path = output_dir / "model_comparison.json"
    with open(comparison_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info("Model comparison saved to %s", comparison_path)

    # Summary table
    logger.info("\nModel comparison summary (test set RMSE):")
    for name, m in all_metrics.items():
        logger.info("  %-20s RMSE=%.4f  R2=%.4f", name, m.get("rmse", float("nan")), m.get("r2", float("nan")))

    if args.run_loio:
        logger.info("Running leave-one-isotope-out cross-validation (this may take a while)...")
        # TODO: implement LOIO-CV pass using train+val combined
        logger.info("LOIO-CV not yet implemented in this script. Use notebooks/03_model_training.ipynb.")


if __name__ == "__main__":
    main()
