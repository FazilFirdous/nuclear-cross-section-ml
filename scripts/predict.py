#!/usr/bin/env python3
"""
Generate cross-section predictions for unmeasured isotopes.

Loads a trained model and produces predictions with uncertainty estimates
for a user-specified list of isotopes at specified energies.

Usage:
    python scripts/predict.py --model results/models/xgboost.pkl \
        --z 79 --a 197 --reaction n-gamma \
        --energies 1e-3 1e-2 0.1 1.0 1e6

    python scripts/predict.py --model results/models/ensemble.pkl \
        --isotopes-file isotopes.txt \
        --energies-file energies.txt \
        --output results/predictions/Au197_n-gamma.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.nuclear_properties import NuclearPropertiesLoader
from data.dataset_builder import FEATURE_COLUMNS
from loguru import logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict neutron cross-sections for unmeasured isotopes."
    )

    # Model
    parser.add_argument("--model", required=True, help="Path to trained model (.pkl)")
    parser.add_argument(
        "--model-type", default="auto",
        choices=["auto", "xgboost", "random_forest", "neural_network", "ensemble"],
        help="Model type (auto-detected from file if 'auto')"
    )

    # Target isotope(s)
    isotope_group = parser.add_mutually_exclusive_group()
    isotope_group.add_argument("--z", type=int, help="Proton number Z")
    parser.add_argument("--a", type=int, help="Mass number A")
    isotope_group.add_argument(
        "--isotopes-file", help="Text file with one Z A per line"
    )

    # Energies
    energy_group = parser.add_mutually_exclusive_group()
    energy_group.add_argument(
        "--energies", nargs="+", type=float,
        help="Incident neutron energies in eV"
    )
    energy_group.add_argument(
        "--energies-file", help="Text file with one energy (eV) per line"
    )
    energy_group.add_argument(
        "--energy-grid", nargs=3, type=float, metavar=("E_MIN_EV", "E_MAX_EV", "N_POINTS"),
        help="Log-uniform energy grid: E_min E_max N_points (in eV)"
    )

    # Nuclear properties
    parser.add_argument(
        "--ame-path", default="data/external/ame2020_mass.txt"
    )
    parser.add_argument(
        "--nubase-path", default="data/external/nubase2020.txt"
    )
    parser.add_argument(
        "--frdm-path", default="data/external/frdm2012_deformation.dat"
    )

    # Reaction
    parser.add_argument("--reaction", default="n-gamma")

    # Output
    parser.add_argument(
        "--output", default="results/predictions/predictions.csv",
        help="Output file path (.csv or .h5)"
    )

    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()


def load_model(model_path: str, model_type: str = "auto"):
    """Load a trained model from disk."""
    import pickle

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    logger.info("Loaded model: %s", type(model).__name__)
    return model


def build_feature_vector(
    Z: int, A: int,
    energy_eV: float,
    reaction: str,
    nuclear_props_df: pd.DataFrame,
    props_loader: NuclearPropertiesLoader,
) -> dict:
    """Build a single feature vector for a (Z, A, energy) prediction point."""
    props = props_loader.get_features_for_isotope(Z, A, nuclear_props_df)

    features = {}
    for col in FEATURE_COLUMNS:
        features[col] = props.get(col, np.nan)

    features["log10_energy_eV"] = np.log10(max(energy_eV, 1e-15))
    features["Z"] = Z
    features["N"] = A - Z
    features["A"] = A

    # Q-value (approximate from mass table if available)
    features["Q_value_MeV"] = np.nan  # Will be filled by DatasetBuilder if needed

    return features


def main():
    args = parse_args()

    log_level = "DEBUG" if args.verbose else "INFO"
    logger.remove()
    logger.add(sys.stderr, level=log_level, colorize=True)

    # Load model
    model = load_model(args.model, args.model_type)

    # Load nuclear properties
    logger.info("Loading nuclear properties...")
    props_loader = NuclearPropertiesLoader(
        ame_path=args.ame_path,
        nubase_path=args.nubase_path,
        frdm_path=args.frdm_path if Path(args.frdm_path).exists() else None,
    )
    nuclear_props_df = props_loader.load_all()

    # Build isotope list
    if args.isotopes_file:
        isotopes = []
        with open(args.isotopes_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    isotopes.append((int(parts[0]), int(parts[1])))
    else:
        if args.z is None or args.a is None:
            logger.error("Specify --z and --a, or --isotopes-file")
            sys.exit(1)
        isotopes = [(args.z, args.a)]

    # Build energy grid
    if args.energy_grid:
        e_min, e_max, n_pts = args.energy_grid
        energies = np.logspace(np.log10(e_min), np.log10(e_max), int(n_pts))
    elif args.energies_file:
        with open(args.energies_file) as f:
            energies = np.array([float(line.strip()) for line in f if line.strip()])
    elif args.energies:
        energies = np.array(args.energies)
    else:
        # Default energy grid: thermal to 20 MeV in 100 log-spaced points
        energies = np.logspace(-2, 7.3, 100)

    logger.info(
        "Predicting for %d isotopes at %d energy points (%d total)",
        len(isotopes), len(energies), len(isotopes) * len(energies)
    )

    # Build feature matrix
    records = []
    for Z, A in isotopes:
        for energy_eV in energies:
            feat = build_feature_vector(
                Z, A, energy_eV, args.reaction, nuclear_props_df, props_loader
            )
            feat["energy_eV"] = energy_eV
            records.append(feat)

    feat_df = pd.DataFrame(records)
    X = feat_df[FEATURE_COLUMNS].values.astype(np.float32)

    # Handle NaN (simple median imputation for prediction)
    for j in range(X.shape[1]):
        col = X[:, j]
        nan_mask = np.isnan(col)
        if nan_mask.any():
            median_val = np.nanmedian(col)
            X[nan_mask, j] = median_val if not np.isnan(median_val) else 0.0

    # Predict
    logger.info("Running model inference...")
    try:
        preds = model.predict_with_uncertainty(X)
    except AttributeError:
        # Fallback for models without uncertainty
        y_pred = model.predict(X)
        preds = {"mean": y_pred}

    # Build output DataFrame
    pred_key = "mean" if "mean" in preds else "median"
    results = feat_df[["Z", "N", "A"]].copy()
    results["energy_eV"] = feat_df["energy_eV"]
    results["reaction"] = args.reaction
    results[f"log10_sigma_barn_pred"] = preds[pred_key]
    results["sigma_barn_pred"] = 10 ** preds[pred_key]

    if "lower_68" in preds:
        results["sigma_barn_lower_68"] = 10 ** preds["lower_68"]
        results["sigma_barn_upper_68"] = 10 ** preds["upper_68"]
    if "lower_95" in preds:
        results["sigma_barn_lower_95"] = 10 ** preds["lower_95"]
        results["sigma_barn_upper_95"] = 10 ** preds["upper_95"]
    if "std" in preds:
        results["log10_sigma_std"] = preds["std"]

    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix == ".h5":
        results.to_hdf(str(output_path), key="predictions", mode="w")
    else:
        results.to_csv(output_path, index=False)

    logger.info("Predictions saved to %s (%d rows)", output_path, len(results))

    # Print summary for single-isotope predictions
    if len(isotopes) == 1:
        Z, A = isotopes[0]
        logger.info("\nPredicted cross-sections for Z=%d A=%d (%s):", Z, A, args.reaction)
        sample = results[["energy_eV", "sigma_barn_pred"]].head(10)
        for _, row in sample.iterrows():
            logger.info("  E = %.3e eV: sigma = %.3e barn", row["energy_eV"], row["sigma_barn_pred"])


if __name__ == "__main__":
    main()
