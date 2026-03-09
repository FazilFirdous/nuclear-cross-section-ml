#!/usr/bin/env python3
"""
Build ML-ready training dataset from raw EXFOR and nuclear properties data.

Usage:
    python scripts/build_dataset.py --reaction n-gamma
    python scripts/build_dataset.py --reaction n-gamma --output data/processed/ --cv-strategy leave_one_isotope_out
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.dataset_builder import DatasetBuilder, FEATURE_COLUMNS
from data.nuclear_properties import NuclearPropertiesLoader
from loguru import logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build feature-engineered ML training dataset from EXFOR data."
    )
    parser.add_argument(
        "--reaction", default="n-gamma",
        help="Reaction type to process (default: n-gamma)"
    )
    parser.add_argument(
        "--exfor-dir", default="data/raw/exfor",
        help="Directory containing raw EXFOR parquet files"
    )
    parser.add_argument(
        "--ame-path", default="data/external/ame2020_mass.txt",
        help="Path to AME2020 mass table"
    )
    parser.add_argument(
        "--nubase-path", default="data/external/nubase2020.txt",
        help="Path to NuBase2020 ground-state properties"
    )
    parser.add_argument(
        "--frdm-path", default="data/external/frdm2012_deformation.dat",
        help="Path to FRDM deformation table (optional)"
    )
    parser.add_argument(
        "--output", default="data/processed",
        help="Output directory for processed datasets"
    )
    parser.add_argument(
        "--test-fraction", type=float, default=0.15,
        help="Fraction of data for test set"
    )
    parser.add_argument(
        "--val-fraction", type=float, default=0.10,
        help="Fraction of data for validation set"
    )
    parser.add_argument(
        "--cv-strategy", default="leave_one_isotope_out",
        choices=["leave_one_isotope_out", "random"],
        help="Train/test split strategy"
    )
    parser.add_argument(
        "--min-points", type=int, default=3,
        help="Minimum data points per isotope to include"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed"
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    log_level = "DEBUG" if args.verbose else "INFO"
    logger.remove()
    logger.add(sys.stderr, level=log_level, colorize=True)

    # Load EXFOR data
    reaction_slug = args.reaction.replace(",", "_").replace("-", "_")
    exfor_path = Path(args.exfor_dir) / f"{reaction_slug}_all.parquet"

    if not exfor_path.exists():
        logger.error(
            "EXFOR data not found at %s. Run download_data.py first.", exfor_path
        )
        sys.exit(1)

    logger.info("Loading EXFOR data from %s", exfor_path)
    exfor_df = pd.read_parquet(exfor_path)
    logger.info("Loaded %d raw data points", len(exfor_df))

    # Load nuclear properties
    logger.info("Loading nuclear properties (AME2020, NuBase2020, FRDM)")
    props_loader = NuclearPropertiesLoader(
        ame_path=args.ame_path,
        nubase_path=args.nubase_path,
        frdm_path=args.frdm_path if Path(args.frdm_path).exists() else None,
    )
    nuclear_props = props_loader.load_all()
    logger.info("Loaded properties for %d isotopes", len(nuclear_props))

    # Build dataset
    builder = DatasetBuilder(nuclear_props=nuclear_props, output_dir=args.output)
    dataset = builder.build(
        exfor_data=exfor_df,
        reaction=args.reaction,
        min_points_per_isotope=args.min_points,
    )

    if dataset.empty:
        logger.error("Dataset is empty after processing.")
        sys.exit(1)

    logger.info("Built dataset with %d samples and %d features", len(dataset), len(FEATURE_COLUMNS))

    # Log feature completeness
    missing = dataset[FEATURE_COLUMNS].isnull().mean().sort_values(ascending=False)
    high_missing = missing[missing > 0.1]
    if not high_missing.empty:
        logger.warning("Features with >10%% missing values:")
        for feat, frac in high_missing.items():
            logger.warning("  %s: %.1f%%", feat, 100 * frac)

    # Split
    train_df, val_df, test_df = builder.split_train_val_test(
        dataset,
        test_fraction=args.test_fraction,
        val_fraction=args.val_fraction,
        strategy=args.cv_strategy,
        random_seed=args.seed,
    )

    # Save to HDF5
    output_path = builder.save_to_hdf5(train_df, val_df, test_df, reaction=args.reaction)
    logger.info("Dataset saved to %s", output_path)

    # Also save CSV for inspection
    csv_path = Path(args.output) / f"{reaction_slug}_dataset_train.csv"
    train_df.to_csv(csv_path, index=False)
    logger.info("Training set CSV saved to %s (for inspection)", csv_path)

    # Summary statistics
    logger.info("\nDataset summary:")
    logger.info("  Train: %d samples from %d isotopes",
                len(train_df), train_df[["Z", "A"]].drop_duplicates().shape[0])
    logger.info("  Val:   %d samples from %d isotopes",
                len(val_df), val_df[["Z", "A"]].drop_duplicates().shape[0])
    logger.info("  Test:  %d samples from %d isotopes",
                len(test_df), test_df[["Z", "A"]].drop_duplicates().shape[0])
    logger.info(
        "  Target range: [%.2f, %.2f] log10(barn)",
        dataset["log10_cross_section_barn"].min(),
        dataset["log10_cross_section_barn"].max(),
    )


if __name__ == "__main__":
    main()
