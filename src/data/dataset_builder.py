"""
Training dataset builder.

Combines EXFOR experimental cross-section measurements with nuclear structure
properties to create the feature matrix and target vector used for ML training.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

logger = logging.getLogger(__name__)

# Feature columns used in the ML model input vector
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

TARGET_COLUMN = "log10_cross_section_barn"

# Neutron mass excess in keV for Q-value computation
NEUTRON_MASS_EXCESS_KEV = 8071.317


class DatasetBuilder:
    """
    Builds training datasets by merging EXFOR cross-sections with nuclear properties.

    Parameters
    ----------
    nuclear_props : pd.DataFrame
        Table of nuclear properties from NuclearPropertiesLoader.
    output_dir : str or Path
        Directory to write processed dataset files.
    """

    def __init__(
        self,
        nuclear_props: pd.DataFrame,
        output_dir: str = "data/processed",
    ):
        self.nuclear_props = nuclear_props
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build(
        self,
        exfor_data: pd.DataFrame,
        reaction: str,
        energy_bins_per_decade: int = 5,
        min_points_per_isotope: int = 3,
    ) -> pd.DataFrame:
        """
        Build a feature-engineered dataset from EXFOR data for one reaction type.

        Parameters
        ----------
        exfor_data : pd.DataFrame
            Raw EXFOR data for the reaction (from EXFORDownloader.download_reaction).
        reaction : str
            Reaction label (e.g., "n-gamma").
        energy_bins_per_decade : int
            For isotopes with many measurements, bin data to this density per decade.
        min_points_per_isotope : int
            Minimum data points required to include an isotope in the dataset.

        Returns
        -------
        pd.DataFrame
            Feature matrix with TARGET_COLUMN appended.
        """
        if exfor_data.empty:
            logger.warning("Empty EXFOR data for reaction %s", reaction)
            return pd.DataFrame()

        logger.info(
            "Building dataset for %s: %d raw data points", reaction, len(exfor_data)
        )

        # Merge nuclear properties
        df = exfor_data.merge(
            self.nuclear_props,
            on=["Z", "A", "N"],
            how="left",
            suffixes=("", "_props"),
        )

        # Add log-energy feature
        df["log10_energy_eV"] = np.log10(df["energy_eV"].clip(lower=1e-10))

        # Add log-cross-section target
        df[TARGET_COLUMN] = np.log10(df["cross_section_barn"].clip(lower=1e-15))

        # Compute Q-values (reaction energy release)
        df["Q_value_MeV"] = self._compute_q_values(df, reaction)

        # Filter isotopes with too few data points
        point_counts = df.groupby(["Z", "A"]).size()
        sufficient = point_counts[point_counts >= min_points_per_isotope].index
        df = df[df.set_index(["Z", "A"]).index.isin(sufficient)].copy()

        logger.info(
            "After filtering: %d points from %d isotopes",
            len(df),
            df[["Z", "A"]].drop_duplicates().shape[0],
        )

        # Bin energy grid if requested (reduce measurement density bias)
        if energy_bins_per_decade > 0:
            df = self._bin_energy_grid(df, energy_bins_per_decade)

        # Keep only feature columns + target + identifiers
        keep_cols = (
            ["Z", "N", "A", "energy_eV", "cross_section_barn", "reaction", "entry_id"]
            + FEATURE_COLUMNS
            + [TARGET_COLUMN]
        )
        keep_cols = [c for c in keep_cols if c in df.columns]
        df = df[keep_cols].dropna(subset=[TARGET_COLUMN, "log10_energy_eV"])

        logger.info(
            "Final dataset: %d samples, %d features",
            len(df),
            len(FEATURE_COLUMNS),
        )
        return df

    def _compute_q_values(self, df: pd.DataFrame, reaction: str) -> pd.Series:
        """
        Compute reaction Q-values from mass excesses.

        Q = (sum of reactant mass excesses) - (sum of product mass excesses) in MeV.
        For n-capture: Q = ME(Z,A) + ME(n) - ME(Z, A+1)
        For n-2n: Q = ME(Z,A) + ME(n) - ME(Z, A-1) - 2*ME(n)
        """
        q_values = pd.Series(np.nan, index=df.index)

        if "mass_excess_keV" not in df.columns:
            logger.debug("mass_excess_keV not in dataset; Q-values will be NaN.")
            return q_values

        # Build mass excess lookup
        me_lookup = self.nuclear_props.set_index(["Z", "A"])["mass_excess_keV"].to_dict()
        me_n = NEUTRON_MASS_EXCESS_KEV

        for idx, row in df.iterrows():
            Z, A = int(row["Z"]), int(row["A"])
            me_target = me_lookup.get((Z, A), np.nan)

            if np.isnan(me_target):
                continue

            if reaction == "n-gamma":
                me_compound = me_lookup.get((Z, A + 1), np.nan)
                q = (me_target + me_n - me_compound) / 1000.0 if not np.isnan(me_compound) else np.nan
            elif reaction == "n-fission":
                # Approximate: use binding energy difference for actinides
                q = np.nan  # Full Q-value requires fission fragment masses
            elif reaction == "n-2n":
                me_product = me_lookup.get((Z, A - 1), np.nan)
                q = (me_target + me_n - me_product - me_n) / 1000.0 if not np.isnan(me_product) else np.nan
            elif reaction == "n-p":
                from .nuclear_properties import NEUTRON_MASS_EXCESS_KEV as ME_N
                # Proton mass excess = 7288.971 keV
                me_proton = 7288.971
                me_product = me_lookup.get((Z - 1, A), np.nan)
                q = (me_target + me_n - me_product - me_proton) / 1000.0 if not np.isnan(me_product) else np.nan
            elif reaction == "n-alpha":
                # Alpha mass excess = 2424.916 keV
                me_alpha = 2424.916
                me_product = me_lookup.get((Z - 2, A - 3), np.nan)
                q = (me_target + me_n - me_product - me_alpha) / 1000.0 if not np.isnan(me_product) else np.nan
            else:
                q = np.nan

            q_values[idx] = q

        return q_values

    def _bin_energy_grid(
        self, df: pd.DataFrame, bins_per_decade: int
    ) -> pd.DataFrame:
        """
        Bin measurements to a uniform log-energy grid per isotope.

        When many experiments have measured the same isotope at similar energies,
        retain only the measurement with the smallest uncertainty in each energy bin.
        This prevents well-measured isotopes from dominating the training loss.
        """
        e_min = df["energy_eV"].min()
        e_max = df["energy_eV"].max()
        if e_min <= 0:
            e_min = 1e-5

        n_decades = np.log10(e_max / e_min)
        n_bins = max(10, int(n_decades * bins_per_decade))
        bin_edges = np.logspace(np.log10(e_min), np.log10(e_max), n_bins + 1)

        df = df.copy()
        df["energy_bin"] = pd.cut(df["energy_eV"], bins=bin_edges, labels=False)

        # For each (Z, A, energy_bin), keep the row with smallest uncertainty
        def select_best_in_bin(group):
            if "uncertainty_percent" in group.columns:
                not_null = group["uncertainty_percent"].notna()
                if not_null.any():
                    return group.loc[group.loc[not_null, "uncertainty_percent"].idxmin()]
            return group.iloc[0]

        binned = (
            df.groupby(["Z", "A", "energy_bin"], group_keys=False)
            .apply(select_best_in_bin)
            .reset_index(drop=True)
        )

        binned = binned.drop(columns=["energy_bin"], errors="ignore")
        logger.info(
            "Energy binning: %d -> %d data points (%.1f%% retained)",
            len(df),
            len(binned),
            100.0 * len(binned) / len(df),
        )
        return binned

    def split_train_val_test(
        self,
        df: pd.DataFrame,
        test_fraction: float = 0.15,
        val_fraction: float = 0.10,
        strategy: str = "leave_one_isotope_out",
        random_seed: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split dataset into train/validation/test sets.

        Two strategies are supported:
        - "random": Standard random split by data points.
        - "leave_one_isotope_out": Splits by isotope identity so that all
          measurements of a given isotope go to the same split. This is the
          recommended strategy as it tests generalization to unmeasured nuclei.

        Parameters
        ----------
        df : pd.DataFrame
            Full feature-engineered dataset.
        test_fraction, val_fraction : float
            Fractions for test and validation sets.
        strategy : str
            Split strategy ("random" or "leave_one_isotope_out").
        random_seed : int
            Random state for reproducibility.

        Returns
        -------
        (train_df, val_df, test_df)
        """
        rng = np.random.RandomState(random_seed)

        if strategy == "leave_one_isotope_out":
            # Create a group label per isotope
            isotope_ids = df[["Z", "A"]].apply(
                lambda row: f"{int(row['Z'])}_{int(row['A'])}", axis=1
            ).values

            unique_isotopes = np.unique(isotope_ids)
            rng.shuffle(unique_isotopes)

            n_test = max(1, int(len(unique_isotopes) * test_fraction))
            n_val = max(1, int(len(unique_isotopes) * val_fraction))

            test_isotopes = set(unique_isotopes[:n_test])
            val_isotopes = set(unique_isotopes[n_test:n_test + n_val])

            test_mask = np.isin(isotope_ids, list(test_isotopes))
            val_mask = np.isin(isotope_ids, list(val_isotopes))
            train_mask = ~(test_mask | val_mask)

        else:  # random
            n = len(df)
            indices = rng.permutation(n)
            n_test = int(n * test_fraction)
            n_val = int(n * val_fraction)

            test_idx = set(indices[:n_test])
            val_idx = set(indices[n_test:n_test + n_val])

            test_mask = np.array([i in test_idx for i in range(n)])
            val_mask = np.array([i in val_idx for i in range(n)])
            train_mask = ~(test_mask | val_mask)

        train_df = df[train_mask].copy()
        val_df = df[val_mask].copy()
        test_df = df[test_mask].copy()

        logger.info(
            "Split: train=%d, val=%d, test=%d (strategy=%s)",
            len(train_df), len(val_df), len(test_df), strategy,
        )
        return train_df, val_df, test_df

    def save_to_hdf5(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        reaction: str,
    ) -> Path:
        """
        Save train/val/test splits to an HDF5 file.

        Parameters
        ----------
        train_df, val_df, test_df : pd.DataFrame
            Dataset splits.
        reaction : str
            Reaction label used in the filename.

        Returns
        -------
        Path
            Path to the saved HDF5 file.
        """
        output_path = self.output_dir / f"{reaction.replace(',', '_')}_dataset.h5"

        with h5py.File(output_path, "w") as f:
            for split_name, split_df in [
                ("train", train_df), ("val", val_df), ("test", test_df)
            ]:
                grp = f.create_group(split_name)
                X = split_df[FEATURE_COLUMNS].values.astype(np.float32)
                y = split_df[TARGET_COLUMN].values.astype(np.float32)
                grp.create_dataset("X", data=X, compression="gzip")
                grp.create_dataset("y", data=y, compression="gzip")
                grp.attrs["feature_names"] = FEATURE_COLUMNS
                grp.attrs["n_samples"] = len(split_df)

                # Save Z, A identifiers for leave-one-isotope analysis
                if "Z" in split_df.columns:
                    grp.create_dataset("Z", data=split_df["Z"].values.astype(np.int32))
                    grp.create_dataset("A", data=split_df["A"].values.astype(np.int32))

            f.attrs["reaction"] = reaction
            f.attrs["n_features"] = len(FEATURE_COLUMNS)

        logger.info("Saved dataset to %s", output_path)
        return output_path

    @staticmethod
    def load_from_hdf5(
        path: str,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Load train/val/test splits from an HDF5 dataset file.

        Parameters
        ----------
        path : str
            Path to the HDF5 file.

        Returns
        -------
        dict
            Keys "train", "val", "test"; values are (X, y) tuples.
        """
        splits = {}
        with h5py.File(path, "r") as f:
            for split in ["train", "val", "test"]:
                if split in f:
                    X = f[split]["X"][:]
                    y = f[split]["y"][:]
                    splits[split] = (X, y)
        return splits
