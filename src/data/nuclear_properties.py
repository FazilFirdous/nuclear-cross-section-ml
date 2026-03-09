"""
Nuclear structure properties loader.

Loads and parses the Atomic Mass Evaluation (AME2020), NuBase2020 ground-state
properties, and FRDM deformation tables. These provide the nuclear structure
observables used as input features for cross-section prediction.

References:
  AME2020: W.J. Huang et al., Chinese Phys. C 45, 030002 (2021)
           M. Wang et al., Chinese Phys. C 45, 030003 (2021)
  NuBase2020: F.G. Kondev et al., Chinese Phys. C 45, 030001 (2021)
  FRDM: P. Moller et al., At. Data Nucl. Data Tables 109-110, 1 (2016)
"""

import logging
import re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Magic numbers for shell-closure proximity features
MAGIC_PROTON = np.array([2, 8, 20, 28, 50, 82])
MAGIC_NEUTRON = np.array([2, 8, 20, 28, 50, 82, 126])


def dist_to_nearest_magic(x: float, magic: np.ndarray) -> float:
    """Return the minimum distance from x to any magic number."""
    return float(np.min(np.abs(magic - x)))


class NuclearPropertiesLoader:
    """
    Loads nuclear structure data from AME2020, NuBase2020, and FRDM tables.

    Provides a unified interface for accessing nuclear properties needed
    for cross-section feature vectors.

    Parameters
    ----------
    ame_path : str or Path
        Path to the AME2020 mass table file.
    nubase_path : str or Path
        Path to the NuBase2020 ground-state properties file.
    frdm_path : str or Path, optional
        Path to the FRDM deformation parameter table.
    """

    def __init__(
        self,
        ame_path: str = "data/external/ame2020_mass.txt",
        nubase_path: str = "data/external/nubase2020.txt",
        frdm_path: Optional[str] = "data/external/frdm2012_deformation.dat",
    ):
        self.ame_path = Path(ame_path)
        self.nubase_path = Path(nubase_path)
        self.frdm_path = Path(frdm_path) if frdm_path else None

        self._ame = None
        self._nubase = None
        self._frdm = None

    def load_all(self) -> pd.DataFrame:
        """
        Load and merge all nuclear property tables.

        Returns
        -------
        pd.DataFrame
            Merged table indexed by (Z, A) with all available nuclear properties.
        """
        ame = self.load_ame2020()
        nubase = self.load_nubase2020()

        merged = ame.merge(nubase, on=["Z", "A", "N"], how="outer")

        if self.frdm_path and self.frdm_path.exists():
            frdm = self.load_frdm()
            merged = merged.merge(frdm, on=["Z", "A"], how="left")
        else:
            logger.warning(
                "FRDM deformation table not found at %s; deformation features will be NaN.",
                self.frdm_path,
            )
            merged["beta2"] = np.nan
            merged["beta4"] = np.nan

        merged = self._compute_derived_features(merged)
        merged = merged.drop_duplicates(subset=["Z", "A"]).reset_index(drop=True)

        logger.info(
            "Loaded nuclear properties for %d isotopes (Z=1..%d)",
            len(merged),
            merged["Z"].max(),
        )
        return merged

    def load_ame2020(self) -> pd.DataFrame:
        """
        Parse the AME2020 atomic mass table.

        The AME2020 fixed-width format is described in:
        https://www-nds.iaea.org/amdc/ame2020/

        Returns
        -------
        pd.DataFrame
            Columns: Z, A, N, mass_excess_keV, mass_excess_unc_keV,
                     binding_energy_per_A_keV, Sn_keV, S2n_keV, Sp_keV, S2p_keV.
        """
        if self._ame is not None:
            return self._ame

        if not self.ame_path.exists():
            logger.warning("AME2020 file not found: %s. Returning empty DataFrame.", self.ame_path)
            return self._empty_ame()

        records = []
        with open(self.ame_path, "r", encoding="latin-1") as f:
            lines = f.readlines()

        # Skip header lines (lines starting with '#' or blank)
        data_lines = [l for l in lines if not l.startswith("#") and len(l.strip()) > 10]

        for line in data_lines:
            try:
                record = self._parse_ame_line(line)
                if record:
                    records.append(record)
            except Exception as exc:
                logger.debug("AME parse error: %s | line: %s", exc, line[:40])

        if not records:
            logger.warning("No AME2020 records parsed from %s", self.ame_path)
            return self._empty_ame()

        df = pd.DataFrame(records)
        self._ame = df
        logger.info("Parsed %d AME2020 mass entries", len(df))
        return df

    def _parse_ame_line(self, line: str) -> Optional[dict]:
        """Parse one AME2020 fixed-width data line."""
        # AME2020 mass_1.mas20 format (columns):
        # 1: cc (continuation flag), 2-4: N-Z, 5-7: N, 8-10: Z, 11-14: A, 15-18: el,
        # 19-21: origin, 22-41: mass_excess(keV), 42-52: unc, 53-63: BE/A, 64-72: unc,
        # 73-75: beta decay type, 76-86: Sn, 87-92: unc, 93-103: S2n, 104-109: unc,
        # 110-120: Sp, 121-126: unc, 127-137: S2p, 138-143: unc, 144-161: atomic_mass_u, ...
        if len(line) < 80:
            return None

        # Extract fields using fixed column widths from AME2020 documentation
        try:
            n_z = int(line[1:5].strip() or "0")
            N = int(line[5:9].strip() or "0")
            Z = int(line[9:13].strip() or "0")
            A = int(line[13:17].strip() or "0")

            if Z < 1 or A < 1 or N < 0:
                return None

            def parse_value_unc(val_str: str, unc_str: str) -> Tuple[Optional[float], Optional[float]]:
                val_str = val_str.replace("*", "").strip()
                unc_str = unc_str.replace("*", "").strip()
                val = float(val_str) if val_str else None
                unc = float(unc_str) if unc_str else None
                return val, unc

            me, me_unc = parse_value_unc(line[18:31], line[31:42])
            be_a, be_a_unc = parse_value_unc(line[42:52], line[52:63])
            sn, sn_unc = parse_value_unc(line[63:75], line[75:82]) if len(line) > 82 else (None, None)
            s2n, s2n_unc = parse_value_unc(line[82:94], line[94:101]) if len(line) > 101 else (None, None)
            sp, sp_unc = parse_value_unc(line[101:113], line[113:120]) if len(line) > 120 else (None, None)
            s2p, s2p_unc = parse_value_unc(line[120:132], line[132:139]) if len(line) > 139 else (None, None)

            return {
                "Z": Z,
                "N": N,
                "A": A,
                "mass_excess_keV": me,
                "mass_excess_unc_keV": me_unc,
                "binding_energy_per_A_keV": be_a,
                "binding_energy_per_A_unc_keV": be_a_unc,
                "Sn_keV": sn,
                "Sn_unc_keV": sn_unc,
                "S2n_keV": s2n,
                "S2n_unc_keV": s2n_unc,
                "Sp_keV": sp,
                "Sp_unc_keV": sp_unc,
                "S2p_keV": s2p,
                "S2p_unc_keV": s2p_unc,
            }
        except (ValueError, IndexError):
            return None

    def _empty_ame(self) -> pd.DataFrame:
        return pd.DataFrame(columns=[
            "Z", "N", "A", "mass_excess_keV", "mass_excess_unc_keV",
            "binding_energy_per_A_keV", "binding_energy_per_A_unc_keV",
            "Sn_keV", "S2n_keV", "Sp_keV", "S2p_keV",
        ])

    def load_nubase2020(self) -> pd.DataFrame:
        """
        Parse the NuBase2020 ground-state properties table.

        Extracts spin, parity, half-life, and ground-state energy for each isotope.

        Returns
        -------
        pd.DataFrame
            Columns: Z, A, N, spin, parity, half_life_s, ground_state_energy_keV.
        """
        if self._nubase is not None:
            return self._nubase

        if not self.nubase_path.exists():
            logger.warning(
                "NuBase2020 file not found: %s. Returning empty DataFrame.", self.nubase_path
            )
            return pd.DataFrame(columns=["Z", "A", "N", "spin", "parity"])

        records = []
        with open(self.nubase_path, "r", encoding="latin-1") as f:
            lines = f.readlines()

        for line in lines:
            if line.startswith("#") or len(line.strip()) < 20:
                continue
            try:
                record = self._parse_nubase_line(line)
                if record:
                    records.append(record)
            except Exception as exc:
                logger.debug("NuBase parse error: %s", exc)

        if not records:
            return pd.DataFrame(columns=["Z", "A", "N", "spin", "parity"])

        df = pd.DataFrame(records)
        # Keep only ground states (isomer flag = 0)
        df = df[df.get("isomer", 0) == 0].copy()
        self._nubase = df
        logger.info("Parsed %d NuBase2020 ground-state entries", len(df))
        return df

    def _parse_nubase_line(self, line: str) -> Optional[dict]:
        """Parse one NuBase2020 fixed-width data line."""
        try:
            A = int(line[0:3].strip())
            Z = int(line[4:8].strip() or "0")
            N = A - Z
            isomer_flag = 0 if line[7:9].strip() == "0" or not line[7:9].strip() else 1

            # Spin-parity string (approximately columns 79-93 in NuBase2020 format)
            spin_parity_str = line[79:93].strip() if len(line) > 93 else ""

            spin, parity = self._parse_spin_parity(spin_parity_str)

            return {
                "Z": Z,
                "N": N,
                "A": A,
                "spin": spin,
                "parity": parity,
                "spin_parity_str": spin_parity_str,
                "isomer": isomer_flag,
            }
        except (ValueError, IndexError):
            return None

    def _parse_spin_parity(self, sp_str: str) -> Tuple[Optional[float], Optional[int]]:
        """
        Parse a spin-parity string like '3/2+', '0+', '5-', '(1/2+)' into
        numeric (spin, parity) = (float, +/-1) values.
        """
        if not sp_str:
            return None, None

        sp_str = sp_str.strip().strip("()")
        if not sp_str:
            return None, None

        parity_match = re.search(r"([+-])$", sp_str)
        parity = None
        if parity_match:
            parity = 1 if parity_match.group(1) == "+" else -1
            sp_str = sp_str[:-1]

        spin = None
        if "/" in sp_str:
            parts = sp_str.split("/")
            if len(parts) == 2:
                try:
                    spin = float(parts[0]) / float(parts[1])
                except ValueError:
                    pass
        else:
            try:
                spin = float(sp_str)
            except ValueError:
                pass

        return spin, parity

    def load_frdm(self) -> pd.DataFrame:
        """
        Parse the FRDM2012 nuclear deformation parameter table.

        The FRDM table provides quadrupole (beta2) and hexadecapole (beta4)
        deformation parameters for ground states across the nuclear chart.

        Returns
        -------
        pd.DataFrame
            Columns: Z, A, beta2, beta4, epsilon2, epsilon4.
        """
        if self._frdm is not None:
            return self._frdm

        if not self.frdm_path or not self.frdm_path.exists():
            logger.warning("FRDM file not found; returning empty deformation table.")
            return pd.DataFrame(columns=["Z", "A", "beta2", "beta4"])

        records = []
        with open(self.frdm_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            try:
                Z = int(parts[0])
                A = int(parts[1])
                # Typical FRDM column order: Z A N epsilon2 epsilon4 beta2 beta4 ...
                # (varies by version; adapt as needed)
                beta2 = float(parts[5]) if len(parts) > 5 else float(parts[3])
                beta4 = float(parts[6]) if len(parts) > 6 else float(parts[4])
                records.append({"Z": Z, "A": A, "beta2": beta2, "beta4": beta4})
            except (ValueError, IndexError):
                continue

        if not records:
            logger.warning("No FRDM records parsed from %s", self.frdm_path)
            return pd.DataFrame(columns=["Z", "A", "beta2", "beta4"])

        df = pd.DataFrame(records).drop_duplicates(subset=["Z", "A"])
        self._frdm = df
        logger.info("Parsed %d FRDM deformation entries", len(df))
        return df

    def _compute_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute derived nuclear features from primary observables.

        Adds: Z_even, N_even, isospin_asym, surface_area,
              delta_n (neutron pairing gap), delta_p (proton pairing gap),
              dist_to_magic_Z, dist_to_magic_N.
        """
        df = df.copy()

        df["Z_even"] = (df["Z"] % 2 == 0).astype(int)
        df["N_even"] = (df["N"] % 2 == 0).astype(int)
        df["isospin_asym"] = (df["N"] - df["Z"]) / df["A"]
        df["surface_area"] = df["A"] ** (2.0 / 3.0)

        # Binding energy in MeV (AME gives keV)
        if "binding_energy_per_A_keV" in df.columns:
            df["binding_energy_per_A_MeV"] = df["binding_energy_per_A_keV"] / 1000.0
        else:
            df["binding_energy_per_A_MeV"] = np.nan

        # Separation energies in MeV
        for col in ["Sn_keV", "S2n_keV", "Sp_keV", "S2p_keV"]:
            mev_col = col.replace("_keV", "_MeV")
            if col in df.columns:
                df[mev_col] = df[col] / 1000.0
            else:
                df[mev_col] = np.nan

        # Empirical pairing gaps: delta = (-1)^N * (M(N-1) - 2*M(N) + M(N+1)) / 2
        # Approximated from Sn differences: delta_n ~ (S2n - 2*Sn) using AME data
        if "S2n_MeV" in df.columns and "Sn_MeV" in df.columns:
            df["delta_n"] = (df["S2n_MeV"] - 2 * df["Sn_MeV"]).abs() / 2.0
        else:
            df["delta_n"] = np.nan

        if "S2p_MeV" in df.columns and "Sp_MeV" in df.columns:
            df["delta_p"] = (df["S2p_MeV"] - 2 * df["Sp_MeV"]).abs() / 2.0
        else:
            df["delta_p"] = np.nan

        # Shell-closure proximity
        df["dist_to_magic_Z"] = df["Z"].apply(
            lambda z: dist_to_nearest_magic(z, MAGIC_PROTON)
        )
        df["dist_to_magic_N"] = df["N"].apply(
            lambda n: dist_to_nearest_magic(n, MAGIC_NEUTRON)
        )

        return df

    def get_features_for_isotope(
        self, Z: int, A: int, properties_df: Optional[pd.DataFrame] = None
    ) -> dict:
        """
        Return a feature dictionary for a single isotope (Z, A).

        Parameters
        ----------
        Z, A : int
            Proton and mass number.
        properties_df : pd.DataFrame, optional
            Pre-loaded nuclear properties table. If None, loads all tables.

        Returns
        -------
        dict
            Feature dictionary suitable for ML model input.
        """
        if properties_df is None:
            properties_df = self.load_all()

        row = properties_df[(properties_df["Z"] == Z) & (properties_df["A"] == A)]
        if row.empty:
            logger.warning("No properties found for Z=%d, A=%d; using estimated values.", Z, A)
            return self._estimate_features(Z, A)

        return row.iloc[0].to_dict()

    def _estimate_features(self, Z: int, A: int) -> dict:
        """
        Estimate nuclear features for isotopes not in the AME/NuBase tables
        using simple empirical formulas (liquid drop model, etc.).
        """
        N = A - Z
        # Liquid drop model binding energy: Bethe-Weizsacker formula
        a_v, a_s, a_c, a_a = 15.75, 17.8, 0.711, 23.7
        if N % 2 == 0 and Z % 2 == 0:
            a_p = 12.0
        elif N % 2 == 1 and Z % 2 == 1:
            a_p = -12.0
        else:
            a_p = 0.0

        be_per_a = (
            a_v
            - a_s * A ** (-1.0 / 3.0)
            - a_c * Z * (Z - 1) * A ** (-4.0 / 3.0)
            - a_a * ((N - Z) / A) ** 2
            + a_p * A ** (-3.0 / 2.0)
        )

        return {
            "Z": Z, "N": N, "A": A,
            "Z_even": int(Z % 2 == 0),
            "N_even": int(N % 2 == 0),
            "isospin_asym": (N - Z) / A,
            "surface_area": A ** (2.0 / 3.0),
            "binding_energy_per_A_MeV": be_per_a,
            "Sn_MeV": np.nan,
            "S2n_MeV": np.nan,
            "Sp_MeV": np.nan,
            "S2p_MeV": np.nan,
            "delta_n": np.nan,
            "delta_p": np.nan,
            "spin": np.nan,
            "parity": np.nan,
            "beta2": np.nan,
            "beta4": np.nan,
            "dist_to_magic_Z": dist_to_nearest_magic(Z, MAGIC_PROTON),
            "dist_to_magic_N": dist_to_nearest_magic(N, MAGIC_NEUTRON),
        }
