"""
ENDF/B-VIII.0 evaluated nuclear data processor.

Parses ENDF-6 formatted files from the ENDF/B-VIII.0 library to extract
evaluated cross-section data for training and comparison purposes.

The ENDF-6 format is described in:
  ENDF-6 Formats Manual, NNDC, Brookhaven National Laboratory (2018)
  https://www.nndc.bnl.gov/endf/endf6guide.html

Reference library: https://www.nndc.bnl.gov/endf/b8.0/
"""

import logging
import re
import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ENDF-6 MF/MT codes relevant to neutron cross-sections
ENDF_MT_MAP = {
    1: "n-total",
    2: "n-elastic",
    4: "n-inelastic-total",
    16: "n-2n",
    18: "n-fission",
    102: "n-gamma",
    103: "n-p",
    107: "n-alpha",
}

NNDC_ENDF_BASE = "https://www.nndc.bnl.gov/endf/b8.0/zips"


class ENDFProcessor:
    """
    Downloads and processes ENDF/B-VIII.0 evaluated cross-section files.

    ENDF evaluated data represents the best-estimate cross-sections from
    expert evaluation of experimental data combined with theoretical model
    calculations, and serves as ground-truth for nuclear engineering codes.

    Parameters
    ----------
    data_dir : str or Path
        Directory to store downloaded and processed ENDF files.
    """

    def __init__(self, data_dir: str = "data/raw/endf"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def parse_endf_file(self, filepath: Path) -> Dict[str, pd.DataFrame]:
        """
        Parse an ENDF-6 formatted file and extract cross-section data.

        This parser handles the ENDF-6 TAB1 record format used in MF=3
        (reaction cross-sections) sections.

        Parameters
        ----------
        filepath : Path
            Path to an ENDF-6 formatted file.

        Returns
        -------
        dict mapping MT label string to DataFrame with columns:
            energy_eV, cross_section_barn.
        """
        results = {}

        with open(filepath, "r") as f:
            lines = f.readlines()

        Z, A = self._extract_za_from_header(lines)
        if Z is None:
            logger.warning("Could not parse Z/A from %s", filepath)
            return results

        mf3_blocks = self._extract_mf3_blocks(lines)

        for mt, block_lines in mf3_blocks.items():
            if mt not in ENDF_MT_MAP:
                continue

            energies, cross_sections = self._parse_tab1_record(block_lines)
            if energies is None or len(energies) == 0:
                continue

            label = ENDF_MT_MAP[mt]
            results[label] = pd.DataFrame({
                "Z": Z,
                "A": A,
                "N": A - Z,
                "reaction": label,
                "energy_eV": energies,
                "cross_section_barn": cross_sections,
                "source": "ENDF/B-VIII.0",
            })

        return results

    def _extract_za_from_header(
        self, lines: List[str]
    ) -> Tuple[Optional[int], Optional[int]]:
        """Extract Z and A from ENDF-6 file header (TPID/CONT record)."""
        for line in lines[:10]:
            # ENDF-6 CONT record: ZA in columns 1-11
            try:
                za_field = line[:11].strip()
                za = float(za_field)
                if za > 0:
                    A = int(za % 1000)
                    Z = int(za // 1000)
                    if 1 <= Z <= 118 and A >= Z:
                        return Z, A
            except (ValueError, IndexError):
                continue
        return None, None

    def _extract_mf3_blocks(
        self, lines: List[str]
    ) -> Dict[int, List[str]]:
        """Extract MF=3 (cross-section) blocks indexed by MT number."""
        blocks = {}
        current_mt = None
        current_block = []
        in_mf3 = False

        for line in lines:
            if len(line) < 75:
                continue
            try:
                mf = int(line[70:72].strip() or "0")
                mt = int(line[72:75].strip() or "0")
            except ValueError:
                continue

            if mf == 3 and mt != 0:
                if mt != current_mt:
                    if current_mt is not None and current_block:
                        blocks[current_mt] = current_block
                    current_mt = mt
                    current_block = []
                    in_mf3 = True
                current_block.append(line)
            elif in_mf3 and mf != 3:
                if current_mt is not None and current_block:
                    blocks[current_mt] = current_block
                in_mf3 = False
                current_mt = None
                current_block = []

        if current_mt is not None and current_block:
            blocks[current_mt] = current_block

        return blocks

    def _parse_tab1_record(
        self, block_lines: List[str]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Parse an ENDF-6 TAB1 record to extract (energy, cross-section) pairs.

        TAB1 records use a fixed-width 6-field format with 11 characters per field.
        """
        values = []
        # Skip CONT header lines; collect all numeric fields
        for line in block_lines[2:]:
            # ENDF-6 data lines: 6 fields of 11 chars each in columns 1-66
            for i in range(6):
                field = line[i * 11:(i + 1) * 11].strip()
                if not field:
                    continue
                try:
                    values.append(float(field.replace("+", "e+").replace("-", "e-")
                                        if "e" not in field.lower() and
                                           ("+" in field[1:] or
                                            ("-" in field[1:] and field[0] != "-"))
                                        else field))
                except ValueError:
                    pass

        if len(values) < 4:
            return None, None

        # First CONT record gives NR (number of interpolation ranges) and NP (number of points)
        try:
            # NP is the total number of (E, sigma) pairs
            np_val = int(values[1]) if len(values) > 1 else 0
            if np_val < 1:
                return None, None

            # Skip NBT and INT arrays (interpolation tables); find start of (E, sigma) pairs
            nr_val = int(values[0]) if len(values) > 0 else 0
            data_start = 2 + 2 * nr_val  # skip NR pairs of (NBT, INT)

            if data_start + 2 * np_val > len(values):
                # Truncate to available data
                n_pairs = (len(values) - data_start) // 2
            else:
                n_pairs = np_val

            energies = np.array(values[data_start:data_start + 2 * n_pairs:2])
            cross_sections = np.array(values[data_start + 1:data_start + 2 * n_pairs:1:2])

            # Filter physical values
            mask = (energies > 0) & (cross_sections >= 0)
            return energies[mask], cross_sections[mask]

        except (IndexError, ValueError, OverflowError) as exc:
            logger.debug("TAB1 parse error: %s", exc)
            return None, None

    def process_directory(self, endf_dir: Optional[Path] = None) -> pd.DataFrame:
        """
        Process all ENDF files in a directory and combine into a single DataFrame.

        Parameters
        ----------
        endf_dir : Path, optional
            Directory containing ENDF-6 files. Defaults to self.data_dir.

        Returns
        -------
        pd.DataFrame
            Combined evaluated cross-section data.
        """
        if endf_dir is None:
            endf_dir = self.data_dir

        endf_files = list(endf_dir.glob("*.endf")) + list(endf_dir.glob("*.dat"))
        logger.info("Found %d ENDF files in %s", len(endf_files), endf_dir)

        all_frames = []
        for filepath in tqdm(endf_files, desc="Processing ENDF files"):
            try:
                file_data = self.parse_endf_file(filepath)
                for label, df in file_data.items():
                    all_frames.append(df)
            except Exception as exc:
                logger.warning("Failed to parse %s: %s", filepath, exc)

        if not all_frames:
            logger.warning("No ENDF data parsed from %s", endf_dir)
            return pd.DataFrame()

        combined = pd.concat(all_frames, ignore_index=True)
        logger.info("Parsed %d evaluated cross-section records", len(combined))
        return combined

    def compare_with_exfor(
        self,
        endf_df: pd.DataFrame,
        exfor_df: pd.DataFrame,
        reaction: str,
        Z: int,
        A: int,
    ) -> pd.DataFrame:
        """
        Compare ENDF evaluated values against EXFOR experimental measurements.

        Interpolates ENDF values at the energies of EXFOR data points to
        enable direct comparison and residual analysis.

        Parameters
        ----------
        endf_df : pd.DataFrame
            Evaluated ENDF data (output of process_directory).
        exfor_df : pd.DataFrame
            Experimental EXFOR data.
        reaction, Z, A : str, int, int
            Isotope and reaction to compare.

        Returns
        -------
        pd.DataFrame
            Merged comparison with columns for both ENDF and EXFOR values.
        """
        endf_iso = endf_df[
            (endf_df["Z"] == Z) & (endf_df["A"] == A) & (endf_df["reaction"] == reaction)
        ].sort_values("energy_eV")

        exfor_iso = exfor_df[
            (exfor_df["Z"] == Z) & (exfor_df["A"] == A) & (exfor_df["reaction"] == reaction)
        ].copy()

        if endf_iso.empty or exfor_iso.empty:
            return pd.DataFrame()

        # Interpolate ENDF at EXFOR energy points (log-log interpolation)
        endf_energies = endf_iso["energy_eV"].values
        endf_sigma = endf_iso["cross_section_barn"].values

        valid = (endf_sigma > 0) & (endf_energies > 0)
        log_e = np.log10(endf_energies[valid])
        log_s = np.log10(endf_sigma[valid])

        exfor_e = exfor_iso["energy_eV"].values
        in_range = (exfor_e >= endf_energies[valid].min()) & (exfor_e <= endf_energies[valid].max())

        endf_interp = np.full(len(exfor_e), np.nan)
        endf_interp[in_range] = 10 ** np.interp(
            np.log10(exfor_e[in_range]), log_e, log_s
        )

        exfor_iso["endf_cross_section_barn"] = endf_interp
        exfor_iso["ratio_exfor_to_endf"] = (
            exfor_iso["cross_section_barn"] / exfor_iso["endf_cross_section_barn"]
        )

        return exfor_iso.dropna(subset=["endf_cross_section_barn"])
