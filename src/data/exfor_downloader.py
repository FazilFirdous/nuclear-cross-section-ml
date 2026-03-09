"""
EXFOR database downloader.

Downloads experimental neutron cross-section data from the EXFOR database
via the IAEA Nuclear Data Services REST API.

Reference: https://nds.iaea.org/exfor/
Documentation: https://nds.iaea.org/exfor/services/
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)


# EXFOR reaction notation maps to our internal labels
REACTION_MAP = {
    "n-total": "N,TOT",
    "n-elastic": "N,EL",
    "n-gamma": "N,G",
    "n-fission": "N,F",
    "n-2n": "N,2N",
    "n-alpha": "N,A",
    "n-p": "N,P",
    "n-inelastic": "N,INL",
}

# EXFOR API base URL (IAEA NDS)
EXFOR_API_BASE = "https://nds.iaea.org/exfor/services"
EXFOR_ENTRY_URL = f"{EXFOR_API_BASE}/entries/json"
EXFOR_DATASET_URL = f"{EXFOR_API_BASE}/datasets/json"
EXFOR_SEARCH_URL = f"{EXFOR_API_BASE}/search"


class EXFORDownloader:
    """
    Downloads and caches experimental cross-section data from EXFOR.

    The EXFOR (Experimental Nuclear Reaction Data) database is maintained
    by the IAEA Nuclear Data Section and contains over 22,000 experimental
    datasets from nuclear reaction measurements worldwide.

    Parameters
    ----------
    output_dir : str or Path
        Directory to save downloaded data files.
    max_retries : int
        Maximum number of retry attempts for failed HTTP requests.
    retry_delay : float
        Base delay in seconds between retries (exponential backoff applied).
    """

    def __init__(
        self,
        output_dir: str = "data/raw/exfor",
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "nuclear-cross-section-ml/0.1"})

    def _get_with_retry(self, url: str, params: Optional[Dict] = None) -> requests.Response:
        """Perform HTTP GET with exponential backoff retry."""
        delay = self.retry_delay
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                return response
            except requests.RequestException as exc:
                if attempt == self.max_retries:
                    raise RuntimeError(
                        f"Failed to fetch {url} after {self.max_retries} retries: {exc}"
                    ) from exc
                logger.warning(
                    "Request failed (attempt %d/%d): %s. Retrying in %.0fs.",
                    attempt + 1,
                    self.max_retries,
                    exc,
                    delay,
                )
                time.sleep(delay)
                delay *= 2

    def search_exfor(
        self,
        reaction: str,
        quantity: str = "SIG",
        z_min: int = 1,
        z_max: int = 118,
    ) -> List[Dict]:
        """
        Search EXFOR for datasets matching a reaction type.

        Parameters
        ----------
        reaction : str
            Reaction notation, e.g. "N,G" for (n,gamma).
        quantity : str
            EXFOR quantity code; "SIG" for cross-section.
        z_min, z_max : int
            Atomic number range to search.

        Returns
        -------
        list of dict
            List of dataset metadata records.
        """
        exfor_reaction = REACTION_MAP.get(reaction, reaction)
        logger.info("Searching EXFOR for reaction %s (quantity %s)", exfor_reaction, quantity)

        # EXFOR REST API search parameters
        params = {
            "reaction": exfor_reaction,
            "quantity": quantity,
            "target_Z_min": z_min,
            "target_Z_max": z_max,
            "format": "json",
        }

        try:
            response = self._get_with_retry(EXFOR_SEARCH_URL, params=params)
            data = response.json()
            entries = data.get("entries", [])
            logger.info("Found %d EXFOR entries for %s", len(entries), exfor_reaction)
            return entries
        except Exception as exc:
            logger.error("EXFOR search failed: %s", exc)
            return []

    def download_entry(self, entry_id: str) -> Optional[Dict]:
        """
        Download a specific EXFOR entry by its accession number.

        Parameters
        ----------
        entry_id : str
            EXFOR entry accession number (e.g., "10001").

        Returns
        -------
        dict or None
            Parsed entry data, or None if download fails.
        """
        cache_path = self.output_dir / f"{entry_id}.json"
        if cache_path.exists():
            with open(cache_path) as f:
                return json.load(f)

        url = f"{EXFOR_ENTRY_URL}/{entry_id}"
        try:
            response = self._get_with_retry(url)
            data = response.json()
            with open(cache_path, "w") as f:
                json.dump(data, f, indent=2)
            return data
        except Exception as exc:
            logger.warning("Failed to download entry %s: %s", entry_id, exc)
            return None

    def parse_entry_to_dataframe(self, entry_data: Dict) -> pd.DataFrame:
        """
        Parse an EXFOR entry JSON into a tidy DataFrame.

        Each row represents a single (energy, cross-section, uncertainty) measurement.

        Parameters
        ----------
        entry_data : dict
            Raw EXFOR entry data from the API.

        Returns
        -------
        pd.DataFrame
            Columns: entry_id, Z, A, reaction, energy_eV, cross_section_barn,
                     uncertainty_barn, uncertainty_percent, reference.
        """
        records = []
        entry_id = entry_data.get("entry", "")

        for dataset in entry_data.get("datasets", []):
            target = dataset.get("target", {})
            Z = target.get("Z", None)
            A = target.get("A", None)
            reaction = dataset.get("reaction", "")
            reference = dataset.get("reference", "")

            for point in dataset.get("data", []):
                energy_eV = point.get("EN", None)
                sigma_barn = point.get("DATA", None)
                err_barn = point.get("ERR-T", point.get("DATA-ERR", None))

                if energy_eV is None or sigma_barn is None:
                    continue

                err_pct = (err_barn / sigma_barn * 100) if (err_barn and sigma_barn > 0) else None

                records.append({
                    "entry_id": entry_id,
                    "Z": Z,
                    "A": A,
                    "N": (A - Z) if (A and Z) else None,
                    "reaction": reaction,
                    "energy_eV": energy_eV,
                    "cross_section_barn": sigma_barn,
                    "uncertainty_barn": err_barn,
                    "uncertainty_percent": err_pct,
                    "reference": reference,
                })

        if not records:
            return pd.DataFrame()

        return pd.DataFrame(records)

    def download_reaction(
        self,
        reaction: str,
        z_range: Tuple[int, int] = (1, 118),
        energy_range_eV: Tuple[float, float] = (1e-5, 2e7),
    ) -> pd.DataFrame:
        """
        Download all EXFOR data for a given reaction type.

        Parameters
        ----------
        reaction : str
            Reaction label, e.g. "n-gamma".
        z_range : tuple
            (z_min, z_max) for the search.
        energy_range_eV : tuple
            (e_min, e_max) energy filter in eV.

        Returns
        -------
        pd.DataFrame
            Combined dataset of all measurements.
        """
        parquet_path = self.output_dir / f"{reaction.replace(',', '_')}_all.parquet"
        if parquet_path.exists():
            logger.info("Loading cached %s data from %s", reaction, parquet_path)
            return pd.read_parquet(parquet_path)

        entries = self.search_exfor(
            reaction=reaction,
            z_min=z_range[0],
            z_max=z_range[1],
        )

        all_frames = []
        failed = 0
        for entry in tqdm(entries, desc=f"Downloading {reaction}"):
            entry_id = entry.get("entry_id", entry.get("id", ""))
            if not entry_id:
                continue
            data = self.download_entry(entry_id)
            if data is None:
                failed += 1
                continue
            df = self.parse_entry_to_dataframe(data)
            if not df.empty:
                all_frames.append(df)
            time.sleep(0.05)  # polite rate limiting

        logger.info(
            "Downloaded %d datasets for %s (%d failures)",
            len(all_frames),
            reaction,
            failed,
        )

        if not all_frames:
            logger.warning("No data downloaded for reaction %s", reaction)
            return pd.DataFrame()

        combined = pd.concat(all_frames, ignore_index=True)

        # Apply energy filter
        e_min, e_max = energy_range_eV
        combined = combined[
            combined["energy_eV"].between(e_min, e_max, inclusive="both")
        ]

        # Remove clearly unphysical values
        combined = combined[combined["cross_section_barn"] > 0]
        combined = combined.dropna(subset=["Z", "A", "energy_eV", "cross_section_barn"])
        combined["Z"] = combined["Z"].astype(int)
        combined["A"] = combined["A"].astype(int)

        combined.to_parquet(parquet_path, index=False)
        logger.info("Saved %d records to %s", len(combined), parquet_path)
        return combined

    def download_all_reactions(
        self,
        reactions: Optional[List[str]] = None,
        z_range: Tuple[int, int] = (1, 118),
    ) -> Dict[str, pd.DataFrame]:
        """
        Download EXFOR data for multiple reaction types.

        Parameters
        ----------
        reactions : list of str, optional
            Reactions to download. Defaults to the standard set.
        z_range : tuple
            Atomic number range.

        Returns
        -------
        dict mapping reaction label to DataFrame.
        """
        if reactions is None:
            reactions = list(REACTION_MAP.keys())

        results = {}
        for reaction in reactions:
            logger.info("Processing reaction: %s", reaction)
            df = self.download_reaction(reaction, z_range=z_range)
            results[reaction] = df
            logger.info(
                "  %s: %d data points for %d unique isotopes",
                reaction,
                len(df),
                df[["Z", "A"]].drop_duplicates().shape[0] if not df.empty else 0,
            )

        return results

    def summarize_coverage(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Summarize EXFOR data coverage across reactions and isotopes.

        Parameters
        ----------
        data : dict
            Output of download_all_reactions.

        Returns
        -------
        pd.DataFrame
            Coverage summary with columns: reaction, n_isotopes, n_points,
            z_range, a_range, median_uncertainty_percent.
        """
        rows = []
        for reaction, df in data.items():
            if df.empty:
                rows.append({
                    "reaction": reaction,
                    "n_isotopes": 0,
                    "n_points": 0,
                    "z_min": None,
                    "z_max": None,
                    "a_min": None,
                    "a_max": None,
                    "median_uncertainty_pct": None,
                })
                continue

            unique_isotopes = df[["Z", "A"]].drop_duplicates()
            rows.append({
                "reaction": reaction,
                "n_isotopes": len(unique_isotopes),
                "n_points": len(df),
                "z_min": int(df["Z"].min()),
                "z_max": int(df["Z"].max()),
                "a_min": int(df["A"].min()),
                "a_max": int(df["A"].max()),
                "median_uncertainty_pct": df["uncertainty_percent"].median(),
            })

        return pd.DataFrame(rows)
