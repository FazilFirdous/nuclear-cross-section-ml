#!/usr/bin/env python3
"""
Download nuclear data from EXFOR and AME/NuBase sources.

Usage:
    python scripts/download_data.py --reactions n-gamma n-total n-elastic
    python scripts/download_data.py --reactions all --z-min 1 --z-max 118
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.exfor_downloader import EXFORDownloader, REACTION_MAP
from loguru import logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download nuclear cross-section data from EXFOR and nuclear property tables."
    )
    parser.add_argument(
        "--reactions",
        nargs="+",
        default=["n-gamma"],
        choices=list(REACTION_MAP.keys()) + ["all"],
        help="Reaction types to download. Use 'all' for all supported reactions.",
    )
    parser.add_argument(
        "--z-min", type=int, default=1, help="Minimum atomic number Z (default: 1)"
    )
    parser.add_argument(
        "--z-max", type=int, default=118, help="Maximum atomic number Z (default: 118)"
    )
    parser.add_argument(
        "--energy-min", type=float, default=1e-5,
        help="Minimum neutron energy in eV (default: 1e-5, thermal)"
    )
    parser.add_argument(
        "--energy-max", type=float, default=2e7,
        help="Maximum neutron energy in eV (default: 2e7, ~20 MeV)"
    )
    parser.add_argument(
        "--output-dir", default="data/raw/exfor",
        help="Output directory for EXFOR data files"
    )
    parser.add_argument(
        "--download-nuclear-props", action="store_true",
        help="Also download AME2020 and NuBase2020 tables"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose logging"
    )
    return parser.parse_args()


def download_nuclear_properties():
    """Download AME2020 and NuBase2020 tables from IAEA NDS."""
    import requests
    from pathlib import Path

    external_dir = Path("data/external")
    external_dir.mkdir(parents=True, exist_ok=True)

    sources = {
        "ame2020_mass.txt": "https://www-nds.iaea.org/amdc/ame2020/mass_1.mas20.txt",
        "nubase2020.txt": "https://www-nds.iaea.org/amdc/ame2020/nubase_4.mas20.txt",
    }

    session = requests.Session()
    session.headers.update({"User-Agent": "nuclear-cross-section-ml/0.1"})

    for filename, url in sources.items():
        output_path = external_dir / filename
        if output_path.exists():
            logger.info("Already exists: %s", output_path)
            continue

        logger.info("Downloading %s from %s", filename, url)
        try:
            resp = session.get(url, timeout=60)
            resp.raise_for_status()
            output_path.write_text(resp.text)
            logger.info("Saved %s (%d bytes)", output_path, len(resp.content))
        except Exception as exc:
            logger.warning("Failed to download %s: %s", filename, exc)
            logger.info(
                "Please download manually from https://www-nds.iaea.org/amdc/ "
                "and place at %s", output_path
            )


def main():
    args = parse_args()

    log_level = "DEBUG" if args.verbose else "INFO"
    logger.remove()
    logger.add(sys.stderr, level=log_level, colorize=True)

    if args.reactions == ["all"]:
        reactions = list(REACTION_MAP.keys())
    else:
        reactions = args.reactions

    logger.info("Starting EXFOR data download")
    logger.info("  Reactions: %s", reactions)
    logger.info("  Z range: %d - %d", args.z_min, args.z_max)
    logger.info("  Energy range: %.2e - %.2e eV", args.energy_min, args.energy_max)

    downloader = EXFORDownloader(output_dir=args.output_dir)

    results = {}
    for reaction in reactions:
        logger.info("Downloading reaction: %s", reaction)
        df = downloader.download_reaction(
            reaction=reaction,
            z_range=(args.z_min, args.z_max),
            energy_range_eV=(args.energy_min, args.energy_max),
        )
        results[reaction] = df
        if not df.empty:
            n_isotopes = df[["Z", "A"]].drop_duplicates().shape[0]
            logger.info(
                "  %s: %d data points for %d isotopes",
                reaction, len(df), n_isotopes
            )

    # Print coverage summary
    summary = downloader.summarize_coverage(results)
    logger.info("\nData coverage summary:")
    logger.info("\n%s", summary.to_string(index=False))

    if args.download_nuclear_props:
        logger.info("Downloading nuclear property tables (AME2020, NuBase2020)")
        download_nuclear_properties()

    logger.info("Download complete.")


if __name__ == "__main__":
    main()
