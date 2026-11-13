"""Download the behavioral image-matching data from Zenodo and extract it under BONNER_CACHING_HOME.

The data are Zenodo record 21937066 (https://doi.org/10.5281/zenodo.21937066; all versions:
https://doi.org/10.5281/zenodo.21937065). The record holds one file, behavior.zip, rooted at behavior/, so
extracting it into BONNER_CACHING_HOME gives

    behavior/image_matching_experiment/v00_time=02/data/responses/          (200 ms)
    behavior/image_matching_experiment/v01_time=01_pct=65/data/responses/   (100 ms)

which notebooks/figures/fig_5c.ipynb reads. The download is checked against the checksum the record publishes,
and no existing file is overwritten.

Usage: python scripts/download_behavior_data.py [--output-dir DIR]
"""

import argparse
import hashlib
import io
import os
import zipfile
from pathlib import Path

import requests
from dotenv import load_dotenv

RECORD_URL = "https://zenodo.org/api/records/21937066"
FILE_NAME = "behavior.zip"


def download_behavior_data(output_dir: Path) -> None:
    """Download FILE_NAME from the record, verify its checksum, and extract it into ``output_dir``."""
    record = requests.get(RECORD_URL, timeout=60)
    record.raise_for_status()
    entry = next(f for f in record.json()["files"] if f["key"] == FILE_NAME)
    algorithm, expected = entry["checksum"].split(":")

    response = requests.get(entry["links"]["self"], timeout=60)
    response.raise_for_status()
    digest = hashlib.new(algorithm, response.content).hexdigest()
    if digest != expected:
        raise RuntimeError(f"{FILE_NAME}: {algorithm} {digest} does not match the record's {expected}")

    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        existing = [name for name in archive.namelist() if (output_dir / name).exists()]
        if existing:
            raise FileExistsError(
                f"{len(existing)} of the archive's files already exist under {output_dir}, e.g. {existing[0]}"
            )
        archive.extractall(output_dir)
    print(f"Extracted {FILE_NAME} into {output_dir / 'behavior'}")


if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser(description="Download the behavioral image-matching data from Zenodo.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="base directory to extract into (default: $BONNER_CACHING_HOME)",
    )
    args = parser.parse_args()
    download_behavior_data(args.output_dir or Path(os.environ["BONNER_CACHING_HOME"]))
