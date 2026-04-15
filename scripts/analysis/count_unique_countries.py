#!/usr/bin/env python3
"""Count unique COUNTRY values in SDG6 Tracker CSV files."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


def unique_countries(csv_path: Path) -> set[str]:
    countries: set[str] = set()
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "COUNTRY" not in reader.fieldnames:
            raise ValueError(f"'COUNTRY' column not found in {csv_path}")
        for row in reader:
            value = (row.get("COUNTRY") or "").strip()
            if value:
                countries.add(value)
    return countries


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    data_root = Path(os.environ.get("SDG6_DATA_ROOT", repo_root / "data")).expanduser()
    default_files = [data_root / "R7.csv", data_root / "R8.csv", data_root / "R9.csv"]

    parser = argparse.ArgumentParser(
        description="Count unique COUNTRY values in R7/R8/R9 CSV files."
    )
    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        default=default_files,
        help="CSV files to analyze (defaults: R7.csv R8.csv R9.csv).",
    )
    args = parser.parse_args()

    all_countries: set[str] = set()
    for csv_path in args.files:
        countries = unique_countries(csv_path)
        all_countries.update(countries)
        print(f"{csv_path.name}: {len(countries)} unique COUNTRY values")
    print(f"All files combined: {len(all_countries)} unique COUNTRY values")


if __name__ == "__main__":
    main()
