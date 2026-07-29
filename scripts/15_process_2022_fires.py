from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(
        0,
        str(Path(__file__).resolve().parents[1]),
    )

from scripts.utils.firms_daily import (
    process_firms_daily,
)


def main() -> None:
    process_firms_daily(
        fire_path=Path(
            "data/raw/raw_fires/2022/"
            "DL_FIRE_SV-C2_779511 (Unzipped Files)/"
            "fire_archive_SV-C2_779511.csv"
        ),
        grid_reference_path=Path(
            "data/processed/wildfire_training_2023.parquet"
        ),
        output_path=Path(
            "data/processed/california_fires_daily_2022.parquet"
        ),
        expected_year=2022,
        expected_satellite="SNPP",
    )


if __name__ == "__main__":
    main()
