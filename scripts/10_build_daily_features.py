from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(
        0,
        str(Path(__file__).resolve().parents[1]),
    )

from scripts.utils.next_day_features import (
    build_next_day_features,
    read_weather_files,
)


DEFAULT_WEATHER_INPUT = Path("data/current/weather")
DEFAULT_REFERENCE_PATH = Path(
    "data/processed/wildfire_training_2024.parquet"
)
DEFAULT_LANDFIRE_PATH = Path(
    "data/processed/california_landfire_by_grid_2024.parquet"
)
DEFAULT_HISTORICAL_PATH = Path(
    "data/processed/historical_firms_by_grid_2020_2023.parquet"
)
DEFAULT_OUTPUT_PATH = Path(
    "data/current/features/next_day_features.parquet"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build date-D features used to predict next-day "
            "FIRMS activity on D+1."
        )
    )
    parser.add_argument(
        "--weather",
        type=Path,
        nargs="+",
        default=[DEFAULT_WEATHER_INPUT],
        help=(
            "Daily weather Parquet file(s), or directories containing "
            "them. Include the feature date and at least seven prior "
            "dates."
        ),
    )
    parser.add_argument(
        "--feature-date",
        help=(
            "Feature date in YYYY-MM-DD format. Defaults to the latest "
            "date in the weather inputs."
        ),
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=DEFAULT_REFERENCE_PATH,
        help="Grid reference containing elevation and centroid columns.",
    )
    parser.add_argument(
        "--landfire",
        type=Path,
        default=DEFAULT_LANDFIRE_PATH,
    )
    parser.add_argument(
        "--historical-firms",
        type=Path,
        default=DEFAULT_HISTORICAL_PATH,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weather = read_weather_files(args.weather)
    reference = pd.read_parquet(args.reference)
    landfire = pd.read_parquet(args.landfire)
    historical_firms = pd.read_parquet(
        args.historical_firms
    )

    features = build_next_day_features(
        weather=weather,
        landfire=landfire,
        historical_firms=historical_firms,
        reference_data=reference,
        feature_date=args.feature_date,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(args.output, index=False)

    feature_date = features["feature_date"].iloc[0].date()
    prediction_date = (
        features["prediction_date"].iloc[0].date()
    )
    print(f"Feature date: {feature_date}")
    print(f"Prediction date: {prediction_date}")
    print(f"Grid cells: {len(features):,}")
    print(f"Saved features to: {args.output}")


if __name__ == "__main__":
    main()
