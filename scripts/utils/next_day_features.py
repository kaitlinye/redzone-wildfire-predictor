from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


WEATHER_COLUMNS = [
    "grid_id",
    "date",
    "temperature_max",
    "temperature_min",
    "humidity_mean",
    "humidity_min",
    "precipitation_total",
    "wind_speed_max",
    "wind_direction_dominant",
]

LANDFIRE_COLUMNS = [
    "grid_id",
    "vegetation_cover_mean",
    "fuel_model_dominant",
    "landfire_missing",
]

HISTORICAL_COUNT_COLUMN = (
    "historical_firms_detection_count_2020_2023"
)

REFERENCE_COLUMNS = [
    "grid_id",
    "elevation",
    "centroid_lat",
    "centroid_lon",
]


def discover_parquet_files(
    inputs: Iterable[Path],
) -> list[Path]:
    """Resolve explicit Parquet files and directories of Parquet files."""
    files: list[Path] = []

    for input_path in inputs:
        path = Path(input_path)

        if path.is_dir():
            files.extend(sorted(path.glob("*.parquet")))
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(
                f"Weather input was not found: {path}"
            )

    unique_files = list(dict.fromkeys(files))

    if not unique_files:
        raise FileNotFoundError(
            "No Parquet weather files were found."
        )

    return unique_files


def read_weather_files(
    inputs: Iterable[Path],
) -> pd.DataFrame:
    """Read and concatenate daily weather files."""
    files = discover_parquet_files(inputs)
    frames = []
    for source_order, path in enumerate(files):
        frame = pd.read_parquet(path)
        missing = sorted(
            set(WEATHER_COLUMNS) - set(frame.columns)
        )
        if missing:
            raise ValueError(
                f"{path} is missing weather columns: {missing}"
            )
        duplicate_rows = int(
            frame.duplicated(["grid_id", "date"]).sum()
        )
        if duplicate_rows:
            raise ValueError(
                f"{path} contains {duplicate_rows:,} duplicate "
                "grid/date rows."
            )
        frame["_source_order"] = source_order
        frames.append(frame)

    weather = pd.concat(frames, ignore_index=True)

    weather["date"] = (
        pd.to_datetime(weather["date"])
        .dt.tz_localize(None)
        .dt.normalize()
    )

    # Daily downloads intentionally overlap by 30 days. Prefer the newest
    # file so revised forecast values replace older copies.
    weather = (
        weather.sort_values("_source_order")
        .drop_duplicates(
            ["grid_id", "date"],
            keep="last",
        )
        .drop(columns="_source_order")
        .reset_index(drop=True)
    )

    return weather


def _one_row_per_grid(
    data: pd.DataFrame,
    columns: list[str],
    source_name: str,
) -> pd.DataFrame:
    missing = sorted(set(columns) - set(data.columns))
    if missing:
        raise ValueError(
            f"{source_name} data is missing columns: {missing}"
        )

    result = data[columns].copy()
    duplicate_grids = int(
        result["grid_id"].duplicated().sum()
    )
    if duplicate_grids:
        raise ValueError(
            f"{source_name} data contains "
            f"{duplicate_grids:,} duplicate grid IDs."
        )

    return result


def _build_reference_grid(
    weather: pd.DataFrame,
    reference_data: pd.DataFrame | None,
) -> pd.DataFrame:
    available_weather_columns = [
        column
        for column in REFERENCE_COLUMNS
        if column in weather.columns
    ]
    reference = (
        weather[available_weather_columns]
        .drop_duplicates("grid_id", keep="last")
        .copy()
    )

    if reference_data is not None:
        missing_reference_columns = sorted(
            set(REFERENCE_COLUMNS) - set(reference_data.columns)
        )
        if missing_reference_columns:
            raise ValueError(
                "Grid reference data is missing columns: "
                f"{missing_reference_columns}"
            )
        conflicting_reference_values = (
            reference_data[REFERENCE_COLUMNS]
            .groupby("grid_id")
            .nunique(dropna=False)
            .gt(1)
            .any(axis=1)
        )
        if conflicting_reference_values.any():
            raise ValueError(
                "Grid reference data contains changing coordinates "
                "or elevation for "
                f"{int(conflicting_reference_values.sum()):,} grids."
            )
        external_reference = (
            reference_data[REFERENCE_COLUMNS]
            .drop_duplicates("grid_id", keep="last")
        )
        if reference.empty:
            reference = external_reference
        else:
            reference = reference.merge(
                external_reference,
                on="grid_id",
                how="outer",
                suffixes=("", "_reference"),
                validate="one_to_one",
            )
            for column in REFERENCE_COLUMNS[1:]:
                fallback = f"{column}_reference"
                if fallback in reference:
                    reference[column] = (
                        reference[column]
                        .fillna(reference[fallback])
                    )
                    reference = reference.drop(
                        columns=fallback
                    )

    missing = sorted(
        set(REFERENCE_COLUMNS) - set(reference.columns)
    )
    if missing:
        raise ValueError(
            "Grid coordinates/elevation are unavailable. "
            f"Missing columns: {missing}"
        )

    return reference


def _window_aggregate(
    history: pd.DataFrame,
    feature_date: pd.Timestamp,
    days: int,
    column: str,
    aggregation: str,
    minimum_days: int,
    output_column: str,
) -> pd.DataFrame:
    start_date = feature_date - timedelta(days=days)
    window = history.loc[
        (history["date"] >= start_date)
        & (history["date"] < feature_date),
        ["grid_id", "date", column],
    ]

    grouped = window.groupby("grid_id")[column]
    if aggregation == "sum":
        values = grouped.sum(min_count=minimum_days)
    elif aggregation == "max":
        values = grouped.max()
    elif aggregation == "min":
        values = grouped.min()
    else:
        raise ValueError(
            f"Unsupported aggregation: {aggregation}"
        )

    counts = grouped.count()
    values = values.where(counts >= minimum_days)
    return values.rename(output_column).reset_index()


def build_next_day_features(
    weather: pd.DataFrame,
    landfire: pd.DataFrame,
    historical_firms: pd.DataFrame,
    reference_data: pd.DataFrame | None = None,
    feature_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Build model-ready features for date D, which predict activity on D+1.

    Rolling features use only dates before D. Grids without enough recent
    weather history are rejected instead of being scored with stale rows.
    """
    weather = weather.copy()
    weather["date"] = (
        pd.to_datetime(weather["date"])
        .dt.tz_localize(None)
        .dt.normalize()
    )

    if feature_date is None:
        selected_date = weather["date"].max()
    else:
        selected_date = pd.Timestamp(feature_date).normalize()

    if pd.isna(selected_date):
        raise ValueError("Weather data contains no usable dates.")

    current = weather.loc[
        weather["date"] == selected_date,
        WEATHER_COLUMNS,
    ].copy()
    if current.empty:
        raise ValueError(
            "No weather rows were found for feature date "
            f"{selected_date.date()}."
        )

    duplicate_current = int(
        current["grid_id"].duplicated().sum()
    )
    if duplicate_current:
        raise ValueError(
            "Feature-date weather contains "
            f"{duplicate_current:,} duplicate grid IDs."
        )

    reference = _build_reference_grid(
        weather,
        reference_data,
    )
    expected_grid_ids = set(reference["grid_id"])
    current_grid_ids = set(current["grid_id"])
    missing_current_grids = sorted(
        expected_grid_ids - current_grid_ids
    )
    unexpected_current_grids = sorted(
        current_grid_ids - expected_grid_ids
    )
    if missing_current_grids or unexpected_current_grids:
        raise ValueError(
            "Feature-date weather grid coverage does not match "
            "the prediction reference. "
            f"Missing grids: {len(missing_current_grids):,}; "
            f"unexpected grids: {len(unexpected_current_grids):,}. "
            "Example missing grid IDs: "
            f"{missing_current_grids[:10]}"
        )

    landfire_features = _one_row_per_grid(
        landfire,
        LANDFIRE_COLUMNS,
        "LANDFIRE",
    )
    historical_features = _one_row_per_grid(
        historical_firms,
        ["grid_id", HISTORICAL_COUNT_COLUMN],
        "Historical FIRMS",
    )

    aggregate_specs = [
        (
            7,
            "precipitation_total",
            "sum",
            3,
            "rain_previous_7d",
        ),
        (
            30,
            "precipitation_total",
            "sum",
            7,
            "rain_previous_30d",
        ),
        (
            3,
            "temperature_max",
            "max",
            1,
            "temperature_max_previous_3d",
        ),
        (
            3,
            "humidity_min",
            "min",
            1,
            "humidity_min_previous_3d",
        ),
    ]

    features = current
    for spec in aggregate_specs:
        aggregate = _window_aggregate(
            weather,
            selected_date,
            *spec,
        )
        features = features.merge(
            aggregate,
            on="grid_id",
            how="left",
            validate="one_to_one",
        )

    features = (
        features.merge(
            reference,
            on="grid_id",
            how="left",
            validate="one_to_one",
        )
        .merge(
            landfire_features,
            on="grid_id",
            how="left",
            validate="one_to_one",
        )
        .merge(
            historical_features,
            on="grid_id",
            how="left",
            validate="one_to_one",
        )
    )

    historical_count = features[
        HISTORICAL_COUNT_COLUMN
    ].fillna(0)
    if (historical_count < 0).any():
        raise ValueError(
            "Historical FIRMS detection counts cannot be negative."
        )
    features[HISTORICAL_COUNT_COLUMN] = (
        historical_count.astype("int64")
    )
    features["historical_firms_detection_count_log"] = (
        np.log1p(features[HISTORICAL_COUNT_COLUMN])
    )

    required_complete = [
        *WEATHER_COLUMNS[2:],
        *REFERENCE_COLUMNS[1:],
        *LANDFIRE_COLUMNS[1:],
        "rain_previous_7d",
        "rain_previous_30d",
        "temperature_max_previous_3d",
        "humidity_min_previous_3d",
    ]
    incomplete = features[required_complete].isna().any(axis=1)
    if incomplete.any():
        examples = (
            features.loc[incomplete, "grid_id"]
            .head(10)
            .tolist()
        )
        raise ValueError(
            f"{int(incomplete.sum()):,} feature-date grids have "
            "missing static, current, or rolling inputs. Ensure at "
            "least seven recent prior weather days are available. "
            f"Example grid IDs: {examples}"
        )

    features["feature_date"] = selected_date
    features["prediction_date"] = (
        selected_date + timedelta(days=1)
    )

    return features.sort_values("grid_id").reset_index(drop=True)
