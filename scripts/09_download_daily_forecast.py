from datetime import datetime
import os
from pathlib import Path
import time
from zoneinfo import ZoneInfo

import pandas as pd
import requests


INFERENCE_GRID_FILE = Path(
    "data/inference/california_prediction_grid.parquet"
)
LEGACY_GRID_FILE = Path(
    "data/interim/grid/california_grid_centroids.csv"
)
GRID_REFERENCE_FILE = Path(
    "data/processed/wildfire_training_2024.parquet"
)

OUTPUT_FILE = Path(
    "data/current/weather/recent_weather.parquet"
)
LEGACY_WEATHER_FOLDER = Path("data/current/weather")

API_URL = "https://api.open-meteo.com/v1/forecast"

BATCH_SIZE = int(
    os.environ.get("OPEN_METEO_BATCH_SIZE", "25")
)
WAIT_SECONDS = float(
    os.environ.get("OPEN_METEO_WAIT_SECONDS", "1")
)
PAST_DAYS = 7
HISTORY_DAYS = 30

DAILY_VARIABLES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "relative_humidity_2m_mean",
    "relative_humidity_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
    "wind_direction_10m_dominant",
]


def download_batch(
    batch: pd.DataFrame,
    batch_number: int,
) -> pd.DataFrame:
    latitudes = ",".join(
        batch["centroid_lat"].astype(str)
    )

    longitudes = ",".join(
        batch["centroid_lon"].astype(str)
    )

    params = {
        "latitude": latitudes,
        "longitude": longitudes,
        "daily": ",".join(DAILY_VARIABLES),
        "timezone": "America/Los_Angeles",
        "past_days": PAST_DAYS,
        "forecast_days": 1,
    }

    max_attempts = 5

    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(
                API_URL,
                params=params,
                timeout=300,
            )

            if response.status_code == 429:
                print(
                    f"Batch {batch_number} was rate-limited."
                )
                wait_seconds = min(60 * attempt, 300)
                print(
                    f"Waiting {wait_seconds} seconds..."
                )
                time.sleep(wait_seconds)
                continue

            response.raise_for_status()
            payload = response.json()

            if isinstance(payload, dict):
                payload = [payload]

            if len(payload) != len(batch):
                raise ValueError(
                    "Open-Meteo returned "
                    f"{len(payload)} locations for a "
                    f"{len(batch)}-location batch."
                )

            tables = []

            for position, location_data in enumerate(payload):
                grid_row = batch.iloc[position]

                daily = pd.DataFrame(
                    location_data["daily"]
                )

                daily["grid_id"] = grid_row["grid_id"]
                daily["centroid_lat"] = grid_row[
                    "centroid_lat"
                ]
                daily["centroid_lon"] = grid_row[
                    "centroid_lon"
                ]
                daily["weather_latitude"] = location_data.get(
                    "latitude"
                )
                daily["weather_longitude"] = location_data.get(
                    "longitude"
                )
                daily["elevation"] = location_data.get(
                    "elevation"
                )

                tables.append(daily)

            return pd.concat(
                tables,
                ignore_index=True,
            )

        except requests.exceptions.Timeout:
            print(
                f"Batch {batch_number} timed out. "
                f"Attempt {attempt} of {max_attempts}."
            )

            if attempt < max_attempts:
                print("Waiting 60 seconds before retrying...")
                time.sleep(60)

        except requests.exceptions.RequestException as error:
            print(
                f"Batch {batch_number} failed: {error}"
            )

            if attempt < max_attempts:
                print("Waiting 60 seconds before retrying...")
                time.sleep(60)

    raise RuntimeError(
        f"Batch {batch_number} failed after "
        f"{max_attempts} attempts."
    )


def load_existing_history() -> pd.DataFrame:
    if OUTPUT_FILE.exists():
        return pd.read_parquet(OUTPUT_FILE)

    legacy_files = sorted(
        path
        for path in LEGACY_WEATHER_FOLDER.glob(
            "weather_through_*.parquet"
        )
        if path != OUTPUT_FILE
    )
    if not legacy_files:
        raise FileNotFoundError(
            "No recent weather history was found. The recurring "
            f"pipeline requires {OUTPUT_FILE} or a legacy "
            "weather_through_YYYY-MM-DD.parquet bootstrap file."
        )

    bootstrap_path = legacy_files[-1]
    print(
        "Using legacy weather history as bootstrap: "
        f"{bootstrap_path}"
    )
    return pd.read_parquet(bootstrap_path)


def update_weather_history(
    existing: pd.DataFrame,
    downloaded: pd.DataFrame,
    forecast_date: str,
) -> pd.DataFrame:
    combined = pd.concat(
        [existing, downloaded],
        ignore_index=True,
    )
    combined["date"] = (
        pd.to_datetime(combined["date"])
        .dt.tz_localize(None)
        .dt.normalize()
    )
    combined = (
        combined.drop_duplicates(
            ["grid_id", "date"],
            keep="last",
        )
        .sort_values(["grid_id", "date"])
        .reset_index(drop=True)
    )

    end_date = pd.Timestamp(forecast_date)
    start_date = end_date - pd.Timedelta(
        days=HISTORY_DAYS
    )
    return combined.loc[
        (combined["date"] >= start_date)
        & (combined["date"] <= end_date)
    ].reset_index(drop=True)


def main() -> None:
    forecast_date = (
        datetime.now(ZoneInfo("America/Los_Angeles"))
        .date()
        .isoformat()
    )

    existing_history = load_existing_history()
    existing_history["date"] = pd.to_datetime(
        existing_history["date"]
    )

    if INFERENCE_GRID_FILE.exists():
        grid = pd.read_parquet(
            INFERENCE_GRID_FILE,
            columns=[
                "grid_id",
                "centroid_lat",
                "centroid_lon",
            ],
        )
    elif LEGACY_GRID_FILE.exists():
        grid = pd.read_csv(LEGACY_GRID_FILE)
    elif GRID_REFERENCE_FILE.exists():
        grid = (
            pd.read_parquet(
                GRID_REFERENCE_FILE,
                columns=[
                    "grid_id",
                    "centroid_lat",
                    "centroid_lon",
                ],
            )
            .drop_duplicates("grid_id")
            .reset_index(drop=True)
        )
        print(
            "Grid CSV was not found; using coordinates from "
            f"{GRID_REFERENCE_FILE}."
        )
    else:
        raise FileNotFoundError(
            "No prediction grid was found. Checked "
            f"{INFERENCE_GRID_FILE}, {LEGACY_GRID_FILE}, and "
            f"{GRID_REFERENCE_FILE}."
        )

    required_columns = {
        "grid_id",
        "centroid_lat",
        "centroid_lon",
    }

    missing_columns = required_columns.difference(
        grid.columns
    )

    if missing_columns:
        raise ValueError(
            "The grid file is missing columns: "
            + ", ".join(sorted(missing_columns))
        )

    current_rows = existing_history.loc[
        existing_history["date"].dt.normalize()
        == pd.Timestamp(forecast_date)
    ]
    if (
        current_rows["grid_id"].nunique()
        == grid["grid_id"].nunique()
    ):
        print("Current forecast is already complete:")
        print(OUTPUT_FILE)
        return

    total_batches = (
        len(grid) + BATCH_SIZE - 1
    ) // BATCH_SIZE

    all_batches = []

    print(
        "Downloading weather through feature date "
        f"{forecast_date} ({PAST_DAYS} prior days plus today)"
    )
    print(f"Grid cells: {len(grid):,}")
    print(f"Total batches: {total_batches}")

    for start in range(
        0,
        len(grid),
        BATCH_SIZE,
    ):
        batch_number = (
            start // BATCH_SIZE
        ) + 1

        batch = grid.iloc[
            start:start + BATCH_SIZE
        ].copy()

        print(
            f"Downloading batch "
            f"{batch_number} of {total_batches}..."
        )

        batch_data = download_batch(
            batch=batch,
            batch_number=batch_number,
        )

        all_batches.append(batch_data)

        if batch_number < total_batches:
            time.sleep(WAIT_SECONDS)

    forecast = pd.concat(
        all_batches,
        ignore_index=True,
    )

    forecast = forecast.rename(
        columns={
            "time": "date",
            "temperature_2m_max":
                "temperature_max",
            "temperature_2m_min":
                "temperature_min",
            "relative_humidity_2m_mean":
                "humidity_mean",
            "relative_humidity_2m_min":
                "humidity_min",
            "precipitation_sum":
                "precipitation_total",
            "wind_speed_10m_max":
                "wind_speed_max",
            "wind_direction_10m_dominant":
                "wind_direction_dominant",
        }
    )

    forecast["date"] = pd.to_datetime(
        forecast["date"]
    )

    history = update_weather_history(
        existing=existing_history,
        downloaded=forecast,
        forecast_date=forecast_date,
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = OUTPUT_FILE.with_suffix(
        ".parquet.tmp"
    )
    history.to_parquet(temporary_file, index=False)
    temporary_file.replace(OUTPUT_FILE)

    print()
    print("Download complete.")
    print(f"Saved to: {OUTPUT_FILE}")
    print(f"Downloaded rows: {len(forecast):,}")
    print(f"Retained history rows: {len(history):,}")
    print(
        "Unique grid cells: "
        f"{forecast['grid_id'].nunique():,}"
    )


if __name__ == "__main__":
    main()
