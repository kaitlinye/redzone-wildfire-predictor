from pathlib import Path
import time

import pandas as pd
import requests


GRID_FILE = Path(
    "data/interim/grid/california_grid_centroids.csv"
)

OUTPUT_FOLDER = Path(
    "data/raw/weather/california_2024"
)
OUTPUT_FOLDER.mkdir(
    parents=True,
    exist_ok=True,
)

FAILURE_FILE = OUTPUT_FOLDER / "failed_batches.csv"

START_DATE = "2024-05-01"
END_DATE = "2024-10-31"

BATCH_SIZE = 25
WAIT_SECONDS = 20

URL = "https://archive-api.open-meteo.com/v1/archive"

DAILY_VARIABLES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "relative_humidity_2m_mean",
    "relative_humidity_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
    "wind_direction_10m_dominant",
]


def save_failure(
    batch: pd.DataFrame,
    batch_number: int,
    error_message: str,
) -> None:
    failure = pd.DataFrame(
        {
            "batch_number": [batch_number],
            "first_grid_id": [batch.iloc[0]["grid_id"]],
            "last_grid_id": [batch.iloc[-1]["grid_id"]],
            "error": [error_message],
        }
    )

    failure.to_csv(
        FAILURE_FILE,
        mode="a",
        header=not FAILURE_FILE.exists(),
        index=False,
    )


def download_batch(
    batch: pd.DataFrame,
    batch_number: int,
) -> bool:
    output_file = (
        OUTPUT_FOLDER
        / f"weather_batch_{batch_number:04d}.csv"
    )

    if output_file.exists():
        print(
            f"Skipping batch {batch_number}: "
            "file already exists"
        )
        return True

    latitudes = ",".join(
        batch["centroid_lat"].astype(str)
    )

    longitudes = ",".join(
        batch["centroid_lon"].astype(str)
    )

    params = {
        "latitude": latitudes,
        "longitude": longitudes,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "daily": ",".join(DAILY_VARIABLES),
        "timezone": "UTC",
    }

    while True:
        max_attempts = 3

        for attempt in range(
            1,
            max_attempts + 1,
        ):
            try:
                response = requests.get(
                    URL,
                    params=params,
                    timeout=180,
                )

                if response.status_code == 429:
                    if attempt < max_attempts:
                        wait_time = 120 * attempt

                        print(
                            f"Batch {batch_number} "
                            "was rate-limited."
                        )
                        print(
                            f"Waiting {wait_time} "
                            "seconds before retrying."
                        )

                        time.sleep(wait_time)
                        continue

                    print(
                        f"Batch {batch_number} "
                        "is still rate-limited."
                    )
                    print(
                        "Pausing for 30 minutes, "
                        "then retrying the same batch."
                    )

                    time.sleep(1800)
                    break

                response.raise_for_status()

                payload = response.json()

                if isinstance(payload, dict):
                    payload = [payload]

                if len(payload) != len(batch):
                    raise ValueError(
                        "Open-Meteo returned "
                        f"{len(payload)} locations "
                        f"for a batch of "
                        f"{len(batch)} cells."
                    )

                tables = []

                for position, location_data in enumerate(
                    payload
                ):
                    grid_row = batch.iloc[position]

                    if "daily" not in location_data:
                        raise ValueError(
                            "No daily weather returned for "
                            f"{grid_row['grid_id']}"
                        )

                    daily = pd.DataFrame(
                        location_data["daily"]
                    )

                    daily["grid_id"] = (
                        grid_row["grid_id"]
                    )

                    daily["requested_latitude"] = (
                        grid_row["centroid_lat"]
                    )

                    daily["requested_longitude"] = (
                        grid_row["centroid_lon"]
                    )

                    daily["weather_latitude"] = (
                        location_data.get("latitude")
                    )

                    daily["weather_longitude"] = (
                        location_data.get("longitude")
                    )

                    daily["elevation"] = (
                        location_data.get("elevation")
                    )

                    tables.append(daily)

                combined = pd.concat(
                    tables,
                    ignore_index=True,
                )

                combined.to_csv(
                    output_file,
                    index=False,
                )

                print(
                    f"Saved batch {batch_number}: "
                    f"{len(batch)} grid cells, "
                    f"{len(combined):,} daily rows"
                )

                return True

            except requests.RequestException as error:
                if attempt < max_attempts:
                    wait_time = 60 * attempt

                    print(
                        f"Batch {batch_number} "
                        f"attempt {attempt} failed: "
                        f"{error}"
                    )
                    print(
                        f"Waiting {wait_time} "
                        "seconds before retrying."
                    )

                    time.sleep(wait_time)
                    continue

                print(
                    f"Batch {batch_number} had a "
                    "network error after all retries."
                )
                print(
                    "Pausing for 30 minutes, then "
                    "retrying the same batch."
                )

                time.sleep(1800)
                break

            except Exception as error:
                error_message = str(error)

                print(
                    f"Batch {batch_number} failed: "
                    f"{error_message}"
                )

                save_failure(
                    batch=batch,
                    batch_number=batch_number,
                    error_message=error_message,
                )

                return False


def main() -> None:
    if not GRID_FILE.exists():
        raise FileNotFoundError(
            f"Grid file was not found: {GRID_FILE}"
        )

    grid = pd.read_csv(GRID_FILE)

    required_columns = {
        "grid_id",
        "centroid_lat",
        "centroid_lon",
    }

    missing_columns = (
        required_columns - set(grid.columns)
    )

    if missing_columns:
        raise ValueError(
            "Grid file is missing columns: "
            f"{sorted(missing_columns)}"
        )

    total_batches = (
        len(grid) + BATCH_SIZE - 1
    ) // BATCH_SIZE

    print(
        "Downloading California weather for "
        f"{len(grid):,} grid cells."
    )

    print(
        f"Date range: {START_DATE} to {END_DATE}"
    )

    print(
        f"Number of batches: {total_batches}"
    )

    successful_batches = 0
    failed_batches = 0

    for start_position in range(
        0,
        len(grid),
        BATCH_SIZE,
    ):
        batch_number = (
            start_position // BATCH_SIZE
        ) + 1

        batch = grid.iloc[
            start_position:
            start_position + BATCH_SIZE
        ].copy()

        print()
        print(
            f"Starting batch {batch_number} "
            f"of {total_batches}..."
        )

        success = download_batch(
            batch=batch,
            batch_number=batch_number,
        )

        if success:
            successful_batches += 1
        else:
            failed_batches += 1

        time.sleep(WAIT_SECONDS)

    completed_files = list(
        OUTPUT_FOLDER.glob(
            "weather_batch_*.csv"
        )
    )

    print()
    print(
        "California weather download finished."
    )
    print(
        f"Successful or previously completed "
        f"batches: {successful_batches}"
    )
    print(
        f"Failed batches this run: "
        f"{failed_batches}"
    )
    print(
        f"Completed batch files: "
        f"{len(completed_files)} "
        f"of {total_batches}"
    )


if __name__ == "__main__":
    main()