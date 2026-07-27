from datetime import date, timedelta
from io import StringIO
from pathlib import Path
import os
import time

import pandas as pd
import requests
from dotenv import load_dotenv


load_dotenv("config/secrets.env")

MAP_KEY = os.getenv("FIRMS_MAP_KEY")

if not MAP_KEY:
    raise ValueError(
        "FIRMS_MAP_KEY was not found in config/secrets.env"
    )


SOURCE = "VIIRS_SNPP_SP"

AREA = "-124.48,32.53,-114.13,42.01"

START_DATE = date(2020, 1, 1)
END_DATE = date(2023, 12, 31)

DAY_RANGE = 5

OUTPUT_FOLDER = Path(
    "data/raw/fires/historical"
)

OUTPUT_FOLDER.mkdir(
    parents=True,
    exist_ok=True,
)

CHUNK_FOLDER = OUTPUT_FOLDER / "chunks"

CHUNK_FOLDER.mkdir(
    parents=True,
    exist_ok=True,
)

FINAL_FILE = (
    OUTPUT_FOLDER
    / "california_firms_viirs_snpp_2020_2023.csv"
)

WAIT_SECONDS = 3
MAX_ATTEMPTS = 5


def download_chunk(
    start_date: date,
    day_range: int,
) -> pd.DataFrame:
    url = (
        "https://firms.modaps.eosdis.nasa.gov/"
        f"api/area/csv/{MAP_KEY}/"
        f"{SOURCE}/"
        f"{AREA}/"
        f"{day_range}/"
        f"{start_date.isoformat()}"
    )

    for attempt in range(
        1,
        MAX_ATTEMPTS + 1,
    ):
        try:
            response = requests.get(
                url,
                timeout=180,
            )

            if response.status_code == 429:
                print(
                    "Rate limited. Waiting 10 minutes..."
                )
                time.sleep(600)
                continue

            response.raise_for_status()

            text = response.text.strip()

            if not text:
                return pd.DataFrame()

            if text.lower().startswith(
                "invalid"
            ):
                raise RuntimeError(text)

            return pd.read_csv(
                StringIO(text)
            )

        except requests.exceptions.RequestException as error:
            print(
                f"Attempt {attempt} failed: {error}"
            )

            if attempt < MAX_ATTEMPTS:
                print(
                    "Waiting 60 seconds before retrying..."
                )
                time.sleep(60)

    raise RuntimeError(
        f"Download failed for {start_date}"
    )


def main() -> None:
    current_date = START_DATE

    downloaded_files = []

    while current_date <= END_DATE:
        remaining_days = (
            END_DATE - current_date
        ).days + 1

        day_range = min(
            DAY_RANGE,
            remaining_days,
        )

        chunk_end = (
            current_date
            + timedelta(days=day_range - 1)
        )

        chunk_file = (
            CHUNK_FOLDER
            / (
                f"fires_"
                f"{current_date.isoformat()}_"
                f"{chunk_end.isoformat()}.csv"
            )
        )

        if chunk_file.exists():
            print(
                f"Already downloaded: {chunk_file.name}"
            )

            downloaded_files.append(
                chunk_file
            )

            current_date = (
                current_date
                + timedelta(days=day_range)
            )

            continue

        print(
            f"Downloading "
            f"{current_date.isoformat()} "
            f"through "
            f"{chunk_end.isoformat()}..."
        )

        data = download_chunk(
            current_date,
            day_range,
        )

        data.to_csv(
            chunk_file,
            index=False,
        )

        downloaded_files.append(
            chunk_file
        )

        print(
            f"Saved {len(data):,} detections."
        )

        current_date = (
            current_date
            + timedelta(days=day_range)
        )

        time.sleep(WAIT_SECONDS)

    print()
    print("Combining historical fire files...")

    tables = []

    for file in downloaded_files:
        table = pd.read_csv(file)

        if not table.empty:
            tables.append(table)

    if not tables:
        raise RuntimeError(
            "No historical fire detections were downloaded."
        )

    combined = pd.concat(
        tables,
        ignore_index=True,
    )

    duplicate_columns = [
        column
        for column in [
            "latitude",
            "longitude",
            "acq_date",
            "acq_time",
            "satellite",
        ]
        if column in combined.columns
    ]

    if duplicate_columns:
        before = len(combined)

        combined = combined.drop_duplicates(
            subset=duplicate_columns
        )

        removed = before - len(combined)

        print(
            f"Removed duplicate rows: {removed:,}"
        )

    combined.to_csv(
        FINAL_FILE,
        index=False,
    )

    print()
    print("Historical fire download complete.")
    print(f"Rows: {len(combined):,}")
    print(f"Saved to: {FINAL_FILE}")

    if "acq_date" in combined.columns:
        print(
            "Earliest date: "
            f"{combined['acq_date'].min()}"
        )
        print(
            "Latest date: "
            f"{combined['acq_date'].max()}"
        )


if __name__ == "__main__":
    main()