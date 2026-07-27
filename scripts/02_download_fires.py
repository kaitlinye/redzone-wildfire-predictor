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


OUTPUT_FOLDER = Path("data/raw/fires/california_2024")
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

COMBINED_OUTPUT = (
    OUTPUT_FOLDER
    / "california_firms_viirs_snpp_2024.csv"
)

START_DATE = pd.Timestamp("2024-05-01")
END_DATE = pd.Timestamp("2024-10-31")

SOURCE = "VIIRS_SNPP_SP"

# Approximate California bounding box:
# west, south, east, north
CALIFORNIA_AREA = "-124.48,32.53,-114.13,42.01"

DAYS_PER_REQUEST = 5
WAIT_SECONDS = 2

BASE_URL = (
    "https://firms.modaps.eosdis.nasa.gov/"
    "api/area/csv"
)


def download_date_block(
    block_start: pd.Timestamp,
    day_count: int,
) -> Path:
    start_text = block_start.strftime("%Y-%m-%d")

    output_file = (
        OUTPUT_FOLDER
        / f"firms_{SOURCE}_{start_text}_{day_count}days.csv"
    )

    if output_file.exists():
        print(
            f"Skipping {start_text}: "
            "file already exists"
        )
        return output_file

    url = (
        f"{BASE_URL}/"
        f"{MAP_KEY}/"
        f"{SOURCE}/"
        f"{CALIFORNIA_AREA}/"
        f"{day_count}/"
        f"{start_text}"
    )

    print(
        f"Downloading {start_text} "
        f"for {day_count} day(s)..."
    )

    response = requests.get(
        url,
        timeout=120,
    )

    response.raise_for_status()

    text = response.text.strip()

    if not text:
        raise ValueError(
            f"NASA returned an empty response for {start_text}"
        )

    if text.startswith("<"):
        raise ValueError(
            "NASA returned HTML instead of CSV. "
            "Check the key, source, and requested dates."
        )

    table = pd.read_csv(
        StringIO(text)
    )

    table.to_csv(
        output_file,
        index=False,
    )

    print(
        f"Saved {len(table):,} detections "
        f"to {output_file.name}"
    )

    return output_file


def main() -> None:
    current_date = START_DATE

    downloaded_files = []

    while current_date <= END_DATE:
        remaining_days = (
            END_DATE - current_date
        ).days + 1

        day_count = min(
            DAYS_PER_REQUEST,
            remaining_days,
        )

        file = download_date_block(
            block_start=current_date,
            day_count=day_count,
        )

        downloaded_files.append(file)

        current_date += pd.Timedelta(
            days=day_count
        )

        time.sleep(WAIT_SECONDS)

    print()
    print(
        f"Combining {len(downloaded_files)} "
        "FIRMS files..."
    )

    tables = []

    for file in downloaded_files:
        table = pd.read_csv(file)
        tables.append(table)

    fires = pd.concat(
        tables,
        ignore_index=True,
    )

    before_duplicates = len(fires)

    duplicate_columns = [
        column
        for column in [
            "latitude",
            "longitude",
            "acq_date",
            "acq_time",
            "satellite",
        ]
        if column in fires.columns
    ]

    if duplicate_columns:
        fires = fires.drop_duplicates(
            subset=duplicate_columns
        )

    removed_duplicates = (
        before_duplicates - len(fires)
    )

    fires.to_csv(
        COMBINED_OUTPUT,
        index=False,
    )

    print(
        f"Combined detections: "
        f"{len(fires):,}"
    )

    print(
        f"Duplicate rows removed: "
        f"{removed_duplicates:,}"
    )

    print(
        f"Saved combined file to: "
        f"{COMBINED_OUTPUT}"
    )


if __name__ == "__main__":
    main()