from pathlib import Path

import pandas as pd


INPUT_FOLDER = Path("data/raw/weather/california_2024")
OUTPUT_FOLDER = Path("data/interim/weather_daily")
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

CSV_OUTPUT = OUTPUT_FOLDER / "california_weather_daily_2024.csv"
PARQUET_OUTPUT = OUTPUT_FOLDER / "california_weather_daily_2024.parquet"


files = sorted(INPUT_FOLDER.glob("weather_batch_*.csv"))

if not files:
    raise FileNotFoundError(
        "No weather batch CSV files were found."
    )

print(f"Found {len(files)} weather batch files.")

tables = []

for file in files:
    print(f"Reading {file.name}")
    table = pd.read_csv(file)
    tables.append(table)

weather = pd.concat(
    tables,
    ignore_index=True,
)

weather["time"] = pd.to_datetime(
    weather["time"],
    utc=True,
)

weather = weather.rename(
    columns={
        "time": "date",
        "temperature_2m_max": "temperature_max",
        "temperature_2m_min": "temperature_min",
        "relative_humidity_2m_mean": "humidity_mean",
        "relative_humidity_2m_min": "humidity_min",
        "precipitation_sum": "precipitation_total",
        "wind_speed_10m_max": "wind_speed_max",
        "wind_direction_10m_dominant": "wind_direction_dominant",
    }
)

weather = weather.sort_values(
    ["grid_id", "date"]
).reset_index(drop=True)

duplicate_count = weather.duplicated(
    subset=["grid_id", "date"]
).sum()

print(f"Duplicate grid-date rows: {duplicate_count}")

if duplicate_count > 0:
    weather = weather.drop_duplicates(
        subset=["grid_id", "date"]
    )

weather.to_csv(
    CSV_OUTPUT,
    index=False,
)

weather.to_parquet(
    PARQUET_OUTPUT,
    index=False,
)

print(f"Created {len(weather):,} statewide daily rows.")
print(f"Saved CSV to: {CSV_OUTPUT}")
print(f"Saved Parquet to: {PARQUET_OUTPUT}")