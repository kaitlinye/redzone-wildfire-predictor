from pathlib import Path

import pandas as pd


WEATHER_FILE = Path(
    "data/interim/weather_daily/"
    "california_weather_daily_2024.parquet"
)

FIRE_FILE = Path(
    "data/interim/fires_daily/"
    "fires_by_grid_day_2024.parquet"
)

GRID_FILE = Path(
    "data/interim/grid/"
    "california_grid_centroids.csv"
)

LANDFIRE_FILE = Path(
    "data/interim/vegetation_by_grid/"
    "california_landfire_by_grid_2024.parquet"
)

HISTORICAL_FIRE_COUNT_FILE = Path(
    "data/interim/fires_historical_by_grid/"
    "historical_fire_count_2020_2023.parquet"
)

OUTPUT_FOLDER = Path(
    "data/processed"
)

OUTPUT_FOLDER.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_PARQUET = (
    OUTPUT_FOLDER
    / "wildfire_training_2024.parquet"
)

OUTPUT_CSV = (
    OUTPUT_FOLDER
    / "wildfire_training_2024_sample.csv"
)


print("Reading weather data...")

weather = pd.read_parquet(
    WEATHER_FILE
)

print("Reading fire data...")

fires = pd.read_parquet(
    FIRE_FILE
)

print("Reading grid coordinates...")

grid = pd.read_csv(
    GRID_FILE
)

print("Reading LANDFIRE data...")

landfire = pd.read_parquet(
    LANDFIRE_FILE
)

print("Reading historical fire counts...")

historical_fire_counts = pd.read_parquet(
    HISTORICAL_FIRE_COUNT_FILE
)


weather["date"] = pd.to_datetime(
    weather["date"]
).dt.tz_localize(None)

fires["date"] = pd.to_datetime(
    fires["date"]
).dt.tz_localize(None)


print("Adding grid coordinates...")

weather = weather.merge(
    grid,
    on="grid_id",
    how="left",
)


print("Adding vegetation and fuel data...")

weather = weather.merge(
    landfire[
        [
            "grid_id",
            "vegetation_cover_mean",
            "fuel_model_dominant",
            "landfire_missing",
        ]
    ],
    on="grid_id",
    how="left",
)


print("Adding historical fire counts...")

weather = weather.merge(
    historical_fire_counts[
        [
            "grid_id",
            "historical_fire_count_2020_2023",
        ]
    ],
    on="grid_id",
    how="left",
)


print("Adding fire detections...")

data = weather.merge(
    fires,
    on=[
        "grid_id",
        "date",
    ],
    how="left",
)


data["fire_detection_count"] = (
    data["fire_detection_count"]
    .fillna(0)
    .astype(int)
)

data["fire_today"] = (
    data["fire_today"]
    .fillna(0)
    .astype(int)
)

if "maximum_frp" in data.columns:
    data["maximum_frp"] = (
        data["maximum_frp"]
        .fillna(0)
    )

if "mean_frp" in data.columns:
    data["mean_frp"] = (
        data["mean_frp"]
        .fillna(0)
    )


print("Checking LANDFIRE values...")

data["vegetation_cover_mean"] = (
    data["vegetation_cover_mean"]
    .fillna(0)
)

data["fuel_model_dominant"] = (
    data["fuel_model_dominant"]
    .fillna(0)
    .astype(int)
)

data["landfire_missing"] = (
    data["landfire_missing"]
    .fillna(1)
    .astype(int)
)


print("Checking historical fire counts...")

data["historical_fire_count_2020_2023"] = (
    data["historical_fire_count_2020_2023"]
    .fillna(0)
    .astype(int)
)


print("Creating rolling weather features...")

data = data.sort_values(
    [
        "grid_id",
        "date",
    ]
).reset_index(drop=True)


data["rain_7d"] = (
    data.groupby("grid_id")[
        "precipitation_total"
    ]
    .transform(
        lambda values:
        values.rolling(
            7,
            min_periods=1,
        ).sum()
    )
)

data["rain_30d"] = (
    data.groupby("grid_id")[
        "precipitation_total"
    ]
    .transform(
        lambda values:
        values.rolling(
            30,
            min_periods=1,
        ).sum()
    )
)

data["temperature_max_3d"] = (
    data.groupby("grid_id")[
        "temperature_max"
    ]
    .transform(
        lambda values:
        values.rolling(
            3,
            min_periods=1,
        ).max()
    )
)

data["humidity_min_3d"] = (
    data.groupby("grid_id")[
        "humidity_min"
    ]
    .transform(
        lambda values:
        values.rolling(
            3,
            min_periods=1,
        ).min()
    )
)


print("Creating next-day fire label...")

data["fire_next_day"] = (
    data.groupby("grid_id")[
        "fire_today"
    ]
    .shift(-1)
)

data = data.dropna(
    subset=[
        "fire_next_day",
    ]
)

data["fire_next_day"] = (
    data["fire_next_day"]
    .astype(int)
)


print("Saving final model dataset...")

data.to_parquet(
    OUTPUT_PARQUET,
    index=False,
)

data.head(5000).to_csv(
    OUTPUT_CSV,
    index=False,
)


print()
print("Training dataset complete.")

print(
    f"Final rows: "
    f"{len(data):,}"
)

print(
    f"Positive next-day fire rows: "
    f"{data['fire_next_day'].sum():,}"
)

print(
    f"Positive rate: "
    f"{data['fire_next_day'].mean():.6f}"
)

print(
    "Missing vegetation values: "
    f"{data['vegetation_cover_mean'].isna().sum():,}"
)

print(
    "Missing fuel values: "
    f"{data['fuel_model_dominant'].isna().sum():,}"
)

print(
    "LANDFIRE missing rows: "
    f"{data['landfire_missing'].sum():,}"
)

print(
    "Missing historical fire counts: "
    f"{data[
        'historical_fire_count_2020_2023'
    ].isna().sum():,}"
)

print(
    "Grid cells with historical fires: "
    f"{data.loc[
        data['historical_fire_count_2020_2023'] > 0,
        'grid_id'
    ].nunique():,}"
)

print(
    f"Saved final Parquet to: "
    f"{OUTPUT_PARQUET}"
)

print(
    f"Saved sample CSV to: "
    f"{OUTPUT_CSV}"
)
