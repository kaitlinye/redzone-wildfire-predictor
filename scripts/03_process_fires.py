from pathlib import Path

import geopandas as gpd
import pandas as pd


FIRE_FILE = Path(
    "data/raw/fires/california_2024/"
    "california_firms_viirs_snpp_2024.csv"
)

GRID_FILE = Path(
    "data/interim/grid/"
    "california_grid_10km.geojson"
)

OUTPUT_FOLDER = Path(
    "data/interim/fires_daily"
)

OUTPUT_FOLDER.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_CSV = (
    OUTPUT_FOLDER
    / "fires_by_grid_day_2024.csv"
)

OUTPUT_PARQUET = (
    OUTPUT_FOLDER
    / "fires_by_grid_day_2024.parquet"
)


print("Reading FIRMS fire detections...")

fires = pd.read_csv(FIRE_FILE)

required_columns = {
    "latitude",
    "longitude",
    "acq_date",
    "acq_time",
}

missing_columns = (
    required_columns - set(fires.columns)
)

if missing_columns:
    raise ValueError(
        "Fire data is missing columns: "
        f"{sorted(missing_columns)}"
    )


print("Creating fire timestamps...")

fires["acq_time"] = (
    fires["acq_time"]
    .astype(str)
    .str.zfill(4)
)

fires["fire_timestamp"] = pd.to_datetime(
    fires["acq_date"].astype(str)
    + " "
    + fires["acq_time"].str[:2]
    + ":"
    + fires["acq_time"].str[2:],
    utc=True,
)

fires["date"] = (
    fires["fire_timestamp"]
    .dt.floor("D")
    .dt.tz_localize(None)
)


print("Turning fire detections into map points...")

fire_points = gpd.GeoDataFrame(
    fires,
    geometry=gpd.points_from_xy(
        fires["longitude"],
        fires["latitude"],
    ),
    crs="EPSG:4326",
)


print("Reading California grid...")

grid = gpd.read_file(GRID_FILE)


print("Assigning fires to grid cells...")

joined = gpd.sjoin(
    fire_points,
    grid[
        [
            "grid_id",
            "geometry",
        ]
    ],
    how="inner",
    predicate="within",
)


print("Summarizing fires by grid cell and date...")

aggregation = {
    "fire_detection_count": (
        "grid_id",
        "size",
    ),
}

if "frp" in joined.columns:
    aggregation["maximum_frp"] = (
        "frp",
        "max",
    )

    aggregation["mean_frp"] = (
        "frp",
        "mean",
    )

daily_fires = (
    joined.groupby(
        [
            "grid_id",
            "date",
        ]
    )
    .agg(**aggregation)
    .reset_index()
)

daily_fires["fire_today"] = 1

daily_fires = daily_fires.sort_values(
    [
        "grid_id",
        "date",
    ]
).reset_index(drop=True)


daily_fires.to_csv(
    OUTPUT_CSV,
    index=False,
)

daily_fires.to_parquet(
    OUTPUT_PARQUET,
    index=False,
)


print(
    f"Original fire detections: "
    f"{len(fires):,}"
)

print(
    f"Detections assigned to California grid: "
    f"{len(joined):,}"
)

print(
    f"Grid-date fire rows created: "
    f"{len(daily_fires):,}"
)

print(
    f"Saved CSV to: "
    f"{OUTPUT_CSV}"
)

print(
    f"Saved Parquet to: "
    f"{OUTPUT_PARQUET}"
)