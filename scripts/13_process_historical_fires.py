from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from scripts.utils.historical_firms import (
    build_historical_firms_summary,
)


HISTORICAL_FIRE_FILE = Path(
    "data/raw/fires/historical/"
    "california_firms_viirs_snpp_2020_2023.csv"
)

GRID_FILE = Path(
    "data/interim/grid/"
    "california_grid_10km.geojson"
)

OUTPUT_FOLDER = Path("data/processed")

OUTPUT_FOLDER.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_PARQUET = (
    OUTPUT_FOLDER
    / "historical_firms_by_grid_2020_2023.parquet"
)

OUTPUT_CSV = (
    OUTPUT_FOLDER
    / "historical_firms_by_grid_2020_2023.csv"
)


def main() -> None:
    if not HISTORICAL_FIRE_FILE.exists():
        raise FileNotFoundError(
            f"Historical fire file not found: "
            f"{HISTORICAL_FIRE_FILE}"
        )

    if not GRID_FILE.exists():
        raise FileNotFoundError(
            f"Grid file not found: {GRID_FILE}"
        )

    print("Reading historical FIRMS detections...")

    fires = pd.read_csv(
        HISTORICAL_FIRE_FILE
    )

    required_columns = {
        "latitude",
        "longitude",
        "acq_date",
    }

    missing_columns = required_columns.difference(
        fires.columns
    )

    if missing_columns:
        raise ValueError(
            "Historical fire file is missing: "
            + ", ".join(sorted(missing_columns))
        )

    print(f"Historical detections: {len(fires):,}")

    fires["acq_date"] = pd.to_datetime(
        fires["acq_date"],
        errors="coerce",
    )

    fires = fires.dropna(
        subset=[
            "latitude",
            "longitude",
            "acq_date",
        ]
    )

    fires = fires[
        fires["acq_date"].dt.year.between(
            2020,
            2023,
        )
    ].copy()

    print(
        "Detections after date validation: "
        f"{len(fires):,}"
    )

    print("Creating fire point geometries...")

    fire_geometry = [
        Point(longitude, latitude)
        for longitude, latitude in zip(
            fires["longitude"],
            fires["latitude"],
        )
    ]

    fire_points = gpd.GeoDataFrame(
        fires,
        geometry=fire_geometry,
        crs="EPSG:4326",
    )

    print("Reading California grid...")

    grid = gpd.read_file(
        GRID_FILE
    )

    if "grid_id" not in grid.columns:
        raise ValueError(
            "Grid file does not contain grid_id."
        )

    fire_points = fire_points.to_crs(
        grid.crs
    )

    print("Assigning historical FIRMS detections to grid cells...")

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

    print(
        "Detections assigned to California grid: "
        f"{len(joined):,}"
    )

    print("Building historical FIRMS summary...")

    result = build_historical_firms_summary(
        joined,
        date_column="acq_date",
        all_grid_ids=grid["grid_id"],
    )

    result.to_parquet(
        OUTPUT_PARQUET,
        index=False,
    )

    result.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    print()
    print("Historical FIRMS processing complete.")
    print(f"Grid rows: {len(result):,}")
    print(
        "Grid cells with historical FIRMS detections: "
        f"{(
            result[
                'historical_firms_detection_count_2020_2023'
            ] > 0
        ).sum():,}"
    )
    print(
        "Total assigned historical detections: "
        f"{result[
            'historical_firms_detection_count_2020_2023'
        ].sum():,}"
    )
    print(f"Saved Parquet: {OUTPUT_PARQUET}")
    print(f"Saved CSV: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
