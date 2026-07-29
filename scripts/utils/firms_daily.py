from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree


MAXIMUM_ASSIGNMENT_DISTANCE_METERS = 10_000
EXPECTED_GRID_COUNT = 4_355


def load_grid_reference(path: Path) -> pd.DataFrame:
    grid = pd.read_parquet(
        path,
        columns=[
            "grid_id",
            "centroid_lat",
            "centroid_lon",
        ],
    ).drop_duplicates("grid_id")

    if len(grid) != EXPECTED_GRID_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_GRID_COUNT:,} grid centroids, "
            f"found {len(grid):,}."
        )
    if grid.isna().any().any():
        raise ValueError(
            "Grid reference contains missing identifiers or coordinates."
        )
    return grid.reset_index(drop=True)


def assign_to_nearest_grid(
    fires: pd.DataFrame,
    grid: pd.DataFrame,
) -> pd.DataFrame:
    transformer = Transformer.from_crs(
        "EPSG:4326",
        "EPSG:3310",
        always_xy=True,
    )
    grid_x, grid_y = transformer.transform(
        grid["centroid_lon"].to_numpy(),
        grid["centroid_lat"].to_numpy(),
    )
    fire_x, fire_y = transformer.transform(
        fires["longitude"].to_numpy(),
        fires["latitude"].to_numpy(),
    )

    tree = cKDTree(np.column_stack([grid_x, grid_y]))
    distances, positions = tree.query(
        np.column_stack([fire_x, fire_y]),
        k=1,
    )
    too_far = distances > MAXIMUM_ASSIGNMENT_DISTANCE_METERS
    if too_far.any():
        raise ValueError(
            f"{int(too_far.sum()):,} FIRMS detections are more than "
            f"{MAXIMUM_ASSIGNMENT_DISTANCE_METERS / 1000:g} km "
            "from the nearest grid centroid."
        )

    result = fires.copy()
    result["grid_id"] = (
        grid["grid_id"].to_numpy()[positions]
    )
    result["assignment_distance_m"] = distances
    return result


def process_firms_daily(
    *,
    fire_path: Path,
    grid_reference_path: Path,
    output_path: Path,
    expected_year: int,
    expected_satellite: str = "SNPP",
) -> pd.DataFrame:
    fires = pd.read_csv(fire_path)
    required = {
        "latitude",
        "longitude",
        "acq_date",
        "acq_time",
        "frp",
        "satellite",
        "instrument",
    }
    missing = sorted(required - set(fires.columns))
    if missing:
        raise ValueError(
            f"FIRMS file is missing columns: {missing}"
        )

    satellites = set(
        fires["satellite"].dropna().astype(str).unique()
    )
    if satellites != {expected_satellite}:
        raise ValueError(
            f"Expected only {expected_satellite} detections, "
            f"found satellites: {sorted(satellites)}."
        )

    fires["date"] = pd.to_datetime(
        fires["acq_date"],
        errors="coerce",
    ).dt.normalize()
    invalid = fires[
        ["latitude", "longitude", "date"]
    ].isna().any(axis=1)
    if invalid.any():
        raise ValueError(
            f"{int(invalid.sum()):,} FIRMS rows have invalid "
            "coordinates or dates."
        )
    if not (fires["date"].dt.year == expected_year).all():
        raise ValueError(
            f"The FIRMS input contains dates outside {expected_year}."
        )

    duplicate_key = [
        "latitude",
        "longitude",
        "acq_date",
        "acq_time",
        "satellite",
        "instrument",
    ]
    duplicate_detections = int(
        fires.duplicated(duplicate_key).sum()
    )
    if duplicate_detections:
        raise ValueError(
            f"FIRMS input contains {duplicate_detections:,} "
            "duplicate detections."
        )

    grid = load_grid_reference(grid_reference_path)
    assigned = assign_to_nearest_grid(fires, grid)

    daily = (
        assigned.groupby(["grid_id", "date"])
        .agg(
            fire_count=("grid_id", "size"),
            total_frp=("frp", "sum"),
            mean_frp=("frp", "mean"),
            max_frp=("frp", "max"),
        )
        .reset_index()
    )
    daily["fire_present"] = 1
    daily = daily[
        [
            "grid_id",
            "date",
            "fire_present",
            "fire_count",
            "total_frp",
            "mean_frp",
            "max_frp",
        ]
    ].sort_values(["date", "grid_id"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(output_path, index=False)

    print(
        f"{expected_satellite} FIRMS detections: {len(fires):,}"
    )
    print(
        "Maximum nearest-centroid distance: "
        f"{assigned['assignment_distance_m'].max() / 1000:.2f} km"
    )
    print(f"Positive grid-days: {len(daily):,}")
    print(f"Active grids: {daily['grid_id'].nunique():,}")
    print(f"Saved daily FIRMS data to: {output_path}")

    return daily
