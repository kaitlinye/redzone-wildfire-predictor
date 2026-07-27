from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import box


INPUT_FILE = Path(
    "data/raw/boundaries/"
    "cb_2025_us_state_500k/"
    "cb_2025_us_state_500k.shp"
)

OUTPUT_FOLDER = Path("data/interim/grid")
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_FOLDER / "california_grid_10km.geojson"

CALIFORNIA_FIPS = "06"
GRID_SIZE_METERS = 10_000
CALIFORNIA_PROJECTED_CRS = "EPSG:3310"
OUTPUT_CRS = "EPSG:4326"


print("Reading United States state boundaries...")

states = gpd.read_file(INPUT_FILE)

print("Selecting California...")

california = states[states["STATEFP"] == CALIFORNIA_FIPS].copy()

if california.empty:
    raise ValueError("California was not found in the boundary file.")

california = california.to_crs(CALIFORNIA_PROJECTED_CRS)

min_x, min_y, max_x, max_y = california.total_bounds

print("Creating 10 km grid cells...")

cells = []

x = min_x
while x < max_x:
    y = min_y

    while y < max_y:
        cells.append(
            box(
                x,
                y,
                x + GRID_SIZE_METERS,
                y + GRID_SIZE_METERS,
            )
        )

        y += GRID_SIZE_METERS

    x += GRID_SIZE_METERS

grid = gpd.GeoDataFrame(
    {"geometry": cells},
    crs=CALIFORNIA_PROJECTED_CRS,
)

print("Keeping only cells that intersect California...")

grid = gpd.overlay(
    grid,
    california[["geometry"]],
    how="intersection",
)

grid = grid.reset_index(drop=True)

grid["grid_id"] = [
    f"CA-{number:05d}"
    for number in range(1, len(grid) + 1)
]

centroids = grid.geometry.centroid

centroid_points = gpd.GeoSeries(
    centroids,
    crs=CALIFORNIA_PROJECTED_CRS,
).to_crs(OUTPUT_CRS)

grid["centroid_lon"] = centroid_points.x
grid["centroid_lat"] = centroid_points.y

grid = grid.to_crs(OUTPUT_CRS)

grid = grid[
    [
        "grid_id",
        "centroid_lat",
        "centroid_lon",
        "geometry",
    ]
]

grid.to_file(
    OUTPUT_FILE,
    driver="GeoJSON",
)

print(f"Created {len(grid):,} California grid cells.")
print(f"Saved grid to: {OUTPUT_FILE}")

CENTROIDS_FILE = OUTPUT_FOLDER / "california_grid_centroids.csv"

grid[
    [
        "grid_id",
        "centroid_lat",
        "centroid_lon",
    ]
].to_csv(
    CENTROIDS_FILE,
    index=False,
)

print(f"Saved centroids to: {CENTROIDS_FILE}")