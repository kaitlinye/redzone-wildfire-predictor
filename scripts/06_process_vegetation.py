from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask


GRID_FILE = Path(
    "data/interim/grid/california_grid_10km.geojson"
)

EVC_FILE = Path(
    "data/raw/vegetation/landfire_2024/"
    "LF2024_EVC_CONUS/"
    "LF2024_EVC_CONUS.tif"
)

FBFM40_FILE = Path(
    "data/raw/fuels/landfire_2024/"
    "LF2024_FBFM40_CONUS/"
    "LF2024_FBFM40_CONUS.tif"
)


OUTPUT_FOLDER = Path(
    "data/interim/vegetation_by_grid"
)

OUTPUT_FOLDER.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_CSV = (
    OUTPUT_FOLDER
    / "california_landfire_by_grid_2024.csv"
)

OUTPUT_PARQUET = (
    OUTPUT_FOLDER
    / "california_landfire_by_grid_2024.parquet"
)


def valid_pixels(
    values: np.ndarray,
    nodata_value,
) -> np.ndarray:
    values = values.flatten()

    if nodata_value is not None:
        values = values[
            values != nodata_value
        ]

    values = values[
        ~np.isnan(values)
    ]

    return values


def vegetation_mean(
    raster,
    geometry,
) -> float:
    try:
        clipped, _ = mask(
            raster,
            [geometry],
            crop=True,
            filled=False,
        )

        values = clipped[0].compressed()

        if values.size == 0:
            return np.nan

        values = valid_pixels(
            values,
            raster.nodata,
        )

        if values.size == 0:
            return np.nan

        return float(
            np.mean(values)
        )

    except ValueError:
        return np.nan


def dominant_fuel_model(
    raster,
    geometry,
):
    try:
        clipped, _ = mask(
            raster,
            [geometry],
            crop=True,
            filled=False,
        )

        values = clipped[0].compressed()

        if values.size == 0:
            return np.nan

        values = valid_pixels(
            values,
            raster.nodata,
        )

        if values.size == 0:
            return np.nan

        values = values.astype(int)

        unique_values, counts = np.unique(
            values,
            return_counts=True,
        )

        dominant_index = np.argmax(counts)

        return int(
            unique_values[dominant_index]
        )

    except ValueError:
        return np.nan


def main() -> None:
    if not GRID_FILE.exists():
        raise FileNotFoundError(
            f"Grid file not found: {GRID_FILE}"
        )

    if not EVC_FILE.exists():
        raise FileNotFoundError(
            f"EVC raster not found: {EVC_FILE}"
        )

    if not FBFM40_FILE.exists():
        raise FileNotFoundError(
            f"FBFM40 raster not found: {FBFM40_FILE}"
        )

    print("Reading California grid...")

    grid = gpd.read_file(
        GRID_FILE
    )

    if "grid_id" not in grid.columns:
        raise ValueError(
            "The grid file does not contain grid_id."
        )

    print(
        f"Grid cells: {len(grid):,}"
    )

    results = []

    with rasterio.open(EVC_FILE) as evc_raster:
        with rasterio.open(
            FBFM40_FILE
        ) as fuel_raster:

            print(
                f"EVC CRS: {evc_raster.crs}"
            )
            print(
                f"Fuel CRS: {fuel_raster.crs}"
            )

            evc_grid = grid.to_crs(
                evc_raster.crs
            )

            fuel_grid = grid.to_crs(
                fuel_raster.crs
            )

            total = len(grid)

            for position in range(total):
                if (
                    position % 100 == 0
                    or position == total - 1
                ):
                    print(
                        f"Processing grid cell "
                        f"{position + 1:,} "
                        f"of {total:,}"
                    )

                grid_id = grid.iloc[
                    position
                ]["grid_id"]

                evc_geometry = evc_grid.iloc[
                    position
                ].geometry

                fuel_geometry = fuel_grid.iloc[
                    position
                ].geometry

                cover_mean = vegetation_mean(
                    evc_raster,
                    evc_geometry,
                )

                fuel_dominant = dominant_fuel_model(
                    fuel_raster,
                    fuel_geometry,
                )

                results.append(
                    {
                        "grid_id": grid_id,
                        "vegetation_cover_mean":
                            cover_mean,
                        "fuel_model_dominant":
                            fuel_dominant,
                    }
                )

    result_table = pd.DataFrame(
        results
    )

    result_table.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    result_table.to_parquet(
        OUTPUT_PARQUET,
        index=False,
    )

    print()
    print("LANDFIRE processing complete.")
    print(f"Saved CSV: {OUTPUT_CSV}")
    print(
        f"Saved Parquet: {OUTPUT_PARQUET}"
    )
    print(
        "Missing vegetation values: "
        f"{result_table['vegetation_cover_mean'].isna().sum():,}"
    )
    print(
        "Missing fuel values: "
        f"{result_table['fuel_model_dominant'].isna().sum():,}"
    )


if __name__ == "__main__":
    main()
