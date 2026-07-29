from pathlib import Path

import pandas as pd


SOURCE_PATH = Path(
    "data/current/features/next_day_features.parquet"
)
OUTPUT_PATH = Path(
    "data/inference/california_prediction_grid.parquet"
)

INFERENCE_COLUMNS = [
    "grid_id",
    "centroid_lat",
    "centroid_lon",
    "elevation",
    "vegetation_cover_mean",
    "fuel_model_dominant",
    "landfire_missing",
    "historical_firms_detection_count_2020_2023",
]


def main() -> None:
    source = pd.read_parquet(SOURCE_PATH)
    missing_columns = sorted(
        set(INFERENCE_COLUMNS) - set(source.columns)
    )
    if missing_columns:
        raise ValueError(
            "Inference source is missing columns: "
            f"{missing_columns}"
        )

    grid = (
        source[INFERENCE_COLUMNS]
        .drop_duplicates()
        .sort_values("grid_id")
        .reset_index(drop=True)
    )
    if grid["grid_id"].duplicated().any():
        raise ValueError(
            "Inference source contains conflicting rows for a grid."
        )
    if grid.isna().any().any():
        missing = grid.columns[grid.isna().any()].tolist()
        raise ValueError(
            f"Inference grid contains missing values in: {missing}"
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    grid.to_parquet(OUTPUT_PATH, index=False)

    print(f"Grid cells: {len(grid):,}")
    print(f"Saved inference grid to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
