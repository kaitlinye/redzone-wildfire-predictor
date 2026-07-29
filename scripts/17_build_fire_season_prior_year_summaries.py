from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


FIRE_SEASON_START = (5, 1)
FIRE_SEASON_END = (10, 30)

DAILY_FIRE_PATHS = {
    2022: Path(
        "data/processed/california_fires_daily_2022.parquet"
    ),
    2023: Path(
        "data/processed/california_fires_daily_2023.parquet"
    ),
    2025: Path(
        "data/processed/california_fires_daily_2025.parquet"
    ),
}
TRAINING_2024_PATH = Path(
    "data/processed/wildfire_training_2024.parquet"
)
GRID_REFERENCE_PATH = Path(
    "data/processed/wildfire_training_2023.parquet"
)
OUTPUT_DIRECTORY = Path("data/processed/prior_year_firms")
COMBINED_OUTPUT_PATH = (
    OUTPUT_DIRECTORY
    / "prior_year_firms_by_grid_2022_2025_may_oct.parquet"
)

SUMMARY_COLUMNS = [
    "grid_id",
    "history_year",
    "feature_year",
    "window_start",
    "window_end",
    "prior_year_firms_detection_count",
    "prior_year_firms_detection_count_log",
    "prior_year_firms_active_days",
]


def fire_season_bounds(
    year: int,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(
        year=year,
        month=FIRE_SEASON_START[0],
        day=FIRE_SEASON_START[1],
    )
    end = pd.Timestamp(
        year=year,
        month=FIRE_SEASON_END[0],
        day=FIRE_SEASON_END[1],
    )
    return start, end


def load_grid_ids() -> pd.Series:
    grid_ids = (
        pd.read_parquet(
            GRID_REFERENCE_PATH,
            columns=["grid_id"],
        )["grid_id"]
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )
    if len(grid_ids) != 4_355:
        raise ValueError(
            "Expected 4,355 grid IDs, found "
            f"{len(grid_ids):,}."
        )
    if grid_ids.isna().any():
        raise ValueError("Grid reference contains missing grid IDs.")
    return grid_ids


def load_daily_fires(year: int) -> pd.DataFrame:
    if year == 2024:
        data = pd.read_parquet(
            TRAINING_2024_PATH,
            columns=[
                "grid_id",
                "date",
                "fire_detection_count",
                "fire_today",
            ],
        ).rename(
            columns={
                "fire_detection_count": "fire_count",
                "fire_today": "fire_present",
            }
        )
        data = data[data["fire_present"] == 1].copy()
    else:
        path = DAILY_FIRE_PATHS[year]
        data = pd.read_parquet(
            path,
            columns=[
                "grid_id",
                "date",
                "fire_count",
                "fire_present",
            ],
        )

    data["date"] = (
        pd.to_datetime(data["date"], utc=True)
        .dt.tz_localize(None)
        .dt.normalize()
    )
    if not (data["date"].dt.year == year).all():
        raise ValueError(
            f"{year} daily FIRMS input contains other years."
        )
    duplicate_rows = int(
        data.duplicated(["grid_id", "date"]).sum()
    )
    if duplicate_rows:
        raise ValueError(
            f"{year} daily FIRMS input contains "
            f"{duplicate_rows:,} duplicate grid/date rows."
        )
    if (
        (data["fire_present"] != 1).any()
        or (data["fire_count"] < 1).any()
    ):
        raise ValueError(
            f"{year} daily FIRMS input contains invalid positive rows."
        )
    return data


def build_summary(
    year: int,
    grid_ids: pd.Series,
) -> pd.DataFrame:
    start, end = fire_season_bounds(year)
    daily = load_daily_fires(year)
    season = daily[
        (daily["date"] >= start)
        & (daily["date"] <= end)
    ].copy()

    if season.empty:
        raise ValueError(
            f"No {year} FIRMS rows fall within {start.date()} "
            f"through {end.date()}."
        )

    aggregate = (
        season.groupby("grid_id")
        .agg(
            prior_year_firms_detection_count=(
                "fire_count",
                "sum",
            ),
            prior_year_firms_active_days=(
                "date",
                "nunique",
            ),
        )
        .reset_index()
    )
    summary = pd.DataFrame(
        {"grid_id": grid_ids}
    ).merge(
        aggregate,
        on="grid_id",
        how="left",
        validate="one_to_one",
    )
    count_columns = [
        "prior_year_firms_detection_count",
        "prior_year_firms_active_days",
    ]
    summary[count_columns] = (
        summary[count_columns]
        .fillna(0)
        .astype("int32")
    )
    summary["prior_year_firms_detection_count_log"] = (
        np.log1p(
            summary["prior_year_firms_detection_count"]
        )
    )
    summary["history_year"] = year
    summary["feature_year"] = year + 1
    summary["window_start"] = start
    summary["window_end"] = end
    summary = summary[SUMMARY_COLUMNS]

    if len(summary) != len(grid_ids):
        raise ValueError(
            f"{year} summary changed the grid count."
        )
    if summary["grid_id"].duplicated().any():
        raise ValueError(
            f"{year} summary contains duplicate grid IDs."
        )
    if (
        int(
            summary[
                "prior_year_firms_detection_count"
            ].sum()
        )
        != int(season["fire_count"].sum())
    ):
        raise ValueError(
            f"{year} detection totals changed during aggregation."
        )

    return summary


def main() -> None:
    grid_ids = load_grid_ids()
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    summaries = []

    for year in [2022, 2023, 2024, 2025]:
        summary = build_summary(year, grid_ids)
        output_path = (
            OUTPUT_DIRECTORY
            / f"prior_year_firms_by_grid_{year}_may_oct.parquet"
        )
        summary.to_parquet(output_path, index=False)
        summaries.append(summary)

        detections = int(
            summary[
                "prior_year_firms_detection_count"
            ].sum()
        )
        active_days = int(
            summary[
                "prior_year_firms_active_days"
            ].sum()
        )
        active_grids = int(
            (
                summary[
                    "prior_year_firms_active_days"
                ]
                > 0
            ).sum()
        )
        print(
            f"{year}: {detections:,} detections, "
            f"{active_days:,} positive grid-days, "
            f"{active_grids:,} active grids"
        )
        print(f"Saved: {output_path}")

    combined = pd.concat(summaries, ignore_index=True)
    if len(combined) != 4 * len(grid_ids):
        raise ValueError(
            "Combined summary has an unexpected row count."
        )
    if combined.duplicated(
        ["grid_id", "history_year"]
    ).any():
        raise ValueError(
            "Combined summary contains duplicate grid/year rows."
        )
    combined.to_parquet(
        COMBINED_OUTPUT_PATH,
        index=False,
    )
    print(f"\nSaved combined summary: {COMBINED_OUTPUT_PATH}")
    print(f"Combined rows: {len(combined):,}")


if __name__ == "__main__":
    main()
