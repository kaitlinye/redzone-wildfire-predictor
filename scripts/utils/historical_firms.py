from collections.abc import Iterable

import pandas as pd


HISTORICAL_START_YEAR = 2020
HISTORICAL_END_YEAR = 2023

HISTORICAL_FIRMS_COLUMNS = [
    "historical_firms_detection_count_2020_2023",
    "historical_fire_active_days_2020_2023",
    "historical_fire_active_years_2020_2023",
]


def build_historical_firms_summary(
    firms_data: pd.DataFrame,
    *,
    date_column: str = "date",
    all_grid_ids: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Summarize 2020-2023 FIRMS detections to one row per grid."""
    required_columns = {
        "grid_id",
        date_column,
    }
    missing_columns = required_columns - set(
        firms_data.columns
    )

    if missing_columns:
        raise ValueError(
            "Historical FIRMS data is missing columns: "
            f"{sorted(missing_columns)}"
        )

    data = firms_data[
        [
            "grid_id",
            date_column,
        ]
    ].copy()

    data["date"] = pd.to_datetime(
        data[date_column],
        errors="coerce",
    ).dt.normalize()

    invalid_date_count = int(
        data["date"].isna().sum()
    )

    if invalid_date_count:
        raise ValueError(
            "Historical FIRMS data contains "
            f"{invalid_date_count:,} invalid dates."
        )

    data = data.loc[
        data["date"].dt.year.between(
            HISTORICAL_START_YEAR,
            HISTORICAL_END_YEAR,
        )
    ].copy()

    data["year"] = data["date"].dt.year

    summary = (
        data.groupby(
            "grid_id",
            as_index=False,
        )
        .agg(
            historical_firms_detection_count_2020_2023=(
                "date",
                "size",
            ),
            historical_fire_active_days_2020_2023=(
                "date",
                "nunique",
            ),
            historical_fire_active_years_2020_2023=(
                "year",
                "nunique",
            ),
        )
    )

    if all_grid_ids is not None:
        grid_ids = pd.DataFrame(
            {
                "grid_id": list(all_grid_ids),
            }
        )

        duplicate_grid_count = int(
            grid_ids["grid_id"].duplicated().sum()
        )

        if duplicate_grid_count:
            raise ValueError(
                "The grid contains "
                f"{duplicate_grid_count:,} duplicate grid IDs."
            )

        summary = grid_ids.merge(
            summary,
            on="grid_id",
            how="left",
            validate="one_to_one",
        )

    summary[HISTORICAL_FIRMS_COLUMNS] = (
        summary[HISTORICAL_FIRMS_COLUMNS]
        .fillna(0)
        .astype("int32")
    )

    validate_historical_firms_summary(summary)

    return summary


def validate_historical_firms_summary(
    summary: pd.DataFrame,
) -> None:
    """Validate the schema and one-row-per-grid contract."""
    required_columns = {
        "grid_id",
        *HISTORICAL_FIRMS_COLUMNS,
    }
    missing_columns = required_columns - set(
        summary.columns
    )

    if missing_columns:
        raise ValueError(
            "Historical FIRMS summary is missing columns: "
            f"{sorted(missing_columns)}"
        )

    duplicate_grid_count = int(
        summary["grid_id"].duplicated().sum()
    )

    if duplicate_grid_count:
        raise ValueError(
            "Historical FIRMS summary contains "
            f"{duplicate_grid_count:,} duplicate grid IDs."
        )

    missing_grid_count = int(
        summary["grid_id"].isna().sum()
    )

    if missing_grid_count:
        raise ValueError(
            "Historical FIRMS summary contains "
            f"{missing_grid_count:,} missing grid IDs."
        )

    non_numeric_columns = [
        column
        for column in HISTORICAL_FIRMS_COLUMNS
        if not pd.api.types.is_numeric_dtype(
            summary[column]
        )
    ]

    if non_numeric_columns:
        raise ValueError(
            "Historical FIRMS summary columns must be numeric: "
            f"{non_numeric_columns}"
        )

    missing_value_count = int(
        summary[HISTORICAL_FIRMS_COLUMNS]
        .isna()
        .sum()
        .sum()
    )

    if missing_value_count:
        raise ValueError(
            "Historical FIRMS summary contains "
            f"{missing_value_count:,} missing count values."
        )

    negative_value_count = int(
        (summary[HISTORICAL_FIRMS_COLUMNS] < 0)
        .sum()
        .sum()
    )

    if negative_value_count:
        raise ValueError(
            "Historical FIRMS summary contains negative counts."
        )

    invalid_active_year_count = int(
        (
            summary[
                "historical_fire_active_years_2020_2023"
            ]
            > (
                HISTORICAL_END_YEAR
                - HISTORICAL_START_YEAR
                + 1
            )
        ).sum()
    )

    if invalid_active_year_count:
        raise ValueError(
            "Historical FIRMS summary contains active-year "
            "counts greater than four."
        )

    invalid_active_day_count = int(
        (
            summary[
                "historical_fire_active_days_2020_2023"
            ]
            > summary[
                "historical_firms_detection_count_2020_2023"
            ]
        ).sum()
    )

    if invalid_active_day_count:
        raise ValueError(
            "Historical FIRMS summary contains active-day "
            "counts greater than detection counts."
        )

    invalid_year_day_count = int(
        (
            summary[
                "historical_fire_active_years_2020_2023"
            ]
            > summary[
                "historical_fire_active_days_2020_2023"
            ]
        ).sum()
    )

    if invalid_year_day_count:
        raise ValueError(
            "Historical FIRMS summary contains active-year "
            "counts greater than active-day counts."
        )
