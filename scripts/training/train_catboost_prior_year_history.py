from __future__ import annotations

from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(
        0,
        str(Path(__file__).resolve().parents[2]),
    )

from scripts.training.train_catboost import (
    prepare_catboost_features,
)
from scripts.training.train_catboost_2023_2024 import (
    CATEGORICAL_FEATURES,
    CATEGORICAL_MISSING_VALUE,
    LANDFIRE_PATH,
    add_year_safe_temporal_features,
    build_model,
    load_2023_grid_days,
    load_2024_grid_days,
    split_data,
)
from scripts.utils.evaluation import (
    choose_validation_threshold,
    evaluate_binary_classifier,
)
from scripts.utils.experiment_logging import (
    save_experiment_results,
    save_test_predictions,
)


EXPERIMENT_NAME = "catboost_prior_year_history_2022_2024"
EXPERIMENT_NOTES = (
    "Unweighted CatBoost trained on 2023 and May-July 2024 "
    "daily rows. Leakage-safe prior-year SNPP FIRMS detection "
    "count and active-day features use 2022 for 2023 rows and "
    "2023 for 2024 rows. Validation and testing are chronological "
    "2024 periods."
)
MINIMUM_RECALL = 0.70

FIRE_PATHS = {
    2022: Path(
        "data/processed/california_fires_daily_2022.parquet"
    ),
    2023: Path(
        "data/processed/california_fires_daily_2023.parquet"
    ),
}
MODEL_PATH = Path(f"models/{EXPERIMENT_NAME}.joblib")
PREDICTION_PATH = Path(
    "data/processed/predictions/"
    f"{EXPERIMENT_NAME}.parquet"
)

NUMERIC_FEATURES = [
    "temperature_max",
    "temperature_min",
    "humidity_mean",
    "humidity_min",
    "precipitation_total",
    "wind_speed_max",
    "elevation",
    "vegetation_cover_mean",
    "rain_previous_7d",
    "rain_previous_30d",
    "temperature_max_previous_3d",
    "humidity_min_previous_3d",
    "prior_year_firms_detection_count_log",
    "prior_year_firms_active_days",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_prior_year_summaries() -> pd.DataFrame:
    summaries = []

    for history_year, path in FIRE_PATHS.items():
        daily = pd.read_parquet(path)
        required = {
            "grid_id",
            "date",
            "fire_count",
            "fire_present",
        }
        missing = sorted(required - set(daily.columns))
        if missing:
            raise ValueError(
                f"{path} is missing columns: {missing}"
            )

        daily["date"] = (
            pd.to_datetime(daily["date"], utc=True)
            .dt.tz_localize(None)
            .dt.normalize()
        )
        if not (daily["date"].dt.year == history_year).all():
            raise ValueError(
                f"{path} contains dates outside {history_year}."
            )
        duplicate_rows = int(
            daily.duplicated(["grid_id", "date"]).sum()
        )
        if duplicate_rows:
            raise ValueError(
                f"{path} contains {duplicate_rows:,} duplicate "
                "grid/date rows."
            )
        if (daily["fire_count"] < 1).any():
            raise ValueError(
                f"{path} contains non-positive FIRMS counts."
            )

        summary = (
            daily.groupby("grid_id")
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
        summary["feature_year"] = history_year + 1
        summaries.append(summary)

        print(
            f"{history_year} prior history: {len(daily):,} "
            f"positive grid-days, "
            f"{int(daily['fire_count'].sum()):,} detections, "
            f"{daily['grid_id'].nunique():,} active grids"
        )

    return pd.concat(summaries, ignore_index=True)


def load_data() -> pd.DataFrame:
    data = pd.concat(
        [
            load_2023_grid_days(),
            load_2024_grid_days(),
        ],
        ignore_index=True,
    )
    data = add_year_safe_temporal_features(data)
    data["feature_year"] = data["date"].dt.year

    prior_history = build_prior_year_summaries()
    data = data.merge(
        prior_history,
        on=["grid_id", "feature_year"],
        how="left",
        validate="many_to_one",
    )
    history_columns = [
        "prior_year_firms_detection_count",
        "prior_year_firms_active_days",
    ]
    data[history_columns] = (
        data[history_columns]
        .fillna(0)
        .astype("int32")
    )
    data["prior_year_firms_detection_count_log"] = np.log1p(
        data["prior_year_firms_detection_count"]
    )

    landfire_columns = [
        "grid_id",
        "vegetation_cover_mean",
        "fuel_model_dominant",
        "landfire_missing",
    ]
    landfire = pd.read_parquet(LANDFIRE_PATH)
    missing_landfire = sorted(
        set(landfire_columns) - set(landfire.columns)
    )
    if missing_landfire:
        raise ValueError(
            "LANDFIRE data is missing columns: "
            f"{missing_landfire}"
        )
    if landfire["grid_id"].duplicated().any():
        raise ValueError(
            "LANDFIRE data contains duplicate grid IDs."
        )

    data = data.merge(
        landfire[landfire_columns],
        on="grid_id",
        how="left",
        validate="many_to_one",
    )

    non_rolling_features = [
        column
        for column in FEATURE_COLUMNS
        if column
        not in {
            "rain_previous_7d",
            "rain_previous_30d",
            "temperature_max_previous_3d",
            "humidity_min_previous_3d",
        }
    ]
    incomplete = data[non_rolling_features].isna().any(axis=1)
    if incomplete.any():
        raise ValueError(
            f"{int(incomplete.sum()):,} rows have missing current, "
            "static, or prior-year features."
        )

    rows_before = len(data)
    data = data.dropna(subset=FEATURE_COLUMNS).copy()
    print(
        "Removed "
        f"{rows_before - len(data):,} early-season rows without "
        "enough prior weather for rolling features."
    )

    return data.sort_values(
        ["date", "grid_id"]
    ).reset_index(drop=True)


def main() -> None:
    data = load_data()
    train_data, validation_data, test_data = split_data(data)

    X_train = prepare_catboost_features(
        train_data[FEATURE_COLUMNS],
        CATEGORICAL_FEATURES,
    )
    X_validation = prepare_catboost_features(
        validation_data[FEATURE_COLUMNS],
        CATEGORICAL_FEATURES,
    )
    X_test = prepare_catboost_features(
        test_data[FEATURE_COLUMNS],
        CATEGORICAL_FEATURES,
    )
    y_train = train_data["fire_next_day"]
    y_validation = validation_data["fire_next_day"]
    y_test = test_data["fire_next_day"]

    model = build_model()
    print("\nTraining prior-year-history CatBoost...")
    model.fit(
        X_train,
        y_train,
        eval_set=(X_validation, y_validation),
        use_best_model=True,
        early_stopping_rounds=100,
        verbose=50,
    )

    validation_scores = model.predict_proba(X_validation)[:, 1]
    threshold = choose_validation_threshold(
        y_true=y_validation,
        probabilities=validation_scores,
        minimum_recall=MINIMUM_RECALL,
    )
    validation_results = evaluate_binary_classifier(
        name="Validation",
        model=model,
        X=X_validation,
        y=y_validation,
        threshold=threshold,
    )
    test_results = evaluate_binary_classifier(
        name="Test",
        model=model,
        X=X_test,
        y=y_test,
        threshold=threshold,
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "threshold": threshold,
            "feature_columns": FEATURE_COLUMNS,
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "categorical_missing_value": (
                CATEGORICAL_MISSING_VALUE
            ),
            "target_column": "fire_next_day",
            "best_iteration": model.get_best_iteration(),
            "training_years": [2023, 2024],
            "prior_history_years": [2022, 2023],
            "prior_history_sensor": "SNPP VIIRS",
            "historical_firms_aggregate_included": False,
        },
        MODEL_PATH,
    )
    print(f"\nSaved trained model to {MODEL_PATH}")

    save_test_predictions(
        test_data=test_data,
        probabilities=test_results["probabilities"],
        predictions=test_results["predictions"],
        output_path=PREDICTION_PATH,
    )
    save_experiment_results(
        experiment_name=EXPERIMENT_NAME,
        model_name="CatBoostClassifier",
        train_data=train_data,
        validation_data=validation_data,
        test_data=test_data,
        threshold=threshold,
        validation_results=validation_results,
        test_results=test_results,
        feature_columns=FEATURE_COLUMNS,
        notes=EXPERIMENT_NOTES,
    )


if __name__ == "__main__":
    main()
