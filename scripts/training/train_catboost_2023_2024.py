from __future__ import annotations

from pathlib import Path
import sys

import joblib
import pandas as pd
from catboost import CatBoostClassifier

if __package__ in (None, ""):
    sys.path.insert(
        0,
        str(Path(__file__).resolve().parents[2]),
    )

from scripts.training.train_catboost import (
    prepare_catboost_features,
)
from scripts.training.train_logistic_regression import (
    create_next_day_label,
    create_rolling_weather_features,
)
from scripts.utils.evaluation import (
    choose_validation_threshold,
    evaluate_binary_classifier,
)
from scripts.utils.experiment_logging import (
    save_experiment_results,
    save_test_predictions,
)


EXPERIMENT_NAME = "catboost_2023_2024_rolling_weather"
EXPERIMENT_NOTES = (
    "Unweighted CatBoost trained on May-October 2023 and "
    "May-July 2024; validated and tested chronologically in "
    "2024. The fixed 2020-2023 historical FIRMS aggregate is "
    "omitted because it would leak 2023 outcomes into 2023 rows."
)
MINIMUM_RECALL = 0.70
CATEGORICAL_MISSING_VALUE = "__MISSING__"

WEATHER_2023_PATH = Path(
    "data/processed/california_weather_daily_2023.parquet"
)
FIRES_2023_PATH = Path(
    "data/processed/california_fires_daily_2023.parquet"
)
TRAINING_2024_PATH = Path(
    "data/processed/wildfire_training_2024.parquet"
)
LANDFIRE_PATH = Path(
    "data/processed/california_landfire_by_grid_2024.parquet"
)
MODEL_PATH = Path(f"models/{EXPERIMENT_NAME}.joblib")
PREDICTION_PATH = Path(
    "data/processed/predictions/"
    f"{EXPERIMENT_NAME}.parquet"
)

BASE_COLUMNS = [
    "date",
    "grid_id",
    "temperature_max",
    "temperature_min",
    "humidity_mean",
    "humidity_min",
    "precipitation_total",
    "wind_speed_max",
    "wind_direction_dominant",
    "elevation",
    "fire_today",
]

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
]

CATEGORICAL_FEATURES = [
    "wind_direction_dominant",
    "fuel_model_dominant",
    "landfire_missing",
]

FEATURE_COLUMNS = (
    NUMERIC_FEATURES + CATEGORICAL_FEATURES
)


def normalize_dates(values: pd.Series) -> pd.Series:
    return (
        pd.to_datetime(values, utc=True)
        .dt.tz_localize(None)
        .dt.normalize()
    )


def load_2023_grid_days() -> pd.DataFrame:
    weather = pd.read_parquet(WEATHER_2023_PATH)
    fires = pd.read_parquet(FIRES_2023_PATH)

    required_weather = set(BASE_COLUMNS) - {"fire_today"}
    missing_weather = sorted(
        required_weather - set(weather.columns)
    )
    required_fires = {"grid_id", "date", "fire_present"}
    missing_fires = sorted(
        required_fires - set(fires.columns)
    )
    if missing_weather:
        raise ValueError(
            "2023 weather is missing columns: "
            f"{missing_weather}"
        )
    if missing_fires:
        raise ValueError(
            "2023 FIRMS data is missing columns: "
            f"{missing_fires}"
        )

    weather["date"] = normalize_dates(weather["date"])
    fires["date"] = normalize_dates(fires["date"])

    duplicate_weather = int(
        weather.duplicated(["grid_id", "date"]).sum()
    )
    duplicate_fires = int(
        fires.duplicated(["grid_id", "date"]).sum()
    )
    if duplicate_weather or duplicate_fires:
        raise ValueError(
            "Duplicate 2023 grid/date rows were found: "
            f"weather={duplicate_weather:,}, "
            f"FIRMS={duplicate_fires:,}."
        )

    result = weather.merge(
        fires[["grid_id", "date", "fire_present"]],
        on=["grid_id", "date"],
        how="left",
        validate="one_to_one",
    )
    result["fire_today"] = (
        result["fire_present"]
        .fillna(0)
        .astype("int8")
    )
    return result[BASE_COLUMNS].copy()


def load_2024_grid_days() -> pd.DataFrame:
    data = pd.read_parquet(TRAINING_2024_PATH)
    missing = sorted(set(BASE_COLUMNS) - set(data.columns))
    if missing:
        raise ValueError(
            "2024 training data is missing columns: "
            f"{missing}"
        )
    data["date"] = normalize_dates(data["date"])
    return data[BASE_COLUMNS].copy()


def add_year_safe_temporal_features(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate rolling features and labels without crossing year gaps."""
    yearly_frames = []
    for year, year_data in data.groupby(
        data["date"].dt.year,
        sort=True,
    ):
        print(f"Creating rolling features and labels for {year}...")
        featured = create_rolling_weather_features(year_data)
        featured = create_next_day_label(featured)
        yearly_frames.append(featured)

    return pd.concat(yearly_frames, ignore_index=True)


def load_multiyear_data() -> pd.DataFrame:
    data_2023 = load_2023_grid_days()
    data_2024 = load_2024_grid_days()
    data = pd.concat(
        [data_2023, data_2024],
        ignore_index=True,
    )
    data = add_year_safe_temporal_features(data)

    landfire = pd.read_parquet(LANDFIRE_PATH)
    required_landfire = [
        "grid_id",
        "vegetation_cover_mean",
        "fuel_model_dominant",
        "landfire_missing",
    ]
    missing_landfire = sorted(
        set(required_landfire) - set(landfire.columns)
    )
    if missing_landfire:
        raise ValueError(
            "LANDFIRE data is missing columns: "
            f"{missing_landfire}"
        )
    duplicate_landfire = int(
        landfire["grid_id"].duplicated().sum()
    )
    if duplicate_landfire:
        raise ValueError(
            "LANDFIRE data contains "
            f"{duplicate_landfire:,} duplicate grids."
        )

    data = data.merge(
        landfire[required_landfire],
        on="grid_id",
        how="left",
        validate="many_to_one",
    )
    missing_features = sorted(
        set(FEATURE_COLUMNS) - set(data.columns)
    )
    if missing_features:
        raise ValueError(
            "Multi-year data is missing model features: "
            f"{missing_features}"
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
            f"{int(incomplete.sum()):,} rows have missing model "
            "features after the multi-year merge."
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


def split_data(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = data[data["date"] < "2024-08-01"].copy()
    validation = data[
        (data["date"] >= "2024-08-01")
        & (data["date"] < "2024-09-16")
    ].copy()
    test = data[data["date"] >= "2024-09-16"].copy()

    for name, frame in [
        ("Training", train),
        ("Validation", validation),
        ("Testing", test),
    ]:
        if frame.empty:
            raise ValueError(f"{name} split is empty.")
        positives = int(frame["fire_next_day"].sum())
        print(
            f"{name}: {frame['date'].min().date()} to "
            f"{frame['date'].max().date()}, {len(frame):,} rows, "
            f"{positives:,} positives, "
            f"{positives / len(frame):.4%} positive"
        )

    return train, validation, test


def build_model() -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=2000,
        learning_rate=0.03,
        depth=8,
        loss_function="Logloss",
        eval_metric="PRAUC",
        cat_features=CATEGORICAL_FEATURES,
        l2_leaf_reg=5.0,
        random_strength=1.0,
        bootstrap_type="Bernoulli",
        subsample=0.8,
        random_seed=42,
        thread_count=-1,
        allow_writing_files=False,
    )


def main() -> None:
    data = load_multiyear_data()
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
    print("\nTraining multi-year CatBoost...")
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
