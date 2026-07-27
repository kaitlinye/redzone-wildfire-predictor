from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from scripts.utils.evaluation import (
    choose_validation_threshold,
    evaluate_binary_classifier,
)
from scripts.utils.experiment_logging import (
    save_experiment_results,
    save_test_predictions,
)

EXPERIMENT_NAME = "logistic_rolling_weather_2024"

TRAINING_PATH = Path(
    "data/processed/wildfire_training_2024.parquet"
)

LANDFIRE_PATH = Path(
    "data/processed/california_landfire_by_grid_2024.parquet"
)

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_and_merge_data() -> pd.DataFrame:
    training_data = pd.read_parquet(TRAINING_PATH)
    landfire_data = pd.read_parquet(LANDFIRE_PATH)

    required_training_columns = [
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

    required_landfire_columns = [
        "grid_id",
        "vegetation_cover_mean",
        "fuel_model_dominant",
        "landfire_missing",
    ]

    missing_training_columns = [
        column
        for column in required_training_columns
        if column not in training_data.columns
    ]

    missing_landfire_columns = [
        column
        for column in required_landfire_columns
        if column not in landfire_data.columns
    ]

    if missing_training_columns:
        raise ValueError(
            "Training data is missing columns: "
            f"{missing_training_columns}"
        )

    if missing_landfire_columns:
        raise ValueError(
            "LANDFIRE data is missing columns: "
            f"{missing_landfire_columns}"
        )

    duplicate_landfire_grids = (
        landfire_data["grid_id"]
        .duplicated()
        .sum()
    )

    if duplicate_landfire_grids > 0:
        raise ValueError(
            "LANDFIRE data contains "
            f"{duplicate_landfire_grids} duplicate grid IDs."
        )

    training_data["date"] = pd.to_datetime(
        training_data["date"]
    )

    data = training_data.merge(
        landfire_data,
        on="grid_id",
        how="left",
        validate="many_to_one",
    )

    if len(data) != len(training_data):
        raise ValueError(
            "The LANDFIRE merge changed the number of rows."
        )

    print(f"Training rows before merge: {len(training_data):,}")
    print(f"Rows after merge: {len(data):,}")

    print("\nLANDFIRE missingness:")
    print(
        data[
            [
                "vegetation_cover_mean",
                "fuel_model_dominant",
                "landfire_missing",
            ]
        ]
        .isna()
        .mean()
        .sort_values(ascending=False)
    )

    return data

def create_rolling_weather_features(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create historical weather features separately for each grid cell.

    The input must contain:
    - grid_id
    - date
    - precipitation_total
    - temperature_max
    - humidity_min

    A new DataFrame is returned; the original is not modified.
    """
    required_columns: set[str] = {
        "grid_id",
        "date",
        "precipitation_total",
        "temperature_max",
        "humidity_min",
    }

    missing_columns: set[str] = (
        required_columns - set(data.columns)
    )

    if missing_columns:
        raise ValueError(
            "Cannot create rolling weather features. "
            f"Missing columns: {sorted(missing_columns)}"
        )

    result: pd.DataFrame = data.sort_values(
        ["grid_id", "date"]
    ).copy()

    grouped = result.groupby(
        "grid_id",
        group_keys=False,
        sort=False,
    )

    result["rain_previous_7d"] = (
        grouped["precipitation_total"]
        .transform(
            lambda values: (
                values.shift(1)
                .rolling(
                    window=7,
                    min_periods=3,
                )
                .sum()
            )
        )
    )

    result["rain_previous_30d"] = (
        grouped["precipitation_total"]
        .transform(
            lambda values: (
                values.shift(1)
                .rolling(
                    window=30,
                    min_periods=7,
                )
                .sum()
            )
        )
    )

    result["temperature_max_previous_3d"] = (
        grouped["temperature_max"]
        .transform(
            lambda values: (
                values.shift(1)
                .rolling(
                    window=3,
                    min_periods=1,
                )
                .max()
            )
        )
    )

    result["humidity_min_previous_3d"] = (
        grouped["humidity_min"]
        .transform(
            lambda values: (
                values.shift(1)
                .rolling(
                    window=3,
                    min_periods=1,
                )
                .min()
            )
        )
    )

    return result

def create_next_day_label(
    data: pd.DataFrame,
) -> pd.DataFrame:
    data = data.sort_values(
        ["grid_id", "date"]
    ).copy()

    data["next_date"] = (
        data.groupby("grid_id")["date"]
        .shift(-1)
    )

    data["next_fire_value"] = (
        data.groupby("grid_id")["fire_today"]
        .shift(-1)
    )

    data["days_to_next_row"] = (
        data["next_date"] - data["date"]
    ).dt.days

    data["fire_next_day"] = (
        data["next_fire_value"]
        .where(data["days_to_next_row"] == 1)
    )

    rows_before = len(data)

    data = data.dropna(
        subset=["fire_next_day"]
    ).copy()

    removed_rows = rows_before - len(data)

    data["fire_next_day"] = (
        data["fire_next_day"]
        .astype(int)
    )

    data = data.drop(
        columns=[
            "next_date",
            "next_fire_value",
            "days_to_next_row",
        ]
    )

    print(
        "\nRemoved "
        f"{removed_rows:,} rows because the next calendar day "
        "was missing or unknown."
    )

    print("\nNext-day label distribution:")
    print(
        data["fire_next_day"]
        .value_counts(dropna=False)
        .sort_index()
    )

    print("\nNext-day positive percentage:")
    print(
        data["fire_next_day"]
        .value_counts(normalize=True)
        .sort_index()
        .mul(100)
        .round(4)
    )

    return data

def prepare_features(
    data: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    list[str],
    list[str],
    list[str],
]:
    target_column = "fire_next_day"

    numeric_features: list[str] = [
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

    categorical_features = [
        "wind_direction_dominant",
        "fuel_model_dominant",
        "landfire_missing",
    ]

    feature_columns = (
        numeric_features
        + categorical_features
    )

    missing_features = [
        column
        for column in feature_columns
        if column not in data.columns
    ]

    if missing_features:
        raise ValueError(
            "Merged data is missing model features: "
            f"{missing_features}"
        )

    data = data.dropna(
        subset=[target_column]
    ).copy()

    data[target_column] = (
        data[target_column]
        .astype(int)
    )

    data["wind_direction_dominant"] = (
        data["wind_direction_dominant"]
        .astype("string")
    )

    data["fuel_model_dominant"] = (
        data["fuel_model_dominant"]
        .astype("string")
    )

    data["landfire_missing"] = (
        data["landfire_missing"]
        .fillna(True)
        .astype("string")
    )

    return (
        data,
        numeric_features,
        categorical_features,
        feature_columns,
    )


def split_data_by_time(
    data: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    train_data = data[
        data["date"] < "2024-08-01"
    ].copy()

    validation_data = data[
        (data["date"] >= "2024-08-01")
        & (data["date"] < "2024-09-16")
    ].copy()

    test_data = data[
        data["date"] >= "2024-09-16"
    ].copy()

    splits = [
        ("Training", train_data),
        ("Validation", validation_data),
        ("Testing", test_data),
    ]

    for name, split in splits:
        if split.empty:
            raise ValueError(
                f"{name} data is empty. "
                "Adjust the date boundaries."
            )

        positives = int(
            split["fire_next_day"].sum()
        )

        total = len(split)

        positive_rate = (
            positives / total
            if total > 0
            else 0
        )

        start_date = split["date"].min().date()
        end_date = split["date"].max().date()

        print(
            f"{name}: "
            f"{start_date} to {end_date}, "
            f"{total:,} rows, "
            f"{positives:,} positives, "
            f"{positive_rate:.4%} positive"
        )

        if positives == 0:
            raise ValueError(
                f"{name} split has no positive fire rows. "
                "Choose different date boundaries."
            )

    return (
        train_data,
        validation_data,
        test_data,
    )

def build_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
) -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                ),
            ),
        ]
    )

    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ]
    )

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessing",
                preprocessing,
            ),
            (
                "model",
                model,
            ),
        ]
    )

    return pipeline


def evaluate_model(
    name: str,
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float,
) -> dict[str, float]:
    probabilities = (
        pipeline.predict_proba(X)[:, 1]
    )

    predictions = (
        probabilities >= threshold
    ).astype(int)

    pr_auc = average_precision_score(
        y,
        probabilities,
    )

    roc_auc = roc_auc_score(
        y,
        probabilities,
    )

    print(f"\n{name} results")
    print("-" * 50)
    print(f"Threshold: {threshold:.4f}")
    print(f"PR AUC: {pr_auc:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y,
            predictions,
            digits=4,
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    print(
        confusion_matrix(
            y,
            predictions,
        )
    )

    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "probabilities": probabilities,
        "predictions": predictions,
    }



def main() -> None:
    data = load_and_merge_data()

    data = create_rolling_weather_features(data)
    data = create_next_day_label(data)

    monthly_counts = (
        data.groupby(data["date"].dt.to_period("M"))
        ["fire_next_day"]
        .agg(
            total_rows="count",
            positive_rows="sum",
        )
    )

    monthly_counts["positive_percent"] = (
        100
        * monthly_counts["positive_rows"]
        / monthly_counts["total_rows"]
    )

    print(monthly_counts)

    (
        data,
        numeric_features,
        categorical_features,
        feature_columns,
    ) = prepare_features(data)

    print("\nSample next-day labels:")
    print(
        data[
            [
                "grid_id",
                "date",
                "fire_today",
                "fire_next_day",
            ]
        ].head(20)
    )

    (
        train_data,
        validation_data,
        test_data,
    ) = split_data_by_time(data)

    target_column = "fire_next_day"

    X_train = train_data[
        feature_columns
    ]
    y_train = train_data[
        target_column
    ]

    X_validation = validation_data[
        feature_columns
    ]
    y_validation = validation_data[
        target_column
    ]

    X_test = test_data[
        feature_columns
    ]
    y_test = test_data[
        target_column
    ]

    pipeline = build_pipeline(
        numeric_features,
        categorical_features,
    )

    print(
        "\nTraining logistic regression..."
    )

    pipeline.fit(
        X_train,
        y_train,
    )

    validation_probabilities = (
        pipeline.predict_proba(X_validation)[:, 1]
    )

    chosen_threshold = choose_validation_threshold(
        y_true=y_validation,
        probabilities=validation_probabilities,
        minimum_recall=0.70,
    )

    validation_results = evaluate_binary_classifier(
        name="Validation",
        model=pipeline,
        X=X_validation,
        y=y_validation,
        threshold=chosen_threshold,
    )

    test_results = evaluate_binary_classifier(
        name="Test",
        model=pipeline,
        X=X_test,
        y=y_test,
        threshold=chosen_threshold,
    )

    save_experiment_results(
        experiment_name=(
            EXPERIMENT_NAME
        ),
        model_name="LogisticRegression",
        train_data=train_data,
        validation_data=validation_data,
        test_data=test_data,
        threshold=chosen_threshold,
        validation_results=validation_results,
        test_results=test_results,
        feature_columns=feature_columns,
        notes=(
            "Balanced class weights; daily and rolling "
            "weather features; LANDFIRE features."
        ),
    )

    save_test_predictions(
        test_data=test_data,
        probabilities=test_results["probabilities"],
        predictions=test_results["predictions"],
        output_path=Path(
            "data/processed/predictions/"
            f"{EXPERIMENT_NAME}.parquet"
        ),
    )

    model_path = (
        MODEL_DIR
        / f"{EXPERIMENT_NAME}.joblib"
    )

    joblib.dump(
        {
            "pipeline": pipeline,
            "threshold": chosen_threshold,
            "feature_columns": feature_columns,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "target_column": target_column,
        },
        model_path,
    )

    print(
        f"\nSaved trained model to {model_path}"
    )



if __name__ == "__main__":
    main()