from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from scripts.training.train_logistic_regression import (
    create_next_day_label,
    create_rolling_weather_features,
    load_and_merge_data,
    prepare_features,
    split_data_by_time,
)
from scripts.utils.evaluation import (
    choose_validation_threshold,
    evaluate_binary_classifier,
)
from scripts.utils.experiment_logging import (
    save_experiment_results,
    save_test_predictions,
)


EXPERIMENT_NAME = (
    "gradient_boosting_rolling_history_2024"
)
EXPERIMENT_NOTES = (
    "Histogram gradient boosting with balanced class "
    "weights; daily and rolling weather; LANDFIRE; "
    "historical FIRMS features from 2020-2023."
)
MINIMUM_RECALL = 0.70
WIND_DIRECTION_COLUMN = (
    "wind_direction_dominant"
)

MODEL_PATH = Path(
    f"models/{EXPERIMENT_NAME}.joblib"
)
PREDICTION_PATH = Path(
    "data/processed/predictions/"
    f"{EXPERIMENT_NAME}.parquet"
)


def prepare_gradient_boosting_features(
    data: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
) -> tuple[
    pd.DataFrame,
    list[str],
    list[str],
    list[str],
]:
    """Represent wind degrees numerically for histogram boosting."""
    result = data.copy()
    result[WIND_DIRECTION_COLUMN] = (
        pd.to_numeric(
            result[WIND_DIRECTION_COLUMN],
            errors="coerce",
        )
    )

    gradient_numeric_features = [
        *numeric_features,
        WIND_DIRECTION_COLUMN,
    ]
    gradient_categorical_features = [
        column
        for column in categorical_features
        if column != WIND_DIRECTION_COLUMN
    ]
    feature_columns = (
        gradient_numeric_features
        + gradient_categorical_features
    )

    return (
        result,
        gradient_numeric_features,
        gradient_categorical_features,
        feature_columns,
    )


def build_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
) -> Pipeline:
    """Build a memory-efficient histogram-boosting pipeline."""
    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                ),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent",
                ),
            ),
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown=(
                        "use_encoded_value"
                    ),
                    unknown_value=-1,
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

    categorical_indices = list(
        range(
            len(numeric_features),
            len(numeric_features)
            + len(categorical_features),
        )
    )

    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=1.0,
        categorical_features=categorical_indices,
        class_weight="balanced",
        early_stopping=False,
        random_state=42,
    )

    return Pipeline(
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


def main() -> None:
    data = load_and_merge_data()
    data = create_rolling_weather_features(
        data
    )
    data = create_next_day_label(
        data
    )

    (
        data,
        numeric_features,
        categorical_features,
        _,
    ) = prepare_features(data)

    (
        data,
        numeric_features,
        categorical_features,
        feature_columns,
    ) = prepare_gradient_boosting_features(
        data=data,
        numeric_features=numeric_features,
        categorical_features=(
            categorical_features
        ),
    )

    (
        train_data,
        validation_data,
        test_data,
    ) = split_data_by_time(data)

    target_column = "fire_next_day"

    X_train = train_data[feature_columns]
    y_train = train_data[target_column]

    X_validation = validation_data[
        feature_columns
    ]
    y_validation = validation_data[
        target_column
    ]

    X_test = test_data[feature_columns]
    y_test = test_data[target_column]

    pipeline = build_pipeline(
        numeric_features=numeric_features,
        categorical_features=(
            categorical_features
        ),
    )

    print(
        "\nTraining histogram gradient boosting..."
    )

    pipeline.fit(
        X_train,
        y_train,
    )

    validation_scores = (
        pipeline.predict_proba(
            X_validation
        )[:, 1]
    )

    chosen_threshold = (
        choose_validation_threshold(
            y_true=y_validation,
            probabilities=validation_scores,
            minimum_recall=MINIMUM_RECALL,
        )
    )

    validation_results = (
        evaluate_binary_classifier(
            name="Validation",
            model=pipeline,
            X=X_validation,
            y=y_validation,
            threshold=chosen_threshold,
        )
    )

    test_results = (
        evaluate_binary_classifier(
            name="Test",
            model=pipeline,
            X=X_test,
            y=y_test,
            threshold=chosen_threshold,
        )
    )

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        {
            "pipeline": pipeline,
            "threshold": chosen_threshold,
            "feature_columns": feature_columns,
            "numeric_features": numeric_features,
            "categorical_features": (
                categorical_features
            ),
            "target_column": target_column,
        },
        MODEL_PATH,
    )

    print(
        f"\nSaved trained model to {MODEL_PATH}"
    )

    save_test_predictions(
        test_data=test_data,
        probabilities=(
            test_results["probabilities"]
        ),
        predictions=(
            test_results["predictions"]
        ),
        output_path=PREDICTION_PATH,
    )

    save_experiment_results(
        experiment_name=EXPERIMENT_NAME,
        model_name=(
            "HistGradientBoostingClassifier"
        ),
        train_data=train_data,
        validation_data=validation_data,
        test_data=test_data,
        threshold=chosen_threshold,
        validation_results=validation_results,
        test_results=test_results,
        feature_columns=feature_columns,
        notes=EXPERIMENT_NOTES,
    )


if __name__ == "__main__":
    main()
