from pathlib import Path

import joblib

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

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
    "random_forest_rolling_history_2024"
)
EXPERIMENT_NOTES = (
    "Random forest with balanced subsample weights; "
    "daily and rolling weather; LANDFIRE; historical "
    "FIRMS features from 2020-2023."
)
MINIMUM_RECALL = 0.70

MODEL_PATH = Path(
    f"models/{EXPERIMENT_NAME}.joblib"
)
PREDICTION_PATH = Path(
    "data/processed/predictions/"
    f"{EXPERIMENT_NAME}.parquet"
)


def build_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
) -> Pipeline:
    """Build the random-forest preprocessing and model pipeline."""
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
        ],
        sparse_threshold=1.0,
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_leaf=10,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
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
        feature_columns,
    ) = prepare_features(data)

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

    print("\nTraining random forest...")

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
        model_name="RandomForestClassifier",
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
