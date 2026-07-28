from pathlib import Path

import joblib
import pandas as pd
from catboost import CatBoostClassifier

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


EXPERIMENT_NAME = "catboost_rolling_history_2024"
EXPERIMENT_NOTES = (
    "Unweighted CatBoost with native categorical features; "
    "daily and rolling weather; LANDFIRE; historical FIRMS "
    "features from 2020-2023."
)
MINIMUM_RECALL = 0.70
CATEGORICAL_MISSING_VALUE = "__MISSING__"

MODEL_PATH = Path(
    f"models/{EXPERIMENT_NAME}.joblib"
)
PREDICTION_PATH = Path(
    "data/processed/predictions/"
    f"{EXPERIMENT_NAME}.parquet"
)


def prepare_catboost_features(
    features: pd.DataFrame,
    categorical_features: list[str],
) -> pd.DataFrame:
    """Convert categorical values to CatBoost-compatible strings."""
    result = features.copy()

    for column in categorical_features:
        result[column] = (
            result[column]
            .astype("string")
            .fillna(CATEGORICAL_MISSING_VALUE)
            .astype(str)
        )

    return result


def build_model(
    categorical_features: list[str],
) -> CatBoostClassifier:
    """Build the unweighted CatBoost classifier."""
    return CatBoostClassifier(
        iterations=2000,
        learning_rate=0.03,
        depth=8,
        loss_function="Logloss",
        eval_metric="PRAUC",
        cat_features=categorical_features,
        l2_leaf_reg=5.0,
        random_strength=1.0,
        bootstrap_type="Bernoulli",
        subsample=0.8,
        random_seed=42,
        thread_count=-1,
        allow_writing_files=False,
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

    X_train = prepare_catboost_features(
        train_data[feature_columns],
        categorical_features,
    )
    y_train = train_data[target_column]

    X_validation = prepare_catboost_features(
        validation_data[feature_columns],
        categorical_features,
    )
    y_validation = validation_data[
        target_column
    ]

    X_test = prepare_catboost_features(
        test_data[feature_columns],
        categorical_features,
    )
    y_test = test_data[target_column]

    model = build_model(
        categorical_features=(
            categorical_features
        )
    )

    print("\nTraining CatBoost...")

    model.fit(
        X_train,
        y_train,
        eval_set=(
            X_validation,
            y_validation,
        ),
        use_best_model=True,
        early_stopping_rounds=100,
        verbose=50,
    )

    validation_scores = (
        model.predict_proba(
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
            model=model,
            X=X_validation,
            y=y_validation,
            threshold=chosen_threshold,
        )
    )

    test_results = (
        evaluate_binary_classifier(
            name="Test",
            model=model,
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
            "model": model,
            "threshold": chosen_threshold,
            "feature_columns": feature_columns,
            "numeric_features": numeric_features,
            "categorical_features": (
                categorical_features
            ),
            "categorical_missing_value": (
                CATEGORICAL_MISSING_VALUE
            ),
            "target_column": target_column,
            "best_iteration": (
                model.get_best_iteration()
            ),
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
        model_name="CatBoostClassifier",
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
