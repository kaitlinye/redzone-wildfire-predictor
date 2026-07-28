import math
from pathlib import Path

import joblib
import lightgbm as lgb
import pandas as pd
from lightgbm import LGBMClassifier

from scripts.training.train_logistic_regression import (
    create_next_day_label,
    create_rolling_weather_features,
    load_and_merge_data,
    prepare_features,
    split_data_by_time,
)
from scripts.utils.categorical_features import (
    apply_training_categories,
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
    "lightgbm_weight_tuned_rolling_history_2024"
)
EXPERIMENT_NOTES = (
    "LightGBM with validation-selected scale_pos_weight; "
    "native categorical features; daily and rolling "
    "weather; LANDFIRE; historical FIRMS features "
    "from 2020-2023."
)
MINIMUM_RECALL = 0.70

MODEL_PATH = Path(
    f"models/{EXPERIMENT_NAME}.joblib"
)
PREDICTION_PATH = Path(
    "data/processed/predictions/"
    f"{EXPERIMENT_NAME}.parquet"
)
WEIGHT_RESULTS_PATH = Path(
    "metadata/experiments/"
    "lightgbm_scale_pos_weight_validation.csv"
)


def build_model(
    scale_pos_weight: float,
) -> LGBMClassifier:
    """Build the LightGBM classifier."""
    return LGBMClassifier(
        objective="binary",
        metric="None",
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=50,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )


def fit_candidate(
    *,
    scale_pos_weight: float,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    categorical_features: list[str],
) -> tuple[
    LGBMClassifier,
    float,
    dict[str, object],
]:
    """Fit and evaluate one weight using validation data only."""
    print(
        "\nTesting scale_pos_weight: "
        f"{scale_pos_weight:.4f}"
    )

    model = build_model(
        scale_pos_weight=scale_pos_weight
    )

    model.fit(
        X_train,
        y_train,
        eval_X=X_validation,
        eval_y=y_validation,
        eval_metric="average_precision",
        categorical_feature=(
            categorical_features
        ),
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=100,
                first_metric_only=True,
            ),
            lgb.log_evaluation(
                period=50,
            ),
        ],
    )

    validation_scores = (
        model.predict_proba(
            X_validation
        )[:, 1]
    )

    threshold = choose_validation_threshold(
        y_true=y_validation,
        probabilities=validation_scores,
        minimum_recall=MINIMUM_RECALL,
    )

    validation_results = (
        evaluate_binary_classifier(
            name=(
                "Validation "
                f"(weight={scale_pos_weight:.4f})"
            ),
            model=model,
            X=X_validation,
            y=y_validation,
            threshold=threshold,
        )
    )

    return (
        model,
        threshold,
        validation_results,
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

    (
        X_train,
        X_validation,
        X_test,
        categorical_levels,
    ) = apply_training_categories(
        X_train=X_train,
        X_validation=X_validation,
        X_test=X_test,
        categorical_features=(
            categorical_features
        ),
    )

    positive_count = int(y_train.sum())
    negative_count = int(
        len(y_train) - positive_count
    )

    if positive_count == 0:
        raise ValueError(
            "Training data contains no positive rows."
        )

    imbalance_ratio = (
        negative_count / positive_count
    )

    print(
        "\nTraining imbalance ratio: "
        f"{imbalance_ratio:.4f}"
    )

    weight_candidates = [
        (
            "unweighted",
            1.0,
        ),
        (
            "sqrt_imbalance",
            math.sqrt(imbalance_ratio),
        ),
        (
            "moderate_weight",
            40.0,
        ),
    ]

    best_model: LGBMClassifier | None = None
    best_threshold = 0.0
    best_validation_results: (
        dict[str, object] | None
    ) = None
    best_weight_name = ""
    best_weight = 0.0
    best_selection_key = (
        float("-inf"),
        float("-inf"),
    )
    validation_records: list[
        dict[str, object]
    ] = []

    for weight_name, weight in weight_candidates:
        (
            candidate_model,
            candidate_threshold,
            candidate_results,
        ) = fit_candidate(
            scale_pos_weight=weight,
            X_train=X_train,
            y_train=y_train,
            X_validation=X_validation,
            y_validation=y_validation,
            categorical_features=(
                categorical_features
            ),
        )

        confusion = candidate_results[
            "confusion_matrix"
        ]
        false_positives = int(
            confusion[0, 1]
        )
        false_negatives = int(
            confusion[1, 0]
        )

        validation_records.append(
            {
                "weight_name": weight_name,
                "scale_pos_weight": weight,
                "best_iteration": (
                    candidate_model.best_iteration_
                ),
                "threshold": (
                    candidate_threshold
                ),
                "pr_auc": (
                    candidate_results["pr_auc"]
                ),
                "roc_auc": (
                    candidate_results["roc_auc"]
                ),
                "precision": (
                    candidate_results["precision"]
                ),
                "recall": (
                    candidate_results["recall"]
                ),
                "f1": (
                    candidate_results["f1"]
                ),
                "false_positives": (
                    false_positives
                ),
                "false_negatives": (
                    false_negatives
                ),
            }
        )

        selection_key = (
            float(
                candidate_results[
                    "precision"
                ]
            ),
            float(
                candidate_results[
                    "pr_auc"
                ]
            ),
        )

        if selection_key > best_selection_key:
            best_selection_key = (
                selection_key
            )
            best_model = candidate_model
            best_threshold = (
                candidate_threshold
            )
            best_validation_results = (
                candidate_results
            )
            best_weight_name = weight_name
            best_weight = weight

    if (
        best_model is None
        or best_validation_results is None
    ):
        raise RuntimeError(
            "No LightGBM weight candidate was selected."
        )

    WEIGHT_RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    validation_table = pd.DataFrame(
        validation_records
    ).sort_values(
        by=[
            "precision",
            "pr_auc",
        ],
        ascending=False,
    )
    validation_table.to_csv(
        WEIGHT_RESULTS_PATH,
        index=False,
    )

    print(
        "\nValidation weight comparison"
    )
    print("-" * 50)
    print(
        validation_table.to_string(
            index=False
        )
    )
    print(
        "\nSelected weight: "
        f"{best_weight_name} "
        f"({best_weight:.4f})"
    )
    print(
        "Saved validation comparison to "
        f"{WEIGHT_RESULTS_PATH}"
    )

    test_results = (
        evaluate_binary_classifier(
            name="Test",
            model=best_model,
            X=X_test,
            y=y_test,
            threshold=best_threshold,
        )
    )

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        {
            "model": best_model,
            "threshold": best_threshold,
            "feature_columns": feature_columns,
            "numeric_features": numeric_features,
            "categorical_features": (
                categorical_features
            ),
            "categorical_levels": (
                categorical_levels
            ),
            "target_column": target_column,
            "scale_pos_weight": best_weight,
            "weight_name": best_weight_name,
            "best_iteration": (
                best_model.best_iteration_
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
        model_name="LGBMClassifier",
        train_data=train_data,
        validation_data=validation_data,
        test_data=test_data,
        threshold=best_threshold,
        validation_results=(
            best_validation_results
        ),
        test_results=test_results,
        feature_columns=feature_columns,
        notes=(
            f"{EXPERIMENT_NOTES} Selected "
            f"{best_weight_name}="
            f"{best_weight:.4f}."
        ),
    )


if __name__ == "__main__":
    main()
