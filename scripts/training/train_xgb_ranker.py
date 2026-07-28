from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRanker

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
    evaluate_binary_scores,
    evaluate_daily_ranking,
)
from scripts.utils.experiment_logging import (
    save_experiment_results,
    save_test_predictions,
)


EXPERIMENT_NAME = (
    "xgbranker_daily_rolling_history_2024"
)
EXPERIMENT_NOTES = (
    "XGBoost LambdaMART grouped by prediction date; "
    "rank:ndcg objective; native categorical features; "
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
RANKING_RESULTS_PATH = Path(
    "metadata/experiments/"
    f"{EXPERIMENT_NAME}_daily_ranking.csv"
)


def prepare_ranking_split(
    data: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    np.ndarray,
]:
    """Sort a split into contiguous daily ranking groups."""
    sorted_data = data.sort_values(
        [
            "date",
            "grid_id",
        ]
    ).copy()
    X = sorted_data[
        feature_columns
    ]
    y = sorted_data[
        "fire_next_day"
    ]
    qid = pd.factorize(
        sorted_data["date"],
        sort=True,
    )[0].astype("int32")

    if np.any(
        qid[1:] < qid[:-1]
    ):
        raise ValueError(
            "Ranking query IDs must be sorted."
        )

    return (
        sorted_data,
        X,
        y,
        qid,
    )


def build_model() -> XGBRanker:
    """Build the daily LambdaMART ranking model."""
    return XGBRanker(
        objective="rank:ndcg",
        eval_metric="ndcg@44",
        tree_method="hist",
        device="cpu",
        enable_categorical=True,
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=10,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        max_cat_to_onehot=32,
        lambdarank_pair_method="topk",
        lambdarank_num_pair_per_sample=32,
        early_stopping_rounds=50,
        random_state=42,
        n_jobs=-1,
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

    (
        train_data,
        X_train,
        y_train,
        qid_train,
    ) = prepare_ranking_split(
        train_data,
        feature_columns,
    )
    (
        validation_data,
        X_validation,
        y_validation,
        qid_validation,
    ) = prepare_ranking_split(
        validation_data,
        feature_columns,
    )
    (
        test_data,
        X_test,
        y_test,
        qid_test,
    ) = prepare_ranking_split(
        test_data,
        feature_columns,
    )

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

    model = build_model()

    print(
        "\nTraining XGBoost daily ranker..."
    )
    print(
        "Ranking groups: "
        f"{len(np.unique(qid_train))} training days, "
        f"{len(np.unique(qid_validation))} "
        "validation days"
    )

    model.fit(
        X_train,
        y_train,
        qid=qid_train,
        eval_set=[
            (
                X_validation,
                y_validation,
            )
        ],
        eval_qid=[
            qid_validation,
        ],
        verbose=25,
    )

    validation_scores = model.predict(
        X_validation
    )

    chosen_threshold = (
        choose_validation_threshold(
            y_true=y_validation,
            probabilities=validation_scores,
            minimum_recall=MINIMUM_RECALL,
        )
    )

    validation_results = (
        evaluate_binary_scores(
            name="Validation",
            y=y_validation,
            scores=validation_scores,
            threshold=chosen_threshold,
        )
    )
    validation_ranking = (
        evaluate_daily_ranking(
            dates=validation_data["date"],
            y=y_validation,
            scores=validation_scores,
        )
    )
    validation_ranking.insert(
        0,
        "split",
        "validation",
    )

    test_scores = model.predict(
        X_test
    )
    test_results = (
        evaluate_binary_scores(
            name="Test",
            y=y_test,
            scores=test_scores,
            threshold=chosen_threshold,
        )
    )
    test_ranking = evaluate_daily_ranking(
        dates=test_data["date"],
        y=y_test,
        scores=test_scores,
    )
    test_ranking.insert(
        0,
        "split",
        "test",
    )

    ranking_results = pd.concat(
        [
            validation_ranking,
            test_ranking,
        ],
        ignore_index=True,
    )
    RANKING_RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    ranking_results.to_csv(
        RANKING_RESULTS_PATH,
        index=False,
    )

    print(
        "Saved daily ranking results to "
        f"{RANKING_RESULTS_PATH}"
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
            "categorical_levels": (
                categorical_levels
            ),
            "target_column": "fire_next_day",
            "ranking_group": "date",
            "best_iteration": (
                model.best_iteration
            ),
        },
        MODEL_PATH,
    )

    print(
        f"\nSaved trained model to {MODEL_PATH}"
    )

    save_test_predictions(
        test_data=test_data,
        probabilities=test_scores,
        predictions=(
            test_results["predictions"]
        ),
        output_path=PREDICTION_PATH,
    )

    save_experiment_results(
        experiment_name=EXPERIMENT_NAME,
        model_name="XGBRanker",
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
