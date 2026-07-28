from itertools import product
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from scripts.training.train_catboost import (
    prepare_catboost_features,
)
from scripts.training.train_gradient_boosting import (
    prepare_gradient_boosting_features,
)
from scripts.training.train_logistic_regression import (
    create_next_day_label,
    create_rolling_weather_features,
    load_and_merge_data,
    prepare_features,
    split_data_by_time,
)
from scripts.utils.categorical_features import (
    apply_saved_categories,
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
    "ensemble_rank_blend_rolling_history_2024"
)
MINIMUM_RECALL = 0.70
WEIGHT_UNITS = 4

BASE_MODEL_PATHS = {
    "catboost": Path(
        "models/catboost_rolling_history_2024.joblib"
    ),
    "hist_gradient_boosting": Path(
        "models/"
        "gradient_boosting_rolling_history_2024.joblib"
    ),
    "lightgbm": Path(
        "models/"
        "lightgbm_weight_tuned_rolling_history_2024.joblib"
    ),
    "xgbranker": Path(
        "models/"
        "xgbranker_daily_rolling_history_2024.joblib"
    ),
}

MODEL_PATH = Path(
    f"models/{EXPERIMENT_NAME}.joblib"
)
PREDICTION_PATH = Path(
    "data/processed/predictions/"
    f"{EXPERIMENT_NAME}.parquet"
)
WEIGHT_RESULTS_PATH = Path(
    "metadata/experiments/"
    f"{EXPERIMENT_NAME}_validation_weights.csv"
)
RANKING_RESULTS_PATH = Path(
    "metadata/experiments/"
    f"{EXPERIMENT_NAME}_daily_ranking.csv"
)


def load_artifact(
    path: Path,
) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(
            f"Required model artifact not found: {path}"
        )

    artifact = joblib.load(path)

    if not isinstance(artifact, dict):
        raise ValueError(
            f"Unexpected model artifact format: {path}"
        )

    return artifact


def daily_percentile_ranks(
    dates: pd.Series,
    scores: np.ndarray,
) -> np.ndarray:
    """Convert arbitrary model scores to comparable daily ranks."""
    if len(dates) != len(scores):
        raise ValueError(
            "Dates and model scores must have equal lengths."
        )

    ranking = pd.DataFrame(
        {
            "date": pd.to_datetime(
                dates
            ).to_numpy(),
            "score": scores,
        }
    )

    return (
        ranking.groupby("date")["score"]
        .rank(
            method="average",
            pct=True,
        )
        .to_numpy()
    )


def predict_base_scores(
    *,
    data: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    artifacts: dict[
        str,
        dict[str, object],
    ],
) -> dict[str, np.ndarray]:
    """Generate scores from each previously trained model."""
    scores: dict[
        str,
        np.ndarray,
    ] = {}

    catboost_artifact = artifacts["catboost"]
    catboost_features = catboost_artifact[
        "feature_columns"
    ]
    catboost_X = prepare_catboost_features(
        data[catboost_features],
        catboost_artifact[
            "categorical_features"
        ],
    )
    scores["catboost"] = (
        catboost_artifact["model"]
        .predict_proba(catboost_X)[:, 1]
    )

    hist_artifact = artifacts[
        "hist_gradient_boosting"
    ]
    (
        hist_data,
        _,
        _,
        _,
    ) = prepare_gradient_boosting_features(
        data=data,
        numeric_features=numeric_features,
        categorical_features=(
            categorical_features
        ),
    )
    hist_X = hist_data[
        hist_artifact["feature_columns"]
    ]
    scores["hist_gradient_boosting"] = (
        hist_artifact["pipeline"]
        .predict_proba(hist_X)[:, 1]
    )

    lightgbm_artifact = artifacts["lightgbm"]
    lightgbm_X = apply_saved_categories(
        data[
            lightgbm_artifact[
                "feature_columns"
            ]
        ],
        lightgbm_artifact[
            "categorical_levels"
        ],
    )
    scores["lightgbm"] = (
        lightgbm_artifact["model"]
        .predict_proba(lightgbm_X)[:, 1]
    )

    ranker_artifact = artifacts["xgbranker"]
    ranker_X = apply_saved_categories(
        data[
            ranker_artifact[
                "feature_columns"
            ]
        ],
        ranker_artifact[
            "categorical_levels"
        ],
    )
    scores["xgbranker"] = (
        ranker_artifact["model"]
        .predict(ranker_X)
    )

    return scores


def create_daily_rank_scores(
    *,
    dates: pd.Series,
    base_scores: dict[
        str,
        np.ndarray,
    ],
) -> dict[str, np.ndarray]:
    return {
        name: daily_percentile_ranks(
            dates,
            scores,
        )
        for name, scores in base_scores.items()
    }


def blend_scores(
    daily_ranks: dict[
        str,
        np.ndarray,
    ],
    weights: dict[str, float],
) -> np.ndarray:
    blended = np.zeros(
        len(next(iter(daily_ranks.values()))),
        dtype="float64",
    )

    for name, weight in weights.items():
        blended += (
            weight * daily_ranks[name]
        )

    return blended


def select_validation_weights(
    *,
    y_validation: pd.Series,
    daily_ranks: dict[
        str,
        np.ndarray,
    ],
) -> tuple[
    dict[str, float],
    pd.DataFrame,
]:
    """Select quarter-step model weights by validation PR AUC."""
    model_names = list(
        daily_ranks
    )
    records: list[
        dict[str, float]
    ] = []

    for units in product(
        range(WEIGHT_UNITS + 1),
        repeat=len(model_names),
    ):
        if sum(units) != WEIGHT_UNITS:
            continue

        weights = {
            name: units[index] / WEIGHT_UNITS
            for index, name in enumerate(
                model_names
            )
        }
        scores = blend_scores(
            daily_ranks,
            weights,
        )
        pr_auc = average_precision_score(
            y_validation,
            scores,
        )

        record = {
            f"{name}_weight": weight
            for name, weight in weights.items()
        }
        record["validation_pr_auc"] = float(
            pr_auc
        )
        records.append(record)

    results = pd.DataFrame(records).sort_values(
        by="validation_pr_auc",
        ascending=False,
    )

    if results.empty:
        raise RuntimeError(
            "No ensemble weight candidates were generated."
        )

    best_row = results.iloc[0]
    best_weights = {
        name: float(
            best_row[f"{name}_weight"]
        )
        for name in model_names
    }

    return (
        best_weights,
        results,
    )


def main() -> None:
    artifacts = {
        name: load_artifact(path)
        for name, path in (
            BASE_MODEL_PATHS.items()
        )
    }

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

    validation_base_scores = (
        predict_base_scores(
            data=validation_data,
            numeric_features=numeric_features,
            categorical_features=(
                categorical_features
            ),
            artifacts=artifacts,
        )
    )
    validation_daily_ranks = (
        create_daily_rank_scores(
            dates=validation_data["date"],
            base_scores=(
                validation_base_scores
            ),
        )
    )

    (
        selected_weights,
        weight_results,
    ) = select_validation_weights(
        y_validation=validation_data[
            "fire_next_day"
        ],
        daily_ranks=validation_daily_ranks,
    )

    WEIGHT_RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    weight_results.to_csv(
        WEIGHT_RESULTS_PATH,
        index=False,
    )

    print("\nSelected ensemble weights")
    print("-" * 50)

    for name, weight in (
        selected_weights.items()
    ):
        print(f"{name}: {weight:.2f}")

    print(
        "Validation PR AUC: "
        f"{weight_results.iloc[0]['validation_pr_auc']:.4f}"
    )
    print(
        "Saved weight comparison to "
        f"{WEIGHT_RESULTS_PATH}"
    )

    validation_scores = blend_scores(
        validation_daily_ranks,
        selected_weights,
    )
    chosen_threshold = (
        choose_validation_threshold(
            y_true=validation_data[
                "fire_next_day"
            ],
            probabilities=validation_scores,
            minimum_recall=MINIMUM_RECALL,
        )
    )
    validation_results = (
        evaluate_binary_scores(
            name="Validation",
            y=validation_data[
                "fire_next_day"
            ],
            scores=validation_scores,
            threshold=chosen_threshold,
        )
    )
    validation_ranking = (
        evaluate_daily_ranking(
            dates=validation_data["date"],
            y=validation_data[
                "fire_next_day"
            ],
            scores=validation_scores,
        )
    )
    validation_ranking.insert(
        0,
        "split",
        "validation",
    )

    test_base_scores = predict_base_scores(
        data=test_data,
        numeric_features=numeric_features,
        categorical_features=(
            categorical_features
        ),
        artifacts=artifacts,
    )
    test_daily_ranks = (
        create_daily_rank_scores(
            dates=test_data["date"],
            base_scores=test_base_scores,
        )
    )
    test_scores = blend_scores(
        test_daily_ranks,
        selected_weights,
    )
    test_results = evaluate_binary_scores(
        name="Test",
        y=test_data["fire_next_day"],
        scores=test_scores,
        threshold=chosen_threshold,
    )
    test_ranking = evaluate_daily_ranking(
        dates=test_data["date"],
        y=test_data["fire_next_day"],
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
    ranking_results.to_csv(
        RANKING_RESULTS_PATH,
        index=False,
    )

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    joblib.dump(
        {
            "ensemble_type": (
                "daily_percentile_rank_blend"
            ),
            "base_model_paths": {
                name: str(path)
                for name, path in (
                    BASE_MODEL_PATHS.items()
                )
            },
            "weights": selected_weights,
            "threshold": chosen_threshold,
            "feature_columns": feature_columns,
            "target_column": "fire_next_day",
        },
        MODEL_PATH,
    )

    print(
        f"\nSaved ensemble definition to {MODEL_PATH}"
    )
    print(
        "Saved daily ranking results to "
        f"{RANKING_RESULTS_PATH}"
    )

    save_test_predictions(
        test_data=test_data,
        probabilities=test_scores,
        predictions=(
            test_results["predictions"]
        ),
        output_path=PREDICTION_PATH,
    )

    weight_description = ", ".join(
        f"{name}={weight:.2f}"
        for name, weight in (
            selected_weights.items()
        )
    )
    save_experiment_results(
        experiment_name=EXPERIMENT_NAME,
        model_name="DailyRankBlendEnsemble",
        train_data=train_data,
        validation_data=validation_data,
        test_data=test_data,
        threshold=chosen_threshold,
        validation_results=validation_results,
        test_results=test_results,
        feature_columns=feature_columns,
        notes=(
            "Validation-selected daily percentile rank "
            f"blend: {weight_description}."
        ),
    )


if __name__ == "__main__":
    main()
