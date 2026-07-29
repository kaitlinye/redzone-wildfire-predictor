from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def save_experiment_results(
    *,
    experiment_name: str,
    model_name: str,
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    test_data: pd.DataFrame,
    threshold: float,
    validation_results: dict[str, Any],
    test_results: dict[str, Any],
    feature_columns: list[str],
    notes: str = "",
    results_path: Path = Path(
        "metadata/model_results.csv"
    ),
) -> None:
    results_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    confusion = test_results["confusion_matrix"]

    true_negatives = int(confusion[0, 0])
    false_positives = int(confusion[0, 1])
    false_negatives = int(confusion[1, 0])
    true_positives = int(confusion[1, 1])

    result_row = pd.DataFrame(
        [
            {
                "recorded_at_utc": datetime.now(
                    timezone.utc
                ).isoformat(),
                "experiment_name": experiment_name,
                "model": model_name,
                "train_start": (
                    train_data["date"].min().date()
                ),
                "train_end": (
                    train_data["date"].max().date()
                ),
                "validation_start": (
                    validation_data["date"].min().date()
                ),
                "validation_end": (
                    validation_data["date"].max().date()
                ),
                "test_start": (
                    test_data["date"].min().date()
                ),
                "test_end": (
                    test_data["date"].max().date()
                ),
                "features": "|".join(feature_columns),
                "feature_count": len(feature_columns),
                "threshold": threshold,
                "validation_pr_auc": (
                    validation_results["pr_auc"]
                ),
                "validation_roc_auc": (
                    validation_results["roc_auc"]
                ),
                "test_pr_auc": test_results["pr_auc"],
                "test_roc_auc": test_results["roc_auc"],
                "test_precision": (
                    test_results["precision"]
                ),
                "test_recall": test_results["recall"],
                "test_f1": test_results["f1"],
                "true_negatives": true_negatives,
                "false_positives": false_positives,
                "false_negatives": false_negatives,
                "true_positives": true_positives,
                "notes": notes,
            }
        ]
    )

    write_header = not results_path.exists()

    if not write_header:
        existing_columns = pd.read_csv(
            results_path,
            nrows=0,
        ).columns.tolist()
        expected_columns = result_row.columns.tolist()

        if existing_columns != expected_columns:
            raise ValueError(
                "Experiment results CSV has an incompatible "
                f"header. Expected {expected_columns}, found "
                f"{existing_columns}."
            )

    result_row.to_csv(
        results_path,
        mode="a",
        header=write_header,
        index=False,
    )

    print(
        f"\nSaved experiment results to "
        f"{results_path}"
    )


def save_test_predictions(
    *,
    test_data: pd.DataFrame,
    probabilities: object,
    predictions: object,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    preferred_columns = [
        "date",
        "grid_id",
        "centroid_lat",
        "centroid_lon",
        "fire_today",
        "fire_next_day",
    ]

    available_columns = [
        column
        for column in preferred_columns
        if column in test_data.columns
    ]

    output = test_data[
        available_columns
    ].copy()

    output["predicted_score"] = probabilities
    output["predicted_class"] = predictions

    output.to_parquet(
        output_path,
        index=False,
    )

    print(
        f"Saved test predictions to {output_path}"
    )
