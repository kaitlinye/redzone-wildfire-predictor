from typing import Any

import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline


EvaluationResults = dict[str, Any]


def choose_validation_threshold(
    y_true: pd.Series,
    probabilities: np.ndarray,
    minimum_recall: float = 0.70,
) -> float:
    precision, recall, thresholds = precision_recall_curve(
        y_true,
        probabilities,
    )

    if len(thresholds) == 0:
        raise ValueError(
            "No thresholds were produced. Check the validation labels."
        )

    threshold_table = pd.DataFrame(
        {
            "threshold": thresholds,
            "precision": precision[:-1],
            "recall": recall[:-1],
        }
    )

    acceptable = threshold_table[
        threshold_table["recall"] >= minimum_recall
    ]

    if acceptable.empty:
        best_row = threshold_table.loc[
            threshold_table["recall"].idxmax()
        ]
    else:
        best_row = acceptable.sort_values(
            by=["precision", "threshold"],
            ascending=[False, False],
        ).iloc[0]

    threshold = float(best_row["threshold"])

    print("\nChosen validation threshold")
    print("-" * 50)
    print(f"Threshold: {threshold:.4f}")
    print(
        f"Validation precision: "
        f"{float(best_row['precision']):.4f}"
    )
    print(
        f"Validation recall: "
        f"{float(best_row['recall']):.4f}"
    )

    return threshold


def evaluate_binary_classifier(
    name: str,
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float,
) -> EvaluationResults:
    probabilities = model.predict_proba(X)[:, 1]

    results = evaluate_binary_scores(
        name=name,
        y=y,
        scores=probabilities,
        threshold=threshold,
    )
    results["probabilities"] = probabilities

    return results


def evaluate_binary_scores(
    *,
    name: str,
    y: pd.Series,
    scores: np.ndarray,
    threshold: float,
) -> EvaluationResults:
    """Evaluate arbitrary classifier or ranking scores."""
    predictions = (scores >= threshold).astype(int)

    pr_auc = average_precision_score(
        y,
        scores,
    )

    roc_auc = roc_auc_score(
        y,
        scores,
    )

    precision_value = precision_score(
        y,
        predictions,
        zero_division=0,
    )

    recall_value = recall_score(
        y,
        predictions,
        zero_division=0,
    )

    f1_value = f1_score(
        y,
        predictions,
        zero_division=0,
    )

    confusion = confusion_matrix(
        y,
        predictions,
        labels=[0, 1],
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
    print(confusion)

    return {
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "precision": float(precision_value),
        "recall": float(recall_value),
        "f1": float(f1_value),
        "confusion_matrix": confusion,
        "scores": scores,
        "predictions": predictions,
    }


def evaluate_daily_ranking(
    *,
    dates: pd.Series,
    y: pd.Series,
    scores: np.ndarray,
    top_fractions: tuple[
        float,
        ...,
    ] = (
        0.01,
        0.05,
        0.10,
    ),
) -> pd.DataFrame:
    """Measure positive-row capture within each day's top scores."""
    if not (
        len(dates)
        == len(y)
        == len(scores)
    ):
        raise ValueError(
            "Dates, labels, and scores must have equal lengths."
        )

    ranking = pd.DataFrame(
        {
            "date": pd.to_datetime(
                dates
            ).to_numpy(),
            "label": y.to_numpy(),
            "score": scores,
        }
    )
    ranking["daily_rank"] = (
        ranking.groupby("date")["score"]
        .rank(
            method="first",
            ascending=False,
        )
    )
    daily_row_count = (
        ranking.groupby("date")["score"]
        .transform("size")
    )
    total_positives = int(
        ranking["label"].sum()
    )

    if total_positives == 0:
        raise ValueError(
            "Daily ranking data contains no positive rows."
        )

    records: list[
        dict[str, float | int | str]
    ] = []

    for fraction in top_fractions:
        if not 0 < fraction <= 1:
            raise ValueError(
                "Top fractions must be within (0, 1]."
            )

        daily_cutoff = np.ceil(
            daily_row_count * fraction
        )
        selected = ranking[
            ranking["daily_rank"]
            <= daily_cutoff
        ]
        selected_count = len(selected)
        captured_positives = int(
            selected["label"].sum()
        )

        records.append(
            {
                "risk_group": (
                    f"Top {fraction:.0%}"
                ),
                "fraction": fraction,
                "rows_selected": (
                    selected_count
                ),
                "positive_rows_captured": (
                    captured_positives
                ),
                "capture_rate": (
                    captured_positives
                    / total_positives
                ),
                "precision": (
                    captured_positives
                    / selected_count
                    if selected_count
                    else 0.0
                ),
            }
        )

    results = pd.DataFrame(records)

    print("\nDaily ranking results")
    print("-" * 50)
    print(results.to_string(index=False))

    return results
