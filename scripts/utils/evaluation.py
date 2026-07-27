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
    predictions = (probabilities >= threshold).astype(int)

    pr_auc = average_precision_score(
        y,
        probabilities,
    )

    roc_auc = roc_auc_score(
        y,
        probabilities,
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
        "probabilities": probabilities,
        "predictions": predictions,
    }