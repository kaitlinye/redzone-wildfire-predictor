from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

if __package__ in (None, ""):
    sys.path.insert(
        0,
        str(Path(__file__).resolve().parents[2]),
    )

from scripts.training.train_catboost import (
    prepare_catboost_features,
)
from scripts.training.train_catboost_2023_2024 import (
    CATEGORICAL_FEATURES,
    build_model,
)
from scripts.training.train_catboost_seasonal_prior_year_history import (
    FEATURE_COLUMNS,
    MINIMUM_RECALL,
    SEASONAL_HISTORY_PATH,
    load_data,
)
from scripts.training.train_logistic_regression import (
    create_rolling_weather_features,
)
from scripts.utils.evaluation import (
    choose_validation_threshold,
    evaluate_binary_scores,
    evaluate_daily_ranking,
)


OUTPUT_FOLDER = Path("metadata/experiments")
FOLD_RESULTS_PATH = OUTPUT_FOLDER / (
    "catboost_seasonal_rolling_backtest_2023_2024_folds.csv"
)
MONTHLY_RESULTS_PATH = OUTPUT_FOLDER / (
    "catboost_seasonal_rolling_backtest_2023_2024_monthly.csv"
)
RANKING_RESULTS_PATH = OUTPUT_FOLDER / (
    "catboost_seasonal_rolling_backtest_2023_2024_ranking.csv"
)
LEAKAGE_AUDIT_PATH = OUTPUT_FOLDER / (
    "catboost_seasonal_rolling_backtest_leakage_audit.csv"
)


@dataclass(frozen=True)
class TemporalFold:
    name: str
    validation_start: str
    test_start: str
    test_end_exclusive: str


FOLDS = [
    TemporalFold(
        name="late_fire_season_2023",
        validation_start="2023-08-01",
        test_start="2023-09-16",
        test_end_exclusive="2023-10-30",
    ),
    TemporalFold(
        name="late_fire_season_2024",
        validation_start="2024-08-01",
        test_start="2024-09-16",
        test_end_exclusive="2024-10-30",
    ),
]


def split_fold(
    data: pd.DataFrame,
    fold: TemporalFold,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    validation_start = pd.Timestamp(
        fold.validation_start
    )
    test_start = pd.Timestamp(fold.test_start)
    test_end = pd.Timestamp(
        fold.test_end_exclusive
    )

    train = data[data["date"] < validation_start].copy()
    validation = data[
        (data["date"] >= validation_start)
        & (data["date"] < test_start)
    ].copy()
    test = data[
        (data["date"] >= test_start)
        & (data["date"] < test_end)
    ].copy()

    for split_name, frame in [
        ("training", train),
        ("validation", validation),
        ("testing", test),
    ]:
        if frame.empty:
            raise ValueError(
                f"{fold.name} {split_name} split is empty."
            )
    if not (
        train["date"].max()
        < validation["date"].min()
        <= validation["date"].max()
        < test["date"].min()
    ):
        raise ValueError(
            f"{fold.name} is not strictly chronological."
        )

    return train, validation, test


def metric_record(
    y: pd.Series,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    predictions = (scores >= threshold).astype(int)
    return {
        "pr_auc": float(
            average_precision_score(y, scores)
        ),
        "roc_auc": float(
            roc_auc_score(y, scores)
        ),
        "precision": float(
            precision_score(
                y,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y,
                predictions,
                zero_division=0,
            )
        ),
    }


def monthly_records(
    fold_name: str,
    test: pd.DataFrame,
    scores: np.ndarray,
    threshold: float,
) -> list[dict[str, float | int | str]]:
    scored = test[
        ["date", "grid_id", "fire_next_day"]
    ].copy()
    scored["score"] = scores
    scored["month"] = (
        scored["date"].dt.to_period("M").astype(str)
    )
    records: list[dict[str, float | int | str]] = []

    for month, month_data in scored.groupby(
        "month",
        sort=True,
    ):
        metrics = metric_record(
            month_data["fire_next_day"],
            month_data["score"].to_numpy(),
            threshold,
        )
        ranking = evaluate_daily_ranking(
            dates=month_data["date"],
            y=month_data["fire_next_day"],
            scores=month_data["score"].to_numpy(),
        )
        capture = {
            row.risk_group: float(row.capture_rate)
            for row in ranking.itertuples(index=False)
        }
        records.append(
            {
                "fold": fold_name,
                "month": month,
                "start_date": (
                    month_data["date"].min().date().isoformat()
                ),
                "end_date": (
                    month_data["date"].max().date().isoformat()
                ),
                "rows": len(month_data),
                "positive_rows": int(
                    month_data["fire_next_day"].sum()
                ),
                "positive_rate": float(
                    month_data["fire_next_day"].mean()
                ),
                "threshold": threshold,
                **metrics,
                "top_1_percent_capture": capture["Top 1%"],
                "top_5_percent_capture": capture["Top 5%"],
                "top_10_percent_capture": capture["Top 10%"],
            }
        )

    return records


def run_leakage_audit(
    data: pd.DataFrame,
) -> pd.DataFrame:
    audit_rows: list[dict[str, str | bool]] = []
    history = pd.read_parquet(SEASONAL_HISTORY_PATH)
    history["window_start"] = pd.to_datetime(
        history["window_start"]
    )
    history["window_end"] = pd.to_datetime(
        history["window_end"]
    )
    relevant_history = history[
        history["feature_year"].isin([2023, 2024])
    ].copy()

    mapping_valid = bool(
        (
            relevant_history["history_year"]
            == relevant_history["feature_year"] - 1
        ).all()
    )
    window_precedes_features = bool(
        (
            relevant_history["window_end"]
            < pd.to_datetime(
                relevant_history["feature_year"]
                .astype(str)
                + "-01-01"
            )
        ).all()
    )
    complete_history = (
        relevant_history.groupby("feature_year")[
            "grid_id"
        ]
        .nunique()
        .to_dict()
        == {2023: 4_355, 2024: 4_355}
    )
    audit_rows.extend(
        [
            {
                "check": "prior_year_mapping",
                "passed": mapping_valid,
                "evidence": (
                    "Every history_year equals feature_year - 1."
                ),
            },
            {
                "check": "prior_year_window_precedes_features",
                "passed": window_precedes_features,
                "evidence": (
                    "Every May-October history window ends before "
                    "January 1 of its feature year."
                ),
            },
            {
                "check": "prior_year_grid_coverage",
                "passed": complete_history,
                "evidence": (
                    "2023 and 2024 each map prior-year history to "
                    "4,355 grids."
                ),
            },
        ]
    )

    rolling_columns = [
        "rain_previous_7d",
        "rain_previous_30d",
        "temperature_max_previous_3d",
        "humidity_min_previous_3d",
    ]
    for year in [2023, 2024]:
        year_data = data[
            data["date"].dt.year == year
        ]
        grid_id = year_data["grid_id"].iloc[0]
        sample = (
            year_data[year_data["grid_id"] == grid_id]
            .sort_values("date")
            .copy()
        )
        target_position = len(sample) // 2
        target_date = sample.iloc[target_position]["date"]
        baseline = create_rolling_weather_features(sample)
        perturbed_source = sample.copy()
        perturbed_source.loc[
            perturbed_source["date"] >= target_date,
            [
                "precipitation_total",
                "temperature_max",
                "humidity_min",
            ],
        ] = [999_999.0, 999_999.0, -999_999.0]
        perturbed = create_rolling_weather_features(
            perturbed_source
        )
        baseline_row = baseline.loc[
            baseline["date"] == target_date,
            rolling_columns,
        ].reset_index(drop=True)
        perturbed_row = perturbed.loc[
            perturbed["date"] == target_date,
            rolling_columns,
        ].reset_index(drop=True)
        unchanged = baseline_row.equals(perturbed_row)
        audit_rows.append(
            {
                "check": f"rolling_weather_past_only_{year}",
                "passed": unchanged,
                "evidence": (
                    f"Changing weather on and after {target_date.date()} "
                    "did not change that date's rolling features."
                ),
            }
        )

    audit = pd.DataFrame(audit_rows)
    if not audit["passed"].all():
        failures = audit.loc[
            ~audit["passed"],
            "check",
        ].tolist()
        raise ValueError(
            f"Temporal leakage audit failed: {failures}"
        )
    return audit


def main() -> None:
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    data = load_data()

    leakage_audit = run_leakage_audit(data)
    leakage_audit.to_csv(
        LEAKAGE_AUDIT_PATH,
        index=False,
    )
    print("\nTemporal leakage audit")
    print("-" * 50)
    print(leakage_audit.to_string(index=False))

    if FOLD_RESULTS_PATH.exists():
        fold_records = pd.read_csv(
            FOLD_RESULTS_PATH
        ).to_dict("records")
    else:
        fold_records = []
    if MONTHLY_RESULTS_PATH.exists():
        monthly = pd.read_csv(
            MONTHLY_RESULTS_PATH
        ).to_dict("records")
    else:
        monthly = []
    if RANKING_RESULTS_PATH.exists():
        ranking_records = [
            pd.read_csv(RANKING_RESULTS_PATH)
        ]
    else:
        ranking_records = []
    completed_folds = {
        str(record["fold"])
        for record in fold_records
    }

    for fold in FOLDS:
        if fold.name in completed_folds:
            print(
                f"\nSkipping completed fold: {fold.name}"
            )
            continue

        print("\n" + "=" * 70)
        print(f"Rolling temporal fold: {fold.name}")
        train, validation, test = split_fold(data, fold)
        for name, frame in [
            ("Training", train),
            ("Validation", validation),
            ("Testing", test),
        ]:
            print(
                f"{name}: {frame['date'].min().date()} to "
                f"{frame['date'].max().date()}, "
                f"{len(frame):,} rows, "
                f"{int(frame['fire_next_day'].sum()):,} positives"
            )

        X_train = prepare_catboost_features(
            train[FEATURE_COLUMNS],
            CATEGORICAL_FEATURES,
        )
        X_validation = prepare_catboost_features(
            validation[FEATURE_COLUMNS],
            CATEGORICAL_FEATURES,
        )
        X_test = prepare_catboost_features(
            test[FEATURE_COLUMNS],
            CATEGORICAL_FEATURES,
        )
        y_train = train["fire_next_day"]
        y_validation = validation["fire_next_day"]
        y_test = test["fire_next_day"]

        model = build_model()
        model.fit(
            X_train,
            y_train,
            eval_set=(X_validation, y_validation),
            use_best_model=True,
            early_stopping_rounds=100,
            verbose=100,
        )
        validation_scores = model.predict_proba(
            X_validation
        )[:, 1]
        threshold = choose_validation_threshold(
            y_true=y_validation,
            probabilities=validation_scores,
            minimum_recall=MINIMUM_RECALL,
        )
        validation_metrics = metric_record(
            y_validation,
            validation_scores,
            threshold,
        )
        test_scores = model.predict_proba(X_test)[:, 1]
        test_metrics = evaluate_binary_scores(
            name=f"{fold.name} test",
            y=y_test,
            scores=test_scores,
            threshold=threshold,
        )
        ranking = evaluate_daily_ranking(
            dates=test["date"],
            y=y_test,
            scores=test_scores,
        )
        ranking.insert(0, "fold", fold.name)
        ranking_records.append(ranking)
        capture = {
            row.risk_group: float(row.capture_rate)
            for row in ranking.itertuples(index=False)
        }

        fold_records.append(
            {
                "fold": fold.name,
                "train_start": train["date"].min().date(),
                "train_end": train["date"].max().date(),
                "validation_start": (
                    validation["date"].min().date()
                ),
                "validation_end": (
                    validation["date"].max().date()
                ),
                "test_start": test["date"].min().date(),
                "test_end": test["date"].max().date(),
                "train_rows": len(train),
                "validation_rows": len(validation),
                "test_rows": len(test),
                "train_positives": int(y_train.sum()),
                "validation_positives": int(
                    y_validation.sum()
                ),
                "test_positives": int(y_test.sum()),
                "best_iteration": model.get_best_iteration(),
                "threshold": threshold,
                **{
                    f"validation_{key}": value
                    for key, value in validation_metrics.items()
                },
                **{
                    f"test_{key}": value
                    for key, value in test_metrics.items()
                    if key
                    not in {
                        "confusion_matrix",
                        "scores",
                        "predictions",
                    }
                },
                "test_top_1_percent_capture": capture[
                    "Top 1%"
                ],
                "test_top_5_percent_capture": capture[
                    "Top 5%"
                ],
                "test_top_10_percent_capture": capture[
                    "Top 10%"
                ],
            }
        )
        monthly.extend(
            monthly_records(
                fold.name,
                test,
                test_scores,
                threshold,
            )
        )

        # Persist each completed fold so interrupted runs can resume.
        pd.DataFrame(fold_records).to_csv(
            FOLD_RESULTS_PATH,
            index=False,
        )
        pd.DataFrame(monthly).to_csv(
            MONTHLY_RESULTS_PATH,
            index=False,
        )
        pd.concat(
            ranking_records,
            ignore_index=True,
        ).to_csv(
            RANKING_RESULTS_PATH,
            index=False,
        )
        completed_folds.add(fold.name)
        print(f"Checkpointed completed fold: {fold.name}")

    fold_results = pd.read_csv(FOLD_RESULTS_PATH)
    monthly_results = pd.read_csv(
        MONTHLY_RESULTS_PATH
    )

    print("\nRolling backtest fold summary")
    print("-" * 70)
    print(fold_results.to_string(index=False))
    print("\nMonthly backtest summary")
    print("-" * 70)
    print(monthly_results.to_string(index=False))
    print("\nSaved:")
    print(FOLD_RESULTS_PATH)
    print(MONTHLY_RESULTS_PATH)
    print(RANKING_RESULTS_PATH)
    print(LEAKAGE_AUDIT_PATH)


if __name__ == "__main__":
    main()
