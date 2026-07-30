# Seasonal CatBoost rolling temporal backtest

## Design

The backtest uses two chronological, rolling-origin folds. Thresholds are
selected independently on each fold's validation period with a minimum
70% recall target. No test-period rows are used for model fitting,
hyperparameter selection, or threshold selection.

| Fold | Training | Validation | Testing |
|---|---|---|---|
| Late 2023 | 2023-05-08–2023-07-31 | 2023-08-01–2023-09-15 | 2023-09-16–2023-10-29 |
| Late 2024 | 2023-05-08–2024-07-31 | 2024-08-01–2024-09-15 | 2024-09-16–2024-10-29 |

The 2023 feature rows use the May–October 2022 FIRMS summary. The 2024
feature rows use the May–October 2023 FIRMS summary.

## Fold-level test results

| Fold | PR AUC | Precision | Recall | F1 | Top 1% capture | Top 5% capture | Top 10% capture |
|---|---:|---:|---:|---:|---:|---:|---:|
| Late 2023 | 0.2747 | 2.02% | 73.75% | 3.94% | 34.95% | 53.65% | 59.50% |
| Late 2024 | 0.3675 | 5.26% | 78.27% | 9.86% | 44.64% | 67.55% | 75.79% |

## Monthly test results

| Month | PR AUC | Precision | Recall | Top 1% capture | Top 5% capture | Top 10% capture |
|---|---:|---:|---:|---:|---:|---:|
| 2023-09 (Sep 16–30) | 0.2566 | 1.71% | 72.13% | 36.93% | 54.01% | 59.41% |
| 2023-10 (Oct 1–29) | 0.2845 | 2.18% | 74.40% | 34.15% | 53.51% | 59.54% |
| 2024-09 (Sep 16–30) | 0.3784 | 3.87% | 83.97% | 53.44% | 74.05% | 82.44% |
| 2024-10 (Oct 1–29) | 0.3696 | 6.03% | 76.43% | 41.82% | 65.47% | 73.65% |

Precision remains numerically low because next-day positive grid rows are
rare. PR AUC and daily top-risk capture are more informative than accuracy
for this use case. These scores describe ranking performance, not
calibrated wildfire probability.

## Leakage audit

All automated temporal checks passed:

- every prior-year summary maps `history_year` to `feature_year - 1`;
- every prior-year summary window ends before its feature year;
- both feature years have prior-year coverage for all 4,355 grids;
- changing 2023 weather on and after an evaluated date did not alter that
  date's rolling weather features; and
- the equivalent past-only rolling check passed for 2024.

## Inference and failure-mode checks

The test suite verifies:

- exactly 4,355 unique grids with finite model and percentile scores;
- rejection of an incorrect inference grid count;
- rejection of duplicate inference grid IDs;
- rejection of missing feature-date grids;
- rejection of duplicate feature-date weather rows;
- rejection of incomplete feature-date weather;
- rejection of non-finite model scores;
- next-calendar-day target alignment;
- rolling weather features ignore current and future weather; and
- retained weather history is revised, deduplicated, and trimmed.

The current published prediction artifact passed the structural check with
4,355 unique scored grids. Its date is checked separately by the website's
stale-data behavior and is not evidence of current model performance.
