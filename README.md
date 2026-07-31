# RedZone Wildfire Risk Predictor

RedZone is a group project created for the University of Pennsylvania's
Engineering Summer Academy at Penn (ESAP) Artificial Intelligence course.
The project explores how machine learning can combine weather, vegetation,
terrain, and historical satellite fire detections to rank California grid
cells by next-day wildfire-related risk.

> **Educational prototype:** RedZone is not an emergency-warning system.
> Follow official guidance from CAL FIRE, local authorities, and the
> National Weather Service when making safety decisions.

## Project overview

California is divided into 4,355 analysis grids of approximately 10 km.
For every grid and date, the pipeline combines:

- daily and rolling Open-Meteo weather;
- LANDFIRE vegetation and fuel-model information;
- elevation and grid coordinates; and
- prior-year NASA FIRMS VIIRS hotspot history.

The target is whether the same grid receives a NASA FIRMS satellite hotspot
detection on the next calendar day. A hotspot detection is not necessarily
a distinct wildfire.

The website displays each grid's **within-day relative-risk percentile**.
These percentiles compare grids with one another for a particular forecast
date; they are not calibrated probabilities that a wildfire will occur.

## Website features

- Interactive California map with a continuous risk surface.
- Clustered pins at low zoom and individual grid pins at close zoom.
- Filters for Low, Medium, High, and Extreme pin tiers.
- Adjustable map-color opacity for viewing terrain underneath.
- Grid details including weather, fuel model, coordinates, and relative
  ranking.
- Static GitHub Pages deployment with browser-safe JSON predictions.

## Modeling

The repository includes experiments with logistic regression, random
forest, histogram gradient boosting, LightGBM, CatBoost, XGBoost ranking,
and rank-blended ensembles. CatBoost was selected as the primary website
model because it performed well on the highly imbalanced next-day target
and supported the project's numeric and categorical features.

The final seasonal CatBoost evaluation uses chronological rolling-origin
backtests. Thresholds are selected only on each fold's validation period.

| Test period | PR AUC | Precision | Recall | Top 1% capture | Top 5% capture | Top 10% capture |
|---|---:|---:|---:|---:|---:|---:|
| Late 2023 | 0.2747 | 2.02% | 73.75% | 34.95% | 53.65% | 59.50% |
| Late 2024 | 0.3675 | 5.26% | 78.27% | 44.64% | 67.55% | 75.79% |

Because positive grid-days are rare, PR AUC and daily top-risk capture are
more informative than accuracy. See the
[full rolling-backtest report](metadata/experiments/catboost_seasonal_rolling_backtest_2023_2024_report.md)
for monthly results and methodology.

## Data sources

- [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/) VIIRS active-fire
  detections.
- [Open-Meteo](https://open-meteo.com/) daily and forecast weather.
- [LANDFIRE](https://landfire.gov/) vegetation and fuel-model data.

Large raw and processed datasets are intentionally excluded from Git.
Tracked inference assets and trained artifacts allow the website pipeline
to run without committing the complete training datasets.

## Repository layout

```text
docs/                 Static website and published prediction JSON
scripts/              Data processing, training, evaluation, and inference
scripts/training/     Model training and rolling-backtest scripts
scripts/utils/        Shared feature, evaluation, and data utilities
models/               Saved model artifacts
metadata/             Data notes, experiment results, and reports
tests/                Automated integrity and inference tests
data/                 Raw, processed, current, and inference data directories
```

## Local setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the automated tests:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

The test suite checks next-day alignment, temporal leakage, duplicate and
missing grids, incomplete weather, finite model scores, weather-history
updates, and the expected 4,355-grid website output.

## Run the website locally

From the repository root:

```bash
python3 -m http.server 8000 --directory docs
```

Open <http://localhost:8000>. Do not open `docs/index.html` directly,
because browsers may block its prediction-file request under `file://`.

For prediction-generation commands and deployment details, see the
[website documentation](docs/README.md).

## Reproduce the temporal backtest

With the ignored processed training datasets available locally, run:

```bash
.venv/bin/python scripts/training/backtest_catboost_2023_2024.py
```

The script performs leakage checks, resumes from completed fold
checkpoints, and writes fold, monthly, ranking, and audit reports under
`metadata/experiments/`.

## Limitations

- Matching 2025 grid-level historical weather was not available before the
  project deadline, so an independent 2025 evaluation was not completed.
- The 2023-2024 results are chronological development backtests rather than
  external validation on a new year.
- FIRMS detections can include repeated observations and are not a count of
  distinct wildfire incidents.
- The current model produces relative rankings, not calibrated wildfire
  probabilities.
- Weather forecasts, satellite detections, and static land-cover data all
  contain measurement uncertainty.

## Course context

This repository documents the research, engineering, testing, and website
work completed by the project group for the UPenn ESAP Artificial
Intelligence course. It is intended for education, demonstration, and
continued experimentation.
