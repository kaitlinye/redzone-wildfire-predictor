# RedZone next-day prediction interface

This static site reads generated next-day CatBoost predictions from
`docs/data/predictions.json`.

## Generate predictions

The model uses weather and other features dated **D** to predict a FIRMS
hotspot detection on **D+1**. The weather inputs must contain D and enough
recent prior dates to calculate rolling weather features.

From the repository root:

```bash
.venv/bin/python scripts/09_download_daily_forecast.py

.venv/bin/python scripts/10_build_daily_features.py \
  --weather data/current/weather

.venv/bin/python scripts/14_generate_website_predictions.py
```

The first command writes:

```text
data/current/weather/weather_through_YYYY-MM-DD.parquet
```

It requests 30 prior days plus the current day from Open-Meteo so the
rolling features can be calculated on the first run. The feature builder
writes:

```text
data/current/features/next_day_features.parquet
```

The prediction generator writes the browser-safe output:

```text
docs/data/predictions.json
```

Only load a trusted project-created `.joblib` model. Joblib artifacts use
Python pickle internally and must not be accepted from website users.

To test the pipeline with a historical feature date:

```bash
.venv/bin/python scripts/10_build_daily_features.py \
  --weather data/processed/wildfire_training_2024.parquet \
  --feature-date 2024-10-29

.venv/bin/python scripts/14_generate_website_predictions.py
```

This historical command is only an integration check. The website will mark
an old prediction date as stale.

## Run the site

Serve the repository's `docs` directory rather than opening `index.html`
directly:

```bash
python3 -m http.server 8000 --directory docs
```

Then open `http://localhost:8000`.

## Interpretation

- The displayed 0–100 score is a within-day relative-risk percentile.
- It is not a calibrated probability that a wildfire will occur.
- The prediction target is a satellite FIRMS hotspot detection in a grid
  cell on the next calendar day.
- Every analyzed grid is available as a pin. Pins are clustered at lower
  zoom levels and can be filtered by risk tier. Low-tier pins are hidden by
  default, while Medium, High, and Extreme are enabled. The colored risk
  surface always uses all scored grids.
- Temperature is shown in °C, wind in km/h, and precipitation in mm.

The current model supports one next-calendar-day horizon. The site does not
claim sub-daily, 48-hour, or 72-hour predictions.

## Automatic daily updates

`.github/workflows/update-daily-predictions.yml` runs every day at
08:00 UTC (12:00 AM PST or 01:00 AM PDT) and can also be started manually
from the repository's **Actions** tab. It:

1. downloads the current Open-Meteo forecast and recent weather history;
2. builds the rolling and static model features;
3. runs the saved CatBoost model;
4. commits the updated `docs/data/predictions.json`.

The workflow performs inference only. It does not retrain or replace the
saved model.

`.github/workflows/deploy-site.yml` handles deployment separately. It
deploys the `docs` directory after every push to `main`, after a successful
daily prediction update, or when manually started from the Actions tab.
The `workflow_run` trigger is needed because commits pushed with the default
GitHub Actions token do not trigger another push workflow.

Before the first run, open the repository settings:

- Under **Pages**, set the deployment source to **GitHub Actions**.
- Under **Actions → General → Workflow permissions**, select
  **Read and write permissions**.

If a scheduled update fails, the last successful Pages deployment remains
online. Inspect the failed run in the Actions tab and rerun it after fixing
the download or configuration problem.
