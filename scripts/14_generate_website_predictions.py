from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


DEFAULT_FEATURES_PATH = Path(
    "data/current/features/next_day_features.parquet"
)
DEFAULT_MODEL_PATH = Path(
    "models/catboost_rolling_history_2024.joblib"
)
DEFAULT_OUTPUT_PATH = Path(
    "docs/data/predictions.json"
)

DISPLAY_COLUMNS = [
    "grid_id",
    "centroid_lat",
    "centroid_lon",
    "temperature_max",
    "humidity_min",
    "wind_speed_max",
    "precipitation_total",
    "fuel_model_dominant",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score next-day grid features with the saved CatBoost "
            "artifact and publish website JSON."
        )
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=DEFAULT_FEATURES_PATH,
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Load only a trusted local joblib artifact.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
    )
    return parser.parse_args()


def daily_percentile(scores: np.ndarray) -> np.ndarray:
    """Convert model scores to ascending 0-100 daily percentiles."""
    return (
        pd.Series(scores)
        .rank(method="average", pct=True)
        .mul(100)
        .to_numpy()
    )


def risk_level(percentile: float) -> str:
    if percentile >= 99:
        return "Extreme"
    if percentile >= 95:
        return "High"
    if percentile >= 90:
        return "Medium"
    return "Low"


def _single_date(
    features: pd.DataFrame,
    column: str,
) -> pd.Timestamp:
    if column not in features:
        raise ValueError(
            f"Inference features are missing {column}."
        )
    dates = pd.to_datetime(features[column]).dt.normalize()
    unique_dates = dates.drop_duplicates()
    if len(unique_dates) != 1:
        raise ValueError(
            f"Inference features must contain exactly one {column}."
        )
    return unique_dates.iloc[0]


def build_prediction_payload(
    features: pd.DataFrame,
    artifact: dict,
    model_name: str,
) -> dict:
    required_artifact_keys = {
        "model",
        "threshold",
        "feature_columns",
        "categorical_features",
    }
    missing_artifact_keys = sorted(
        required_artifact_keys - set(artifact)
    )
    if missing_artifact_keys:
        raise ValueError(
            "Model artifact is missing keys: "
            f"{missing_artifact_keys}"
        )

    feature_columns = artifact["feature_columns"]
    required_columns = set(
        feature_columns + DISPLAY_COLUMNS
    )
    missing_columns = sorted(
        required_columns - set(features.columns)
    )
    if missing_columns:
        raise ValueError(
            "Inference features are missing columns: "
            f"{missing_columns}"
        )

    feature_date = _single_date(features, "feature_date")
    prediction_date = _single_date(
        features,
        "prediction_date",
    )
    if prediction_date != feature_date + pd.Timedelta(days=1):
        raise ValueError(
            "prediction_date must be exactly one calendar day "
            "after feature_date."
        )

    X = features[feature_columns].copy()
    missing_token = artifact.get(
        "categorical_missing_value",
        "__MISSING__",
    )
    for column in artifact["categorical_features"]:
        X[column] = (
            X[column]
            .astype("string")
            .fillna(missing_token)
            .astype(str)
        )

    model_scores = artifact["model"].predict_proba(X)[:, 1]
    percentiles = daily_percentile(model_scores)
    threshold = float(artifact["threshold"])

    locations = []
    for row, score, percentile in zip(
        features.itertuples(index=False),
        model_scores,
        percentiles,
        strict=True,
    ):
        level = risk_level(float(percentile))
        fuel_model = str(row.fuel_model_dominant)
        locations.append(
            {
                "id": str(row.grid_id),
                "name": f"Grid {row.grid_id}",
                "area": "California analysis grid",
                "lat": round(float(row.centroid_lat), 6),
                "lng": round(float(row.centroid_lon), 6),
                "forest_type": f"LANDFIRE fuel model {fuel_model}",
                "risk_score": round(float(percentile), 1),
                "risk_level": level,
                "model_score": round(float(score), 8),
                "above_model_threshold": bool(
                    score >= threshold
                ),
                "conditions": {
                    "temperature_max_c": round(
                        float(row.temperature_max),
                        1,
                    ),
                    "humidity_min_percent": round(
                        float(row.humidity_min),
                        1,
                    ),
                    "wind_speed_max_kmh": round(
                        float(row.wind_speed_max),
                        1,
                    ),
                    "precipitation_mm": round(
                        float(row.precipitation_total),
                        2,
                    ),
                },
            }
        )

    return {
        "schema_version": 1,
        "status": "ready",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "feature_date": feature_date.date().isoformat(),
        "prediction_date": prediction_date.date().isoformat(),
        "model": model_name,
        "target": "next-day FIRMS hotspot detection",
        "score_semantics": (
            "Within-day relative risk percentile, not calibrated "
            "wildfire probability."
        ),
        "grid_count": len(locations),
        "locations": locations,
    }


def write_payload(payload: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )
    temporary_path.write_text(
        json.dumps(
            payload,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(output_path)


def main() -> None:
    args = parse_args()
    features = pd.read_parquet(args.features)

    # joblib/pickle can execute code while loading. This path must refer
    # only to a model artifact created and controlled by this project.
    artifact = joblib.load(args.model)
    if not isinstance(artifact, dict):
        raise ValueError(
            "Expected the model artifact to contain a dictionary."
        )

    payload = build_prediction_payload(
        features=features,
        artifact=artifact,
        model_name=args.model.stem,
    )
    write_payload(payload, args.output)

    print(f"Feature date: {payload['feature_date']}")
    print(f"Prediction date: {payload['prediction_date']}")
    print(f"Grid cells: {payload['grid_count']:,}")
    print(f"Saved website predictions to: {args.output}")


if __name__ == "__main__":
    main()
