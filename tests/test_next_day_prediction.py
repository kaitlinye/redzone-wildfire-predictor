from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from scripts.utils.next_day_features import (
    build_next_day_features,
)


GENERATOR_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "14_generate_website_predictions.py"
)
SPEC = importlib.util.spec_from_file_location(
    "website_prediction_generator",
    GENERATOR_PATH,
)
assert SPEC is not None
assert SPEC.loader is not None
GENERATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GENERATOR)

DOWNLOADER_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "09_download_daily_forecast.py"
)
DOWNLOADER_SPEC = importlib.util.spec_from_file_location(
    "daily_weather_downloader",
    DOWNLOADER_PATH,
)
assert DOWNLOADER_SPEC is not None
assert DOWNLOADER_SPEC.loader is not None
DOWNLOADER = importlib.util.module_from_spec(
    DOWNLOADER_SPEC
)
DOWNLOADER_SPEC.loader.exec_module(DOWNLOADER)


class FakeModel:
    def predict_proba(
        self,
        features: pd.DataFrame,
    ) -> np.ndarray:
        scores = (
            features["temperature_max"]
            .rank(method="average", pct=True)
            .to_numpy()
            * 0.1
        )
        return np.column_stack([1 - scores, scores])


class NextDayPredictionTests(unittest.TestCase):
    def setUp(self) -> None:
        dates = pd.date_range(
            "2024-01-01",
            periods=31,
            freq="D",
        )
        rows = []
        for grid_number, grid_id in enumerate(
            ["CA-00001", "CA-00002"],
            start=1,
        ):
            for day_number, date in enumerate(dates):
                rows.append(
                    {
                        "grid_id": grid_id,
                        "date": date,
                        "temperature_max": 20 + grid_number,
                        "temperature_min": 10,
                        "humidity_mean": 50,
                        "humidity_min": 30 + grid_number,
                        "precipitation_total": (
                            1000
                            if day_number == 30
                            else 1
                        ),
                        "wind_speed_max": 15,
                        "wind_direction_dominant": 270,
                        "elevation": 100 * grid_number,
                        "centroid_lat": 35 + grid_number,
                        "centroid_lon": -120 - grid_number,
                    }
                )
        self.weather = pd.DataFrame(rows)
        self.landfire = pd.DataFrame(
            {
                "grid_id": ["CA-00001", "CA-00002"],
                "vegetation_cover_mean": [50.0, 60.0],
                "fuel_model_dominant": [101, 102],
                "landfire_missing": [0, 0],
            }
        )
        self.historical = pd.DataFrame(
            {
                "grid_id": ["CA-00001", "CA-00002"],
                "historical_firms_detection_count_2020_2023": [
                    0,
                    9,
                ],
            }
        )

    def test_feature_date_predicts_exactly_next_day(self) -> None:
        features = build_next_day_features(
            weather=self.weather,
            landfire=self.landfire,
            historical_firms=self.historical,
            feature_date="2024-01-31",
        )

        self.assertEqual(len(features), 2)
        self.assertTrue(
            (
                features["feature_date"]
                == pd.Timestamp("2024-01-31")
            ).all()
        )
        self.assertTrue(
            (
                features["prediction_date"]
                == pd.Timestamp("2024-02-01")
            ).all()
        )
        self.assertTrue(
            (features["rain_previous_7d"] == 7).all()
        )
        self.assertTrue(
            (features["rain_previous_30d"] == 30).all()
        )
        self.assertAlmostEqual(
            features.loc[
                features["grid_id"] == "CA-00002",
                "historical_firms_detection_count_log",
            ].iloc[0],
            np.log1p(9),
        )

    def test_payload_uses_percentiles_and_expected_units(self) -> None:
        features = build_next_day_features(
            weather=self.weather,
            landfire=self.landfire,
            historical_firms=self.historical,
            feature_date="2024-01-31",
        )
        feature_columns = [
            "temperature_max",
            "temperature_min",
            "humidity_mean",
            "humidity_min",
            "precipitation_total",
            "wind_speed_max",
            "elevation",
            "vegetation_cover_mean",
            "rain_previous_7d",
            "rain_previous_30d",
            "temperature_max_previous_3d",
            "humidity_min_previous_3d",
            "historical_firms_detection_count_log",
            "wind_direction_dominant",
            "fuel_model_dominant",
            "landfire_missing",
        ]
        artifact = {
            "model": FakeModel(),
            "threshold": 0.05,
            "feature_columns": feature_columns,
            "categorical_features": [
                "wind_direction_dominant",
                "fuel_model_dominant",
                "landfire_missing",
            ],
            "categorical_missing_value": "__MISSING__",
        }

        payload = GENERATOR.build_prediction_payload(
            features,
            artifact,
            "test_model",
        )

        self.assertEqual(payload["prediction_date"], "2024-02-01")
        self.assertEqual(payload["grid_count"], 2)
        self.assertEqual(
            [item["risk_score"] for item in payload["locations"]],
            [50.0, 100.0],
        )
        self.assertIn(
            "temperature_max_c",
            payload["locations"][0]["conditions"],
        )
        self.assertIn(
            "not calibrated",
            payload["score_semantics"],
        )

    def test_weather_history_is_merged_revised_and_trimmed(
        self,
    ) -> None:
        existing = pd.DataFrame(
            {
                "grid_id": ["CA-00001", "CA-00001"],
                "date": ["2024-01-01", "2024-01-31"],
                "temperature_max": [10.0, 20.0],
            }
        )
        downloaded = pd.DataFrame(
            {
                "grid_id": ["CA-00001", "CA-00001"],
                "date": ["2024-01-31", "2024-02-01"],
                "temperature_max": [21.0, 22.0],
            }
        )

        result = DOWNLOADER.update_weather_history(
            existing,
            downloaded,
            forecast_date="2024-02-01",
        )

        self.assertEqual(
            result["date"].dt.date.astype(str).tolist(),
            ["2024-01-31", "2024-02-01"],
        )
        self.assertEqual(
            result["temperature_max"].tolist(),
            [21.0, 22.0],
        )


if __name__ == "__main__":
    unittest.main()
