"""
test_predict_pipeline.py

Verifies two things without requiring trained model artifacts:

1. CustomData.get_data_as_dataframe() produces a DataFrame with the exact
   expected column names and order.

2. PredictPipeline.predict_advanced() returns dicts containing the 5 expected
   keys: prediction, probability, confidence, risk_level, risk_color.

load_object() is mocked so no .pkl artifacts are needed — the test is
fully self-contained and runs in CI without any pre-trained model.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from src.weatherprediction.pipeline.predict_pipeline import (
    CustomData,
    PredictPipeline,
    FEATURE_COLUMNS,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

SAMPLE_KWARGS = dict(
    temperature_2m_max=34.5,
    temperature_2m_min=24.1,
    apparent_temperature_max=38.0,
    apparent_temperature_min=26.5,
    precipitation_sum=0.0,
    weather_code=1,
    wind_speed_10m_max=15.3,
    wind_gusts_10m_max=22.1,
    wind_direction_10m_dominant=180.0,
    rain_today=0,
    day=15,
    month=8,
    year=2024,
    city="Delhi",
)

EXPECTED_COLUMNS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "apparent_temperature_max",
    "apparent_temperature_min",
    "precipitation_sum",
    "weather_code",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "wind_direction_10m_dominant",
    "rain_today",
    "Day",
    "Month",
    "Year",
    "city",
]

EXPECTED_RESULT_KEYS = {"prediction", "probability", "confidence", "risk_level", "risk_color"}


# ── CustomData tests ───────────────────────────────────────────────────────────

class TestCustomData:

    def test_dataframe_has_expected_columns(self):
        """get_data_as_dataframe() must return exactly the 14 expected columns."""
        cd = CustomData(**SAMPLE_KWARGS)
        df = cd.get_data_as_dataframe()

        assert list(df.columns) == EXPECTED_COLUMNS, (
            f"Column mismatch.\n"
            f"  Expected: {EXPECTED_COLUMNS}\n"
            f"  Got:      {list(df.columns)}"
        )

    def test_dataframe_has_one_row(self):
        """Single-inference path must return exactly one row."""
        cd = CustomData(**SAMPLE_KWARGS)
        df = cd.get_data_as_dataframe()
        assert len(df) == 1, f"Expected 1 row, got {len(df)}"

    def test_columns_match_feature_columns_constant(self):
        """DataFrame columns must exactly match the FEATURE_COLUMNS constant."""
        cd = CustomData(**SAMPLE_KWARGS)
        df = cd.get_data_as_dataframe()
        assert list(df.columns) == FEATURE_COLUMNS, (
            "CustomData columns diverged from FEATURE_COLUMNS constant. "
            "Update one or the other to keep them in sync."
        )

    def test_city_value_preserved(self):
        """City string must survive the round-trip to DataFrame unchanged."""
        cd = CustomData(**SAMPLE_KWARGS)
        df = cd.get_data_as_dataframe()
        assert df["city"].iloc[0] == "Delhi"

    def test_numeric_values_preserved(self):
        """Numeric feature values must survive the round-trip unchanged."""
        cd = CustomData(**SAMPLE_KWARGS)
        df = cd.get_data_as_dataframe()
        assert df["temperature_2m_max"].iloc[0] == pytest.approx(34.5)
        assert df["Month"].iloc[0] == 8
        assert df["Year"].iloc[0] == 2024


# ── PredictPipeline tests ──────────────────────────────────────────────────────

class TestPredictPipeline:
    """
    Uses unittest.mock to stub load_object() so no .pkl artifacts are needed.
    The mock preprocessor returns the input unchanged; the mock model returns
    a known probability so we can assert on the output dict values.
    """

    def _make_mocks(self, rain_probability: float = 0.80):
        """Build mock preprocessor + model for a given rain probability."""
        mock_preprocessor = MagicMock()
        mock_preprocessor.transform.side_effect = lambda x: x.values

        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array(
            [[1 - rain_probability, rain_probability]]
        )
        return mock_preprocessor, mock_model

    def _get_sample_df(self) -> pd.DataFrame:
        cd = CustomData(**SAMPLE_KWARGS)
        return cd.get_data_as_dataframe()

    def test_predict_advanced_returns_list_of_dicts(self):
        """predict_advanced() must return a list."""
        mock_pre, mock_model = self._make_mocks()
        with patch("src.weatherprediction.pipeline.predict_pipeline.load_object",
                   side_effect=[mock_pre, mock_model]):
            pipeline = PredictPipeline()
            result = pipeline.predict_advanced(self._get_sample_df())

        assert isinstance(result, list), f"Expected list, got {type(result)}"
        assert len(result) == 1, f"Expected 1 result dict, got {len(result)}"

    def test_predict_advanced_result_has_required_keys(self):
        """Each result dict must contain the 5 required keys."""
        mock_pre, mock_model = self._make_mocks(rain_probability=0.80)
        with patch("src.weatherprediction.pipeline.predict_pipeline.load_object",
                   side_effect=[mock_pre, mock_model]):
            pipeline = PredictPipeline()
            result = pipeline.predict_advanced(self._get_sample_df())

        result_keys = set(result[0].keys())
        assert EXPECTED_RESULT_KEYS.issubset(result_keys), (
            f"Missing keys in predict_advanced() output.\n"
            f"  Expected (at minimum): {EXPECTED_RESULT_KEYS}\n"
            f"  Got: {result_keys}"
        )

    def test_high_probability_gives_rain_yes(self):
        """Probability above DECISION_THRESHOLD (0.375) → prediction = 'Yes'."""
        mock_pre, mock_model = self._make_mocks(rain_probability=0.90)
        with patch("src.weatherprediction.pipeline.predict_pipeline.load_object",
                   side_effect=[mock_pre, mock_model]):
            pipeline = PredictPipeline()
            result = pipeline.predict_advanced(self._get_sample_df())

        assert result[0]["prediction"] == "Yes", (
            f"Expected 'Yes' for p=0.90, got '{result[0]['prediction']}'"
        )

    def test_low_probability_gives_rain_no(self):
        """Probability below DECISION_THRESHOLD (0.375) → prediction = 'No'."""
        mock_pre, mock_model = self._make_mocks(rain_probability=0.10)
        with patch("src.weatherprediction.pipeline.predict_pipeline.load_object",
                   side_effect=[mock_pre, mock_model]):
            pipeline = PredictPipeline()
            result = pipeline.predict_advanced(self._get_sample_df())

        assert result[0]["prediction"] == "No", (
            f"Expected 'No' for p=0.10, got '{result[0]['prediction']}'"
        )

    def test_probability_field_is_percentage(self):
        """probability field must be in 0–100 range (not 0–1)."""
        mock_pre, mock_model = self._make_mocks(rain_probability=0.80)
        with patch("src.weatherprediction.pipeline.predict_pipeline.load_object",
                   side_effect=[mock_pre, mock_model]):
            pipeline = PredictPipeline()
            result = pipeline.predict_advanced(self._get_sample_df())

        prob = result[0]["probability"]
        assert 0.0 <= prob <= 100.0, f"probability out of 0–100 range: {prob}"
        assert prob == pytest.approx(80.0, abs=0.1), f"Expected 80.0, got {prob}"

    def test_risk_level_is_valid_string(self):
        """risk_level must be one of the four defined categories."""
        valid_levels = {"Low", "Moderate", "High", "Very High"}
        mock_pre, mock_model = self._make_mocks(rain_probability=0.80)
        with patch("src.weatherprediction.pipeline.predict_pipeline.load_object",
                   side_effect=[mock_pre, mock_model]):
            pipeline = PredictPipeline()
            result = pipeline.predict_advanced(self._get_sample_df())

        assert result[0]["risk_level"] in valid_levels, (
            f"Unexpected risk_level: '{result[0]['risk_level']}'. "
            f"Must be one of {valid_levels}"
        )
