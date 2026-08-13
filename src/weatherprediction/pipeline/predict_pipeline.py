import os
import sys
import numpy as np
import pandas as pd

from weatherprediction.exception import WeatherException
from weatherprediction.logger import logger
from weatherprediction.utils import load_object


PREPROCESSOR_PATH = os.path.join("artifacts", "preprocessor.pkl")
MODEL_PATH        = os.path.join("artifacts", "model.pkl")

# ── Feature columns ────────────────────────────────────────────────────────────
# These are the columns fed into the fitted ColumnTransformer.
# Order here matches how DataTransformation builds x_train (numeric first via
# select_dtypes, then 'city' as the only categorical).  The ColumnTransformer
# uses column names, so positional order in the DataFrame doesn't matter —
# but keeping them consistent avoids confusion.
FEATURE_COLUMNS = [
    # ── Numeric (13) ──────────────────────────────────────────────────────────
    "temperature_2m_max",           # daily high temp (°C)
    "temperature_2m_min",           # daily low temp (°C)
    "apparent_temperature_max",     # feels-like high (°C)
    "apparent_temperature_min",     # feels-like low (°C)
    "precipitation_sum",            # total precipitation (mm)
    "weather_code",                 # WMO weather code (0–65)
    "wind_speed_10m_max",           # max 10m wind speed (km/h)
    "wind_gusts_10m_max",           # max 10m wind gust (km/h)
    "wind_direction_10m_dominant",  # dominant wind direction (°)
    "rain_today",                   # 1 if precipitation_sum > 0
    "Day",                          # day of month (1–31)
    "Month",                        # month (1–12)
    "Year",                         # year (2000–2024)
    # ── Categorical (1 → 9 OHE columns after drop='first') ───────────────────
    "city",                         # one of 10 Indian cities in training data
]


# Optimal global decision threshold tuned via F1 sweep (0.375 yields 88.3% rain recall & 86.6% F1)
DECISION_THRESHOLD = 0.375


class PredictPipeline:
    """Loads saved artifacts and runs inference on a feature DataFrame."""

    def predict(self, features: pd.DataFrame) -> list[str]:
        """
        Returns list of 'Yes' / 'No' strings (one per row).
        """
        try:
            logger.info("─── Prediction pipeline started ──────────────────────────")
            preprocessor = load_object(PREPROCESSOR_PATH)
            model         = load_object(MODEL_PATH)

            features_transformed = preprocessor.transform(features)
            try:
                proba = model.predict_proba(features_transformed)[:, 1]
                preds = (proba >= DECISION_THRESHOLD).astype(int)
            except AttributeError:
                preds = model.predict(features_transformed)

            results = ["Yes" if p == 1 else "No" for p in preds]
            logger.info(f"Predictions: {results}")
            logger.info("─── Prediction pipeline complete ─────────────────────────")
            return results

        except Exception as e:
            raise WeatherException(e, sys)

    def predict_advanced(self, features: pd.DataFrame) -> list[dict]:
        """
        Returns enriched prediction dicts with:
          - prediction:   'Yes' / 'No'
          - probability:  float  0-100  (rain probability %)
          - confidence:   float  0-100  (distance from 0.5 decision boundary, scaled)
          - risk_level:   str    'Low' / 'Moderate' / 'High' / 'Very High'
          - risk_color:   str    CSS hex color
        """
        try:
            logger.info("─── Advanced Prediction pipeline started ─────────────────")
            preprocessor = load_object(PREPROCESSOR_PATH)
            model         = load_object(MODEL_PATH)

            features_transformed = preprocessor.transform(features)

            # Try to get probability — fallback to hard 0/1 if model doesn't support it
            try:
                proba = model.predict_proba(features_transformed)
                rain_proba = proba[:, 1]  # probability of class 1 (rain)
            except AttributeError:
                raw_preds = model.predict(features_transformed)
                rain_proba = np.array([1.0 if p == 1 else 0.0 for p in raw_preds])

            results = []
            for i in range(len(features)):
                prob_val = float(rain_proba[i])
                prob_pct = round(prob_val * 100, 1)
                label    = "Yes" if prob_val >= DECISION_THRESHOLD else "No"

                # Confidence = distance from tuned decision threshold (scaled 0-100)
                confidence = round(abs(prob_val - DECISION_THRESHOLD) * 160, 1)
                confidence = min(100.0, max(50.0, 50.0 + confidence))

                # Risk level based on rain probability
                if prob_pct < 25.0:
                    risk_level = "Low"
                    risk_color = "#22c55e"
                    risk_bg    = "rgba(34,197,94,0.12)"
                    risk_border= "rgba(34,197,94,0.35)"
                elif prob_pct < 37.5:
                    risk_level = "Moderate"
                    risk_color = "#f59e0b"
                    risk_bg    = "rgba(245,158,11,0.12)"
                    risk_border= "rgba(245,158,11,0.35)"
                elif prob_pct < 70.0:
                    risk_level = "High"
                    risk_color = "#f97316"
                    risk_bg    = "rgba(249,115,22,0.12)"
                    risk_border= "rgba(249,115,22,0.35)"
                else:
                    risk_level = "Very High"
                    risk_color = "#ef4444"
                    risk_bg    = "rgba(239,68,68,0.12)"
                    risk_border= "rgba(239,68,68,0.35)"

                results.append({
                    "prediction":   label,
                    "probability":  prob_pct,
                    "confidence":   confidence,
                    "risk_level":   risk_level,
                    "risk_color":   risk_color,
                    "risk_bg":      risk_bg,
                    "risk_border":  risk_border,
                })

            logger.info(f"Advanced predictions: {results}")
            logger.info("─── Advanced Prediction pipeline complete ────────────────")
            return results

        except Exception as e:
            raise WeatherException(e, sys)


class CustomData:
    """
    Maps raw derived-feature values coming from derive_features_from_owm()
    into a properly typed pandas DataFrame for inference.

    Feature set matches india_2000_2024_daily_weather.csv columns (post-
    engineering) that the ColumnTransformer was fitted on.
    """

    def __init__(
        self,
        temperature_2m_max:          float,
        temperature_2m_min:          float,
        apparent_temperature_max:    float,
        apparent_temperature_min:    float,
        precipitation_sum:           float,
        weather_code:                int,
        wind_speed_10m_max:          float,
        wind_gusts_10m_max:          float,
        wind_direction_10m_dominant: float,
        rain_today:                  int,    # 0 or 1
        day:                         int,
        month:                       int,
        year:                        int,
        city:                        str,    # e.g. "Delhi"
    ):
        self.temperature_2m_max          = temperature_2m_max
        self.temperature_2m_min          = temperature_2m_min
        self.apparent_temperature_max    = apparent_temperature_max
        self.apparent_temperature_min    = apparent_temperature_min
        self.precipitation_sum           = precipitation_sum
        self.weather_code                = weather_code
        self.wind_speed_10m_max          = wind_speed_10m_max
        self.wind_gusts_10m_max          = wind_gusts_10m_max
        self.wind_direction_10m_dominant = wind_direction_10m_dominant
        self.rain_today                  = rain_today
        self.day                         = day
        self.month                       = month
        self.year                        = year
        self.city                        = city

    def get_data_as_dataframe(self) -> pd.DataFrame:
        """Return a single-row DataFrame ready for the preprocessor."""
        try:
            data = {
                "temperature_2m_max":          [self.temperature_2m_max],
                "temperature_2m_min":          [self.temperature_2m_min],
                "apparent_temperature_max":    [self.apparent_temperature_max],
                "apparent_temperature_min":    [self.apparent_temperature_min],
                "precipitation_sum":           [self.precipitation_sum],
                "weather_code":                [self.weather_code],
                "wind_speed_10m_max":          [self.wind_speed_10m_max],
                "wind_gusts_10m_max":          [self.wind_gusts_10m_max],
                "wind_direction_10m_dominant": [self.wind_direction_10m_dominant],
                "rain_today":                  [self.rain_today],
                "Day":                         [self.day],
                "Month":                       [self.month],
                "Year":                        [self.year],
                "city":                        [self.city],
            }
            return pd.DataFrame(data)
        except Exception as e:
            raise WeatherException(e, sys)
