import os
import sys
import numpy as np
import pandas as pd

from weatherprediction.exception import WeatherException
from weatherprediction.logger import logger
from weatherprediction.utils import load_object


PREPROCESSOR_PATH = os.path.join("artifacts", "preprocessor.pkl")
MODEL_PATH        = os.path.join("artifacts", "model.pkl")

# Feature order must exactly match what DataTransformation produces
FEATURE_COLUMNS = [
    "MinTemp", "MaxTemp", "Rainfall", "Evaporation", "Sunshine",
    "WindGustSpeed", "WindSpeed9am", "WindSpeed3pm",
    "Humidity9am", "Humidity3pm",
    "Pressure9am", "Pressure3pm",
    "Temp9am", "Temp3pm",
    "RainToday",          # already label-encoded to 0/1
    "Day", "Month", "Year",
]


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
            preds = model.predict(features_transformed)

            # Try to get probability — fallback to hard 0/1 if model doesn't support it
            try:
                proba = model.predict_proba(features_transformed)
                rain_proba = proba[:, 1]  # probability of class 1 (rain)
            except AttributeError:
                rain_proba = np.array([1.0 if p == 1 else 0.0 for p in preds])

            results = []
            for i, p in enumerate(preds):
                prob_pct = round(float(rain_proba[i]) * 100, 1)
                label    = "Yes" if p == 1 else "No"

                # Confidence = how far from 0.5 boundary, scaled 0-100
                confidence = round(abs(float(rain_proba[i]) - 0.5) * 200, 1)

                # Risk level based on rain probability
                if prob_pct < 25:
                    risk_level = "Low"
                    risk_color = "#22c55e"
                    risk_bg    = "rgba(34,197,94,0.12)"
                    risk_border= "rgba(34,197,94,0.35)"
                elif prob_pct < 50:
                    risk_level = "Moderate"
                    risk_color = "#f59e0b"
                    risk_bg    = "rgba(245,158,11,0.12)"
                    risk_border= "rgba(245,158,11,0.35)"
                elif prob_pct < 75:
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
    Maps raw form-field values (strings/floats) coming from the Flask
    request into a properly typed pandas DataFrame for inference.
    """

    def __init__(
        self,
        min_temp:       float,
        max_temp:       float,
        rainfall:       float,
        evaporation:    float,
        sunshine:       float,
        wind_gust_speed: float,
        wind_speed_9am: float,
        wind_speed_3pm: float,
        humidity_9am:   float,
        humidity_3pm:   float,
        pressure_9am:   float,
        pressure_3pm:   float,
        temp_9am:       float,
        temp_3pm:       float,
        rain_today:     int,   # 0 or 1
        day:            int,
        month:          int,
        year:           int,
    ):
        self.min_temp        = min_temp
        self.max_temp        = max_temp
        self.rainfall        = rainfall
        self.evaporation     = evaporation
        self.sunshine        = sunshine
        self.wind_gust_speed = wind_gust_speed
        self.wind_speed_9am  = wind_speed_9am
        self.wind_speed_3pm  = wind_speed_3pm
        self.humidity_9am    = humidity_9am
        self.humidity_3pm    = humidity_3pm
        self.pressure_9am    = pressure_9am
        self.pressure_3pm    = pressure_3pm
        self.temp_9am        = temp_9am
        self.temp_3pm        = temp_3pm
        self.rain_today      = rain_today
        self.day             = day
        self.month           = month
        self.year            = year

    def get_data_as_dataframe(self) -> pd.DataFrame:
        """Return a single-row DataFrame ready for the preprocessor."""
        try:
            data = {
                "MinTemp":       [self.min_temp],
                "MaxTemp":       [self.max_temp],
                "Rainfall":      [self.rainfall],
                "Evaporation":   [self.evaporation],
                "Sunshine":      [self.sunshine],
                "WindGustSpeed": [self.wind_gust_speed],
                "WindSpeed9am":  [self.wind_speed_9am],
                "WindSpeed3pm":  [self.wind_speed_3pm],
                "Humidity9am":   [self.humidity_9am],
                "Humidity3pm":   [self.humidity_3pm],
                "Pressure9am":   [self.pressure_9am],
                "Pressure3pm":   [self.pressure_3pm],
                "Temp9am":       [self.temp_9am],
                "Temp3pm":       [self.temp_3pm],
                "RainToday":     [self.rain_today],
                "Day":           [self.day],
                "Month":         [self.month],
                "Year":          [self.year],
            }
            return pd.DataFrame(data)
        except Exception as e:
            raise WeatherException(e, sys)
