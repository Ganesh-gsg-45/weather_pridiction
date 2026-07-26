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
        Parameters
        ----------
        features : pd.DataFrame
            One or more rows with columns matching FEATURE_COLUMNS.

        Returns
        -------
        list of "Yes" / "No" strings (one per row)
        """
        try:
            logger.info("─── Prediction pipeline started ──────────────────────────")
            preprocessor = load_object(PREPROCESSOR_PATH)
            model         = load_object(MODEL_PATH)

            features_transformed = preprocessor.transform(features)
            preds = model.predict(features_transformed)

            # Map 1 → "Yes", 0 → "No"
            results = ["Yes" if p == 1 else "No" for p in preds]
            logger.info(f"Predictions: {results}")
            logger.info("─── Prediction pipeline complete ─────────────────────────")
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
