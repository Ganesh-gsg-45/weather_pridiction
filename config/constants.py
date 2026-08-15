"""
constants.py — Project-wide constants reference for GOOD WEATHER.

This file documents the key thresholds and configuration values used
across the pipeline. It serves as the single source of truth for
values that are currently inline in their respective modules.

NOTE: These values are NOT imported by the app in this hygiene pass.
      Future refactor: import these in predict_pipeline.py,
      data_ingestion.py, and app.py to centralize config.
"""

# ── Model Inference ────────────────────────────────────────────────────────────
# Source: src/weatherprediction/pipeline/predict_pipeline.py
# Optimal decision threshold tuned via F1 sweep (0.375 yields 88.3% rain
# recall & 86.6% F1 on the 2022–2024 hold-out test set).
DECISION_THRESHOLD = 0.375

# Risk probability bands (rain_probability % → risk label)
RISK_BANDS = {
    "Low":       (0.0,  25.0),   # color: #22c55e (green)
    "Moderate":  (25.0, 37.5),   # color: #f59e0b (amber)
    "High":      (37.5, 70.0),   # color: #f97316 (orange)
    "Very High": (70.0, 100.0),  # color: #ef4444 (red)
}

# ── Data Split ─────────────────────────────────────────────────────────────────
# Source: src/weatherprediction/components/data_ingestion.py
# Rows with date <= TRAIN_CUTOFF → train; rows after → test.
# Train: 2000-01-01 → 2021-12-31 (~22 years, ~80%)
# Test:  2022-01-01 → 2024-xx-xx (~3 years,  ~20%)
TRAIN_CUTOFF = "2021-12-31"

# ── City Coordinates ──────────────────────────────────────────────────────────
# Source: app.py — used for geocoding & Open-Meteo API calls.
# These are the 10 Indian cities covered by the training dataset.
CITY_COORDS = {
    "Delhi":     {"lat": 28.6139, "lon": 77.2090},
    "Mumbai":    {"lat": 19.0760, "lon": 72.8777},
    "Chennai":   {"lat": 13.0827, "lon": 80.2707},
    "Kolkata":   {"lat": 22.5726, "lon": 88.3639},
    "Bangalore": {"lat": 12.9716, "lon": 77.5946},
    "Hyderabad": {"lat": 17.3850, "lon": 78.4867},
    "Pune":      {"lat": 18.5204, "lon": 73.8567},
    "Ahmedabad": {"lat": 23.0225, "lon": 72.5714},
    "Jaipur":    {"lat": 26.9124, "lon": 75.7873},
    "Surat":     {"lat": 21.1702, "lon": 72.8311},
}

# ── Feature Columns ────────────────────────────────────────────────────────────
# Source: src/weatherprediction/pipeline/predict_pipeline.py
# Exact column order fed into the fitted ColumnTransformer.
FEATURE_COLUMNS = [
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
