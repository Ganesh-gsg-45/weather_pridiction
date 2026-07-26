import sys
import os
import requests
from flask import Flask, request, render_template, jsonify
from dotenv import load_dotenv

from weatherprediction.exception import WeatherException
from weatherprediction.logger import logger
from weatherprediction.pipeline.predict_pipeline import CustomData, PredictPipeline

load_dotenv()
app = Flask(__name__)

WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "")
OWM_BASE = "https://api.openweathermap.org/data/2.5"
OWM_GEO  = "https://api.openweathermap.org/geo/1.0"


# ── Helpers ────────────────────────────────────────────────────────────────────
def owm_get(endpoint, params: dict):
    """Call OpenWeatherMap API and return JSON or None on error."""
    params["appid"] = WEATHER_API_KEY
    try:
        r = requests.get(endpoint, params=params, timeout=8)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        logger.warning(f"OWM API error: {exc}")
        return None


def weather_icon_class(icon_code: str) -> str:
    """Map OWM icon codes to CSS icon class names."""
    mapping = {
        "01d": "sun", "01n": "moon",
        "02d": "partly-cloudy", "02n": "partly-cloudy-night",
        "03d": "cloudy", "03n": "cloudy",
        "04d": "overcast", "04n": "overcast",
        "09d": "rain", "09n": "rain",
        "10d": "rain-sun", "10n": "rain",
        "11d": "thunder", "11n": "thunder",
        "13d": "snow", "13n": "snow",
        "50d": "fog", "50n": "fog",
    }
    return mapping.get(icon_code, "sun")


def aqi_label(aqi: int) -> dict:
    labels = {
        1: {"text": "Good",        "color": "#22c55e"},
        2: {"text": "Fair",        "color": "#84cc16"},
        3: {"text": "Moderate",    "color": "#f59e0b"},
        4: {"text": "Poor",        "color": "#ef4444"},
        5: {"text": "Very Poor",   "color": "#7c3aed"},
    }
    return labels.get(aqi, {"text": "Unknown", "color": "#6b7280"})


# ── Pages ──────────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    """AccuWeather-style landing page."""
    return render_template("index.html")


@app.route("/weather", methods=["GET"])
def weather_dashboard():
    """Full weather dashboard for a given city."""
    city = request.args.get("city", "London")
    lat  = request.args.get("lat", None)
    lon  = request.args.get("lon", None)

    # If lat/lon provided (geolocation), use them directly
    if lat and lon:
        loc_params = {"lat": lat, "lon": lon, "units": "metric"}
    else:
        loc_params = {"q": city, "units": "metric"}

    # ── Current weather ────────────────────────────────────────────────────────
    current = owm_get(f"{OWM_BASE}/weather", {**loc_params})
    if not current:
        return render_template("weather.html", error=f"City '{city}' not found.", city=city)

    # Extract coordinates for subsequent calls
    coord_lat = current["coord"]["lat"]
    coord_lon = current["coord"]["lon"]

    # ── 5-day / 3-hour forecast ────────────────────────────────────────────────
    forecast_data = owm_get(f"{OWM_BASE}/forecast", {"lat": coord_lat, "lon": coord_lon, "units": "metric"})

    # ── Air quality ────────────────────────────────────────────────────────────
    aqi_data = owm_get(f"{OWM_BASE}/air_pollution", {"lat": coord_lat, "lon": coord_lon})

    # ── Process current weather ────────────────────────────────────────────────
    icon_code   = current["weather"][0]["icon"]
    description = current["weather"][0]["description"].title()
    temp        = round(current["main"]["temp"])
    feels_like  = round(current["main"]["feels_like"])
    temp_min    = round(current["main"]["temp_min"])
    temp_max    = round(current["main"]["temp_max"])
    humidity    = current["main"]["humidity"]
    pressure    = current["main"]["pressure"]
    wind_speed  = round(current["wind"]["speed"] * 3.6, 1)   # m/s → km/h
    wind_deg    = current["wind"].get("deg", 0)
    visibility  = round(current.get("visibility", 10000) / 1000, 1)
    sunrise_ts  = current["sys"]["sunrise"]
    sunset_ts   = current["sys"]["sunset"]
    city_name   = current["name"]
    country     = current["sys"]["country"]
    timezone    = current["timezone"]          # seconds offset from UTC

    # Wind direction text
    dirs = ["N","NNE","NE","ENE","E","ESE","SE","SSE","S","SSW","SW","WSW","W","WNW","NW","NNW"]
    wind_dir = dirs[round(wind_deg / 22.5) % 16]

    # ── Process 5-day forecast → daily summary ─────────────────────────────────
    daily = {}
    if forecast_data:
        for item in forecast_data["list"]:
            date_str = item["dt_txt"].split(" ")[0]
            if date_str not in daily:
                daily[date_str] = {
                    "date":  date_str,
                    "highs": [],
                    "lows":  [],
                    "icons": [],
                    "desc":  item["weather"][0]["description"].title(),
                    "precip":   0,
                }
            daily[date_str]["highs"].append(item["main"]["temp_max"])
            daily[date_str]["lows"].append(item["main"]["temp_min"])
            daily[date_str]["icons"].append(item["weather"][0]["icon"])
            daily[date_str]["precip"] = max(daily[date_str]["precip"], item.get("pop", 0))

    daily_list = []
    for d in list(daily.values())[:7]:
        icon_c = max(set(d["icons"]), key=d["icons"].count)
        daily_list.append({
            "date":       d["date"],
            "high":       round(max(d["highs"])),
            "low":        round(min(d["lows"])),
            "icon_code":  icon_c,
            "icon_class": weather_icon_class(icon_c),
            "desc":       d["desc"],
            "precip":     round(d["precip"] * 100),
        })

    # ── Process hourly (next 24h from 3-hour intervals = 8 slots) ─────────────
    hourly_list = []
    if forecast_data:
        for item in forecast_data["list"][:8]:
            ic = item["weather"][0]["icon"]
            hourly_list.append({
                "time":       item["dt_txt"].split(" ")[1][:5],
                "temp":       round(item["main"]["temp"]),
                "icon_code":  ic,
                "icon_class": weather_icon_class(ic),
                "precip":     round(item.get("pop", 0) * 100),
            })

    # ── AQI ────────────────────────────────────────────────────────────────────
    aqi_index = 1
    aqi_info  = aqi_label(1)
    aqi_components = {}
    if aqi_data and aqi_data.get("list"):
        aqi_index      = aqi_data["list"][0]["main"]["aqi"]
        aqi_info       = aqi_label(aqi_index)
        aqi_components = aqi_data["list"][0]["components"]

    ctx = {
        "city":           city_name,
        "country":        country,
        "lat":            coord_lat,
        "lon":            coord_lon,
        "temp":           temp,
        "feels_like":     feels_like,
        "temp_min":       temp_min,
        "temp_max":       temp_max,
        "humidity":       humidity,
        "pressure":       pressure,
        "wind_speed":     wind_speed,
        "wind_deg":       wind_deg,
        "wind_dir":       wind_dir,
        "visibility":     visibility,
        "description":    description,
        "icon_code":      icon_code,
        "icon_class":     weather_icon_class(icon_code),
        "sunrise_ts":     sunrise_ts,
        "sunset_ts":      sunset_ts,
        "timezone":       timezone,
        "daily":          daily_list,
        "hourly":         hourly_list,
        "aqi_index":      aqi_index,
        "aqi_text":       aqi_info["text"],
        "aqi_color":      aqi_info["color"],
        "aqi_components": aqi_components,
        "api_key":        WEATHER_API_KEY,
    }
    return render_template("weather.html", **ctx)


# ── API endpoints (JSON) ───────────────────────────────────────────────────────
@app.route("/api/search", methods=["GET"])
def api_search():
    """City autocomplete search."""
    q = request.args.get("q", "")
    if len(q) < 2:
        return jsonify([])
    data = owm_get(f"{OWM_GEO}/direct", {"q": q, "limit": 5})
    if not data:
        return jsonify([])
    results = [
        {"name": item["name"], "country": item["country"],
         "state": item.get("state", ""),
         "lat": item["lat"], "lon": item["lon"]}
        for item in data
    ]
    return jsonify(results)


@app.route("/api/weather", methods=["GET"])
def api_weather():
    """Current weather JSON proxy."""
    lat = request.args.get("lat")
    lon = request.args.get("lon")
    city = request.args.get("city", "London")
    if lat and lon:
        params = {"lat": lat, "lon": lon, "units": "metric"}
    else:
        params = {"q": city, "units": "metric"}
    data = owm_get(f"{OWM_BASE}/weather", params)
    return jsonify(data or {"error": "not found"})


# ── ML Predict ────────────────────────────────────────────────────────────────
@app.route("/predict-form", methods=["GET"])
def predict_form():
    """Serve the ML prediction form (original index page)."""
    return render_template("predict_form.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        form = request.form

        data = CustomData(
            min_temp        = float(form["min_temp"]),
            max_temp        = float(form["max_temp"]),
            rainfall        = float(form["rainfall"]),
            evaporation     = float(form["evaporation"]),
            sunshine        = float(form["sunshine"]),
            wind_gust_speed = float(form["wind_gust_speed"]),
            wind_speed_9am  = float(form["wind_speed_9am"]),
            wind_speed_3pm  = float(form["wind_speed_3pm"]),
            humidity_9am    = float(form["humidity_9am"]),
            humidity_3pm    = float(form["humidity_3pm"]),
            pressure_9am    = float(form["pressure_9am"]),
            pressure_3pm    = float(form["pressure_3pm"]),
            temp_9am        = float(form["temp_9am"]),
            temp_3pm        = float(form["temp_3pm"]),
            rain_today      = int(form["rain_today"]),
            day             = int(form["day"]),
            month           = int(form["month"]),
            year            = int(form["year"]),
        )

        df = data.get_data_as_dataframe()
        logger.info(f"Input DataFrame:\n{df.to_string()}")

        pipeline = PredictPipeline()
        results  = pipeline.predict(df)
        prediction = results[0]

        return render_template("result.html", prediction=prediction)

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise WeatherException(e, sys)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
