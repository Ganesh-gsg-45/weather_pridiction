import sys
import os
import math
from datetime import datetime, date, timedelta
import requests
from flask import Flask, request, render_template, jsonify, Response
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


def compute_moon_phase(dt: datetime = None) -> dict:
    """
    Compute moon phase using a standard lunar cycle formula.
    Returns phase_name, phase_icon, and illumination (0-1).
    No external API needed.
    """
    if dt is None:
        dt = datetime.utcnow()

    # Known new moon reference: Jan 6, 2000
    known_new_moon = datetime(2000, 1, 6, 18, 14)
    lunar_cycle = 29.53058867  # days

    delta = dt - known_new_moon
    days_since = delta.total_seconds() / 86400.0
    phase_position = (days_since % lunar_cycle) / lunar_cycle  # 0.0 – 1.0

    # Illumination fraction (simple cosine model)
    illumination = (1 - math.cos(2 * math.pi * phase_position)) / 2

    # Phase name buckets
    if phase_position < 0.0625:
        name, icon = "New Moon", "🌑"
    elif phase_position < 0.1875:
        name, icon = "Waxing Crescent", "🌒"
    elif phase_position < 0.3125:
        name, icon = "First Quarter", "🌓"
    elif phase_position < 0.4375:
        name, icon = "Waxing Gibbous", "🌔"
    elif phase_position < 0.5625:
        name, icon = "Full Moon", "🌕"
    elif phase_position < 0.6875:
        name, icon = "Waning Gibbous", "🌖"
    elif phase_position < 0.8125:
        name, icon = "Last Quarter", "🌗"
    elif phase_position < 0.9375:
        name, icon = "Waning Crescent", "🌘"
    else:
        name, icon = "New Moon", "🌑"

    return {
        "name": name,
        "icon": icon,
        "illumination": round(illumination * 100),
        "position": round(phase_position * 100),
    }


# ── Pages ──────────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    """GOOD WEATHER landing page."""
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

    # Last updated timestamp (local time of the city)
    from datetime import timezone as tz
    import datetime as dt_module
    utc_now = dt_module.datetime.utcnow()
    city_offset = dt_module.timedelta(seconds=timezone)
    city_now = utc_now + city_offset
    last_updated = city_now.strftime("%H:%M, %a %d %b")

    # Wind direction text
    dirs = ["N","NNE","NE","ENE","E","ESE","SE","SSE","S","SSW","SW","WSW","W","WNW","NW","NNW"]
    wind_dir = dirs[round(wind_deg / 22.5) % 16]

    # ── Process forecast into: daily summary + per-day hourly breakdown ────────
    today_str = utc_now.strftime("%Y-%m-%d")

    # Build a dict: date_str → list of 3-hour slots (hourly_items)
    day_slots_map = {}   # date → [slot, ...]
    if forecast_data:
        for item in forecast_data["list"]:
            date_str = item["dt_txt"].split(" ")[0]
            if date_str not in day_slots_map:
                day_slots_map[date_str] = []
            day_slots_map[date_str].append(item)

    # Build daily summary list (max 5 days)
    daily_list = []
    for date_str, slots in list(day_slots_map.items())[:5]:
        highs = [s["main"]["temp_max"] for s in slots]
        lows  = [s["main"]["temp_min"] for s in slots]
        icons = [s["weather"][0]["icon"] for s in slots]
        precips = [s.get("pop", 0) for s in slots]
        icon_c = max(set(icons), key=icons.count)

        # Per-day hourly breakdown for expandable rows
        day_hourly = []
        for s in slots:
            ic = s["weather"][0]["icon"]
            day_hourly.append({
                "time":       s["dt_txt"].split(" ")[1][:5],
                "temp":       round(s["main"]["temp"]),
                "icon_code":  ic,
                "icon_class": weather_icon_class(ic),
                "precip":     round(s.get("pop", 0) * 100),
                "desc":       s["weather"][0]["description"].title(),
            })

        daily_list.append({
            "date":       date_str,
            "is_today":   date_str == today_str,
            "high":       round(max(highs)),
            "low":        round(min(lows)),
            "icon_code":  icon_c,
            "icon_class": weather_icon_class(icon_c),
            "desc":       slots[0]["weather"][0]["description"].title(),
            "precip":     round(max(precips) * 100),
            "hourly":     day_hourly,
        })

    # Today hi/lo from forecast (more accurate than current weather min/max)
    today_forecast = day_slots_map.get(today_str)
    if today_forecast:
        today_high = round(max(s["main"]["temp_max"] for s in today_forecast))
        today_low  = round(min(s["main"]["temp_min"] for s in today_forecast))
    else:
        today_high = temp_max
        today_low  = temp_min

    # ── Process hourly (next 24h = 8 slots of 3-hour intervals) ──────────────
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
                "desc":       item["weather"][0]["description"].title(),
            })

    # ── AQI ────────────────────────────────────────────────────────────────────
    aqi_index = 1
    aqi_info  = aqi_label(1)
    aqi_components = {}
    if aqi_data and aqi_data.get("list"):
        aqi_index      = aqi_data["list"][0]["main"]["aqi"]
        aqi_info       = aqi_label(aqi_index)
        aqi_components = aqi_data["list"][0]["components"]

    # ── Moon phase (pure math, no API) ────────────────────────────────────────
    moon = compute_moon_phase(utc_now)

    ctx = {
        "city":           city_name,
        "country":        country,
        "lat":            coord_lat,
        "lon":            coord_lon,
        "temp":           temp,
        "feels_like":     feels_like,
        "temp_min":       temp_min,
        "temp_max":       temp_max,
        "today_high":     today_high,
        "today_low":      today_low,
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
        "last_updated":   last_updated,
        "daily":          daily_list,
        "hourly":         hourly_list,
        "aqi_index":      aqi_index,
        "aqi_text":       aqi_info["text"],
        "aqi_color":      aqi_info["color"],
        "aqi_components": aqi_components,
        "moon":           moon,
    }
    return render_template("weather.html", **ctx)




# ── API endpoints (JSON & Proxy) ─────────────────────────────────────────────
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


@app.route("/api/map/precipitation/<int:z>/<int:x>/<int:y>.png", methods=["GET"])
def api_map_precipitation(z, x, y):
    """Proxy OpenWeatherMap precipitation tiles server-side to protect API key."""
    url = f"https://tile.openweathermap.org/map/precipitation_new/{z}/{x}/{y}.png"
    try:
        r = requests.get(url, params={"appid": WEATHER_API_KEY}, timeout=5)
        if r.status_code == 200:
            return Response(r.content, mimetype="image/png", headers={"Cache-Control": "public, max-age=3600"})
        return ("Tile not found", r.status_code)
    except Exception as e:
        logger.warning(f"Map tile error: {e}")
        return ("Tile server error", 500)




# ── ML Feature Derivation ──────────────────────────────────────────────────────
def derive_features_from_owm(city: str, lat: float = None, lon: float = None, target_date_str: str = None):
    """
    Fetch OWM current & forecast weather data and derive all 18 model features.
    """
    if not target_date_str:
        target_date_str = datetime.now().strftime("%Y-%m-%d")

    target_dt = datetime.strptime(target_date_str, "%Y-%m-%d").date()

    if lat and lon:
        loc_params = {"lat": lat, "lon": lon, "units": "metric"}
    else:
        loc_params = {"q": city, "units": "metric"}

    current = owm_get(f"{OWM_BASE}/weather", {**loc_params})
    if not current:
        raise ValueError(f"Could not retrieve weather data for '{city}'.")

    coord_lat = current["coord"]["lat"]
    coord_lon = current["coord"]["lon"]
    resolved_city = current.get("name", city)

    forecast_data = owm_get(f"{OWM_BASE}/forecast", {"lat": coord_lat, "lon": coord_lon, "units": "metric"})

    # Filter forecast slots for target_date_str
    day_slots = []
    if forecast_data and "list" in forecast_data:
        for slot in forecast_data["list"]:
            slot_date = slot["dt_txt"].split(" ")[0]
            if slot_date == target_date_str:
                day_slots.append(slot)

    # Fallback to current weather if target date has no forecast slots
    if not day_slots:
        day_slots = [{
            "main": current["main"],
            "wind": current.get("wind", {}),
            "clouds": current.get("clouds", {}),
            "rain": current.get("rain", {}),
            "dt_txt": f"{target_date_str} 12:00:00"
        }]

    # Min/Max temps
    min_temp = min(s["main"]["temp_min"] for s in day_slots)
    max_temp = max(s["main"]["temp_max"] for s in day_slots)

    # 9am and 3pm slots
    def closest_slot(target_hour):
        return min(day_slots, key=lambda s: abs(int(s["dt_txt"].split(" ")[1].split(":")[0]) - target_hour))

    slot_9am = closest_slot(9)
    slot_3pm = closest_slot(15)

    temp_9am = slot_9am["main"]["temp"]
    temp_3pm = slot_3pm["main"]["temp"]

    humidity_9am = float(slot_9am["main"]["humidity"])
    humidity_3pm = float(slot_3pm["main"]["humidity"])

    pressure_9am = float(slot_9am["main"]["pressure"])
    pressure_3pm = float(slot_3pm["main"]["pressure"])

    # Wind (convert m/s to km/h)
    wind_speed_9am = round(slot_9am["wind"].get("speed", 0.0) * 3.6, 1)
    wind_speed_3pm = round(slot_3pm["wind"].get("speed", 0.0) * 3.6, 1)

    max_gust_ms = max(s["wind"].get("gust", s["wind"].get("speed", 0.0) * 1.3) for s in day_slots)
    wind_gust_speed = round(max_gust_ms * 3.6, 1)

    # Rainfall (mm)
    rainfall = sum(s.get("rain", {}).get("3h", 0.0) for s in day_slots)
    rain_today = 1 if rainfall > 0.0 else 0

    # Day length (hours)
    sunrise_ts = current.get("sys", {}).get("sunrise", 0)
    sunset_ts = current.get("sys", {}).get("sunset", 0)
    if sunset_ts > sunrise_ts:
        day_length_hours = (sunset_ts - sunrise_ts) / 3600.0
    else:
        day_length_hours = 12.0

    # Sunshine (hours) estimated from cloud cover
    avg_clouds = sum(s.get("clouds", {}).get("all", 50) for s in day_slots) / float(len(day_slots))
    sunshine = round(max(0.0, day_length_hours * (1.0 - (avg_clouds / 100.0))), 1)

    # Evaporation (mm) estimated with Hargreaves equation
    mean_temp = (max_temp + min_temp) / 2.0
    temp_range = max(0.1, max_temp - min_temp)
    evap_estimate = 0.0023 * (mean_temp + 17.8) * math.sqrt(temp_range) * (day_length_hours / 12.0)
    evaporation = round(max(0.0, evap_estimate), 1)

    return {
        "min_temp": round(min_temp, 1),
        "max_temp": round(max_temp, 1),
        "rainfall": round(rainfall, 1),
        "evaporation": evaporation,
        "sunshine": sunshine,
        "wind_gust_speed": wind_gust_speed,
        "wind_speed_9am": wind_speed_9am,
        "wind_speed_3pm": wind_speed_3pm,
        "humidity_9am": humidity_9am,
        "humidity_3pm": humidity_3pm,
        "pressure_9am": pressure_9am,
        "pressure_3pm": pressure_3pm,
        "temp_9am": round(temp_9am, 1),
        "temp_3pm": round(temp_3pm, 1),
        "rain_today": rain_today,
        "day": target_dt.day,
        "month": target_dt.month,
        "year": target_dt.year,
        "resolved_city": resolved_city,
        "country": current.get("sys", {}).get("country", ""),
        "date_str": target_date_str,
    }


# ── ML Predict ────────────────────────────────────────────────────────────────
@app.route("/predict-form", methods=["GET"])
def predict_form():
    """Serve the redesigned city & date picker ML prediction form."""
    return render_template("predict_form.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        form = request.form
        city = form.get("city", "London").strip()
        lat = float(form["lat"]) if form.get("lat") else None
        lon = float(form["lon"]) if form.get("lon") else None
        target_date = form.get("date", datetime.now().strftime("%Y-%m-%d"))

        feat = derive_features_from_owm(city, lat, lon, target_date)

        data = CustomData(
            min_temp        = feat["min_temp"],
            max_temp        = feat["max_temp"],
            rainfall        = feat["rainfall"],
            evaporation     = feat["evaporation"],
            sunshine        = feat["sunshine"],
            wind_gust_speed = feat["wind_gust_speed"],
            wind_speed_9am  = feat["wind_speed_9am"],
            wind_speed_3pm  = feat["wind_speed_3pm"],
            humidity_9am    = feat["humidity_9am"],
            humidity_3pm    = feat["humidity_3pm"],
            pressure_9am    = feat["pressure_9am"],
            pressure_3pm    = feat["pressure_3pm"],
            temp_9am        = feat["temp_9am"],
            temp_3pm        = feat["temp_3pm"],
            rain_today      = feat["rain_today"],
            day             = feat["day"],
            month           = feat["month"],
            year            = feat["year"],
        )

        df = data.get_data_as_dataframe()
        logger.info(f"Auto-derived Input DataFrame:\n{df.to_string()}")

        pipeline = PredictPipeline()
        adv = pipeline.predict_advanced(df)
        result = adv[0]

        return render_template(
            "result.html",
            prediction   = result["prediction"],
            probability  = result["probability"],
            confidence   = result["confidence"],
            risk_level   = result["risk_level"],
            risk_color   = result["risk_color"],
            risk_bg      = result["risk_bg"],
            risk_border  = result["risk_border"],
            features     = feat,
            city         = feat["resolved_city"],
            country      = feat["country"],
            date         = feat["date_str"],
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise WeatherException(e, sys)


@app.route("/predict-multi", methods=["POST"])
def predict_multi():
    """5-day AI rain prediction timeline."""
    try:
        form = request.form
        city = form.get("city", "London").strip()
        lat = float(form["lat"]) if form.get("lat") else None
        lon = float(form["lon"]) if form.get("lon") else None

        pipeline = PredictPipeline()
        days_results = []

        for offset in range(5):
            target_dt = datetime.now() + timedelta(days=offset)
            target_date = target_dt.strftime("%Y-%m-%d")
            try:
                feat = derive_features_from_owm(city, lat, lon, target_date)
                data = CustomData(
                    min_temp        = feat["min_temp"],
                    max_temp        = feat["max_temp"],
                    rainfall        = feat["rainfall"],
                    evaporation     = feat["evaporation"],
                    sunshine        = feat["sunshine"],
                    wind_gust_speed = feat["wind_gust_speed"],
                    wind_speed_9am  = feat["wind_speed_9am"],
                    wind_speed_3pm  = feat["wind_speed_3pm"],
                    humidity_9am    = feat["humidity_9am"],
                    humidity_3pm    = feat["humidity_3pm"],
                    pressure_9am    = feat["pressure_9am"],
                    pressure_3pm    = feat["pressure_3pm"],
                    temp_9am        = feat["temp_9am"],
                    temp_3pm        = feat["temp_3pm"],
                    rain_today      = feat["rain_today"],
                    day             = feat["day"],
                    month           = feat["month"],
                    year            = feat["year"],
                )
                df = data.get_data_as_dataframe()
                adv = pipeline.predict_advanced(df)
                result = adv[0]
                result["features"]    = feat
                result["date"]        = target_date
                result["day_label"]   = "Today" if offset == 0 else target_dt.strftime("%A")
                result["date_short"]  = target_dt.strftime("%d %b")
                days_results.append(result)
            except Exception as ex:
                logger.warning(f"Could not predict for {target_date}: {ex}")
                days_results.append({
                    "prediction": "N/A", "probability": 0, "confidence": 0,
                    "risk_level": "Unknown", "risk_color": "#6b7280",
                    "risk_bg": "rgba(107,114,128,0.12)", "risk_border": "rgba(107,114,128,0.35)",
                    "date": target_date, "day_label": target_dt.strftime("%A"),
                    "date_short": target_dt.strftime("%d %b"), "features": {},
                })

        return render_template(
            "predict_multi.html",
            city    = city,
            country = days_results[0].get("features", {}).get("country", "") if days_results else "",
            days    = days_results,
        )

    except Exception as e:
        logger.error(f"Multi-day prediction error: {e}")
        raise WeatherException(e, sys)


if __name__ == "__main__":
    port = 5000
    print("\n" + "="*50)
    print("  Weather Prediction App is starting...")
    print(f"  >> Local:   http://127.0.0.1:{port}")
    print(f"  >> Network: http://0.0.0.0:{port}")
    print("="*50 + "\n")
    app.run(host="0.0.0.0", port=port, debug=True)

