"""
Production Feedback Loop Evaluator.
Reads logs/prediction_feedback.csv, fetches actual historical rain outcome
for past predicted dates via Open-Meteo Archive API, updates the feedback log,
and computes real-world production accuracy, precision, and recall metrics.
"""
import os
import sys
import pandas as pd
import requests
from datetime import datetime, timedelta

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding="utf-8")

LOG_FILE = os.path.join("logs", "prediction_feedback.csv")

# City lat/lon mapping for actual outcome lookups
CITY_COORDS = {
    "Delhi": (28.6139, 77.2090),
    "Mumbai": (19.0760, 72.8777),
    "Bangalore": (12.9716, 77.5946),
    "Chennai": (13.0827, 80.2707),
    "Kolkata": (22.5726, 88.3639),
    "Hyderabad": (17.3850, 78.4867),
    "Ahmedabad": (23.0225, 72.5714),
    "Jaipur": (26.9124, 75.7873),
    "Lucknow": (26.8467, 80.9462),
    "Pune": (18.5204, 73.8567),
    "Cairo": (30.0444, 31.2357),
    "Dubai": (25.2048, 55.2708),
    "Riyadh": (24.7136, 46.6753),
}


def fetch_actual_rain(city: str, date_str: str) -> float:
    """
    Fetch actual daily precipitation (mm) from Open-Meteo Historical Archive API.
    """
    coords = CITY_COORDS.get(city)
    if not coords:
        return float("nan")

    lat, lon = coords
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}&start_date={date_str}&end_date={date_str}"
        f"&daily=precipitation_sum&timezone=auto"
    )
    try:
        r = requests.get(url, timeout=6)
        if r.status_code == 200:
            data = r.json()
            precip_list = data.get("daily", {}).get("precipitation_sum", [])
            if precip_list and precip_list[0] is not None:
                return float(precip_list[0])
    except Exception as exc:
        print(f"   [!] Error fetching actuals for {city} on {date_str}: {exc}")
    return float("nan")


def evaluate_production_feedback():
    SEP = "=" * 70
    print(f"\n{SEP}")
    print("  PRODUCTION PREDICTION FEEDBACK EVALUATOR")
    print(SEP)

    if not os.path.exists(LOG_FILE):
        print(f"  [!] Log file not found at: {LOG_FILE}")
        return

    df = pd.read_csv(LOG_FILE)
    if df.empty:
        print("  [!] Feedback log is empty. No predictions logged yet.")
        return

    print(f"  Total logged predictions: {len(df)}")
    
    # Ensure feedback columns exist
    if "actual_precip_mm" not in df.columns:
        df["actual_precip_mm"] = float("nan")
    if "actual_rain" not in df.columns:
        df["actual_rain"] = float("nan")
    if "is_correct" not in df.columns:
        df["is_correct"] = float("nan")

    today_str = datetime.utcnow().strftime("%Y-%m-%d")
    updated_count = 0

    for idx, row in df.iterrows():
        pdate = str(row["predict_date"]).split(" ")[0]
        # Only evaluate dates that are strictly in the past
        if pdate < today_str and pd.isna(row["actual_rain"]):
            city = str(row["city"])
            print(f"  Fetching actual outcome for {city} on {pdate}...")
            actual_mm = fetch_actual_rain(city, pdate)
            
            if not pd.isna(actual_mm):
                actual_rain = 1 if actual_mm > 0.0 else 0
                pred_label = 1 if str(row["prediction"]).strip().lower() == "yes" else 0
                is_correct = 1 if pred_label == actual_rain else 0
                
                df.at[idx, "actual_precip_mm"] = actual_mm
                df.at[idx, "actual_rain"] = actual_rain
                df.at[idx, "is_correct"] = is_correct
                updated_count += 1

    # Save updated CSV log
    df.to_csv(LOG_FILE, index=False)
    print(f"\n  Updated {updated_count} past prediction row(s) with actual outcomes.")

    # Compute production evaluation metrics
    eval_df = df.dropna(subset=["actual_rain"])
    print(f"\n{SEP}")
    print("  PRODUCTION ACCURACY REPORT")
    print(SEP)
    
    if eval_df.empty:
        print("  No past prediction records with verified actual outcomes yet.")
        print("  (Future predictions logged today will be evaluated once the date passes).")
    else:
        y_true = eval_df["actual_rain"].astype(int).values
        y_pred = (eval_df["prediction"].str.strip().str.lower() == "yes").astype(int).values
        
        acc = (y_true == y_pred).mean()
        rain_mask = (y_true == 1)
        rain_recall = (y_pred[rain_mask] == 1).mean() if rain_mask.sum() > 0 else float("nan")
        
        print(f"  Verified Past Predictions : {len(eval_df)}")
        print(f"  Production Accuracy       : {acc*100:.2f}%")
        print(f"  Production Rain Recall    : {rain_recall*100:.2f}%" if not pd.isna(rain_recall) else "  Production Rain Recall    : N/A (no actual rain days yet)")
        print()
        print("  Sample Feedback Log:")
        print(eval_df[["timestamp", "city", "predict_date", "prediction", "probability", "actual_precip_mm", "actual_rain", "is_correct"]].to_string(index=False))

    print(f"\n{SEP}\n")


if __name__ == "__main__":
    evaluate_production_feedback()
