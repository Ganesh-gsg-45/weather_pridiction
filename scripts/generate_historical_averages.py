"""
scripts/generate_historical_averages.py
=======================================
PRECOMPUTE STEP — run once (or when dataset changes), not on every app startup.

Reads  : dataset/india_2000_2024_daily_weather.csv
Writes : artifacts/historical_daily_averages.json

Output format:
{
  "Delhi": {
    "1":   {"tmax": 19.8, "tmin": 7.1, "precip": 0.3, "wind": 10.2, "samples": 25},
    "2":   {...},
    ...
    "365": {...}
  },
  "Mumbai": { ... },
  ...
}

Notes:
- day_of_year is 1–365 for non-leap years, 1–366 for leap years.
- Feb 29 (day 60 in a 366-day year) is remapped to day 60 alongside Feb 28
  so non-leap-year lookups for day 60 still work. It is averaged in with fewer
  samples; the sample count is stored so callers can optionally weight it.
- Values are rounded to 2 dp to keep file size compact.
"""

import os
import sys
import json
import time
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
CSV_PATH     = os.path.join(PROJECT_ROOT, "dataset", "india_2000_2024_daily_weather.csv")
OUT_PATH     = os.path.join(PROJECT_ROOT, "artifacts", "historical_daily_averages.json")

COLS_NEEDED  = ["city", "date", "temperature_2m_max", "temperature_2m_min",
                "precipitation_sum", "wind_speed_10m_max"]

METRICS = {
    "tmax":   "temperature_2m_max",
    "tmin":   "temperature_2m_min",
    "precip": "precipitation_sum",
    "wind":   "wind_speed_10m_max",
}


def main():
    t0 = time.time()
    print(f"[1/4] Loading {CSV_PATH} ...")

    if not os.path.exists(CSV_PATH):
        print(f"ERROR: Dataset not found at {CSV_PATH}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(CSV_PATH, usecols=COLS_NEEDED, parse_dates=["date"])
    print(f"      {len(df):,} rows loaded | cities: {sorted(df['city'].unique().tolist())}")

    # ── Day-of-year: always use 1–365 non-leap numbering ─────────────────────
    # For leap years, day 366 (Dec 31) doesn't exist in a normal year.
    # Strategy: remap Feb 29 (leap-only day 60) to day 60 — it averages in
    # with Feb 28 data from non-leap years, weighted by fewer samples.
    # All days > 59 in leap years are shifted back by 1 to stay in 1-365 range.
    print("[2/4] Computing day-of-year (normalised to 1-365) ...")

    is_leap = df["date"].dt.is_leap_year
    raw_doy = df["date"].dt.dayofyear        # 1-366 in leap years

    # Feb 29 in a leap year: raw_doy == 60 AND is_leap AND month == 2 AND day == 29
    is_feb29 = is_leap & (df["date"].dt.month == 2) & (df["date"].dt.day == 29)

    # For leap years, days after Feb 28 (doy > 59) shift back by 1 unless Feb 29
    adj_doy = raw_doy.copy()
    adj_doy[is_leap & (raw_doy > 59) & ~is_feb29] -= 1
    # Feb 29 stays at 60 (same slot as Feb 28 non-leap)

    df["doy"] = adj_doy.astype(int)

    # Sanity check: all doy should be 1-365
    assert df["doy"].between(1, 365).all(), "doy out of 1-365 range!"
    print(f"      doy range: {df['doy'].min()}-{df['doy'].max()} OK")
    print(f"      Feb-29 rows merged into doy 60: {is_feb29.sum()}")

    # ── Group by city + doy ────────────────────────────────────────────────────
    print("[3/4] Grouping by city + day_of_year and computing means ...")
    grp = df.groupby(["city", "doy"])

    result = {}
    for (city, doy), g in grp:
        if city not in result:
            result[city] = {}
        result[city][str(doy)] = {
            "tmax":    round(float(g["temperature_2m_max"].mean()), 2),
            "tmin":    round(float(g["temperature_2m_min"].mean()), 2),
            "precip":  round(float(g["precipitation_sum"].mean()), 2),
            "wind":    round(float(g["wind_speed_10m_max"].mean()), 2),
            "samples": int(len(g)),
        }

    # ── Spot-check ────────────────────────────────────────────────────────────
    print("\n  -- Spot-check (city=Delhi, a few doys) --")
    for doy in [1, 60, 180, 250, 365]:
        v = result.get("Delhi", {}).get(str(doy), {})
        print(f"    Delhi doy={doy:3d}: tmax={v.get('tmax','?')}  "
              f"tmin={v.get('tmin','?')}  "
              f"precip={v.get('precip','?')}mm  "
              f"samples={v.get('samples','?')}")

    print("\n  -- Spot-check (city=Mumbai, monsoon doy ~180) --")
    for doy in [170, 180, 200]:
        v = result.get("Mumbai", {}).get(str(doy), {})
        print(f"    Mumbai doy={doy}: tmax={v.get('tmax','?')}  "
              f"precip={v.get('precip','?')}mm  samples={v.get('samples','?')}")

    # ── Write output ───────────────────────────────────────────────────────────
    print(f"\n[4/4] Writing {OUT_PATH} ...")
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, separators=(",", ":"))  # compact, no pretty indent

    size_kb = os.path.getsize(OUT_PATH) / 1024
    total_entries = sum(len(v) for v in result.values())
    elapsed = time.time() - t0

    print(f"\nDone in {elapsed:.1f}s")
    print(f"   File : {OUT_PATH}")
    print(f"   Size : {size_kb:.1f} KB")
    print(f"   Cities: {len(result)}")
    print(f"   Total entries (city x doy): {total_entries:,}")


if __name__ == "__main__":
    main()
