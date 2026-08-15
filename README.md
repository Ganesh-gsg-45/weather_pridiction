# 🌦️ GOOD WEATHER

**Rain-tomorrow classifier for 10 major Indian cities** — a production-grade MLOps pipeline built on 25 years of real meteorological data (2000–2024), with a live inference API, threshold-tuned CatBoost model, and a continuous feedback loop.

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML Model | CatBoost (gradient boosting) |
| Preprocessing | scikit-learn ColumnTransformer |
| Web Framework | Flask 2.3 |
| Dataset | Open-Meteo historical data — 10 Indian cities, 2000–2024 |
| Live Weather | OpenWeatherMap API |
| Serialization | dill (model + preprocessor .pkl) |

---

## Project Structure

```
weather_prediction/
├── app.py                        # Flask app — all routes
├── config/constants.py           # Documented project constants (reference)
├── dataset/                      # Raw training data (CSV)
├── docs/                         # Additional documentation
├── logs/                         # App logs + prediction_feedback.csv
├── notebook/
│   └── eda_fe_model_training.ipynb
├── scripts/
│   ├── evaluate_feedback.py      # Evaluate live prediction outcomes
│   └── generate_historical_averages.py
├── src/weatherprediction/
│   ├── components/               # data_ingestion, data_transformation, model_trainer
│   ├── pipeline/                 # train_pipeline, predict_pipeline
│   ├── exception.py
│   ├── logger.py
│   └── utils.py
├── static/
├── templates/
├── tests/                        # Unit tests
├── .env.example                  # Env var reference (copy to .env)
├── requirements.txt
└── setup.py
```

---

## Setup

### 1. Clone the repo

```bash
git clone <repo-url>
cd "weather pridiction"
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
# Edit .env and set WEATHER_API_KEY to your OpenWeatherMap key
```

### 5. Train the model (first run)

The app requires trained artifacts (`artifacts/model.pkl`, `artifacts/preprocessor.pkl`).
Run the full training pipeline:

```bash
python -m src.weatherprediction.pipeline.train_pipeline
```

This runs data ingestion → transformation → model training and saves artifacts to `artifacts/`.

Optionally pre-compute historical averages for the trends dashboard:

```bash
python scripts/generate_historical_averages.py
```

### 6. Run the app

```bash
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

---

## How Predictions Work

### Real Data, Real Geography

The model was trained on **25 years of daily meteorological observations** (Open-Meteo, 2000–2024) across 10 major Indian cities: Delhi, Mumbai, Chennai, Kolkata, Bangalore, Hyderabad, Pune, Ahmedabad, Jaipur, and Surat. The dataset covers approximately 91,000 city-days.

### Time-Based Train/Test Split

To prevent data leakage, the split is **strictly temporal** — no random shuffling:

- **Train:** 2000-01-01 → 2021-12-31 (~22 years, ~80% of data)
- **Test:** 2022-01-01 → 2024-xx-xx (~3 years, ~20% of data)

This mirrors real deployment conditions: the model is evaluated only on future dates it never saw during training.

### Threshold Tuning

Standard classifiers default to a 0.5 decision threshold, which optimises accuracy. Rain prediction is **asymmetric** — a missed rain event ("false negative") is far worse than a false alarm. We performed an F1 sweep across thresholds and landed on **0.375**, which yields:

- **88.3% rain recall** (catches most actual rain events)
- **86.6% F1 score** on the hold-out test set

### Production Feedback Loop

Every live inference is logged to `logs/prediction_feedback.csv` (timestamp, city, coordinates, features, prediction, probability). Run `scripts/evaluate_feedback.py` to fetch actual weather outcomes via the OpenWeatherMap history API and compare them against logged predictions — closing the loop on real-world accuracy.

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Re-evaluating Feedback

```bash
python scripts/evaluate_feedback.py
```

Fetches actual outcomes for past predictions and prints accuracy metrics.
