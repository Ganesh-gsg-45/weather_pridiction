import os
import sys
import pandas as pd
from dataclasses import dataclass

from weatherprediction.exception import WeatherException
from weatherprediction.logger import logger

# Cutoff date for the time-based split.
# Rows with date <= TRAIN_CUTOFF go to train; rows after go to test.
TRAIN_CUTOFF = "2021-12-31"


@dataclass
class DataIngestionConfig:
    """Paths for all raw / split data artifacts."""
    raw_data_path:   str = os.path.join("artifacts", "raw.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path:  str = os.path.join("artifacts", "test.csv")


class DataIngestion:
    """
    Reads india_2000_2024_daily_weather.csv, engineers the target column
    (rain_tomorrow), and performs a time-based train/test split:

        train : date <= 2021-12-31  (≈ 22 years, ~80 %)
        test  : date  > 2021-12-31  (≈  3 years, ~20 %)

    A time-based split prevents data leakage — the model is evaluated
    only on dates it has never seen during training, matching real-world
    deployment conditions.
    """

    TARGET_COL = "rain_tomorrow"

    def __init__(self):
        self.config = DataIngestionConfig()

    def initiate_data_ingestion(self) -> tuple[str, str]:
        """
        Returns
        -------
        (train_data_path, test_data_path)
        """
        logger.info("─── Data Ingestion started ───────────────────────────────")
        try:
            # ── Locate dataset relative to project root ────────────────────
            dataset_path = os.path.join(
                os.path.dirname(__file__),       # components/
                "..", "..", "..",                # → project root
                "dataset", "india_2000_2024_daily_weather.csv",
            )
            dataset_path = os.path.abspath(dataset_path)
            logger.info(f"Reading dataset from: {dataset_path}")

            df = pd.read_csv(dataset_path)
            logger.info(f"Dataset loaded — shape: {df.shape}")

            # ── Parse date column ──────────────────────────────────────────
            df["date"] = pd.to_datetime(df["date"])

            # ── Sort chronologically per city ─────────────────────────────
            # Required so the shift below gives the *next* day per city.
            df = df.sort_values(["city", "date"]).reset_index(drop=True)

            # ── Engineer target: rain_tomorrow ─────────────────────────────
            # Within each city group, shift precipitation_sum by -1 day so
            # each row knows whether it rained the following day.
            df[self.TARGET_COL] = (
                df.groupby("city")["precipitation_sum"]
                  .shift(-1)
                  .apply(lambda x: 1 if x > 0 else 0)
            )
            # The last row of every city group has no next day → drop it.
            df = df.dropna(subset=[self.TARGET_COL])
            df[self.TARGET_COL] = df[self.TARGET_COL].astype(int)
            logger.info(
                f"Target column '{self.TARGET_COL}' engineered. "
                f"Shape after dropping last-per-city rows: {df.shape}"
            )

            # ── Create artifacts directory ─────────────────────────────────
            os.makedirs(os.path.dirname(self.config.raw_data_path), exist_ok=True)

            # ── Save raw copy (with engineered target) ─────────────────────
            df.to_csv(self.config.raw_data_path, index=False)
            logger.info(f"Raw data saved → {self.config.raw_data_path}")

            # ── Time-based split (no stratify — preserves temporal order) ──
            # Train: 2000-01-01 → 2021-12-31  (~22 yrs, ≈ 80 %)
            # Test : 2022-01-01 → 2024-xx-xx  (~3 yrs,  ≈ 20 %)
            train_df = df[df["date"] <= TRAIN_CUTOFF].copy()
            test_df  = df[df["date"] >  TRAIN_CUTOFF].copy()

            logger.info(
                f"Time-based split (cutoff={TRAIN_CUTOFF}) — "
                f"train: {train_df.shape}, test: {test_df.shape}"
            )
            logger.info(
                f"Train date range: {train_df['date'].min().date()} → "
                f"{train_df['date'].max().date()}"
            )
            logger.info(
                f"Test  date range: {test_df['date'].min().date()} → "
                f"{test_df['date'].max().date()}"
            )

            train_df.to_csv(self.config.train_data_path, index=False)
            test_df.to_csv(self.config.test_data_path,   index=False)
            logger.info(
                f"Train data saved → {self.config.train_data_path}\n"
                f"Test  data saved → {self.config.test_data_path}"
            )

            logger.info("─── Data Ingestion completed ─────────────────────────────")
            return self.config.train_data_path, self.config.test_data_path

        except Exception as e:
            raise WeatherException(e, sys)
