import os
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from weatherprediction.exception import WeatherException
from weatherprediction.logger import logger


@dataclass
class DataIngestionConfig:
    """Paths for all raw / split data artifacts."""
    raw_data_path:   str = os.path.join("artifacts", "raw.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path:  str = os.path.join("artifacts", "test.csv")


class DataIngestion:
    """
    Reads the raw weatherAUS.csv, performs a stratified train/test split,
    and persists both splits to the artifacts directory.
    """

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
                "dataset", "weatherAUS.csv",
            )
            dataset_path = os.path.abspath(dataset_path)
            logger.info(f"Reading dataset from: {dataset_path}")

            df = pd.read_csv(dataset_path)
            logger.info(f"Dataset loaded — shape: {df.shape}")

            # ── Create artifacts directory ─────────────────────────────────
            os.makedirs(os.path.dirname(self.config.raw_data_path), exist_ok=True)

            # ── Save raw copy ──────────────────────────────────────────────
            df.to_csv(self.config.raw_data_path, index=False)
            logger.info(f"Raw data saved → {self.config.raw_data_path}")

            # ── Drop missing target values before split ────────────────────
            df = df.dropna(subset=["RainTomorrow"])
            logger.info(f"Dataset shape after dropping missing targets: {df.shape}")

            # ── Stratified train / test split ──────────────────────────────
            train_df, test_df = train_test_split(
                df,
                test_size=0.2,
                random_state=42,
                stratify=df["RainTomorrow"],   # preserve class balance
            )
            logger.info(
                f"Split complete — train: {train_df.shape}, test: {test_df.shape}"
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
