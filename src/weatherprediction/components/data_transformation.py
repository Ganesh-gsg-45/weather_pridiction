import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from weatherprediction.exception import WeatherException
from weatherprediction.logger import logger
from weatherprediction.utils import save_object


@dataclass
class DataTransformationConfig:
    """Path where the fitted preprocessor will be saved."""
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    """
    Handles all feature engineering and preprocessing steps for the
    india_2000_2024_daily_weather dataset:

      Feature engineering (done in _feature_engineer):
        - Parse 'date'  → Day, Month, Year  (date column then dropped)
        - rain_today    = 1 if precipitation_sum > 0 else 0  (new binary feature)

      Preprocessing (done via ColumnTransformer):
        - Numeric  : median imputation + StandardScaler
        - Categorical ('city'): mode imputation + OneHotEncoder(drop='first')

      Target:
        - 'rain_tomorrow' is already int 0/1 from DataIngestion — no encoding needed.
    """

    # ── Columns to drop before modelling ──────────────────────────────────────
    # 'date' is parsed into Day/Month/Year so the raw column is removed.
    # 'rain_tomorrow' is the target, handled separately.
    COLS_TO_DROP = [
        "date",       # replaced by Day / Month / Year
        "rain_sum",   # exact 1.0 correlation duplicate of precipitation_sum
    ]

    TARGET_COL = "rain_tomorrow"

    def __init__(self):
        self.config = DataTransformationConfig()

    # ──────────────────────────────────────────────────────────────────────────
    def _feature_engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature-engineering steps.

        Steps
        -----
        1. Parse 'date' → Day, Month, Year (numeric temporal features).
        2. Derive 'rain_today' from precipitation_sum (1 if > 0 else 0).
        """
        df = df.copy()

        # ── 1. Parse date → temporal components ────────────────────────────
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df["Day"]   = df["date"].dt.day
            df["Month"] = df["date"].dt.month
            df["Year"]  = df["date"].dt.year
            # 'date' will be dropped via COLS_TO_DROP in the caller

        # ── 2. rain_today: did it rain on THIS day? ─────────────────────────
        # Uses the raw precipitation_sum before any scaling so the threshold
        # is meaningful (> 0 mm = rain today).
        if "precipitation_sum" in df.columns:
            df["rain_today"] = (df["precipitation_sum"] > 0).astype(int)
            logger.info("Derived 'rain_today' from precipitation_sum > 0")

        return df

    # ──────────────────────────────────────────────────────────────────────────
    def get_data_transformer_object(
        self,
        numeric_features: list[str],
        categorical_features: list[str],
    ) -> ColumnTransformer:
        """
        Build and return the sklearn ColumnTransformer (not yet fitted).
        """
        try:
            numeric_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler",  StandardScaler()),
            ])

            categorical_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ])

            preprocessor = ColumnTransformer(transformers=[
                ("num", numeric_pipeline,         numeric_features),
                ("cat", categorical_pipeline,     categorical_features),
            ])

            logger.info(f"Numeric features  : {numeric_features}")
            logger.info(f"Categorical features: {categorical_features}")
            return preprocessor

        except Exception as e:
            raise WeatherException(e, sys)

    # ──────────────────────────────────────────────────────────────────────────
    def initiate_data_transformation(
        self,
        train_path: str,
        test_path: str,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        """
        Reads train/test CSVs, applies feature engineering, fits the
        preprocessor on train, transforms both splits, saves the fitted
        preprocessor, and returns:
            (train_arr, test_arr, preprocessor_path)
        """
        logger.info("─── Data Transformation started ──────────────────────────")
        try:
            train_df = pd.read_csv(train_path)
            test_df  = pd.read_csv(test_path)
            logger.info(f"Loaded train {train_df.shape}, test {test_df.shape}")

            # ── Feature engineering ────────────────────────────────────────
            # Adds Day/Month/Year and rain_today before anything is dropped.
            train_df = self._feature_engineer(train_df)
            test_df  = self._feature_engineer(test_df)

            # ── Separate target ────────────────────────────────────────────
            # rain_tomorrow is already int 0/1 from DataIngestion — no
            # LabelEncoder needed.
            y_train = train_df[self.TARGET_COL].values
            y_test  = test_df[self.TARGET_COL].values
            logger.info(
                f"Target distribution — train: "
                f"rain={y_train.sum():,}/{len(y_train):,} "
                f"({y_train.mean()*100:.1f}%)  "
                f"test: rain={y_test.sum():,}/{len(y_test):,} "
                f"({y_test.mean()*100:.1f}%)"
            )

            # ── Drop unwanted columns ──────────────────────────────────────
            cols_to_drop_existing = [
                c for c in self.COLS_TO_DROP if c in train_df.columns
            ]
            x_train = train_df.drop(columns=cols_to_drop_existing + [self.TARGET_COL])
            x_test  = test_df.drop(
                columns=[c for c in cols_to_drop_existing + [self.TARGET_COL]
                         if c in test_df.columns]
            )

            logger.info(f"Feature matrix shape — train: {x_train.shape}, test: {x_test.shape}")

            # ── Identify feature types ─────────────────────────────────────
            numeric_features     = x_train.select_dtypes(exclude="object").columns.tolist()
            categorical_features = x_train.select_dtypes(include="object").columns.tolist()

            # ── Build & fit preprocessor ───────────────────────────────────
            preprocessor = self.get_data_transformer_object(
                numeric_features, categorical_features
            )
            x_train_transformed = preprocessor.fit_transform(x_train)
            x_test_transformed  = preprocessor.transform(x_test)
            logger.info("Preprocessing complete.")

            # ── Concatenate features + target ──────────────────────────────
            train_arr = np.c_[x_train_transformed, y_train]
            test_arr  = np.c_[x_test_transformed,  y_test]

            # ── Save preprocessor ──────────────────────────────────────────
            save_object(self.config.preprocessor_obj_file_path, preprocessor)

            logger.info("─── Data Transformation completed ────────────────────────")
            return train_arr, test_arr, self.config.preprocessor_obj_file_path

        except Exception as e:
            raise WeatherException(e, sys)
