import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from weatherprediction.exception import WeatherException
from weatherprediction.logger import logger
from weatherprediction.utils import save_object


@dataclass
class DataTransformationConfig:
    """Path where the fitted preprocessor will be saved."""
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    """
    Handles all feature engineering and preprocessing steps that were
    performed inside the notebook:
      - Drop high-cardinality / low-signal categorical columns
      - Parse Date → Day, Month, Year
      - Label-encode the target (RainTomorrow)
      - Median imputation + StandardScaler for numeric features
      - Mode imputation + OneHotEncoder for categorical features
    """

    # ── Columns to drop before modelling (match notebook) ─────────────────────
    COLS_TO_DROP = [
        "Location",
        "WindGustDir",
        "WindDir9am",
        "WindDir3pm",
        "Cloud9am",
        "Cloud3pm",
        "Date",          # replaced by Day / Month / Year
    ]

    TARGET_COL = "RainTomorrow"

    def __init__(self):
        self.config = DataTransformationConfig()

    # ──────────────────────────────────────────────────────────────────────────
    def _feature_engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature-engineering steps in place."""
        df = df.copy()

        # Parse Date → numeric components
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df["Day"]   = df["Date"].dt.day
            df["Month"] = df["Date"].dt.month
            df["Year"]  = df["Date"].dt.year

        # Label-encode RainToday (Yes/No → 1/0)
        if "RainToday" in df.columns:
            le = LabelEncoder()
            df["RainToday"] = le.fit_transform(df["RainToday"].astype(str))

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
            train_df = self._feature_engineer(train_df)
            test_df  = self._feature_engineer(test_df)

            # ── Separate target & encode it ────────────────────────────────
            le = LabelEncoder()
            y_train = le.fit_transform(train_df[self.TARGET_COL].astype(str))
            y_test  = le.transform(test_df[self.TARGET_COL].astype(str))

            # ── Drop unwanted columns ──────────────────────────────────────
            cols_to_drop_existing = [
                c for c in self.COLS_TO_DROP if c in train_df.columns
            ]
            x_train = train_df.drop(columns=cols_to_drop_existing + [self.TARGET_COL])
            x_test  = test_df.drop(columns=[c for c in cols_to_drop_existing + [self.TARGET_COL] if c in test_df.columns])

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
