import sys

from weatherprediction.components.data_ingestion import DataIngestion
from weatherprediction.components.data_transformation import DataTransformation
from weatherprediction.components.model_trainer import ModelTrainer
from weatherprediction.exception import WeatherException
from weatherprediction.logger import logger


def run_training_pipeline() -> float:
    """
    Orchestrates the full training pipeline:
        DataIngestion → DataTransformation → ModelTrainer

    Returns
    -------
    best_test_accuracy : float
    """
    try:
        logger.info("══════════════════════════════════════════════════════════")
        logger.info("          WEATHER PREDICTION — TRAINING PIPELINE          ")
        logger.info("══════════════════════════════════════════════════════════")

        # ── Step 1: Data Ingestion ─────────────────────────────────────────
        data_ingestion = DataIngestion()
        train_path, test_path = data_ingestion.initiate_data_ingestion()

        # ── Step 2: Data Transformation ────────────────────────────────────
        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
            train_path, test_path
        )

        # ── Step 3: Model Training ─────────────────────────────────────────
        model_trainer = ModelTrainer()
        best_accuracy = model_trainer.initiate_model_trainer(train_arr, test_arr)

        logger.info("══════════════════════════════════════════════════════════")
        logger.info(f"  Training pipeline complete — best accuracy: {best_accuracy:.4f}")
        logger.info("══════════════════════════════════════════════════════════")

        return best_accuracy

    except Exception as e:
        raise WeatherException(e, sys)


if __name__ == "__main__":
    accuracy = run_training_pipeline()
    print(f"\nTraining complete! Best model accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
