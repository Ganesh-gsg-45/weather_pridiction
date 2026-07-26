import os
import sys
import numpy as np
from dataclasses import dataclass

from catboost import CatBoostClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from weatherprediction.exception import WeatherException
from weatherprediction.logger import logger
from weatherprediction.utils import evaluate_models, get_best_model, save_object


@dataclass
class ModelTrainerConfig:
    """Path where the best trained model will be saved."""
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    """
    Trains all candidate classifiers from the notebook, compares their
    test-set accuracy, and saves the best model to disk.
    """

    # ── Minimum acceptable test accuracy ──────────────────────────────────────
    MIN_ACCURACY_THRESHOLD = 0.60

    # ── Candidate models (same set as the notebook) ───────────────────────────
    MODELS = {
        "Logistic Regression":      LogisticRegression(max_iter=500),
        "K-Neighbors Classifier":   KNeighborsClassifier(),
        "Decision Tree Classifier": DecisionTreeClassifier(),
        "Random Forest Classifier": RandomForestClassifier(),
        "XGBClassifier":            XGBClassifier(eval_metric="logloss", verbosity=0),
        "CatBoost Classifier":      CatBoostClassifier(verbose=False),
        "AdaBoost Classifier":      AdaBoostClassifier(),
    }

    def __init__(self):
        self.config = ModelTrainerConfig()

    def initiate_model_trainer(
        self,
        train_arr: np.ndarray,
        test_arr:  np.ndarray,
    ) -> float:
        """
        Parameters
        ----------
        train_arr : np.ndarray  shape (n_train, n_features + 1)
        test_arr  : np.ndarray  shape (n_test,  n_features + 1)
            Last column is the target label.

        Returns
        -------
        best_test_accuracy : float
        """
        logger.info("─── Model Training started ───────────────────────────────")
        try:
            # ── Split features / target ────────────────────────────────────
            x_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            x_test,  y_test  = test_arr[:,  :-1], test_arr[:,  -1]

            logger.info(
                f"x_train: {x_train.shape}, y_train: {y_train.shape} | "
                f"x_test: {x_test.shape},  y_test: {y_test.shape}"
            )

            # ── Train & evaluate all models ────────────────────────────────
            report = evaluate_models(x_train, y_train, x_test, y_test, self.MODELS)

            # ── Log full comparison table ──────────────────────────────────
            logger.info("\n{:<30} {:>14} {:>12} {:>9} {:>9}".format(
                "Model", "Train Acc", "Test Acc", "Precision", "F1"
            ))
            logger.info("-" * 80)
            for name, metrics in sorted(
                report.items(), key=lambda kv: kv[1]["test_accuracy"], reverse=True
            ):
                logger.info(
                    "{:<30} {:>14.4f} {:>12.4f} {:>9.4f} {:>9.4f}".format(
                        name,
                        metrics["train_accuracy"],
                        metrics["test_accuracy"],
                        metrics["precision"],
                        metrics["f1_score"],
                    )
                )

            # ── Pick best model ────────────────────────────────────────────
            best_name, best_model, best_acc = get_best_model(report)
            logger.info(f"\nBest model: [{best_name}]  test_accuracy={best_acc:.4f}")

            # ── Sanity check ───────────────────────────────────────────────
            if best_acc < self.MIN_ACCURACY_THRESHOLD:
                raise WeatherException(
                    f"No model exceeded the minimum accuracy threshold of "
                    f"{self.MIN_ACCURACY_THRESHOLD:.0%}. Best was {best_acc:.4f}.",
                    sys,
                )

            # ── Save best model ────────────────────────────────────────────
            save_object(self.config.trained_model_file_path, best_model)
            logger.info(f"Best model saved → {self.config.trained_model_file_path}")

            logger.info("─── Model Training completed ─────────────────────────────")
            return best_acc

        except Exception as e:
            raise WeatherException(e, sys)
