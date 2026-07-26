import os
import sys
import dill
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from weatherprediction.exception import WeatherException
from weatherprediction.logger import logger


def save_object(file_path: str, obj) -> None:
    """Serialise any Python object to disk using dill."""
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as f:
            dill.dump(obj, f)
        logger.info(f"Object saved to: {file_path}")
    except Exception as e:
        raise WeatherException(e, sys)


def load_object(file_path: str):
    """Deserialise a Python object from disk."""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Artifact not found at: {file_path}")
        with open(file_path, "rb") as f:
            obj = dill.load(f)
        logger.info(f"Object loaded from: {file_path}")
        return obj
    except Exception as e:
        raise WeatherException(e, sys)


def evaluate_models(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    models: dict,
) -> dict:
    """
    Train each model, evaluate on train + test sets, and return a
    report dict: {model_name: {"model": obj, "test_accuracy": float, ...}}
    """
    try:
        report = {}

        for name, model in models.items():
            logger.info(f"Training model: {name}")
            model.fit(x_train, y_train)

            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)

            train_acc = accuracy_score(y_train, y_train_pred)
            test_acc  = accuracy_score(y_test,  y_test_pred)
            test_prec = precision_score(y_test, y_test_pred, zero_division=0)
            test_rec  = recall_score(y_test,    y_test_pred, zero_division=0)
            test_f1   = f1_score(y_test,        y_test_pred, zero_division=0)

            logger.info(
                f"{name} → train_acc={train_acc:.4f} | "
                f"test_acc={test_acc:.4f} | precision={test_prec:.4f} | "
                f"recall={test_rec:.4f} | f1={test_f1:.4f}"
            )

            report[name] = {
                "model":          model,
                "train_accuracy": train_acc,
                "test_accuracy":  test_acc,
                "precision":      test_prec,
                "recall":         test_rec,
                "f1_score":       test_f1,
            }

        return report

    except Exception as e:
        raise WeatherException(e, sys)


def get_best_model(report: dict) -> tuple[str, object, float]:
    """
    Return (best_model_name, best_model_object, best_test_accuracy)
    from the evaluation report.
    """
    best_name = max(report, key=lambda k: report[k]["test_accuracy"])
    best_info = report[best_name]
    return best_name, best_info["model"], best_info["test_accuracy"]
