import os
import sys
import dill
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
)

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
    report dict keyed by model name.

    Metrics computed per model
    --------------------------
    train_accuracy   : accuracy on training set (watch for overfitting)
    test_accuracy    : overall accuracy on test set
    precision        : precision for class 1 (rain)
    recall           : recall for class 1 (rain)  — same as rain_recall
    rain_recall      : alias of recall; highlighted separately so it's
                       impossible to miss when comparing models
    f1_score         : harmonic mean of precision & recall
    roc_auc          : area under ROC curve (requires predict_proba;
                       falls back to 0.5 for models that lack it)
    """
    try:
        report = {}

        for name, model in models.items():
            logger.info(f"Training model: {name}")
            model.fit(x_train, y_train)

            y_train_pred = model.predict(x_train)
            y_test_pred  = model.predict(x_test)

            train_acc  = accuracy_score(y_train, y_train_pred)
            test_acc   = accuracy_score(y_test,  y_test_pred)
            test_prec  = precision_score(y_test,  y_test_pred, zero_division=0)
            test_rec   = recall_score(y_test,    y_test_pred, zero_division=0)
            test_f1    = f1_score(y_test,        y_test_pred, zero_division=0)

            # ── ROC-AUC (requires predict_proba) ──────────────────────────────
            try:
                y_proba  = model.predict_proba(x_test)[:, 1]
                test_auc = roc_auc_score(y_test, y_proba)
            except AttributeError:
                # Model doesn't support predict_proba (e.g. LinearSVC)
                test_auc = 0.5
                logger.warning(f"{name}: no predict_proba — ROC-AUC set to 0.5")

            logger.info(
                f"{name} → "
                f"train_acc={train_acc:.4f} | test_acc={test_acc:.4f} | "
                f"prec={test_prec:.4f} | recall={test_rec:.4f} | "
                f"f1={test_f1:.4f} | roc_auc={test_auc:.4f}"
            )

            report[name] = {
                "model":          model,
                "train_accuracy": train_acc,
                "test_accuracy":  test_acc,
                "precision":      test_prec,
                "recall":         test_rec,
                "rain_recall":    test_rec,   # alias — rain-day catch rate
                "f1_score":       test_f1,
                "roc_auc":        test_auc,
            }

        return report

    except Exception as e:
        raise WeatherException(e, sys)


def get_best_model(report: dict) -> tuple[str, object, float]:
    """
    Return (best_model_name, best_model_object, best_f1_score).

    Ranking key: F1 score (not raw accuracy).
    F1 balances precision and recall, making it robust to mild class
    imbalance and ensuring the model actually catches rain days rather
    than coasting on majority-class predictions.
    """
    best_name = max(report, key=lambda k: report[k]["f1_score"])
    best_info = report[best_name]
    return best_name, best_info["model"], best_info["f1_score"]
