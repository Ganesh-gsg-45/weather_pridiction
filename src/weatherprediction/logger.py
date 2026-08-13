import logging
import os
from logging.handlers import RotatingFileHandler

# ── Log directory ──────────────────────────────────────────────────────────────
logs_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_dir, exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_dir, "weather_app.log")

# ── Logger configuration ───────────────────────────────────────────────────────
logger = logging.getLogger("weatherprediction")
logger.setLevel(logging.INFO)

if not logger.handlers:
    # Rotate log files when they reach 5 MB, keeping up to 3 backup files
    handler = RotatingFileHandler(LOG_FILE_PATH, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    formatter = logging.Formatter("[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
