# logger_config.py
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

BASE_DIR = Path(__file__).parent
LOG_DIR = BASE_DIR / "logs"


def setup_logging():
    # Create a custom logger
    logger = logging.getLogger("bom_api_wrapper")
    logger.setLevel(logging.INFO)

    # Format: Time - Level - Message
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Handler 1: Write to file (rotate after 5MB)
    # if on vercel then do not log to file
    if os.getenv("VERCEL") == "1":
        Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
        log_filename = LOG_DIR / os.getenv("log_file", "bom_wrapper.log")
        file_handler = RotatingFileHandler(
            log_filename, maxBytes=5 * 1024 * 1024, backupCount=3
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Handler 2: Write to console (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


logger = setup_logging()
