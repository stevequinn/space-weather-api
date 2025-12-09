# logger_config.py
import logging
import sys
from logging.handlers import RotatingFileHandler
import os

def setup_logging():
    log_filename = os.getenv("LOG_FILE", "bom_wrapper.log")

    # Create a custom logger
    logger = logging.getLogger("bom_api_wrapper")
    logger.setLevel(logging.INFO)

    # Format: Time - Level - Message
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Handler 1: Write to file (rotate after 5MB)
    file_handler = RotatingFileHandler(log_filename, maxBytes=5*1024*1024, backupCount=3)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Handler 2: Write to console (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger

logger = setup_logging()
