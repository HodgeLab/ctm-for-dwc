"""
Utility for repeatable, standard logging
"""

import os
import logging
from datetime import datetime


def make_logger(
    include_stdout: bool = False, log_prefix: str = "logs"
) -> logging.Logger:
    # Create a logger
    logger = logging.getLogger()

    # Capture all message levels
    logger.setLevel(logging.DEBUG)

    # Set the format
    log_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create a file for the logs
    log_dir = os.path.join(os.getcwd(), "logs")
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)
    now = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
    log_filename = os.path.join(log_dir, f"{log_prefix}_{now}.log")

    # Add a handler for files
    file_handler = logging.FileHandler(log_filename, encoding="utf-8")
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)

    # Optionally add a second handler for logging to stdout
    if include_stdout:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(log_formatter)
        logger.addHandler(stream_handler)

    return logger
