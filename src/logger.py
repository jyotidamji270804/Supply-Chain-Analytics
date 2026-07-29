"""
logger.py
-----------
Centralized logging configuration. Every module imports get_logger(__name__)
instead of using print(), giving consistent timestamps, severity levels, and
a single place to control verbosity across the whole project.
"""

import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for the given module name."""
    logger = logging.getLogger(name)

    if not logger.handlers:  # avoid duplicate handlers on re-import
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger







