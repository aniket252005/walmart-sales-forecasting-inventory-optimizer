"""
logger.py
---------
Centralised logging setup. Import get_logger() in any module.

Usage
-----
    from utils.logger import get_logger
    log = get_logger(__name__)
    log.info("Starting pipeline…")
"""

import logging
import sys
from pathlib import Path


def get_logger(name: str, config: dict = None, level: str = "INFO") -> logging.Logger:
    """
    Return a logger that writes to both stdout and an optional log file.

    Parameters
    ----------
    name   : Usually __name__ of the calling module.
    config : Optional config dict. If provided, reads log level and file path.
    level  : Default level if config is not provided.
    """
    if config:
        level    = config.get("logging", {}).get("level", level)
        log_file = config.get("logging", {}).get("log_file")
    else:
        log_file = None

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger   # already configured

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    fmt     = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                                 datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler (optional)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
