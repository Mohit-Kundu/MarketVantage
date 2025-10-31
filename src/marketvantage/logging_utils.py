import logging
import os
import pathlib
from typing import Optional


def setup_logging(verbosity: int = 0, log_file: Optional[str] = "logs/marketvantage.log") -> None:
    """Configure logging to console and to a file, truncating the file each run.

    Parameters
    ----------
    verbosity: int
        0 = WARNING, 1 = INFO, 2+ = DEBUG
    log_file: Optional[str]
        Path to the log file. If None, file logging is disabled.
    """
    level = logging.WARNING if verbosity <= 0 else (logging.INFO if verbosity == 1 else logging.DEBUG)

    root = logging.getLogger()
    root.setLevel(level)

    # Clear existing handlers to avoid duplicates when re-running
    for h in list(root.handlers):
        root.removeHandler(h)

    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    root.addHandler(console)

    if log_file:
        path = pathlib.Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(path), mode="w", encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
        root.addHandler(file_handler)

