"""
Logging utilities for DNA framework
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output."""

    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
    }
    RESET = '\033[0m'

    def format(self, record):
        # Add color
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.RESET}"

        return super().format(record)


def setup_logger(
    name: str = "dna",
    level: str = "INFO",
    log_file: Optional[Path] = None,
    console: bool = True,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Setup a logger with console and/or file handlers.

    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        console: Whether to add console handler
        format_string: Custom format string

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    logger.handlers.clear()

    # Default format
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        console_formatter = ColoredFormatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # Always log everything to file
        file_formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "dna") -> logging.Logger:
    """
    Get an existing logger or create a new one.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)

    # If no handlers, setup with defaults
    if not logger.handlers:
        setup_logger(name)

    return logger


class LogContext:
    """Context manager for temporary log level changes."""

    def __init__(self, logger: logging.Logger, level: str):
        self.logger = logger
        self.new_level = getattr(logging, level.upper())
        self.old_level = logger.level

    def __enter__(self):
        self.logger.setLevel(self.new_level)
        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.setLevel(self.old_level)
        return False


def create_run_logger(output_dir: Path, run_name: Optional[str] = None) -> logging.Logger:
    """
    Create a logger for a specific run.

    Args:
        output_dir: Directory to save logs
        run_name: Optional run name (defaults to timestamp)

    Returns:
        Configured logger
    """
    if run_name is None:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"{run_name}.log"

    logger = setup_logger(
        name=f"dna.{run_name}",
        level="DEBUG",
        log_file=log_file,
        console=True
    )

    logger.info(f"Starting run: {run_name}")
    logger.info(f"Log file: {log_file}")

    return logger


# Progress tracking utilities
class ProgressLogger:
    """Logger with progress tracking."""

    def __init__(self, logger: logging.Logger, total_steps: int, description: str = "Progress"):
        self.logger = logger
        self.total_steps = total_steps
        self.description = description
        self.current_step = 0
        self.start_time = datetime.now()

    def update(self, steps: int = 1, message: Optional[str] = None):
        """Update progress."""
        self.current_step += steps
        percentage = (self.current_step / self.total_steps) * 100

        elapsed = (datetime.now() - self.start_time).total_seconds()
        steps_per_sec = self.current_step / elapsed if elapsed > 0 else 0
        eta = (self.total_steps - self.current_step) / steps_per_sec if steps_per_sec > 0 else 0

        log_message = f"{self.description}: {self.current_step}/{self.total_steps} ({percentage:.1f}%) "
        log_message += f"[{elapsed:.1f}s elapsed, ETA: {eta:.1f}s]"

        if message:
            log_message += f" - {message}"

        self.logger.info(log_message)

    def finish(self, message: Optional[str] = None):
        """Mark as finished."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        log_message = f"{self.description}: Complete in {elapsed:.1f}s"

        if message:
            log_message += f" - {message}"

        self.logger.info(log_message)
