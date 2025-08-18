import logging
import os
import sys
from typing import Optional
from datetime import datetime

from zoneinfo import ZoneInfo 
# try:
#     from zoneinfo import ZoneInfo 
# except Exception:  # pragma: no cover
#     ZoneInfo = None  # type: ignore


# ANSI color codes
RESET = "\033[0m"
COLORS = {
    logging.DEBUG: "\033[36m",     # Cyan
    logging.INFO: "\033[32m",      # Green
    logging.WARNING: "\033[33m",   # Yellow
    logging.ERROR: "\033[31m",     # Red
    logging.CRITICAL: "\033[41m\033[97m",  # White on Red background
}
DIM = "\033[2m"
BOLD = "\033[1m"


class ColorFormatter(logging.Formatter):
    """Colorful formatter with optional KST timestamp.

    Adds:
      - %(levelname_colored)s token for colored level name
      - KST-aware timestamp if ZoneInfo is available
    """

    def __init__(self, fmt: Optional[str] = None, datefmt: Optional[str] = None, use_colors: bool = True, use_kst: bool = True):
        default_fmt = (
            f"{DIM}%(asctime)s{RESET} | %(levelname_colored)s | %(name)s:%(lineno)d | %(message)s"
        )
        super().__init__(fmt or default_fmt, datefmt)
        self.use_colors = use_colors and self._supports_color()
        self.use_kst = use_kst and (ZoneInfo is not None)

    def _supports_color(self) -> bool:
        no_color_env = os.getenv("LOG_NO_COLOR", "").lower() in {"1", "true", "yes"}
        return sys.stderr.isatty() and not no_color_env

    def format(self, record: logging.LogRecord) -> str:
        level_color = COLORS.get(record.levelno, "") if self.use_colors else ""
        levelname_colored = f"{BOLD}{level_color}{record.levelname}{RESET}" if level_color else record.levelname
        # inject extra token without mutating original levelname for other handlers
        setattr(record, "levelname_colored", levelname_colored)
        return super().format(record)

    def formatTime(self, record: logging.LogRecord, datefmt: Optional[str] = None) -> str:  # noqa: N802
        if self.use_kst and ZoneInfo is not None:
            dt = datetime.fromtimestamp(record.created, tz=ZoneInfo("Asia/Seoul"))
            if datefmt:
                return dt.strftime(datefmt)
            return dt.isoformat(timespec="seconds")
        # fallback to default behavior (localtime)
        return super().formatTime(record, datefmt)


def _build_stream_handler(level: int) -> logging.Handler:
    handler = logging.StreamHandler()
    handler.setLevel(level)
    formatter = ColorFormatter(datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    return handler

def get_logger(name: str = "app", level: Optional[int] = None) -> logging.Logger:
    """Return a configured colorful console logger.

    - Set LOG_LEVEL env (DEBUG|INFO|WARNING|ERROR|CRITICAL) to override level
    - Set LOG_NO_COLOR=1 to disable ANSI colors
    """
    logger = logging.getLogger(name)
    if level is None:
        level_name = os.getenv("LOG_LEVEL", "INFO").upper()
        level = getattr(logging, level_name, logging.INFO)
    logger.setLevel(level)
    logger.propagate = False
    # Avoid duplicate handlers if called multiple times
    if not logger.handlers:
        logger.addHandler(_build_stream_handler(level))
    return logger

# Convenience default logger
app_logger = get_logger("back")

def logger(name: str):
    return app_logger.getChild(name)  # coreiq.<module path>