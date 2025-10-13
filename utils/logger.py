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
        # 원래 levelname을 변경하지 않고 추가 토큰 삽입
        setattr(record, "levelname_colored", levelname_colored)
        return super().format(record)

    def formatTime(self, record: logging.LogRecord, datefmt: Optional[str] = None) -> str:  # noqa: N802
        if self.use_kst and ZoneInfo is not None:
            dt = datetime.fromtimestamp(record.created, tz=ZoneInfo("Asia/Seoul"))
            if datefmt:
                return dt.strftime(datefmt)
            return dt.isoformat(timespec="seconds")
        # 기본 동작(로컬 시간)으로 대체
        return super().formatTime(record, datefmt)


def _build_stream_handler(level: int) -> logging.Handler:
    handler = logging.StreamHandler()
    handler.setLevel(level)
    formatter = ColorFormatter(datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    return handler

def get_logger(name: str = "app", level: Optional[int] = None) -> logging.Logger:
    """설정된 컬러 콘솔 로거를 반환합니다.

    - LOG_LEVEL 환경 변수 (DEBUG|INFO|WARNING|ERROR|CRITICAL)를 설정하여 레벨을 재정의할 수 있습니다.
    - LOG_NO_COLOR=1을 설정하여 ANSI 색상을 비활성화할 수 있습니다.
    """
    logger = logging.getLogger(name)
    if level is None:
        level_name = os.getenv("LOG_LEVEL", "DEBUG").upper()
        level = getattr(logging, level_name, logging.DEBUG)
    logger.setLevel(level)
    logger.propagate = False
    # 여러 번 호출될 때 중복 핸들러를 방지
    if not logger.handlers:
        logger.addHandler(_build_stream_handler(level))
    return logger


def logger(name: str):
    return get_logger("BE").getChild(name)  


