import logging
import os
import sys
from typing import Optional
from datetime import datetime
from logging.handlers import RotatingFileHandler
from zoneinfo import ZoneInfo

from config import config as app_config 

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
        no_color = app_config.get("logging", {}).get("no_color", False)
        return sys.stderr.isatty() and not no_color

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

def _build_file_handler(level: int, filename: str = "server.log") -> logging.Handler:
    # 로그 디렉토리 생성 (/app/logs 또는 ./logs)
    log_dir = "/app/logs"
    if not os.path.exists(log_dir):
        # 도커가 아니면 로컬 logs/backend 사용
        log_dir = "logs/backend" 
        os.makedirs(log_dir, exist_ok=True)
    
    file_path = os.path.join(log_dir, filename)
    
    # 10MB 단위로 최대 5개 파일 유지
    handler = RotatingFileHandler(file_path, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8')
    handler.setLevel(level)
    
    # 파일에는 색상 코드 없이 깔끔하게 저장
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    return handler


def get_logger(name: str = "app", level: Optional[int] = None) -> logging.Logger:
    """설정된 컬러 콘솔 로거를 반환합니다.

    - LOG_LEVEL 환경 변수 (DEBUG|INFO|WARNING|ERROR|CRITICAL)를 설정하여 레벨을 재정의할 수 있습니다.
    - LOG_NO_COLOR=1을 설정하여 ANSI 색상을 비활성화할 수 있습니다.
    """
    logger = logging.getLogger(name)
    if level is None:
        level_name = app_config.get("logging", {}).get("level", "DEBUG").upper()
        level = getattr(logging, level_name, logging.DEBUG) # 설정값이 잘못되었을 때만 DEBUG
    logger.setLevel(level)
    logger.propagate = False
    # 여러 번 호출될 때 중복 핸들러를 방지
    if not logger.handlers:
        logger.addHandler(_build_stream_handler(level))
        
        # [추가] 파일 핸들러 추가
        try:
            logger.addHandler(_build_file_handler(level))
        except Exception as e:
            print(f"Failed to add file handler: {e}") # 권한 문제 등 대비
            
    return logger


def logger(name: str):
    return get_logger("BE").getChild(name)  


