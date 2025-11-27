"""
TXT/MD 전처리 모듈
텍스트 파일을 추출하는 기능 제공
"""
from __future__ import annotations
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _clean_text(s: str | None) -> str:
    """텍스트 정규화 (공통 유틸리티에서 import)"""
    from service.preprocessing.extension.utils import clean_text as clean_text_util
    return clean_text_util(s)


def extract_plain_text(fp: Path) -> tuple[str, list[dict]]:
    """TXT/MD 파일 추출"""
    try:
        text = fp.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        text = fp.read_text(errors="ignore")
    return _clean_text(text), []

