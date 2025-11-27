"""
전처리 공통 유틸리티 모듈
텍스트 정규화 및 공통 헬퍼 함수 제공
"""
from __future__ import annotations
import re
import unicodedata
from pathlib import Path

# 텍스트 정리용 정규식
ZERO_WIDTH_RE = re.compile(r'[\u200B-\u200D\u2060\uFEFF]')
CONTROL_RE = re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F]')  # \t,\n은 유지
MULTISPACE_LINE_END_RE = re.compile(r'[ \t]+\n')
NEWLINES_RE = re.compile(r'\n{3,}')


def clean_text(s: str | None) -> str:
    """텍스트 정규화 (공통 유틸리티)"""
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    # 흔한 공백/불릿/문장부호 정리
    s = s.replace("\xa0", " ").replace("\u2022", "•").replace("\u2212", "-")
    s = ZERO_WIDTH_RE.sub("", s)
    s = CONTROL_RE.sub("", s)
    s = MULTISPACE_LINE_END_RE.sub("\n", s)
    s = NEWLINES_RE.sub("\n\n", s)
    return s.strip()

