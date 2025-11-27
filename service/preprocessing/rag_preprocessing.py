"""
RAG 전처리 라우터 모듈
확장자별로 적절한 전처리 함수를 호출하는 라우팅 기능 제공
"""
from __future__ import annotations

import logging
import shutil
from collections import defaultdict

from pathlib import Path
from typing import Any, List, Optional

from tqdm import tqdm  # type: ignore

from service.preprocessing.extension.csv_preprocessing import extract_csv
from service.preprocessing.extension.docx_preprocessing import extract_docx
from service.preprocessing.extension.excel_preprocessing import extract_excel
from service.preprocessing.extension.hwp_preprocessing import extract_hwp
from service.preprocessing.extension.pdf_preprocessing import extract_pdf_with_tables
from service.preprocessing.extension.ppt_preprocessing import extract_ppt
from service.preprocessing.extension.pptx_preprocessing import extract_pptx
from service.preprocessing.extension.txt_preprocessing import extract_plain_text

logger = logging.getLogger(__name__)


def ext(p: Path) -> str:
    """파일 확장자 반환 (소문자)"""
    return p.suffix.lower()


def extract_any(path: Path) -> tuple[str, list[dict]]:
    """통합 문서 추출 라우터"""
    file_ext = ext(path)
    if file_ext == ".pdf":
        text, tables, _, _ = extract_pdf_with_tables(path)
        return text, tables
    if file_ext in {".txt", ".text", ".md"}:
        return extract_plain_text(path)
    if file_ext == ".docx":
        return extract_docx(path)
    if file_ext == ".pptx":
        return extract_pptx(path)
    if file_ext == ".csv":
        return extract_csv(path)
    if file_ext in {".xlsx", ".xls"}:
        return extract_excel(path)
    if ext == ".ppt":
        return extract_ppt(path)
    # DOC 파일은 DOCX로 변환 후 처리 (docx_preprocessing에서 자동 변환)
    if ext == ".doc":
        return extract_docx(path)
    # HWP 파일은 현재 지원하지 않음 (Windows 서버 필요)
    if ext == ".hwp":
        logger.warning(f"[Extract] HWP 파일은 현재 지원하지 않습니다: {path.name}")
        return "", []
    # 모르는 확장자는 텍스트로 시도
    return extract_plain_text(path)

