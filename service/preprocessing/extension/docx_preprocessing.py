"""
DOCX 전처리 모듈
DOCX 파일을 텍스트와 표로 추출하는 기능 제공
"""
from __future__ import annotations
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _clean_text(s: str | None) -> str:
    """텍스트 정규화 (공통 유틸리티에서 import)"""
    from service.preprocessing.extension.utils import _clean_text as _clean_text_util
    return _clean_text_util(s)


def _extract_docx(fp: Path) -> tuple[str, list[dict]]:
    """DOCX 파일 추출"""
    try:
        from docx import Document
    except Exception:
        logger.warning(f"python-docx not available, treating {fp} as plain text")
        from service.preprocessing.extension.txt_preprocessing import _extract_plain_text
        return _extract_plain_text(fp)
    
    try:
        d = Document(str(fp))
        paras = [_clean_text(p.text) for p in d.paragraphs if _clean_text(p.text)]
        tables = []
        for tb in d.tables:
            rows = []
            for r in tb.rows:
                rows.append([_clean_text(c.text) for c in r.cells])
            if rows:
                md = "\n".join("| " + " | ".join(row) + " |" for row in rows)
                tables.append({"page": 0, "bbox": [], "text": md})
        return _clean_text("\n\n".join(paras)), tables
    except Exception as e:
        logger.exception(f"Failed to extract DOCX {fp}: {e}")
        from service.preprocessing.extension.txt_preprocessing import _extract_plain_text
        return _extract_plain_text(fp)

