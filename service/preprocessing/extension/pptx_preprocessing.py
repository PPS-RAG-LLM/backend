"""
PPTX 전처리 모듈
PPTX 파일을 텍스트와 표로 추출하는 기능 제공
"""
from __future__ import annotations
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _clean_text(s: str | None) -> str:
    """텍스트 정규화 (공통 유틸리티에서 import)"""
    from service.preprocessing.extension.utils import _clean_text as _clean_text_util
    return _clean_text_util(s)


def _extract_pptx(fp: Path) -> tuple[str, list[dict]]:
    """PPTX 파일 추출"""
    try:
        from pptx import Presentation
    except Exception:
        logger.warning(f"python-pptx not available, skipping {fp}")
        return "", []
    
    try:
        prs = Presentation(str(fp))
        texts = []
        tables = []
        for i, slide in enumerate(prs.slides, start=1):
            for sh in slide.shapes:
                if hasattr(sh, "has_text_frame") and sh.has_text_frame:
                    txt = "\n".join(p.text for p in sh.text_frame.paragraphs if p.text)
                    txt = _clean_text(txt)
                    if txt:
                        texts.append(txt)
                if hasattr(sh, "has_table") and sh.has_table:
                    rows = []
                    for r in sh.table.rows:
                        rows.append([_clean_text(c.text) for c in r.cells])
                    if rows:
                        md = "\n".join("| " + " | ".join(row) + " |" for row in rows)
                        tables.append({"page": i, "bbox": [], "text": md})
        return _clean_text("\n\n".join(texts)), tables
    except Exception as e:
        logger.exception(f"Failed to extract PPTX {fp}: {e}")
        return "", []

