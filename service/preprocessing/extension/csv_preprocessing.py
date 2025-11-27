"""
CSV 전처리 모듈
CSV 파일을 표로 추출하는 기능 제공
"""
from __future__ import annotations
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _clean_text(s: str | None) -> str:
    """텍스트 정규화 (공통 유틸리티에서 import)"""
    from service.preprocessing.extension.utils import clean_text as clean_text_util
    return clean_text_util(s)


def _df_to_markdown(df, max_rows=500) -> str:
    """DataFrame을 마크다운 테이블로 변환"""
    if len(df) > max_rows:
        df = df.head(max_rows)
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_clean_text(str(v)) for v in row.tolist()) + " |")
    return "\n".join(lines)


def extract_csv(fp: Path) -> tuple[str, list[dict]]:
    """CSV 파일 추출"""
    try:
        import pandas as pd
        df = pd.read_csv(fp)
        md = _df_to_markdown(df)
        return "", [{"page": 0, "bbox": [], "text": md}]
    except Exception as e:
        logger.warning(f"Failed to extract CSV {fp}: {e}, trying as plain text")
        from service.preprocessing.extension.txt_preprocessing import _extract_plain_text
        return _extract_plain_text(fp)

