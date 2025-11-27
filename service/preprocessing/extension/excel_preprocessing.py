"""
Excel 전처리 모듈
Excel 파일을 표로 추출하는 기능 제공
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


def extract_excel(fp: Path) -> tuple[str, list[dict]]:
    """Excel 파일 추출"""
    try:
        import pandas as pd
        xls = pd.ExcelFile(fp)
        tables = []
        for name in xls.sheet_names:
            df = xls.parse(name)
            md = f"### {name}\n" + _df_to_markdown(df)
            tables.append({"page": 0, "bbox": [], "text": md})
        return "", tables
    except ImportError as e:
        if "openpyxl" in str(e).lower():
            logger.error(
                f"Excel 파일 처리에 필요한 openpyxl이 설치되지 않았습니다. "
                f"다음 명령어로 설치하세요: pip install openpyxl 또는 conda install openpyxl"
            )
        else:
            logger.error(f"Excel 처리에 필요한 라이브러리가 없습니다: {e}")
        return "", []
    except Exception as e:
        logger.warning(f"Failed to extract Excel {fp}: {e}")
        return "", []

