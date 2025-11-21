"""
DOC 전처리 모듈
DOC 파일을 텍스트와 표로 추출하는 기능 제공 (LibreOffice 변환 사용)
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _convert_via_libreoffice(src: Path, target_ext: str) -> Optional[Path]:
    """LibreOffice를 통한 문서 변환"""
    try:
        import subprocess
        outdir = src.parent
        subprocess.run([
            "libreoffice", "--headless", "--convert-to", target_ext, 
            "--outdir", str(outdir), str(src)
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        cand = src.with_suffix("." + target_ext)
        return cand if cand.exists() else None
    except Exception:
        return None


def _extract_doc(fp: Path) -> tuple[str, list[dict]]:
    """DOC 파일 추출 (LibreOffice 변환 시도)"""
    conv = _convert_via_libreoffice(fp, "docx")
    if conv and conv.exists():
        from service.preprocessing.extension.docx_preprocessing import _extract_docx
        result = _extract_docx(conv)
        try:
            conv.unlink()  # 임시 변환 파일 삭제
        except Exception:
            pass
        return result
    from service.preprocessing.extension.txt_preprocessing import _extract_plain_text
    return _extract_plain_text(fp)

