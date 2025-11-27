from __future__ import annotations

import uuid
from config import config as app_config
from pathlib import Path
import time

_RETRIEVAL_PATHS = app_config.get("retrieval", {}).get("paths", {}) or {}
RAW_DATA_DIR = Path(_RETRIEVAL_PATHS.get("raw_data_root", "storage/raw_files"))

def generate_doc_id() -> str:
    """
    문서 업로드 시 사용하는 공통 doc_id 생성기.
    UUID v4 문자열을 반환한다.
    """
    return str(uuid.uuid4())

def save_raw_file(filename: str, folder: str, content: bytes) -> str:
    name = Path(filename or "uploaded").name 
    folder_path = RAW_DATA_DIR / folder
    folder_path.mkdir(parents=True, exist_ok=True)
    dst = folder_path / name
    if dst.exists():
        stem, ext = dst.stem, dst.suffix
        dst = folder_path / f"{stem}_{int(time.time())}{ext}"
    dst.write_bytes(content)
    return str(dst.relative_to(RAW_DATA_DIR).as_posix())