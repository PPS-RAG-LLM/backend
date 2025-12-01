from __future__ import annotations

import uuid
from config import config as app_config
from pathlib import Path
import time


def generate_doc_id() -> str:
    """
    문서 업로드 시 사용하는 공통 doc_id 생성기.
    UUID v4 문자열을 반환한다.
    """
    return str(uuid.uuid4())

def save_raw_file(filename: str, folder: Path, content: bytes) -> str:
    name = Path(filename or "uploaded").name 
    folder.mkdir(parents=True, exist_ok=True)
    dst = folder / name
    if dst.exists():
        stem, ext = dst.stem, dst.suffix
        dst = folder / f"{stem}_{int(time.time())}{ext}"
    dst.write_bytes(content)
    # folder가 절대 경로일 수 있으므로, relative_to 시 주의 필요
    # 하지만 호출처에서 folder를 기준으로 상대 경로를 기대하므로 그대로 둠.
    # 만약 folder가 절대 경로이고, dst도 절대 경로이면 문제 없음.
    return str(dst.relative_to(folder).as_posix())