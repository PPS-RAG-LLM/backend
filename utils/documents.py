from __future__ import annotations

import uuid


def generate_doc_id() -> str:
    """
    문서 업로드 시 사용하는 공통 doc_id 생성기.
    UUID v4 문자열을 반환한다.
    """
    return str(uuid.uuid4())

