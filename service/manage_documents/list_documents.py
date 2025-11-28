from repository.documents import list_workspace_documents as _list_docs
from errors import NotFoundError
from mimetypes import guess_type
from repository.workspace import get_workspace_id_by_slug_for_user
from typing import Dict, Any, List
from pathlib import Path
import json
from config import config


def _guess_mime_from_title_or_name(title: str, fallback_name: str) -> str:
    # title(원본 파일명) 우선, 없으면 documents-info 파일명 사용
    target = title or fallback_name
    mime, _ = guess_type(target)
    return mime or "application/octet-stream"


def _build_db_file_item(doc_row: Dict[str, Any]) -> Dict[str, Any]:
    """DB 조회 결과(doc_row)를 프론트엔드 응답 스키마로 변환"""
    doc_id = str(doc_row.get("doc_id") or "").strip()
    filename = str(doc_row.get("filename") or "").strip()
    source_path = str(doc_row.get("source_path") or "").strip()
    
    # docpath는 실제 경로 혹은 가상의 식별자일 수 있음
    # 여기서는 filename을 title/name으로 사용
    
    mime = _guess_mime_from_title_or_name(filename, source_path)
    
    return {
        "name": filename or Path(source_path).name or "unknown",
        "type": mime,
        "id": doc_id,
        "url": "",  # S3/Presigned URL 등이 있다면 여기서 처리
        "title": filename,
        "cached": True,  # DB에 있으면 cached로 간주
    }


def list_local_documents_for_workspace(user_id: int, slug: str) -> Dict[str, Any]:
    """
    워크스페이스 슬러그로 등록된 문서 목록을 반환한다.
    - 기존: 파일시스템(documents-info JSON) 조회
    - 변경: DB(documents, workspace_documents) 조회
    """
    workspace_id = get_workspace_id_by_slug_for_user(user_id, slug)
    if not workspace_id:
        raise NotFoundError("요청한 워크스페이스를 찾을 수 없습니다")

    rows = _list_docs(int(workspace_id)) or []

    items: List[Dict[str, Any]] = []
    for r in rows:
        # DB 레코드 기반 아이템 생성
        items.append(_build_db_file_item(r))

    return {"localFiles": items}
