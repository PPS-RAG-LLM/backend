from repository.documents import list_workspace_documents as _list_docs
from errors import NotFoundError
from mimetypes import guess_type
from repository.workspace import get_workspace_id_by_slug_for_user
from typing import Dict, Any, List
from pathlib import Path
import json
from config import config

DOC_INFO_DIR = Path(config["user_documents"]["doc_info_dir"])

def _safe_read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _guess_mime_from_title_or_name(title: str, fallback_name: str) -> str:
    # title(원본 파일명) 우선, 없으면 documents-info 파일명 사용
    name = (title or "").strip() or (fallback_name or "").strip()
    mimetype, _ = guess_type(name)
    return mimetype or "application/octet-stream"

def _resolve_docinfo_path(stored_path: str) -> Path:
    p = Path(stored_path)
    if p.exists():
        return p
    # 상대경로 또는 베이스만 다른 경우: 폴백으로 documents-info 폴더 + basename
    return DOC_INFO_DIR / Path(stored_path).name

def _build_local_file_item(docinfo_path: Path, payload: Dict[str, Any]) -> Dict[str, Any]:
    # documents-info JSON의 기본 필드
    doc_id   = str(payload.get("id") or "").strip()
    title    = (payload.get("title") or "").strip()
    url      = (payload.get("url") or "").strip()
    name     = docinfo_path.name
    mime     = _guess_mime_from_title_or_name(title, name)
    return {
        "name": name,
        "type": mime,
        "id": doc_id,
        "url": url,                 # 현재 업로드 로직은 빈 문자열일 수 있음
        "title": title or name,
        "cached": False,
    }

def list_local_documents_for_workspace(user_id: int, slug: str) -> Dict[str, Any]:
    """
    워크스페이스 슬러그로 등록된 문서 목록을 반환한다.
    - DB(workspace_documents)에서 doc_id/filename/docpath를 조회
    - docpath로 documents-info JSON을 읽어 응답 스키마로 매핑
    """
    workspace_id = get_workspace_id_by_slug_for_user(user_id, slug)
    if not workspace_id:
        raise NotFoundError("요청한 워크스페이스를 찾을 수 없습니다")

    rows = _list_docs(int(workspace_id)) or []

    items: List[Dict[str, Any]] = []
    for r in rows:
        docpath = str(r.get("docpath") or "").strip()
        if not docpath:
            continue
        p = _resolve_docinfo_path(docpath)
        if not p.exists():
            # 파일이 삭제되었을 수 있음 → 최소 정보만 반환
            items.append({
                "name": Path(docpath).name,
                #"type": "application/pdf", # TODO : 파일 타입 추가
                "id": str(r.get("doc_id") or "").strip(),
                "url": "",
                "title": r.get("filename"),
                "cached": False,
            })
            continue
        payload = _safe_read_json(p)
        items.append(_build_local_file_item(p, payload))

    return {"localFiles": items}