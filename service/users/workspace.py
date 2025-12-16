from typing import Dict, Any, Optional
from config import config
from repository.documents import list_doc_ids_by_workspace
from repository.rag_settings import get_rag_settings_row
from utils import generate_unique_slug, generate_thread_slug, logger
from errors import BadRequestError, InternalServerError, NotFoundError
from pathlib import Path
from fastapi import UploadFile
from repository.workspace import (
    delete_workspace_by_workspace_id,
    insert_workspace,
    get_workspace_by_id,
    link_workspace_to_user,
    get_default_system_prompt_content,
    get_workspaces_by_user,
    update_workspace_by_slug_for_user,
    get_workspace_by_workspace_id,
    get_workspace_id_by_slug_for_user,
    update_workspace_name_by_slug_for_user
)
from repository.user import get_api_keys_by_user_id
from repository.workspace_thread import (
    create_default_thread, 
    get_threads_by_workspace_id,
)
from service.manage_documents import delete_documents_by_ids

logger = logger(__name__)

ALLOWED_CATEGORIES = {"qna", "doc_gen", "summary"}
 
def create_workspace_for_user(user_id: int, category: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    defaults = config["workspace"]
    logger.debug({
        "user_id": user_id,
        "category": category,
        "payload": payload,
    })
    if category not in ALLOWED_CATEGORIES:
        raise BadRequestError(f"invalid category: {category}")
    name = payload.get("name")
    if not name or not isinstance(name, str):
        raise BadRequestError("name is required")
    
    try: 
        db_settings = get_rag_settings_row()
        
    except Exception as exc :
        logger.warning(f"Milvus DB 설정 조회 실패 : {exc}")
        return []
  
    if category == "qna":
        system_prompt = get_default_system_prompt_content(category)
        chat_history  = defaults["chat_history"]
    else:
        system_prompt = None
        chat_history  = 0

    slug = generate_unique_slug(name)

    # 기본값 함께 워크스페이스 생성 시 저장
    ws_id = insert_workspace(
        name                    = name,
        slug                    = slug,
        category                = category,
        chat_history            = chat_history,
        system_prompt           = system_prompt,
        temperature             = defaults["temperature"],
        provider                = defaults["provider"],
        similarity_threshold    = defaults["similarity_threshold"],
        top_n                   = defaults["top_n"],
        chat_mode               = defaults["chat_mode"],
        query_refusal_response  = defaults["query_refusal_response"],
        vector_search_mode      = db_settings.get("searchType"),
    )

    link_workspace_to_user(user_id=user_id, workspace_id=ws_id)

    logger.debug(f"Creating default thread for {category} workspace: name={name}")
    if category == "qna":
        thread_name = f"thread-{name}"
        thread_slug = generate_thread_slug(thread_name)
        logger.info(f"thread_name: {thread_name}, thread_slug: {thread_slug}")
        thread_id = create_default_thread(user_id=user_id, name=thread_name, thread_slug=thread_slug, workspace_id=ws_id)
        logger.info(f"Default thread created for qna workspace: thread_id={thread_id}")
    else:
        logger.debug(f"No default thread for {category} workspace: name={name}")
        thread_id = None

    ws = get_workspace_by_id(ws_id)
    if not ws:
        raise InternalServerError("workspace retrieval failed")

    result = {
        "id"        : ws["id"],
        "category"  : ws["category"],
        "name"      : ws["name"],
        "slug"      : ws["slug"],
        "createdAt" : ws["created_at"],
    }
    logger.info({"workspace_created": result})
    return result

### 리스트 조회

def list_workspaces(user_id: int) -> list[Dict[str, Any]]:
    
    rows = get_workspaces_by_user(user_id)
    items = []
    for ws in rows:
        if ws["category"]=="qna":
            threads = get_threads_by_workspace_id(ws["id"])
        else:
            threads = []
        items.append({
            "id"        : ws["id"],
            "category"  : ws["category"],
            "name"      : ws["name"],
            "slug"      : ws["slug"],
            "createdAt" : ws["created_at"],
            "updatedAt" : ws["updated_at"],
            "provider"  : ws["provider"],
            "threads": [
                {
                    "id"            : thread["id"],
                    "name"          : thread["name"],
                    "threadSlug"    : thread["slug"],
                    "createdAt"     : thread["created_at"],
                    "updatedAt"     : thread["updated_at"],
                } for thread in threads
            ],
        })
    logger.debug({"workspaces_count": len(items), "user_id": user_id})
    return items

### 워크스페이스 상세 조회

def get_workspace_detail(user_id: int, slug: str) -> Dict[str, Any]:
    workspace_id = get_workspace_id_by_slug_for_user(user_id, slug)
    if not workspace_id:
        raise NotFoundError("요청한 워크스페이스를 찾을 수 없습니다")
    
    ws = get_workspace_by_workspace_id(user_id, workspace_id)
    
    # api_keys가 None일 경우 빈 딕셔너리로 처리 (에러 방지)
    api_keys = get_api_keys_by_user_id(user_id) or {}

    if not ws:
        raise NotFoundError("요청한 워크스페이스를 찾을 수 없습니다")

    return {
        "id"                    : ws["id"],
        "name"                  : ws["name"],
        "category"              : ws["category"],
        "slug"                  : ws["slug"],
        "createdAt"             : ws["created_at"],
        "updatedAt"             : ws["updated_at"],
        "temperature"           : ws["temperature"],
        "chatHistory"           : ws["chat_history"],
        "systemPrompt"          : ws["system_prompt"],
        "provider"              : ws["provider"],
        # 민감 정보 마스킹 처리
        "openaiApiKey"          : api_keys.get("openai_api_key"),
        "anthropicApiKey"       : api_keys.get("anthropic_api_key"),
        "geminiApiKey"          : api_keys.get("gemini_api_key"),
        "chatModel"             : ws["chat_model"],
        "topN"                  : ws["top_n"],
        "chatMode"              : ws["chat_mode"],
        "queryRefusalResponse"  : ws["query_refusal_response"],
        "vectorSearchMode"      : ws["vector_search_mode"],
        "similarityThreshold"   : ws["similarity_threshold"],
        "vectorCount"           : ws["vector_count"],
    }


def delete_workspace(user_id: int, slug: str) -> None:
    workspace_id = get_workspace_id_by_slug_for_user(user_id, slug)
    doc_ids_rows = list_doc_ids_by_workspace(workspace_id)
    doc_ids = [r["doc_id"] for r in doc_ids_rows]
    
    logger.debug(f"doc_ids: {doc_ids}")
    
    if doc_ids:
        delete_documents_by_ids(doc_ids)
        logger.debug("deleted %s documents before workspace removal", len(doc_ids))

    deleted = delete_workspace_by_workspace_id(workspace_id)
    if not deleted:
        logger.error(f"delete workspace failed: user_id={user_id}, slug={slug}")
        raise NotFoundError("삭제 실패")
    return None

def update_workspace_name_service(user_id: int, slug: str, name: str) -> Dict[str, Any]:
    workspace_id = get_workspace_id_by_slug_for_user(user_id, slug)
    if not workspace_id:
        raise NotFoundError("요청한 워크스페이스를 찾을 수 없습니다")
    update_workspace_name_by_slug_for_user(user_id, slug, name)
    return None

def update_workspace(user_id: int, slug: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    updates = {k: v for k, v in payload.items()}
    if not updates:
        return None

    # 1. API Key 필드 분리 (User 테이블용)
    user_updates = {}
    api_key_map = {
        "openaiApiKey": "openai_api_key",
        "anthropicApiKey": "anthropic_api_key",
        "geminiApiKey": "gemini_api_key",
    }
    
    for body_key, db_col in api_key_map.items():
        if body_key in payload:
            val = payload.pop(body_key)
            
            # 중요: 마스킹된 키(**** 포함)가 그대로 돌아온 경우 업데이트 대상에서 제외
            if val and isinstance(val, str) and "****" in val:
                continue
                
            # 빈 문자열이면 None으로 저장 (삭제), 아니면 값 저장
            user_updates[db_col] = val if val and str(val).strip() else None

    # 2. User 테이블 업데이트 (API 키가 있을 경우)
    if user_updates:
        from repository.user import update_user_api_keys # (새로 만들 함수)
        update_user_api_keys(user_id, user_updates)
        
    workspace_id = get_workspace_id_by_slug_for_user(user_id, slug) # 워크스페이스 아이디 조회
    update_workspace_by_slug_for_user(user_id, slug, updates) # 워크스페이스 업데이트
    ws = get_workspace_by_workspace_id(user_id, workspace_id) # 워크스페이스 상세 조회

    if ws is None:
        raise NotFoundError(f"Workspace not found for slug '{slug}' or update failed")
    return {"message": "Workspace updated"}

def upload_and_embed_document(user_id: int, slug: str, file: UploadFile) -> Dict[str, Any]:
    """임시 스텁: 파일 업로드 + 벡터 DB 인제스트 예정 구현."""
    # TODO: 구현
    raise NotImplementedError("upload_and_embed_document is not yet implemented")