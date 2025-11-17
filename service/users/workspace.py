from typing import Dict, Any
from config import config
from repository.documents import delete_workspace_documents_by_doc_ids, list_doc_ids_by_workspace
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
from repository.workspace_thread import (
    create_default_thread, 
    get_threads_by_workspace_id,
)

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

    # 1) llm_models의 기본 모델(필수)
    # model = get_default_llm_model(category)
    # if not model:
    #     raise InternalServerError("no default llm model for category")
    
    if category == "qna":
        system_prompt = get_default_system_prompt_content(category)
        chat_history  = defaults["chat_history"]
    else:
        system_prompt = None
        chat_history  = 0

    # 3) 요청 본문으로 오버라이드 (provider/chat_model은 오버라이드 불가)
    chat_mode               = payload.get("chatMode")
    temperature             = defaults["temperature"]
    similarity_threshold    = defaults["similarity_threshold"]
    top_n                   = defaults["top_n"]
    query_refusal_response  = defaults["query_refusal_response"]

    if chat_mode not in ("chat", "query"):
        raise BadRequestError("chatMode must be 'chat' or 'query'")
    if not isinstance(top_n, int) or top_n <= 0:
        raise BadRequestError("topN must be a positive integer")

    slug = generate_unique_slug(name)

    ws_id = insert_workspace(
        name                    = name,
        slug                    = slug,
        category                = category,
        temperature             = temperature,
        chat_history            = chat_history,
        system_prompt           = system_prompt,
        provider                = "huggingface",
        similarity_threshold    = similarity_threshold,
        top_n                   = top_n,
        chat_mode               = chat_mode,
        query_refusal_response  = query_refusal_response,
    )

    link_workspace_to_user(user_id=user_id, workspace_id=ws_id)

    logger.debug(f"Creating default thread for {category} workspace: name={name}")
    if category =="qna":
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
        "id": ws["id"],
        "category": ws["category"],
        "name": ws["name"],
        "slug": ws["slug"],
        "createdAt": ws["created_at"],
        "temperature": ws["temperature"],
        "UpdatedAt": ws["updated_at"],
        "chatHistory": ws["chat_history"] ,
        "systemPrompt": ws["system_prompt"] or "",
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
            "id": ws["id"],
            "category": ws["category"],
            "name": ws["name"],
            "slug": ws["slug"],
            "createdAt": ws["created_at"],
            "temperature": ws["temperature"],
            "UpdatedAt": ws["updated_at"],
            "chatHistory": ws["chat_history"],
            "systemPrompt": ws["system_prompt"] or "",
            "threads": [
                {
                    "id": thread["id"],
                    "name": thread["name"],
                    "thread_slug": thread["slug"],
                    "createdAt": thread["created_at"],
                    "UpdatedAt": thread["updated_at"],
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

    if not ws:
        raise NotFoundError("요청한 워크스페이스를 찾을 수 없습니다")

    # from repository.documents import get_documents_by_workspace_id
    # threads = get_threads_by_workspace_id(workspace_id)
    return {
        "id": ws["id"],
        "name": ws["name"],
        "category": ws["category"],
        "slug": ws["slug"],
        "createdAt": ws["created_at"],
        "updatedAt": ws["updated_at"],
        "temperature": ws["temperature"],
        "chatHistory": ws["chat_history"],
        "systemPrompt": ws["system_prompt"],
        "provider": ws["provider"],
        "chatModel": ws["chat_model"],
        "topN": ws["top_n"],
        "chatMode": ws["chat_mode"],
        "queryRefusalResponse": ws["query_refusal_response"],
        "vectorSearchMode": ws["vector_search_mode"],
        "similarityThreshold": ws["similarity_threshold"],
        "vectorCount": ws["vector_count"],
    }


def delete_workspace(user_id: int, slug: str) -> None:
    from service.users.documents.documents import delete_document_files
    workspace_id = get_workspace_id_by_slug_for_user(user_id, slug)
    doc_ids_rows = list_doc_ids_by_workspace(workspace_id)
    doc_ids = [r["doc_id"] for r in doc_ids_rows]
    
    logger.debug(f"doc_ids: {doc_ids}")
    
    if doc_ids:
        # 공통 함수 사용해서 파일 삭제
        deleted_files = delete_document_files(doc_ids)
        logger.info(f"File deletion summary: {deleted_files}")
        
        # DB 삭제
        delete_workspace_documents_by_doc_ids(doc_ids, workspace_id)
        logger.debug(f"deleted document records from workspace: {workspace_id}")

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

    # # 이름 변경 시 slug도 자동 갱신
    # if "name" in updates and isinstance(updates["name"], str) and updates["name"].strip():
    #     updates["slug"] = generate_unique_slug(updates["name"]) 

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