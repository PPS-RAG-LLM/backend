from typing import Dict, Any, Optional
from config import config
from utils import generate_unique_slug, logger, to_kst
from errors import BadRequestError, InternalServerError, NotFoundError
from repository.users.workspace import (
    get_default_llm_model,
    insert_workspace,
    get_workspace_by_id,
    link_workspace_to_user,
    get_default_system_prompt_content,
    get_workspaces_by_user,
    get_workspace_by_slug_for_user,
    delete_workspace_by_slug_for_user,
    update_workspace_by_slug_for_user,
)

logger = logger(__name__)

ALLOWED_CATEGORIES = {"qa", "doc_gen", "summary"}
 
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
    model = get_default_llm_model(category)
    if not model:
        raise InternalServerError("no default llm model for category")

    system_prompt = get_default_system_prompt_content(category)

    provider                = model["provider"]
    chat_model              = model["chat_model"]
    # 3) 요청 본문으로 오버라이드 (provider/chat_model은 오버라이드 불가)
    chat_mode               = payload.get("chatMode")
    temperature             = defaults["temperature"]
    chat_history            = defaults["chat_history"]
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
        similarity_threshold    = similarity_threshold,
        provider                = provider,
        chat_model              = chat_model,
        top_n                   = top_n,
        chat_mode               = chat_mode,
        query_refusal_response  = query_refusal_response,
    )

    link_workspace_to_user(user_id=user_id, workspace_id=ws_id)

    ws = get_workspace_by_id(ws_id)
    if not ws:
        raise InternalServerError("workspace retrieval failed")

    result = {
        "id": ws["id"],
        "category": ws["category"],
        "name": ws["name"],
        "slug": ws["slug"],
        "createdAt": to_kst(ws["created_at"]),
        "temperature": ws["temperature"],
        "UpdatedAt": to_kst(ws["updated_at"]),
        "chatHistory": ws["chat_history"],
        "systemPrompt": ws["system_prompt"],
    }
    logger.info({"workspace_created": result})
    return result

### 리스트 조회

def list_workspaces(user_id: int) -> list[Dict[str, Any]]:
    rows = get_workspaces_by_user(user_id)
    items = []
    for ws in rows:
        items.append({
            "id": ws["id"],
            "category": ws["category"],
            "name": ws["name"],
            "slug": ws["slug"],
            "createdAt": to_kst(ws["created_at"]),
            "temperature": ws["temperature"],
            "UpdatedAt": to_kst(ws["updated_at"]),
            "chatHistory": ws["chat_history"],
            "systemPrompt": ws["system_prompt"],
            "threads": [],
        })
    logger.debug({"workspaces_count": len(items), "user_id": user_id})
    return items

### 워크스페이스 상세 조회

def get_workspace_detail(user_id: int, slug: str) -> Dict[str, Any]:
    ws = get_workspace_by_slug_for_user(user_id, slug)
    if not ws:
        raise NotFoundError("요청한 리소스를 찾을 수 없습니다")
    return {
        "id": ws["id"],
        "name": ws["name"],
        "category": ws["category"],
        "slug": ws["slug"],
        "createdAt": to_kst(ws["created_at"]),
        "temperature": ws["temperature"],
        "updatedAt": to_kst(ws["updated_at"]),
        "chatHistory": ws["chat_history"],
        "systemPrompt": ws["system_prompt"],
        "documents": [],
        "threads": [],
    }

def delete_workspace(user_id: int, slug: str) -> None:
    deleted = delete_workspace_by_slug_for_user(user_id, slug)
    if not deleted:
        raise NotFoundError("요청한 리소스를 찾을 수 없습니다")
    return None

def update_workspace(user_id: int, slug: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    # 허용 필드만 전달
    allowed_keys = {"name", "temperature", "chatHistory", "systemPrompt"}
    updates = {k: v for k, v in payload.items() if k in allowed_keys}
    if not updates:
        raise BadRequestError("업데이트할 필드가 없습니다")

    # 이름 변경 시 slug도 자동 갱신
    if "name" in updates and isinstance(updates["name"], str) and updates["name"].strip():
        updates["slug"] = generate_unique_slug(updates["name"]) 

    ws = update_workspace_by_slug_for_user(user_id, slug, updates)
    if not ws:
        raise NotFoundError("요청한 리소스를 찾을 수 없습니다")

    return {
        "workspace": {
            "id": ws["id"],
            "name": ws["name"],
            "category": ws["category"],
            "slug": ws["slug"],
            "createdAt": to_kst(ws["created_at"]),
            "temperature": ws["temperature"],
            "updatedAt": to_kst(ws["updated_at"]),
            "chatHistory": ws["chat_history"],
            "systemPrompt": ws["system_prompt"],
            "documents": [],
        },
        "message": None,
    }


