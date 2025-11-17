"""검증 로직"""
from typing import Dict, Any
from errors import NotFoundError, BadRequestError
from repository.workspace import get_workspace_by_workspace_id, get_workspace_id_by_slug_for_user
from repository.workspace_thread import get_thread_id_by_slug_for_user


def preflight_stream_chat_for_workspace(
    user_id: int,
    slug: str,
    category: str,
    body: Dict[str, Any] | None = None,
    thread_slug: str | None = None,
) -> Dict[str, Any]:
    """
    스트리밍 시작 전 모든 유효성 검사를 수행하고 필요 리소스를 준비한다.
    예외는 여기서 발생시켜 StreamingResponse 시작 전 FastAPI 핸들러로 전달되게 한다.
    """
    # 카테고리 검증
    if category not in ("qna", "doc_gen", "summary"):
        raise BadRequestError("category must be one of: qna, doc_gen, summary")
    
    workspace_id = get_workspace_id_by_slug_for_user(user_id, slug)
    if not workspace_id:
        raise NotFoundError("워크스페이스를 찾을 수 없습니다")

    ws = get_workspace_by_workspace_id(user_id, workspace_id)
    if not ws:
        raise NotFoundError("워크스페이스를 찾을 수 없습니다")

    # QA는 thread_id 필수
    if category == "qna":
        if not thread_slug:
            raise BadRequestError("qna 카테고리는 thread_slug가 필요합니다")
        thread_id = get_thread_id_by_slug_for_user(user_id, thread_slug)
        if not thread_id:
            raise NotFoundError("채팅 스레드를 찾을 수 없습니다")
    else:
        thread_id = None

    # 모드 검증
    mode = (body.get("mode") or ws.get("chat_mode") or "chat").lower()
    if mode not in ("chat", "query"):
        raise BadRequestError("mode must be 'chat' or 'query'")

    return {"ws": ws, "workspace_id": workspace_id, "thread_id": thread_id, "mode": mode}

