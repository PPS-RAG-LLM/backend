from typing import Dict, Any, Generator, List
from errors import NotFoundError, BadRequestError
from utils.llms.registry import LLM
from repository.users.workspace import get_workspace_by_slug_for_user

def _build_messages(ws: Dict[str, Any], body: Dict[str, Any]) -> List[Dict[str, Any]]:
    system_prompt = ws.get("system_prompt")
    provider = (ws.get("provider") or "").lower()
    attachments = body.get("attachments") or []
    content = body["message"]

    msgs: List[Dict[str, Any]] = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})

    if provider == "openai" and attachments:
        parts = [{"type": "text", "text": content}]
        for att in attachments:
            cs = att.get("contentString")
            if cs:
                parts.append({"type": "image_url", "image_url": {"url": cs}})
        msgs.append({"role": "user", "content": parts})
    else:
        msgs.append({"role": "user", "content": content})
    return msgs

def stream_chat_for_workspace(user_id: int, slug: str, body: Dict[str, Any]) -> Generator[str, None, None]:
    """
    """
    ws = get_workspace_by_slug_for_user(user_id, slug)
    if not ws:
        raise NotFoundError("워크스페이스를 찾을 수 없습니다")

    mode = (body.get("mode") or ws.get("chat_mode") or "chat").lower()
    if mode not in ("chat", "query"):
        raise BadRequestError("mode must be 'chat' or 'query'")

    # TODO: 실제 벡터 검색 결과 유무 판단 및 컨텍스트 주입
    if mode == "query":
        has_context = False
        if not has_context:
            yield ws.get("query_refusal_response") or "There is no information about this topic."
            return

    runner = LLM.from_workspace(ws)  # provider/model 라우팅
    messages = _build_messages(ws, body)
    temperature = ws.get("temperature") or 0.7

    for chunk in runner.stream(messages, temperature=temperature):
        yield chunk