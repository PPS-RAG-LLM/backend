from typing import Dict, Any, Generator, List
from errors import NotFoundError, BadRequestError
from utils.llms.registry import LLM
from repository.users.workspace import get_workspace_by_workspace_id, get_workspace_id_by_slug_for_user
from repository.users.workspace_thread import get_thread_id_by_slug_for_user
from repository.users.workspace_chat import (
    get_chat_history_by_workspace_id, 
    get_chat_history_by_thread_id,
    insert_chat_history,
)
from utils import logger
import json

logger = logger(__name__)

def _build_messages(ws: Dict[str, Any], body: Dict[str, Any]) -> List[Dict[str, Any]]:
    system_prompt = ws.get("system_prompt")
    provider = (ws.get("provider") or "").lower()
    attachments = body.get("attachments") or []
    content = body["message"]

    msgs: List[Dict[str, Any]] = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt + ". 반드시 한국어로 대답하세요."})

    if provider == "openai" and attachments:
        parts = [{"type": "text", "text": content}]
        for att in attachments:
            cs = att.get("contentString")
            if cs:
                parts.append({"type": "image_url", "image_url": {"url": cs}})
        msgs.append({"role": "user", "content": parts})
    else:
        msgs.append({"role": "user", "content": content})

    # logger.info(f"msgs: {msgs}")
    return msgs

def stream_chat_for_workspace(
    user_id: int, 
    slug: str, 
    category: str,
    body: Dict[str, Any], 
    thread_slug: str=None
) -> Generator[str, None, None]:
    """"""
    if thread_slug:
        thread_id = get_thread_id_by_slug_for_user(user_id, thread_slug)
        workspace_id = get_workspace_id_by_slug_for_user(slug)
        logger.info(f"thread_id: {thread_id}")
        if not thread_id:
            raise NotFoundError("채팅 스레드를 찾을 수 없습니다")
    else:
        thread_id = None
        workspace_id = get_workspace_id_by_slug_for_user(slug)

    ws = get_workspace_by_workspace_id(user_id, workspace_id)
    messages = []
    if category == "qa":
        limit = ws["chat_history"]
        if limit > 0:
            chat_history = get_chat_history_by_thread_id(user_id, thread_id, limit)
            for chat in chat_history:
                messages.append({"role": "user", "content": chat["prompt"]})
                messages.append({"role": "assistant", "content": chat["response"]})


    if not ws:
        raise NotFoundError("워크스페이스를 찾을 수 없습니다")

    mode = (body.get("mode") or ws.get("chat_mode") or "chat").lower()
    logger.info(f"mode: {mode}")
    
    if mode not in ("chat", "query"):
        raise BadRequestError("mode must be 'chat' or 'query'")

    # TODO: 실제 벡터 검색 결과 유무 판단 및 컨텍스트 주입

    runner = LLM.from_workspace(ws)             # provider/model 라우팅
    messages.extend(_build_messages(ws, body))  # 시스템 프롬프트 주입
    temperature = ws.get("temperature")

    logger.info(f"==========messages: \n{messages}")

    # 4. LLM 호출 및 응답 스트리밍
    response_text = ""
    for chunk in runner.stream(messages, temperature=temperature):
        response_text += chunk
        yield chunk

    # 5. 응답 저장
    response_json = {
        "text": response_text,
        "sources": [],
        "type": "chat",
        "attachments": [],
        "metrics": {
            "completion_tokens": 11,  # 예시 값
            "prompt_tokens": 5782,    # 예시 값
            "total_tokens": 5793,     # 예시 값
            "outputTps": 56.12,       # 예시 값
            "duration": 0.196         # 예시 값
        }
    }

    insert_chat_history(
        user_id = user_id,
        category=category,
        workspace_id=workspace_id,
        prompt=body["message"],
        response=json.dumps(response_json, ensure_ascii=False),
        thread_id=thread_id,
    )
    