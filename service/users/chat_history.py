
from __future__ import annotations
from repository.workspace import get_workspace_id_by_slug_for_user, get_workspace_by_workspace_id
from errors import NotFoundError
from repository.workspace_chat import get_chat_history_by_thread_id
from repository.workspace_thread import get_thread_id_by_slug_for_user
from typing import List, Dict, Any
import json
from utils import logger

logger = logger(__name__)

def list_thread_chats_for_workspace(
    user_id: int,
    slug: str,
    thread_slug: str,
):
    workspace_id = get_workspace_id_by_slug_for_user(user_id, slug)
    workspace = get_workspace_by_workspace_id(user_id, workspace_id)
    if not workspace:
        raise NotFoundError("Workspace not found")
    thread_id = get_thread_id_by_slug_for_user(user_id, thread_slug)

    limit = workspace["chat_history"]
    logger.info(f"limit: {limit}")

    chat_history = get_chat_history_by_thread_id(user_id=user_id, thread_id=thread_id, limit=limit)
    # logger.info(f"chat_history: {chat_history}")

    messages : List[Dict[str, Any]] = []
    if limit > 0 or limit is not None:
        for chat in chat_history[::-1]:  # 오래된 것부터 추가
            messages.append({
                "role": "user", 
                "content": chat["prompt"],
                "sentAt": chat['created_at'],
                "sources": None
            })
            # response는 문자열 -> text만 추출
            assistant_text = chat["response"]
            try:
                assistant_text = json.loads(assistant_text).get("text", assistant_text)
            except Exception:
                pass
            messages.append({
                "role": "assistant", 
                "content": assistant_text, 
                "sentAt": chat['created_at'],
                "sources": None
            })

    # logger.info(f"messages: {messages}")

    return messages