from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from fastapi import APIRouter,Path,Query,Body
from service.users.chat_history import list_thread_chats_for_workspace
from utils import logger
from errors import BadRequestError

logger = logger(__name__)
thread_router = APIRouter(tags=["Workspace Thread"],prefix="/v1/workspace")

class ChatHistoryItem(BaseModel):
    role : str
    content : str
    sentAt : Optional[str] 
    sources : Optional[List[Dict[str, Any]]] = None

class ChatHistoryResponse(BaseModel):
    history : List[ChatHistoryItem]

@thread_router.get("/{slug}/thread/{thread_slug}/chats", summary="워크스페이스에서 쓰레드채팅 히스토리 조회")
def get_workspace_thread_chats(
    slug : str = Path (..., description="워크스페이스 슬러그"),
    thread_slug : str = Path (..., description="쓰레드 슬러그"),
    ):
    user_id = 3
    history = list_thread_chats_for_workspace(
        user_id=user_id,
        slug=slug,
        thread_slug=thread_slug,
    )
    return ChatHistoryResponse(history=history)
    