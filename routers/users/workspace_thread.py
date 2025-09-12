from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from fastapi import APIRouter,Path,Query,Body
from service.users.chat_history import list_thread_chats_for_workspace
from service.users.workspace_thread import update_thread_name_for_workspace
from utils import logger
from errors import BadRequestError

logger = logger(__name__)
thread_router = APIRouter(tags=["Workspace Thread"],prefix="/v1/workspace")


########
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
    


########
class ThreadUpdateRequest(BaseModel):
    name: str = Field(..., min_length=1, description="새 스레드 이름")

class ThreadUpdateResponse(BaseModel):
    thread: Dict[str, Any]
    message: Optional[str] = None

@thread_router.post("/{slug}/thread/{thread_slug}/update", summary="워크스페이스의 스레드 이름 업데이트")
def update_workspace_thread_name(
    slug: str = Path(..., description="워크스페이스 슬러그"),
    thread_slug: str = Path(..., description="쓰레드 슬러그"),
    payload: ThreadUpdateRequest = Body(..., description="업데이트할 스레드 이름"),
):
    user_id = 3
    result = update_thread_name_for_workspace(
        user_id=user_id,
        slug=slug,
        thread_slug=thread_slug,
        name=payload.name,
    )
    return result