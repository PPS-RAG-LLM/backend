from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from fastapi import APIRouter,Path,Query,Body
from service.users.chat_history import list_thread_chats_for_workspace
from service.users.workspace_thread import update_thread_name_for_workspace, create_new_workspace_thread_for_workspace, delete_workspace_thread_for_workspace
from utils import logger
from errors import BadRequestError

logger = logger(__name__)
thread_router = APIRouter(tags=["Workspace Thread"],prefix="/v1/workspace")


########
class ChatHistoryItem(BaseModel):
    chatId : int
    role : str
    content : str
    sentAt : Optional[str] 
    reasoningDuration : Optional[float] = None
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

########
class ThreadInfo(BaseModel):
    id: int
    name: str
    threadSlug: str
    workspaceId: int

class NewThreadResponse(BaseModel):
    thread: ThreadInfo
    message: Optional[str] = None

class NewWorkspaceThreadRequest(BaseModel):
    name: str = Field(..., min_length=1, description="새 스레드 이름")

@thread_router.post("/{slug}/new-thread", summary="워크스페이스에 새로운 스레드 생성", response_model=NewThreadResponse)
def create_new_workspace_thread(
    slug: str = Path(..., description="워크스페이스 슬러그"),
    payload: NewWorkspaceThreadRequest = Body(..., description="새 스레드 이름"),
):
    user_id = 3
    result = create_new_workspace_thread_for_workspace(
        user_id=user_id,
        slug=slug,
        name=payload.name,
    )
    return result

@thread_router.delete("/{slug}/thread/{thread_slug}/delete", summary="스레드 삭제")
def delete_workspace_thread(
    slug: str = Path(..., description="워크스페이스 슬러그"),
    thread_slug: str = Path(..., description="쓰레드 슬러그"),
):
    user_id = 3
    result = delete_workspace_thread_for_workspace(
        user_id=user_id,
        slug=slug,
        thread_slug=thread_slug,
    )
    return result