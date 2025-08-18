from typing import Optional
from fastapi import APIRouter, Query, HTTPException, Depends
from pydantic import BaseModel, Field
from fastapi import File, UploadFile
from service.users.workspace import (
    create_workspace_for_user,
    list_workspaces,
    get_workspace_detail,
    delete_workspace as delete_workspace_service,
    update_workspace as update_workspace_service,
    upload_and_embed_document,
)
from typing import Dict, List, Any
from errors import BadRequestError
from utils import logger
from utils.auth import get_user_id_from_cookie

logger = logger(__name__)

router = APIRouter(tags=["workspace"], prefix="/v1/workspaces")
router_singular = APIRouter(tags=["workspace"], prefix="/v1/workspace")

class NewWorkspaceBody(BaseModel):
    name: str
    similarityThreshold: Optional[float] = Field(0.25)
    temperature: Optional[float] = Field(0.7)
    chatHistory: Optional[int] = Field(20)
    systemPrompt: Optional[str] = Field("")
    queryRefusalResponse: Optional[str] = Field("There is no information about this topic.")
    chatMode: Optional[str] = Field(None, pattern="^(chat|query)$")
    topN: Optional[int] = Field(4, gt=0)

class Workspace(BaseModel):
    id: int
    category: str
    name: str
    slug: str
    createdAt: str
    UpdatedAt: str
    temperature: float
    chatHistory: int
    systemPrompt: str
class NewWorkspaceResponse(BaseModel):
    workspace: Workspace
    message: str
class WorkspaceListItem(BaseModel):
    id: int
    category: str
    name: str
    slug: str
    createdAt: str
    UpdatedAt: str
    temperature: float
    chatHistory: int
    systemPrompt: str
    threads: List[Any] = []
class WorkspaceListResponse(BaseModel):
    workspaces: List[WorkspaceListItem]

class WorkspaceDetailResponse(BaseModel):
    id: int
    name: str
    category: str
    slug: str
    createdAt: str
    temperature: Optional[float] = None
    updatedAt: str
    chatHistory: int
    systemPrompt: Optional[str] = None
    documents: List[Any] = []
    threads: List[Any] = []
class WorkspaceUpdateBody(BaseModel):
    name: Optional[str] = None
    temperature: Optional[float] = None
    chatHistory: Optional[int] = None
    systemPrompt: Optional[str] = None
class WorkspaceUpdateResponse(BaseModel):
    workspace: WorkspaceDetailResponse
    message: Optional[str] = None


@router.get("", response_model = WorkspaceListResponse, summary="로그인한 사용자의 워크스페이스 목록 조회")
# def list_all_workspaces():
def list_all_workspaces(user_id: int = Depends(get_user_id_from_cookie)):
    """로그인한 사용자의 워크스페이스 목록 조회"""
    try:
        logger.info(f"list_all_workspaces: {user_id}")
        items = list_workspaces(user_id)  # 쿠키에서 자동으로 가져온 user_id 사용
        return WorkspaceListResponse(workspaces=items)
    except Exception as e:
        logger.error({"list_workspaces_failed": str(e)})
        raise

@router_singular.post("/new", response_model=NewWorkspaceResponse, summary="새로운 워크스페이스 생성")
def create_new_workspace(
    user_id: int = Depends(get_user_id_from_cookie),
    category: str = Query(..., description="qa | doc_gen | summary"),
    body: NewWorkspaceBody = ...,
):
    logger.debug({"category": category, "body": body.model_dump(exclude_unset=True)})
    try:
        result = create_workspace_for_user(user_id, category, body.model_dump(exclude_unset=True))
    except BadRequestError as e:
        logger.warning({"workspace_create_failed": e.message})
        raise
    logger.info({"workspace_created_id": result["id"]})
    return NewWorkspaceResponse(workspace=result, message="Workspace created")

@router_singular.get("/{slug}", response_model=WorkspaceDetailResponse, summary="워크스페이스 상세 조회")
def get_workspace_by_slug(slug: str):
    user_id = 1
    item = get_workspace_detail(user_id, slug)
    return item

@router_singular.delete("/{slug}", summary="워크스페이스 삭제")
def delete_workspace(slug: str):
    user_id = 1
    delete_workspace_service(user_id, slug)
    return {"message": "Workspace deleted"}

@router_singular.post("/{slug}/update", response_model=WorkspaceUpdateResponse, summary="워크스페이스 업데이트")
def update_workspace(slug: str, body: WorkspaceUpdateBody):
    user_id = 1
    result = update_workspace_service(user_id, slug, body.model_dump(exclude_unset=True))
    return result

from starlette.responses import StreamingResponse

class Attachment(BaseModel):
    name: str           # 파일명
    mime: str           # image/png, image/jpeg, application/pdf, etc.
    contentString: str  # data:image/png;base64,...

class StreamChatRequest(BaseModel):
    message: str
    mode: Optional[str] = Field(None, pattern="^(chat|query)$")
    sessionId: Optional[str] = None
    attachments: Optional[List[Attachment]] = []
    reset: Optional[bool] = False

@router_singular.post("/{slug}/stream-chat", summary="워크스페이스에서 스트리밍 채팅 실행")
def stream_chat_endpoint( slug : str , body: StreamChatRequest, user_id: int = Depends(get_user_id_from_cookie),):
    from service.users.chat import stream_chat_for_workspace
    def to_see(gen):
        for chunk in gen:
            yield f"data: {chunk}\n\n"
    gen = stream_chat_for_workspace(user_id, slug, body.model_dump(exclude_unset=True))
    return StreamingResponse(to_see(gen), media_type="text/event-stream")