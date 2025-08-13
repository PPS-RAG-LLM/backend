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
from utils import logger
from errors import BadRequestError
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


@router.get("", response_model = WorkspaceListResponse)
def list_all_workspaces():
# def list_all_workspaces(user_id: int = Depends(get_user)):
    user_id = 1
    try:
        items = list_workspaces(user_id)
        return WorkspaceListResponse(workspaces=items)
    except Exception as e:
        logger.error({"list_workspaces_failed": str(e)})
        raise

@router_singular.post("/new", response_model=NewWorkspaceResponse)
def create_new_workspace(
    category: str = Query(..., description="qa | doc_gen | summary"),
    body: NewWorkspaceBody = ...,
):
    user_id = 1
    logger.debug({"category": category, "body": body.model_dump(exclude_unset=True)})
    try:
        result = create_workspace_for_user(user_id, category, body.model_dump(exclude_unset=True))
    except BadRequestError as e:
        logger.warning({"workspace_create_failed": e.message})
        raise
    logger.info({"workspace_created_id": result["id"]})
    return NewWorkspaceResponse(workspace=result, message="Workspace created")

@router_singular.get("/{slug}", response_model=WorkspaceDetailResponse)
def get_workspace_by_slug(slug: str):
    user_id = 1
    item = get_workspace_detail(user_id, slug)
    return item

@router_singular.delete("/{slug}")
def delete_workspace(slug: str):
    user_id = 1
    delete_workspace_service(user_id, slug)
    return {"message": "Workspace deleted"}

@router_singular.post("/{slug}/update", response_model=WorkspaceUpdateResponse)
def update_workspace(slug: str, body: WorkspaceUpdateBody):
    user_id = 1
    result = update_workspace_service(user_id, slug, body.model_dump(exclude_unset=True))
    return result

