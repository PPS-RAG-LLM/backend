import uuid, json, itertools, time
from fastapi                   import APIRouter, Path, Body, HTTPException, Query
from fastapi.responses         import StreamingResponse, JSONResponse
from pydantic                  import BaseModel, Field
from typing                    import List, Optional, Generator, Literal
from service.users.workspace   import WorkspaceService
from errors                    import *
from config                    import config

class Attachment(BaseModel):
    name: str
    mime: str
    contentString: str
class ChatRequest(BaseModel):
    model: str                              = config.get("default_model")
    message: str
    mode: Optional[str]                     = "chat"
    sessionId: Optional[str]                = ""
    attachments: Optional[List[Attachment]] = []
    reset: Optional[bool]                   = False
class WorkspaceCreateRequest(BaseModel):
    name: str
    similarityThreshold: float          = 0.5
    openAITemp: float                   = 0.7
    openAiHistory: int                  = 20
    openAiPrompt: Optional[str]         = None
    queryRefusalResponse : Optional[str]= None
    chatMode: str                       = "chat"
    topN: int                           = 4

class WorkspaceUpdateRequest(BaseModel):
    name: Optional[str] = None
    openAiTemp: Optional[float] = None
    openAiHistory: Optional[int] = None
    openAiPrompt: Optional[str] = None

router = APIRouter(tags=["workspace"], prefix="/v1/workspace")


@router.get("/")
async def get_all_workspaces_endpoint():
    """모든 워크스페이스 목록 조회"""
    try:
        # API 키 검증 (403 에러)
        # TODO: 실제 API 키 검증 로직 추가
        
        workspaces = WorkspaceService.get_all_workspaces()
        
        return JSONResponse({
            "workspaces": workspaces
        }, status_code=200)
        
    except Exception as e:
        raise InternalServerError(f"워크스페이스 목록 조회 중 오류 발생: {str(e)}")


@router.post("/new")
async def create_new_workspace(request: WorkspaceCreateRequest = Body(...)):
    """ 새 워크스페이스 생성"""
    workspace = WorkspaceService.create_workspace(
        name=request.name,
        similarity_threshold=request.similarityThreshold,
        open_ai_temp=request.openAITemp,
        open_ai_history=request.openAiHistory,
        system_prompt=request.openAiPrompt,
        query_refusal_response=request.queryRefusalResponse,
        chat_mode=request.chatMode,
        top_n=request.topN,
    )

    return JSONResponse({
        "workspace": workspace,
        "message": "워크스페이스가 생성되었습니다"
    }, status_code=201)


@router.get("/{slug}")
async def get_workspace(slug: str = Path(..., description="워크스페이스 slug")):
    """워크스페이스 정보 조회"""
    workspace = WorkspaceService.get_workspace_by_slug(slug)
    return JSONResponse({
        "workspace": workspace
    }, status_code=200)



@router.delete("/{slug}")
async def delete_workspace_endpoint(
    slug: str = Path(..., description = "삭제할 워크 스페이스 slug")
):
    result = WorkspaceService.delete_workspace(slug)
    return JSONResponse(result)


@router.get("/{slug}/chats")
async def get_workspace_chats(
    slug: str = Path(..., description="워크스페이스 slug"),
    apiSessionId: Optional[str] = Query(None, description="API 세션 ID"),
    limit: Optional[int] = Query(20, description="조회할 채팅 수"),
    orderBy: Optional[str] = Query("createdAt", description="정렬 기준")
):
    """워크스페이스 채팅 히스토리 조회"""
    effective_limit = limit if limit is not None else 20
    effective_order_by = orderBy if orderBy is not None else "createdAt"
    
    history = WorkspaceService.get_workspace_chat_history(
        slug, apiSessionId, effective_limit, effective_order_by
    )
    
    return JSONResponse({
        "history": history
    }, status_code=200)


# 기존 라우터에 새 엔드포인트 추가
@router.post("/{slug}/update")
async def update_workspace_endpoint(
    slug: str = Path(..., description="워크스페이스 slug"),
    request: WorkspaceUpdateRequest = Body(...)
):
    """워크스페이스 설정 업데이트"""
    # API 키 검증 (403 에러)
    # TODO: 실제 API 키 검증 로직 추가
    
    update_data = request.model_dump(exclude_unset=True)
    workspace = WorkspaceService.update_workspace(slug, update_data)
    
    return JSONResponse({
        "workspace": workspace,
        "message": None
    }, status_code=200)


@router.post("/{slug}/stream-chat")
async def chat_endpoint(
    slug        : str         = Path(..., description="workspace slug"),
    chat_request: ChatRequest = Body(...)
):
    """
    • slug(path):     워크스페이스 식별자(workspaceID or slug)
    • chat_request:   명세서에 정의된 요청 body
    """
    try:
        # 서비스에서 스트림 제너레이터 생성
        stream_generator = WorkspaceService.stream_chat(
            slug=slug,
            model=chat_request.model,
            message=chat_request.message,
            session_id=chat_request.sessionId,
            attachments=chat_request.attachments,
            mode=chat_request.mode,
            reset=chat_request.reset
        )

        return StreamingResponse(
            stream_generator, 
            media_type="text/plain; charset=utf-8",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
        )

    except Exception as e:
        return JSONResponse({
            "id"          : None,
            "type"        : "abort",
            "textResponse": "",
            "sources"     : [],
            "close"       : True,
            "error"       : str(e)
        }, status_code=500)