import uuid, json, itertools, time
from fastapi                   import APIRouter, Path, Body, HTTPException, Query
from fastapi.responses         import StreamingResponse, JSONResponse
from pydantic                  import BaseModel, Field
from typing                    import List, Optional, Generator, Literal
from src.models.base           import model_factory
from src.core.workspace_chat   import *
from src.core.workspace_main   import *
from src.errors                import *
from src.config                import config

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
        
        workspaces = get_all_workspaces()
        
        return JSONResponse({
            "workspaces": workspaces
        }, status_code=200)
        
    except Exception as e:
        raise InternalServerError(f"워크스페이스 목록 조회 중 오류 발생: {str(e)}")


@router.post("/new")
async def create_new_workspace(request: WorkspaceCreateRequest = Body(...)):
    """ 새 워크스페이스 생성"""
    try:
        workspace = create_workspace(
            name=request.name,
            similarity_threshold=request.similarityThreshold,  # 그대로 전달
            open_ai_temp=request.openAITemp,
            open_ai_history=request.openAiHistory,
            system_prompt=request.openAiPrompt or "",  # None 가능
            query_refusal_response=request.queryRefusalResponse or "",
            chat_mode=request.chatMode,
            top_n=request.topN,
        )

        return JSONResponse({
            "workspace": {
                "id"            : workspace["id"],
                "name"          : workspace["name"],
                "slug"          : workspace["slug"],
                "createdAt"     : workspace["createdAt"],
                "lastUpdatedAt" : workspace["lastUpdatedAt"],
                "openAiPrompt"  : workspace["openAiPrompt"],
                "openAiHistory" : workspace["openAiHistory"],
                "openAiTemp"    : workspace["openAiTemp"]
            },
            "message": "워크스페이스가 생성되었습니다"
        }, status_code=201)
    except ValueError as e:
        # 이미 존재하는 slug 등의 경우
        if "이미 사용중인 slug" in str(e):
            raise WorkspaceAlreadyExistsError(str(e).split(":")[1].strip())
        raise BadRequestError(str(e))
    except Exception as e:
        raise InternalServerError(f"워크스페이스 생성 중 오류 발생: {str(e)}")


@router.get("/{slug}")
async def get_workspace(slug: str = Path(..., description="워크스페이스 slug")):
    """워크스페이스 정보 조회"""
    try:
        workspace = get_workspace_by_slug(slug)
        if not workspace:
            raise HTTPException(status_code=404, detail="워크스페이스를 찾을 수 없습니다")
            
        return JSONResponse({
            "workspace": workspace
        }, status_code=200)
        
    except WorkspaceNotFoundError:
        raise  # 그대로 전파
    except Exception as e:
        raise InternalServerError(f"워크스페이스 조회 중 오류 발생: {str(e)}")



@router.delete("/{slug}")
async def delete_workspace_endpoint(
    slug: str = Path(..., description = "삭제할 워크 스페이스 slug")
):
    try:
        delete_workspace(slug)
        return JSONResponse({
            "success":True,
            "message":f"워크스페이스 '{slug}'가 성공적으로 삭제완료되었습니다."
        })
    except ValueError as e:
        raise WorkspaceNotFoundError(slug)      # 404 Not Found (워크스페이스가 존재하지 않음)
    except Exception as e:
        raise DatabaseError(f"워크스페이스 삭제 중 오류 발생: {str(e)}")


@router.get("/{slug}/chats")
async def get_workspace_chats(
    slug: str = Path(..., description="워크스페이스 slug"),
    apiSessionId: Optional[str] = Query(None, description="API 세션 ID"),
    limit: Optional[int] = Query(20, description="조회할 채팅 수"),
    orderBy: Optional[str] = Query("createdAt", description="정렬 기준")
):
    """워크스페이스 채팅 히스토리 조회"""
    try:
        if limit and (limit <= 0 or limit > 100):
            raise BadRequestError("limit는 1~100 사이의 값이어야 합니다.")
        effective_limit = limit if limit is not None else 20
        effective_order_by = orderBy if orderBy is not None else "createdAt"
        history = get_workspace_chat_history(slug, apiSessionId, effective_limit, effective_order_by)
        return JSONResponse({
            "history": history
        }, status_code=200)
        
    except BadRequestError:
        raise
    except Exception as e:
        raise InternalServerError(f"채팅 히스토리 조회 중 오류 발생: {str(e)}")


# 기존 라우터에 새 엔드포인트 추가
@router.post("/{slug}/update")
async def update_workspace_endpoint(
    slug: str = Path(..., description="워크스페이스 slug"),
    request: WorkspaceUpdateRequest = Body(...)
):
    """워크스페이스 설정 업데이트"""
    try:
        # API 키 검증 (403 에러)
        # TODO: 실제 API 키 검증 로직 추가
        
        # slug 검증
        if not slug or slug.strip() == "":
            raise BadRequestError("유효한 워크스페이스 slug가 필요합니다")
        
        # 업데이트할 데이터 준비
        update_data = request.model_dump(exclude_unset=True)
        if not update_data:
            raise BadRequestError("업데이트할 데이터가 없습니다")
        
        # 데이터 유효성 검증
        if "openAiTemp" in update_data and update_data["openAiTemp"] is not None:
            if not (0.0 <= update_data["openAiTemp"] <= 2.0):
                raise BadRequestError("openAiTemp는 0.0과 2.0 사이의 값이어야 합니다")
        
        if "openAiHistory" in update_data and update_data["openAiHistory"] is not None:
            if not (1 <= update_data["openAiHistory"] <= 1000):
                raise BadRequestError("openAiHistory는 1과 1000 사이의 값이어야 합니다")
        
        if "name" in update_data and update_data["name"] is not None:
            if not update_data["name"].strip():
                raise BadRequestError("워크스페이스 이름은 비어있을 수 없습니다")
        
        # 워크스페이스 업데이트
        workspace = update_workspace(slug, update_data)
        
        return JSONResponse({
            "workspace": workspace,
            "message": None
        }, status_code=200)
        
    except ValueError as e:
        if "찾을 수 없습니다" in str(e): # 워크스페이스를 찾을 수 없는 경우
            raise NotFoundError(str(e))
        raise BadRequestError(str(e))
    except BadRequestError:
        raise
    except NotFoundError:
        raise
    except Exception as e:
        raise InternalServerError(f"워크스페이스 업데이트 중 오류 발생: {str(e)}")


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
        # 1) 이전 대화 히스토리 + 이번 질문 messages 생성
        messages = get_chat_history(slug, chat_request.sessionId or "")
        messages.append({"role": "user", "content": chat_request.message})

        # 2) 모델 인스턴스 생성
        chat_model = model_factory(chat_request.model)
        generator  = chat_model.stream_chat(messages)

        chat_id   = str(uuid.uuid4())
        sources   = []
        error_msg = None

        # 4) 스트리밍 응답 제너레이터
        def event_stream() -> Generator[str, None, None]:
            full_response = []
            start_time    = time.time()
            token_count   = 0
            try:
                for chunk in generator:
                    if not chunk or chunk.strip() == "":        # 빈 청크는 스킵
                        continue

                    full_response.append(chunk)
                    token_count += 1

                    yield json.dumps({
                        "id"          : chat_id,
                        "type"        : "textResponse",
                        "textResponse": chunk,
                        "sources"     : sources,
                        "close"       : False,
                        "error"       : error_msg
                    }, ensure_ascii=False) + "\n"        # 한국어 정상 출력을 위해 ensure_ascii=False 추가
            finally:
                duration = time.time() - start_time
                # 마지막 패킷(close=True)
                yield json.dumps({
                    "id"          : chat_id,
                    "type"        : "textResponse",
                    "textResponse": "",
                    "sources"     : sources,
                    "close"       : True,
                    "error"       : error_msg
                }, ensure_ascii=False) + "\n"

                # 5) DB 저장 (오류가 나도 스트림은 유지)
                try:
                    response_text = "".join(full_response)
                    metrics = {
                        "completion_tokens" : len(response_text.split()),        # 단어 수로 추정
                        "prompt_tokens"     : len(chat_request.message.split()),
                        "total_tokens"      : len(response_text.split()) + len(chat_request.message.split()),
                        "model"             :chat_request.model,
                        "outputTps"         : token_count / duration if duration > 0 else 0,
                        "duration"          :round(duration, 3)
                    }
                    save_chat(
                        workspace_id= get_workspace_id_from_slug(slug),
                        prompt      = chat_request.message,
                        response    = "".join(full_response),
                        user_id     = 0, # 실제 사용자 ID로 변경
                        session_id  = chat_request.sessionId or "",
                        sources     = sources,
                        attachments = chat_request.attachments or [],
                        metrics     = metrics
                        )
                except Exception as e:
                    print(f"채팅 저장 오류: {e}")  # 로깅으로 대체 가능

        return StreamingResponse(
            event_stream(), 
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