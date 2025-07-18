from fastapi                   import APIRouter, Path, Body
from fastapi.responses         import StreamingResponse, JSONResponse
from pydantic                  import BaseModel
from typing                    import List, Optional, Generator, Literal
from src.models.base           import model_factory
from src.core.workspace        import get_chat_history, save_chat
import uuid, json, itertools


class Attachment(BaseModel):
    name: str
    mime: str
    contentString: str
class ChatRequest(BaseModel):
    model: str = "gpt-4o-mini"
    message: str
    mode: Optional[str]                 = "chat"
    sessionId: Optional[str]            = None
    attachments: Optional[List[Attachment]] = []
    reset: Optional[bool]               = False


router = APIRouter(tags=["workspace"], prefix="/v1/workspace")


@router.post("/{slug}/stream-chat")
async def chat_endpoint(
    slug        : str         = Path(..., description="workspace slug"),
    chat_request: ChatRequest = Body(...)
):
    """
    • slug(path):     워크스페이스 식별자
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
            try:
                for chunk in generator:
                    full_response.append(chunk)
                    yield json.dumps({
                        "id"          : chat_id,
                        "type"        : "textResponse",
                        "textResponse": chunk,
                        "sources"     : sources,
                        "close"       : False,
                        "error"       : error_msg
                    }) + "\n"
            finally:
                # 마지막 패킷(close=True)
                yield json.dumps({
                    "id"          : chat_id,
                    "type"        : "textResponse",
                    "textResponse": "",
                    "sources"     : sources,
                    "close"       : True,
                    "error"       : error_msg
                }) + "\n"

                # 5) DB 저장 (오류가 나도 스트림은 유지)
                try:
                    # workspace_id / user_id 는 예시값(0) → 실제 로직에 맞춰 수정
                    save_chat(workspace_id=0,
                              prompt      = chat_request.message,
                              response    = "".join(full_response),
                              user_id     = 0,
                              session_id  = chat_request.sessionId or "")
                except Exception:
                    pass  # 저장 실패는 서비스 중단 없이 무시

        return StreamingResponse(event_stream(), media_type="application/json")

    except Exception as e:
        return JSONResponse({
            "id"          : None,
            "type"        : "abort",
            "textResponse": "",
            "sources"     : [],
            "close"       : True,
            "error"       : str(e)
        }, status_code=500)