from pydantic import BaseModel, Field
from typing import List, Optional
from fastapi import APIRouter, Depends, Path, Query, Body
from starlette.responses import StreamingResponse
from repository.workspace import get_workspace_by_workspace_id
from service.users.chat import (
    stream_chat_for_qna,
    stream_chat_for_doc_gen,
    stream_chat_for_summary,
)
from service.commons.doc_gen_templates import get_doc_gen_template
from utils import logger
from errors import BadRequestError
import time, json

logger = logger(__name__)

chat_router = APIRouter(tags=["Workspace Chat"], prefix="/v1/workspace")

# 채팅
class Attachment(BaseModel):
    name: str  # 파일명
    mime: str  # image/png, image/jpeg, application/pdf, etc.
    contentString: str  # data:image/png;base64,...


class StreamChatRequest(BaseModel):
    provider: Optional[str] = None
    model: Optional[str] = None
    message: str
    mode: Optional[str] = Field(None, pattern="^(chat|query)$")
    sessionId: Optional[str] = None
    attachments: List[Attachment] = Field(default_factory=list)
    reset: Optional[bool] = False
    rag_enabled: Optional[bool] = True

@chat_router.post(
    "/{slug}/thread/{thread_slug}/stream-chat", summary="QnA 스트리밍 채팅 실행"
)
def stream_chat_qna_endpoint(
    category: str = Query("qna"),
    slug: str = Path(..., description="워크스페이스 슬러그"),
    thread_slug: str = Path(..., description="채팅 스레드 슬러그"),
    body: StreamChatRequest = Body(..., description="채팅 요청 본문"),
):
    user_id = 3
    logger.info(f"\n\n[stream_chat_qna_endpoint] \n\n{body}\n\n")

    gen = stream_chat_for_qna(
        user_id=user_id,
        slug=slug,
        thread_slug=thread_slug,
        category=category,
        body=body.model_dump(exclude_unset=True),
    )
    return StreamingResponse(to_see(gen), media_type="text/event-stream; charset=utf-8")



def to_see(gen):
    logger.info("[stream_chat_qna_endpoint] streaming start")
    buf = []
    last_flush = time.monotonic()
    chat_id = None
    full_response = []

    for chunk in gen:
        if not chunk:
            continue
        if chunk.startswith("__CHAT_ID__:"): 
            chat_id = chunk.split(":",1)[1] # 채팅 ID 추출
            continue
        if not buf:
            chunk = chunk.lstrip()
        buf.append(chunk)
        full_response.append(chunk)
        text = "".join(buf)
        if (
            len(text) >= 32
            or text.endswith((" ","\n", ".", "?", "!", "…", "。", "！", "？"))
            or time.monotonic() - last_flush > 0.2
        ):
            # logger.debug(f"[flush] {repr(text)}")
            yield f'{json.dumps({"content": chunk})}\n\n'
            buf.clear()
            last_flush = time.monotonic()
    if buf:
        text = "".join(buf)
        yield f'{json.dumps({"content": chunk})}\n\n'
    if chat_id:
        yield f'{json.dumps({"chat_id": chat_id, "done":True})}\n\n'

    complete_response = "".join(full_response)
    logger.info("======================\n\n[stream_chat_qna_endpoint] streaming end - Full response "
    f"({len(complete_response)} chars): \n\n{complete_response}\n\n======================\n")


# ====== Unified POST APIs ======
class SummaryRequest(BaseModel):
    provider: Optional[str] = None
    model: Optional[str] = None
    systemPrompt: Optional[str] = None
    originalText: Optional[str] = "텍스트원문"
    userPrompt: Optional[str] = "요청사항"

@chat_router.post("/{slug}/summary/stream", summary="문서 요약 실행 (스트리밍)")
def summary_stream_endpoint(
    slug: str = Path(..., description="워크스페이스 슬러그"),
    body: SummaryRequest = Body(..., description="요약 요청 (시스템프롬프트/내용/요청사항)"),
):
    user_id = 3
    from repository.workspace import get_workspace_id_by_slug_for_user
    workspace_id = get_workspace_id_by_slug_for_user(user_id, slug)

    # Preflight 검증
    ws = get_workspace_by_workspace_id(user_id, workspace_id)
    
    # originalText도 없고 워크스페이스에 문서도 없으면 에러
    if not body.originalText:
        from repository.documents import list_workspace_documents
        workspace_docs = list_workspace_documents(workspace_id)
        if not workspace_docs:
            raise BadRequestError("워크스페이스에 등록된 문서가 없고 originalText도 제공되지 않았습니다.")
    
    gen = stream_chat_for_summary(
        user_id=user_id,
        ws=ws,
        category="summary",
        body={
            "provider": body.provider,
            "model": body.model,
            "systemPrompt": body.systemPrompt,
            "originalText": body.originalText,
            "userPrompt": body.userPrompt,
        },
    )
    return StreamingResponse(to_see(gen), media_type="text/event-stream; charset=utf-8")


class VariableItem(BaseModel):
    key: str
    value: str

class DocGenRequest(BaseModel):
    provider: Optional[str] = None
    model: Optional[str] = None
    templateId: int
    systemPrompt: Optional[str] = None
    userPrompt: Optional[str] = None
    variables: List[VariableItem] = Field(default_factory=list)


@chat_router.post("/{slug}/doc-gen/stream", summary="문서 생성 실행 (스트리밍)")
def doc_gen_stream_endpoint(
    slug: str = Path(..., description="워크스페이스 슬러그"),
    body: DocGenRequest = Body(..., description="문서 생성 요청 (템플릿/변수/요청사항)"),
):
    from repository.workspace import get_workspace_id_by_slug_for_user
    user_id = 3
    tmpl = get_doc_gen_template(int(body.templateId))
    if not tmpl:
        raise BadRequestError("유효하지 않은 templateId 입니다.")
    allowed_keys = {str(v.get("key") or "") for v in (tmpl.get("variables") or []) if v.get("key")}
    variables = [(item.key.strip(), item.value) for item in (body.variables or []) if (item.key or "").strip()]
    provided_keys = {key for key, _ in variables}
    missing = {key for key in allowed_keys if key and key not in provided_keys}
    if missing:
        raise BadRequestError(f"필수 변수 누락: {sorted(missing)}")
    filtered_vars = {key: value for key, value in variables if key in allowed_keys}

    logger.debug(f"allowed_keys: {allowed_keys}")
    logger.debug(f"provided_keys: {provided_keys}")
    logger.debug(f"missing: {missing}")
    logger.debug(f"filtered_vars: {filtered_vars}")
    logger.debug(f"variables: {variables}")

    workspace_id = get_workspace_id_by_slug_for_user(user_id, slug)

    # Preflight 검증
    ws = get_workspace_by_workspace_id(user_id, workspace_id)
    
    message = (body.userPrompt or "").strip() or "요청된 템플릿에 따라 문서를 작성해 주세요."
    gen = stream_chat_for_doc_gen(
        user_id=user_id,
        ws=ws,
        category="doc_gen",
        body={
            "provider": body.provider,
            "model": body.model,
            "message": message,
            "mode": "chat",
            "systemPrompt": body.systemPrompt,
            "templateId": body.templateId,
            "templateVariables": filtered_vars,
        },
    )
    return StreamingResponse(to_see(gen), media_type="text/event-stream; charset=utf-8")


class UpdateMetricsRequest(BaseModel):
    reasoningDuration: float = Field(..., description="추론 시간 (초)", ge=0)

@chat_router.patch(
    "/{slug}/chat/{chat_id}/metrics",
    summary="채팅 메트릭 업데이트",
    description="프론트엔드에서 계산한 reasoning_duration을 업데이트"
)
def update_chat_metrics_endpoint(
    chat_id: int = Path(..., description="채팅 ID"),
    body: UpdateMetricsRequest = Body(..., description="채팅 메트릭 업데이트 요청"),
):
    """
    프론트엔드가 스트리밍 응답 중 계산한 reasoning_duration을 저장
    """
    from service.users.chat.response_metrics import update_reasoning_duration
    user_id = 3
    result = update_reasoning_duration(user_id, chat_id, body.reasoningDuration)
    return {"success": True, "data": result}