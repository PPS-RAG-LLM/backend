from pydantic import BaseModel, Field
from typing import List, Optional
from fastapi import APIRouter, Path, Query, Body
from starlette.responses import StreamingResponse
from repository.workspace import get_workspace_by_workspace_id
from service.users.chat import (
    stream_chat_for_qa,
    stream_chat_for_doc_gen,
    stream_chat_for_summary,
)
from service.commons.doc_gen_templates import get_doc_gen_template
from service.users.chat.common.validators import preflight_stream_chat_for_workspace
from utils import logger
from errors import BadRequestError
import time, json

logger = logger(__name__)

chat_router = APIRouter(tags=["workspace_chat"], prefix="/v1/workspace")

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

@chat_router.post(
    "/{slug}/thread/{thread_slug}/stream-chat", summary="QnA 스트리밍 채팅 실행"
)
def stream_chat_qa_endpoint(
    category: str = Query("qa"),
    slug: str = Path(..., description="워크스페이스 슬러그"),
    thread_slug: str = Path(..., description="채팅 스레드 슬러그"),
    body: StreamChatRequest = Body(..., description="채팅 요청 본문"),
):
    user_id = 3
    logger.info(f"\n\n[stream_chat_qa_endpoint] \n\n{body}\n\n")

    gen = stream_chat_for_qa(
        user_id=user_id,
        slug=slug,
        thread_slug=thread_slug,
        category=category,
        body=body.model_dump(exclude_unset=True),
    )
    return StreamingResponse(to_see(gen), media_type="text/event-stream; charset=utf-8")



def to_see(gen):
    logger.info("[stream_chat_qa_endpoint] streaming start")
    buf = []
    last_flush = time.monotonic()
    for chunk in gen:
        if not chunk:
            continue
        if not buf:
            chunk = chunk.lstrip()
        buf.append(chunk)
        text = "".join(buf)
        if (
            len(text) >= 32
            or text.endswith((" ","\n", ".", "?", "!", "…", "。", "！", "？"))
            or time.monotonic() - last_flush > 0.2
        ):
            # logger.debug(f"[flush] {repr(text)}")
            yield f'data: {json.dumps({"content": chunk})}\n\n'
            buf.clear()
            last_flush = time.monotonic()
    if buf:
        text = "".join(buf)
        yield f'data: {json.dumps({"content": chunk})}\n\n'
    logger.info("[stream_chat_qa_endpoint] streaming end")

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
    user_id = 3
    tmpl = get_doc_gen_template(int(body.templateId))
    if not tmpl:
        raise BadRequestError("유효하지 않은 templateId 입니다.")
    allowed_keys = {str(v.get("key") or "") for v in (tmpl.get("variables") or []) if v.get("key")}
    variables = [item.key.strip() for item in (body.variables or []) if (item.key or "").strip()]
    provided_keys = {key for key, _ in variables}
    missing = {key for key in allowed_keys if key and key not in provided_keys}
    if missing:
        raise BadRequestError(f"필수 변수 누락: {sorted(missing)}")
    filtered_vars = {key: value for key, value in variables if key in allowed_keys}

    message = (body.userPrompt or "").strip() or "요청된 템플릿에 따라 문서를 작성해 주세요."
    gen = stream_chat_for_doc_gen(
        user_id=user_id,
        slug=slug,
        category="doc_gen",
        body={
            "provider": body.provider,
            "model": body.model,
            "message": message,
            "mode": "chat",
            "attachments": [a.model_dump() for a in body.attachments],
            "systemPrompt": body.systemPrompt,
            "templateId": body.templateId,
            "templateVariables": filtered_vars,
        },
    )
    return StreamingResponse(to_see(gen), media_type="text/event-stream; charset=utf-8")




# @chat_router.post("/{slug}/summary", summary="문서 요약 실행")
# def summary_endpoint(
#     slug: str = Path(..., description="워크스페이스 슬러그"),
#     body: SummaryRequest = Body(..., description="요약 요청 (시스템프롬프트/내용/요청사항)"),
# ):
#     user_id = 3
#     # category 고정: summary
#     preflight_stream_chat_for_workspace(
#         user_id=user_id,
#         slug=slug,
#         category="summary",
#         body={
#             "mode": "chat",
#         },
#         thread_slug=None,
#     )
#     # message 빌드: 사용자 프롬프트만 사용
#     message = body.userPrompt.strip()
#     gen = stream_chat_for_workspace(
#         user_id=user_id,
#         slug=slug,
#         category="summary",
#         body={
#             "message": message,
#             "mode": "chat",
#             "attachments": [a.model_dump() for a in body.attachments],
#             "systemPrompt": body.systemPrompt,
#         },
#     )
#     acc = []
#     for chunk in gen:
#         if chunk:
#             acc.append(chunk)
#     return {"text": "".join(acc)}



# @chat_router.post("/{slug}/doc-gen", summary="문서 생성 실행" )
# def doc_gen_endpoint(
#     slug: str = Path(..., description="워크스페이스 슬러그"),
#     body: DocGenRequest = Body(..., description="문서 생성 요청 (템플릿/변수/요청사항)"),
# ):
#     user_id = 3
#     # 템플릿 변수 검증: PromptMapping에 있는 변수만 허용, 누락 시 400
#     tmpl = get_doc_gen_template(int(body.templateId))
#     if not tmpl:
#         raise BadRequestError("유효하지 않은 templateId 입니다.")
#     allowed_keys = {str(v.get("key") or "") for v in (tmpl.get("variables") or []) if v.get("key")}
#     provided_keys = set((body.variables or {}).keys())
#     missing = allowed_keys - provided_keys
#     if missing:
#         raise BadRequestError(f"필수 변수 누락: {sorted(missing)}")
#     filtered_vars = {k: v for k, v in (body.variables or {}).items() if k in allowed_keys}

#     preflight_stream_chat_for_workspace(
#         user_id=user_id,
#         slug=slug,
#         category="doc_gen",
#         body={
#             "mode": "chat",
#         },
#         thread_slug=None,
#     )
#     # message: 요청사항을 사용자 메시지로 전달
#     message = (body.userPrompt or "").strip() or "요청된 템플릿에 따라 문서를 작성해 주세요."
#     gen = stream_chat_for_workspace(
#         user_id=user_id,
#         slug=slug,
#         category="doc_gen",
#         body={
#             "message": message,
#             "mode": "chat",
#             "attachments": [a.model_dump() for a in body.attachments],
#             "systemPrompt": body.systemPrompt,
#             "templateId": body.templateId,
#             "templateVariables": filtered_vars,
#         },
#     )
#     acc = []
#     for chunk in gen:
#         if chunk:
#             acc.append(chunk)
#     return {"text": "".join(acc)}
