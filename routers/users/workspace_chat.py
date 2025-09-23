from pydantic import BaseModel, Field
from typing import List, Optional
from fastapi import APIRouter, Path, Query, Body
from starlette.responses import StreamingResponse
from service.users.chat import (
    stream_chat_for_workspace,
    preflight_stream_chat_for_workspace,
)
from service.users.doc_gen_templates import get_doc_gen_template
from utils import logger
from errors import BadRequestError
import time

logger = logger(__name__)
chat_router = APIRouter(tags=["workspace_chat"], prefix="/v1/workspace")


# 채팅
class Attachment(BaseModel):
    name: str  # 파일명
    mime: str  # image/png, image/jpeg, application/pdf, etc.
    contentString: str  # data:image/png;base64,...


class StreamChatRequest(BaseModel):
    message: str
    mode: Optional[str] = Field(None, pattern="^(chat|query)$")
    sessionId: Optional[str] = None
    attachments: List[Attachment] = Field(default_factory=list)
    reset: Optional[bool] = False
    # 템플릿 통합용(선택): doc_gen/summary 전용
    templateId: Optional[int] = None
    templateVariables: dict[str, str] = Field(default_factory=dict)
    templateText: Optional[str] = None
    # 시스템 프롬프트 오버라이드(선택)
    systemPrompt: Optional[str] = None


@chat_router.post("/{slug}/stream-chat", summary="워크스페이스에서 스트리밍 채팅 실행")
def stream_chat_endpoint(
    category: str = Query(..., description="doc_gen | summary"),
    slug: str = Path(..., description="워크스페이스 슬러그"),
    body: StreamChatRequest = Body(..., description="채팅 요청 본문"),
):
    user_id = 3
    if category in ["doc_gen", "summary"]:
        # 스트리밍 시작 전에 모든 검증 수행 -> 예외는 여기서 FastAPI 핸들러로 감
        preflight_stream_chat_for_workspace(
            user_id=user_id,
            slug=slug,
            category=category,
            body=body.model_dump(exclude_unset=True),
            thread_slug=None,
        )
        gen = stream_chat_for_workspace(
            user_id=user_id,
            slug=slug,
            category=category,
            body=body.model_dump(exclude_unset=True),
        )
        return StreamingResponse(to_see(gen), media_type="text/event-stream")
    else:
        raise BadRequestError(
            "qa category should be used in another endpoint > '/v1/workspace/{slug}/thread/{thread_slug}/stream-chat'"
        )


@chat_router.post(
    "/{slug}/thread/{thread_slug}/stream-chat", summary="QnA 스트리밍 채팅 실행"
)
def stream_chat_qa_endpoint(
    category: str = Query(..., description="only qa"),
    slug: str = Path(..., description="워크스페이스 슬러그"),
    thread_slug: str = Path(..., description="채팅 스레드 슬러그"),
    body: StreamChatRequest = Body(..., description="채팅 요청 본문"),
):
    user_id = 3
    logger.info(f"\n\n[stream_chat_qa_endpoint] \n\n{body}\n\n")
    # 스트리밍 시작 전에 검증
    preflight_stream_chat_for_workspace(
        user_id=user_id,
        slug=slug,
        category=category,
        body=body.model_dump(exclude_unset=True),
        thread_slug=thread_slug,
    )
    gen = stream_chat_for_workspace(
        user_id=user_id,
        slug=slug,
        thread_slug=thread_slug,
        category=category,
        body=body.model_dump(exclude_unset=True),
    )
    return StreamingResponse(to_see(gen), media_type="text/event-stream")


def to_see(gen):
    buf = []
    last_flush = time.monotonic()
    for chunk in gen:
        if not chunk:
            continue
        logger.debug(f"[raw_chunk] {repr(chunk)}")
        if not buf:
            chunk = chunk.lstrip()
        buf.append(chunk)
        text = "".join(buf)
        if (
            len(text) >= 32
            or text.endswith((" ", "\n", ".", "?", "!", "…", "。", "！", "？"))
            or time.monotonic() - last_flush > 0.2
        ):
            # logger.info(f"[flush] {repr(text)}")
            yield f"data: {text}\n\n"
            buf.clear()
            last_flush = time.monotonic()
    if buf:
        text = "".join(buf)
        logger.info(f"[flush-end] {repr(text)}")
        yield f"data: {text}\n\n"

# ====== Unified POST APIs ======
class SummaryRequest(BaseModel):
    systemPrompt: Optional[str] = None
    userPrompt: str
    attachments: List[Attachment] = Field(default_factory=list)


@chat_router.post("/{slug}/summary", summary="문서 요약 실행")
def summary_endpoint(
    slug: str = Path(..., description="워크스페이스 슬러그"),
    body: SummaryRequest = Body(..., description="요약 요청 (시스템프롬프트/내용/요청사항)"),
):
    user_id = 3
    # category 고정: summary
    preflight_stream_chat_for_workspace(
        user_id=user_id,
        slug=slug,
        category="summary",
        body={
            "mode": "chat",
        },
        thread_slug=None,
    )
    # message 빌드: 사용자 프롬프트만 사용
    message = body.userPrompt.strip()
    gen = stream_chat_for_workspace(
        user_id=user_id,
        slug=slug,
        category="summary",
        body={
            "message": message,
            "mode": "chat",
            "attachments": [a.model_dump() for a in body.attachments],
            "systemPrompt": body.systemPrompt,
        },
    )
    acc = []
    for chunk in gen:
        if chunk:
            acc.append(chunk)
    return {"text": "".join(acc)}


class DocGenRequest(BaseModel):
    systemPrompt: Optional[str] = None
    userPrompt: Optional[str] = None
    templateId: int
    variables: dict[str, str] = Field(default_factory=dict)
    attachments: List[Attachment] = Field(default_factory=list)


@chat_router.post("/{slug}/doc-gen", summary="문서 생성 실행" )
def doc_gen_endpoint(
    slug: str = Path(..., description="워크스페이스 슬러그"),
    body: DocGenRequest = Body(..., description="문서 생성 요청 (템플릿/변수/요청사항)"),
):
    user_id = 3
    # 템플릿 변수 검증: PromptMapping에 있는 변수만 허용, 누락 시 400
    tmpl = get_doc_gen_template(int(body.templateId))
    if not tmpl:
        raise BadRequestError("유효하지 않은 templateId 입니다.")
    allowed_keys = {str(v.get("key") or "") for v in (tmpl.get("variables") or []) if v.get("key")}
    provided_keys = set((body.variables or {}).keys())
    missing = allowed_keys - provided_keys
    if missing:
        raise BadRequestError(f"필수 변수 누락: {sorted(missing)}")
    filtered_vars = {k: v for k, v in (body.variables or {}).items() if k in allowed_keys}

    preflight_stream_chat_for_workspace(
        user_id=user_id,
        slug=slug,
        category="doc_gen",
        body={
            "mode": "chat",
        },
        thread_slug=None,
    )
    # message: 요청사항을 사용자 메시지로 전달
    message = (body.userPrompt or "").strip() or "요청된 템플릿에 따라 문서를 작성해 주세요."
    gen = stream_chat_for_workspace(
        user_id=user_id,
        slug=slug,
        category="doc_gen",
        body={
            "message": message,
            "mode": "chat",
            "attachments": [a.model_dump() for a in body.attachments],
            "systemPrompt": body.systemPrompt,
            "templateId": body.templateId,
            "templateVariables": filtered_vars,
        },
    )
    acc = []
    for chunk in gen:
        if chunk:
            acc.append(chunk)
    return {"text": "".join(acc)}


@chat_router.post("/{slug}/summary/stream", summary="문서 요약 실행 (스트리밍)")
def summary_stream_endpoint(
    slug: str = Path(..., description="워크스페이스 슬러그"),
    body: SummaryRequest = Body(..., description="요약 요청 (시스템프롬프트/내용/요청사항)"),
):
    user_id = 3
    preflight_stream_chat_for_workspace(
        user_id=user_id,
        slug=slug,
        category="summary",
        body={"mode": "chat"},
        thread_slug=None,
    )
    message = body.userPrompt.strip()
    gen = stream_chat_for_workspace(
        user_id=user_id,
        slug=slug,
        category="summary",
        body={
            "message": message,
            "mode": "chat",
            "attachments": [a.model_dump() for a in body.attachments],
            "systemPrompt": body.systemPrompt,
        },
    )
    return StreamingResponse(to_see(gen), media_type="text/event-stream")


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
    provided_keys = set((body.variables or {}).keys())
    missing = allowed_keys - provided_keys
    if missing:
        raise BadRequestError(f"필수 변수 누락: {sorted(missing)}")
    filtered_vars = {k: v for k, v in (body.variables or {}).items() if k in allowed_keys}

    preflight_stream_chat_for_workspace(
        user_id=user_id,
        slug=slug,
        category="doc_gen",
        body={"mode": "chat"},
        thread_slug=None,
    )
    message = (body.userPrompt or "").strip() or "요청된 템플릿에 따라 문서를 작성해 주세요."
    gen = stream_chat_for_workspace(
        user_id=user_id,
        slug=slug,
        category="doc_gen",
        body={
            "message": message,
            "mode": "chat",
            "attachments": [a.model_dump() for a in body.attachments],
            "systemPrompt": body.systemPrompt,
            "templateId": body.templateId,
            "templateVariables": filtered_vars,
        },
    )
    return StreamingResponse(to_see(gen), media_type="text/event-stream")



