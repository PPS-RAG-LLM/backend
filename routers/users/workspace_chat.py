from pydantic import BaseModel, Field
from typing import List, Optional
from fastapi import APIRouter, Path, Query, Body
from starlette.responses import StreamingResponse
from service.users.chat import (
    stream_chat_for_workspace,
    preflight_stream_chat_for_workspace,
)
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


@chat_router.post("/{slug}/chat", summary="워크스페이스 채팅 실행 (doc_gen/summary)")
def chat_endpoint(
    category: str = Query(..., description="doc_gen | summary"),
    slug: str = Path(..., description="워크스페이스 슬러그"),
    body: StreamChatRequest = Body(..., description="채팅 요청 본문"),
):
    user_id = 3
    if category not in ["doc_gen", "summary"]:
        raise BadRequestError(
            "doc_gen/summary 카테고리만 지원합니다. QnA는 '/v1/workspace/{slug}/thread/{thread_slug}/stream-chat' 사용"
        )
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
    acc = []
    for chunk in gen:
        if chunk:
            acc.append(chunk)
    return {"text": "".join(acc)}


# ====== Unified POST APIs ======
class SummaryRequest(BaseModel):
    systemPrompt: Optional[str] = None
    content: str
    request: Optional[str] = None
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
    # message 빌드: 요청사항 + 내용
    req_text = (body.request or "").strip()
    content_text = body.content.strip()
    message = (f"요약 요청사항: {req_text}\n\n아래 내용을 요약:\n{content_text}" if req_text else f"아래 내용을 요약:\n{content_text}")
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
    request: Optional[str] = None
    templateId: int
    variables: dict[str, str] = Field(default_factory=dict)
    attachments: List[Attachment] = Field(default_factory=list)


@chat_router.post("/{slug}/doc-gen", summary="문서 생성 실행" )
def doc_gen_endpoint(
    slug: str = Path(..., description="워크스페이스 슬러그"),
    body: DocGenRequest = Body(..., description="문서 생성 요청 (템플릿/변수/요청사항)"),
):
    user_id = 3
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
    message = (body.request or "").strip() or "요청된 템플릿에 따라 문서를 작성해 주세요."
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
            "templateVariables": body.variables,
        },
    )
    acc = []
    for chunk in gen:
        if chunk:
            acc.append(chunk)
    return {"text": "".join(acc)}



