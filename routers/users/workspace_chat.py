from pydantic import BaseModel, Field
from typing import List, Optional
from fastapi import APIRouter,Path,Query,Body
from starlette.responses import StreamingResponse
from service.users.chat import stream_chat_for_workspace
from utils import logger
from errors import BadRequestError
import time

logger = logger(__name__)
chat_router = APIRouter(tags=["workspace_chat"],prefix="/v1/workspace")

#### 채팅
class Attachment(BaseModel):
    name: str           # 파일명
    mime: str           # image/png, image/jpeg, application/pdf, etc.
    contentString: str  # data:image/png;base64,...
class StreamChatRequest(BaseModel):
    message: str
    mode: Optional[str] = Field(None, pattern="^(chat|query)$")
    sessionId: Optional[str] = None
    attachments: List[Attachment] = Field(default_factory=list)
    reset: Optional[bool] = False
    
@chat_router.post("/{slug}/stream-chat", summary="워크스페이스에서 스트리밍 채팅 실행")
def stream_chat_endpoint(
    category: str = Query(...,description="doc_gen | summary"),
    slug    : str = Path(...,description="워크스페이스 슬러그"), 
    body    : StreamChatRequest = Body(...,description="채팅 요청 본문")
    ):
    user_id = 3
    if category in ["doc_gen", "summary"]:
        gen = stream_chat_for_workspace(
            user_id=user_id, 
            slug=slug, 
            body=body.model_dump(exclude_unset=True)
            )
        return StreamingResponse(to_see(gen), media_type="text/event-stream")
    else:
        raise BadRequestError("qa category should be used in another endpoint > '/v1/workspace/{slug}/thread/{thread_slug}/stream-chat'")

@chat_router.post("/{slug}/thread/{thread_slug}/stream-chat", summary="QnA 스트리밍 채팅 실행")
def stream_chat_qa_endpoint(
    category: str = Query(...,description="only qa"),
    slug    : str = Path(...,description="워크스페이스 슬러그"), 
    thread_slug: str = Path(...,description="채팅 스레드 슬러그"),
    body    : StreamChatRequest = Body(...,description="채팅 요청 본문")
    ):
    user_id = 3
    gen = stream_chat_for_workspace(
        user_id=user_id, 
        slug=slug, 
        thread_slug=thread_slug, 
        body=body.model_dump(exclude_unset=True)
        )
    return StreamingResponse(to_see(gen), media_type="text/event-stream")


# qa를 제외한 문서요약, 생성만 처리 가능함
    # if category == "gen_doc" or category == "summary":
    #     def to_see(gen):
    #         buf = []
    #         last_flush = time.monotonic()
    #         for chunk in gen:
    #             if not chunk:
    #                 continue
    #             logger.debug(f"[raw_chunk] {repr(chunk)}")
    #             if not buf:
    #                 chunk = chunk.lstrip()
    #             buf.append(chunk)
    #             text = "".join(buf)
    #             if len(text) >= 32 or text.endswith((" ", "\n", ".", "?", "!", "…", "。", "！", "？")) or time.monotonic() - last_flush > 0.2:
    #                 logger.info(f"[flush] {repr(text)}")
    #                 yield f"data: {text}\n\n"
    #                 buf.clear()
    #                 last_flush = time.monotonic()
    #         if buf:
    #             text = "".join(buf)
    #             logger.info(f"[flush-end] {repr(text)}")
    #             yield f"data: {text}\n\n"
    #     gen = stream_chat_for_workspace(user_id, slug, body.model_dump(exclude_unset=True))
    #     return StreamingResponse(to_see(gen), media_type="text/event-stream")
    # else:
    #     raise BadRequestError("qa category should be used in another endpoint > '/v1/workspace/{slug}/thread/{thread_slug}/stream-chat'")


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
        if len(text) >= 32 or text.endswith((" ", "\n", ".", "?", "!", "…", "。", "！", "？")) or time.monotonic() - last_flush > 0.2:
            logger.info(f"[flush] {repr(text)}")
            yield f"data: {text}\n\n"
            buf.clear()
            last_flush = time.monotonic()
    if buf:
        text = "".join(buf)
        logger.info(f"[flush-end] {repr(text)}")
        yield f"data: {text}\n\n"