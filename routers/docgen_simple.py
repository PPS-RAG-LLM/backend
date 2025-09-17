from typing import Optional, Dict, Any, Generator, List
from fastapi import APIRouter, Body, Query, HTTPException
from starlette.responses import StreamingResponse
from pydantic import BaseModel, Field

from utils import logger
from service.admin.manage_admin_LLM import (
    list_prompts as admin_list_prompts,
    get_active_prompt as admin_get_active_prompt,
    get_prompt as admin_get_prompt,
    test_prompt as admin_test_prompt,
)

log = logger(__name__)
router = APIRouter(tags=["DocGen"], prefix="/v1/doc-gen")


class GenerateBody(BaseModel):
    title: str = Field(..., description="문서 제목")
    variables: Dict[str, Any] = Field(default_factory=dict, description="템플릿 변수들(한글 키 가능)")
    subtask: Optional[str] = Field(None, description="문서 유형(예: travel_plan, report 등)")
    promptId: Optional[int] = Field(None, description="명시적 템플릿 ID")
    modelName: Optional[str] = None
    max_tokens: int = Field(512, gt=0)
    temperature: float = Field(0.7, ge=0, le=2)


class GenerateOnceResponse(BaseModel):
    title: str
    content: str
    promptId: int


@router.get("/templates", summary="문서생성 템플릿 목록(doc_gen)")
def list_templates(subtask: Optional[str] = Query(None)):
    try:
        res = admin_list_prompts("doc_gen", subtask)
        return res
    except Exception as e:
        log.error({"docgen_list_templates_failed": str(e)})
        raise HTTPException(status_code=500, detail="템플릿 목록 조회 실패")


def _resolve_prompt_id(subtask: Optional[str], prompt_id: Optional[int]) -> int:
    if isinstance(prompt_id, int) and prompt_id > 0:
        return prompt_id
    active = admin_get_active_prompt("doc_gen", subtask)
    pid = (active.get("active") or {}).get("promptId")
    if isinstance(pid, int) and pid > 0:
        return pid
    fallback = admin_list_prompts("doc_gen", subtask)
    lst: List[Dict[str, Any]] = fallback.get("promptList", [])
    if not lst:
        raise HTTPException(status_code=404, detail="사용 가능한 템플릿이 없습니다")
    return int(lst[0]["promptId"])


def _sse_stream(text: str) -> Generator[str, None, None]:
    buf: List[str] = []
    for token in text.split(" "):
        if buf:
            buf.append(" ")
        buf.append(token)
        chunk = "".join(buf)
        if len(chunk) >= 24 or chunk.endswith((" ", "\n", ".", "?", "!")):
            yield f"data: {chunk}\n\n"
            buf.clear()
    if buf:
        yield f"data: {''.join(buf)}\n\n"


@router.post("/generate", summary="문서 생성(SSE 스트리밍)")
def generate_stream(body: GenerateBody = Body(...)):
    try:
        # 템플릿/변수 준비
        prompt_id = _resolve_prompt_id(body.subtask, body.promptId)
        prompt_info = admin_get_prompt(prompt_id)
        variables = dict(body.variables or {})
        variables.setdefault("제목", body.title)

        # 프롬프트 테스트(내부적으로 간단 추론 수행)
        result = admin_test_prompt(
            prompt_id,
            {
                "variables": variables,
                "category": "doc_gen",
                "modelName": body.modelName,
                "max_tokens": body.max_tokens,
                "temperature": body.temperature,
            },
        )
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error") or "문서 생성 실패")

        answer = result.get("answer", "")
        return StreamingResponse(_sse_stream(answer), media_type="text/event-stream")
    except HTTPException:
        raise
    except Exception as e:
        log.error({"docgen_generate_failed": str(e)})
        raise HTTPException(status_code=500, detail="문서 생성 중 오류가 발생했습니다")


@router.post("/generate/once", response_model=GenerateOnceResponse, summary="문서 생성(단발 응답)")
def generate_once(body: GenerateBody = Body(...)):
    try:
        prompt_id = _resolve_prompt_id(body.subtask, body.promptId)
        prompt_info = admin_get_prompt(prompt_id)
        variables = dict(body.variables or {})
        variables.setdefault("제목", body.title)

        result = admin_test_prompt(
            prompt_id,
            {
                "variables": variables,
                "category": "doc_gen",
                "modelName": body.modelName,
                "max_tokens": body.max_tokens,
                "temperature": body.temperature,
            },
        )
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error") or "문서 생성 실패")

        return GenerateOnceResponse(title=body.title, content=result.get("answer", ""), promptId=prompt_id)
    except HTTPException:
        raise
    except Exception as e:
        log.error({"docgen_generate_once_failed": str(e)})
        raise HTTPException(status_code=500, detail="문서 생성 중 오류가 발생했습니다")


