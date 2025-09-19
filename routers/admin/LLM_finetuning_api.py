from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
import asyncio
import json

from service.admin.LLM_finetuning import (
    FineTuneRequest,
    start_fine_tuning,
    get_fine_tuning_status,
    get_fine_tuning_logs,
)

router = APIRouter(prefix="/v1/admin/llm", tags=["Admin LLM - FineTuning"], responses={200: {"description": "Success"}})


@router.post("/fine-tuning", summary="파인튜닝 설정 및 실행")
def launch_fine_tuning(body: FineTuneRequest):
    """body.category만 사용 (하위호환 없음)"""
    return start_fine_tuning(body.category, body)


@router.get("/fine-tuning/stream", summary="파인튜닝 진행 상황 SSE")
def stream_fine_tuning(jobId: str):
    async def event_generator():
        last_len = 0
        # 브라우저가 끊기면 3초 후 재접속
        yield "retry: 3000\n\n"
        while True:
            # 1) 새 로그만 보내기
            logs = get_fine_tuning_logs(jobId, tail=2000)
            lines = logs.get("lines", [])
            if len(lines) > last_len:
                for ln in lines[last_len:]:
                    yield f"event: log\ndata: {ln}\n\n"
                last_len = len(lines)

            # 2) 상태 스냅샷
            st = get_fine_tuning_status(jobId)
            yield f"event: status\ndata: {json.dumps(st, ensure_ascii=False)}\n\n"

            # 3) 종료
            if st.get("status") in ("succeeded", "failed"):
                yield "event: end\ndata: done\n\n"
                break

            # 4) keep-alive
            yield ": keepalive\n\n"
            await asyncio.sleep(1)

    headers = {
        "Cache-Control": "no-cache, no-transform",
        "X-Accel-Buffering": "no",
        "Connection": "keep-alive",
    }
    return StreamingResponse(event_generator(), media_type="text/event-stream", headers=headers)
