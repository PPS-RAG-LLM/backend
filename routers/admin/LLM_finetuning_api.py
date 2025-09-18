from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
import asyncio
import json

from service.admin.LLM_finetuning import (
    FineTuneRequest,
    start_fine_tuning,
    get_fine_tuning_status,
    list_train_data_dirs,
    get_fine_tuning_logs,
)

router = APIRouter(prefix="/v1/admin/llm", tags=["Admin LLM - FineTuning"], responses={200: {"description": "Success"}})


@router.post("/fine-tuning", summary="파인튜닝 설정 및 실행 (Body에 category/subcategory 포함)")
def launch_fine_tuning(
    body: FineTuneRequest,
    category: str | None = Query(None, description="(하위호환) qa | doc_gen | summary. 지정 시 body.category 보다 우선하지 않음"),
):
    """
    - 하위호환: Query category가 오더라도, 실제 처리는 body.category를 원천으로 사용.
    - QLORA일 때 quantizationBits(4/8) 검증은 FineTuneRequest validator에서 수행.
    """
    cat = body.category or category
    return start_fine_tuning(cat, body)


@router.get("/fine-tuning", summary="지정된 작업 ID의 파인튜닝 진행 상태와 결과를 조회")
def read_fine_tuning_status(
    jobId: str = Query(...),
    category: str = Query("qa", description="(옵션) qa | doc_gen | summary. 기본값 qa"),
):
    return get_fine_tuning_status(category, jobId)


@router.get("/fine-tuning/train-dirs", summary="학습 데이터(root: storage/train_data) 하위 폴더 목록 조회")
def get_train_dirs():
    return list_train_data_dirs()


@router.get("/fine-tuning/logs", summary="파인튜닝 로그 tail 조회")
def read_fine_tuning_logs(jobId: str = Query(...), tail: int = Query(200, ge=1, le=2000)):
    return get_fine_tuning_logs(jobId, tail)


@router.get("/fine-tuning/stream", summary="파인튜닝 진행 상황을 SSE로 실시간 구독")
def stream_fine_tuning(
    jobId: str,
    category: str = Query("qa", description="(옵션) qa | doc_gen | summary. 기본값 qa"),
):
    async def event_generator():
        last_len = 0
        while True:
            # logs
            logs = get_fine_tuning_logs(jobId, tail=2000)
            lines = logs.get("lines", [])
            if len(lines) > last_len:
                new = lines[last_len:]
                for ln in new:
                    yield f"data: {ln}\n\n"
                last_len = len(lines)
            # status
            st = get_fine_tuning_status(category, jobId)
            yield f"event: status\ndata: {json.dumps(st, ensure_ascii=False)}\n\n"
            if st.get("status") in ("succeeded", "failed"):
                break
            await asyncio.sleep(1)
    return StreamingResponse(event_generator(), media_type="text/event-stream")
