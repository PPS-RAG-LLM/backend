from fastapi import APIRouter, Query

from service.admin.LLM_finetuning import (
    FineTuneRequest,
    start_fine_tuning,
    get_fine_tuning_status,
)

router = APIRouter(prefix="/v1/admin/llm", tags=["Admin LLM - FineTuning"], responses={200: {"description": "Success"}})


@router.post("/fine-tuning", summary="파인튜닝 설정 및 실행")
def launch_fine_tuning(category: str = Query(..., description="qa | doc_gen | summary"), body: FineTuneRequest = ...):
    return start_fine_tuning(category, body)


@router.get("/fine-tuning", summary="지정된 작업 ID의 파인튜닝 진행 상태와 결과를 조회")
def read_fine_tuning_status(category: str = Query(..., description="qa | doc_gen | summary"), jobId: str = Query(...)):
    return get_fine_tuning_status(category, jobId)
