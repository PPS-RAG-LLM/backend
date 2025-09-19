from fastapi import APIRouter, Query, UploadFile, File, Form
from fastapi.responses import StreamingResponse
import asyncio
import json
import os
import uuid
from datetime import datetime
from pathlib import Path

from service.admin.LLM_finetuning import (
    FineTuneRequest,
    start_fine_tuning,
    get_fine_tuning_status,
    list_train_data_dirs,
    get_fine_tuning_logs,
    TRAIN_DATA_ROOT,
)

router = APIRouter(prefix="/v1/admin/llm", tags=["Admin LLM - FineTuning"], responses={200: {"description": "Success"}})

# ---- 업로드 유틸 ----
def _save_upload_csv(file: UploadFile, subdir: str | None = None) -> str:
    """
    업로드된 CSV를 storage/train_data/<subdir or YYYYMMDD>/<uuid>.csv 로 저장.
    반환값은 절대경로.
    """
    assert file.filename, "filename is required"
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in (".csv", ".tsv"):
        # 엄격히 CSV만 받고 싶다면 여기서 에러로 바꿔도 됨
        pass
    day = datetime.now().strftime("%Y%m%d")
    dir_name = subdir or day
    target_dir = os.path.join(TRAIN_DATA_ROOT, dir_name)
    os.makedirs(target_dir, exist_ok=True)
    unique = f"{uuid.uuid4().hex}{ext or '.csv'}"
    target_path = os.path.join(target_dir, unique)

    # 스트리밍 복사
    with open(target_path, "wb") as f:
        while chunk := file.file.read(1024 * 1024):
            f.write(chunk)
    file.file.close()
    return target_path


@router.post("/fine-tuning", summary="파인튜닝 실행(파일 업로드 + 예약 지원)")
async def launch_fine_tuning(
    # ---- 멀티파트 Form 필드들 ----
    category: str = Form(..., description="qa | doc_gen | summary"),
    baseModelName: str = Form(...),
    saveModelName: str = Form(...),
    systemPrompt: str = Form(...),
    batchSize: int = Form(4),
    epochs: int = Form(3),
    learningRate: float = Form(2e-4),
    overfittingPrevention: bool = Form(True),
    gradientAccumulationSteps: int = Form(8),
    tuningType: str = Form("QLORA", description="LORA | QLORA | FULL"),
    quantizationBits: int | None = Form(None, description="QLORA 전용: 4 또는 8"),
    startAt: str | None = Form(None, description="예약 시작 ISO8601 (예: 2025-09-19T13:00:00)"),
    # ---- 업로드 파일 ----
    trainSet: UploadFile = File(..., description="학습 CSV 파일"),
):
    # 1) 파일 저장
    saved_abs = _save_upload_csv(trainSet, subdir=saveModelName)
    # 2) 서비스 요청 모델 생성 (서비스는 '경로'를 받음)
    req = FineTuneRequest(
        category=category,
        subcategory=None,
        baseModelName=baseModelName,
        saveModelName=saveModelName,
        systemPrompt=systemPrompt,
        batchSize=batchSize,
        epochs=epochs,
        learningRate=learningRate,
        overfittingPrevention=overfittingPrevention,
        trainSetFile=saved_abs,
        gradientAccumulationSteps=gradientAccumulationSteps,
        quantizationBits=quantizationBits,
        tuningType=tuningType,
        startAt=startAt,
    )
    return start_fine_tuning(category, req)


@router.get("/fine-tuning", summary="지정된 작업 ID의 파인튜닝 진행 상태와 결과를 조회")
def read_fine_tuning_status(jobId: str = Query(...)):
    return get_fine_tuning_status(job_id=jobId)


@router.get("/fine-tuning/train-dirs", summary="학습 데이터(root: storage/train_data) 하위 폴더 목록 조회")
def get_train_dirs():
    return list_train_data_dirs()


@router.get("/fine-tuning/logs", summary="파인튜닝 로그 tail 조회")
def read_fine_tuning_logs(jobId: str = Query(...), tail: int = Query(200, ge=1, le=2000)):
    return get_fine_tuning_logs(jobId, tail)


@router.get("/fine-tuning/stream", summary="파인튜닝 진행 상황을 SSE로 실시간 구독")
def stream_fine_tuning(jobId: str):
    async def event_generator():
        last_len = 0
        # 재접속 힌트: 3초
        yield "retry: 3000\n\n"
        while True:
            # 1) 로그 증분 전송
            logs = get_fine_tuning_logs(jobId, tail=2000)
            lines = logs.get("lines", [])
            if len(lines) > last_len:
                for ln in lines[last_len:]:
                    yield f"event: log\ndata: {ln}\n\n"
                last_len = len(lines)
            # 2) 상태 스냅샷
            st = get_fine_tuning_status(job_id=jobId)
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
