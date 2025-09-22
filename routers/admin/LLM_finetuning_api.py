from fastapi import APIRouter, Query, UploadFile, File, Form
from pydantic import BaseModel, Field
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
class FineTuneJsonRequest(BaseModel):
    """JSON 바디로 받는 파라미터. Swagger의 Edit Value 사용 가능."""
    trainSetPath: str = Field(..., description="학습 CSV의 절대경로 또는 백엔드 루트 기준 상대경로")
    category: str | None = Field(None, description="qa | doc_gen | summary")
    baseModelName: str | None = Field(None, description="베이스 모델 이름(예: gpt-oss)")
    saveModelName: str | None = Field(None, description="저장될 모델 표시 이름(미지정 시 자동 생성)")
    systemPrompt: str | None = Field(None, description="시스템 프롬프트")
    batchSize: int | None = Field(None)
    epochs: int | None = Field(None)
    learningRate: float | None = Field(None)
    overfittingPrevention: bool | None = Field(None)
    gradientAccumulationSteps: int | None = Field(None)
    tuningType: str | None = Field(None, description="LORA | QLORA | FULL")
    quantizationBits: int | None = Field(None, description="QLORA 전용: 4 또는 8")
    startAt: str | None = Field(None, description="예약 시작 ISO8601 (예: 2025-09-19T13:00:00)")
    startNow: bool | None = Field(None, description="즉시 실행 여부 (True: 바로 시작, False: 예약만 등록)")

    class Config:
        json_schema_extra = {
            "example": {
                "trainSetPath": "./storage/train_data/my_run/train.csv",
                "category": "qa",
                "baseModelName": "gpt-oss",
                "saveModelName": "fine-qa-20250919",
                "systemPrompt": "당신은 도움이 되는 AI 어시스턴트입니다.",
                "batchSize": 4,
                "epochs": 3,
                "learningRate": 0.0002,
                "overfittingPrevention": True,
                "gradientAccumulationSteps": 8,
                "tuningType": "QLORA",
                "quantizationBits": 4,
                "startAt": None,
                "startNow": True,
            }
        }


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


@router.post("/fine-tuning", summary="파인튜닝 실행(텍스트 파라미터 + 학습 파일 업로드)")
async def launch_fine_tuning(
    # 텍스트(옵션) 파라미터들 — 값이 오면 사용, 없으면 기본값
    category: str | None = Form(None, description="qa | doc_gen | summary"),
    baseModelName: str | None = Form(None, description="베이스 모델 이름(예: gpt-oss)"),
    saveModelName: str | None = Form(None, description="저장될 모델 표시 이름(미지정 시 자동 생성)"),
    systemPrompt: str | None = Form(None, description="시스템 프롬프트"),
    batchSize: int | None = Form(None),
    epochs: int | None = Form(None),
    learningRate: float | None = Form(None),
    overfittingPrevention: bool | None = Form(None),
    gradientAccumulationSteps: int | None = Form(None),
    tuningType: str | None = Form(None, description="LORA | QLORA | FULL"),
    quantizationBits: int | None = Form(None, description="QLORA 전용: 4 또는 8"),
    startAt: str | None = Form(None, description="예약 시작 ISO8601 (예: 2025-09-19T13:00:00)"),
    startNow: bool | None = Form(None, description="즉시 실행 여부 (True: 바로 시작, False: 예약만 등록)"),
    # 업로드 파일(필수)
    trainSet: UploadFile = File(..., description="학습 CSV 파일"),
):
    # 기본값 채우기
    _category = (category or "qa").strip()
    _base = (baseModelName or "gpt-oss").strip()
    _save = saveModelName or f"fine-tuned-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    _sys = systemPrompt or "당신은 도움이 되는 AI 어시스턴트입니다."
    _bs = int(batchSize) if batchSize is not None else 4
    _epochs = int(epochs) if epochs is not None else 3
    _lr = float(learningRate) if learningRate is not None else 2e-4
    _ofp = True if overfittingPrevention is None else bool(overfittingPrevention)
    _gas = int(gradientAccumulationSteps) if gradientAccumulationSteps is not None else 8
    _tuning = (tuningType or "QLORA").upper()
    _qbits = quantizationBits if quantizationBits is not None else (4 if _tuning == "QLORA" else None)
    _startNow = startNow if startNow is not None else False

    # 1) 파일 저장
    saved_abs = _save_upload_csv(trainSet, subdir=_save)
    # 2) 서비스 요청 생성
    req = FineTuneRequest(
        category=_category,
        subcategory=None,
        baseModelName=_base,
        saveModelName=_save,
        systemPrompt=_sys,
        batchSize=_bs,
        epochs=_epochs,
        learningRate=_lr,
        overfittingPrevention=_ofp,
        trainSetFile=saved_abs,
        gradientAccumulationSteps=_gas,
        quantizationBits=_qbits,
        tuningType=_tuning,
        startAt=startAt,
        startNow=_startNow,
    )
    return start_fine_tuning(_category, req)


@router.get("/fine-tuning", summary="지정된 작업 ID의 파인튜닝 진행 상태와 결과를 조회")
def read_fine_tuning_status(jobId: str = Query(...)):
    return get_fine_tuning_status(job_id=jobId)


@router.post("/fine-tuning/json", summary="파인튜닝 실행(JSON 바디; Swagger Edit Value로 편리 입력)")
def launch_fine_tuning_json(body: FineTuneJsonRequest):
    # 기본값 채우기
    category = (body.category or "qa").strip()
    baseModelName = (body.baseModelName or "gpt-oss").strip()
    saveModelName = body.saveModelName or f"fine-tuned-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    systemPrompt = body.systemPrompt or "당신은 도움이 되는 AI 어시스턴트입니다."
    batchSize = int(body.batchSize) if body.batchSize is not None else 4
    epochs = int(body.epochs) if body.epochs is not None else 3
    learningRate = float(body.learningRate) if body.learningRate is not None else 2e-4
    overfittingPrevention = True if body.overfittingPrevention is None else bool(body.overfittingPrevention)
    gradientAccumulationSteps = int(body.gradientAccumulationSteps) if body.gradientAccumulationSteps is not None else 8
    tuningType = (body.tuningType or "QLORA").upper()
    quantizationBits = body.quantizationBits if body.quantizationBits is not None else (4 if tuningType == "QLORA" else None)
    startNow = body.startNow if body.startNow is not None else False

    # 학습 파일 경로 확인
    train_path = body.trainSetPath
    if train_path.startswith("./"):
        train_path = str(Path(__file__).resolve().parents[2] / train_path.lstrip("./"))

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
        trainSetFile=train_path,
        gradientAccumulationSteps=gradientAccumulationSteps,
        quantizationBits=quantizationBits,
        tuningType=tuningType,
        startAt=body.startAt,
        startNow=startNow,
    )
    return start_fine_tuning(category, req)

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
