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
    _gas = int(gradientAccumulationSteps) if gradientAccumulationSteps is not None else 16
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



