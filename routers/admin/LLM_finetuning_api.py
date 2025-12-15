from fastapi import APIRouter, Query, UploadFile, File, Form, HTTPException
from pydantic import BaseModel, Field
from fastapi.responses import StreamingResponse, FileResponse
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
    BadRequestError,
    list_feedback_datasets,
    resolve_feedback_download,

)

router = APIRouter(prefix="/v1/admin/llm", tags=["Admin LLM - FineTuning"], responses={200: {"description": "Success"}})


# ---- Response Schemas (for enriched docs) ----
class FineTuneLaunchResponse(BaseModel):
    jobId: str = Field(..., description="생성된 작업 ID (ft-job-xxxx)")
    started: bool = Field(..., description="즉시 실행 여부(True면 즉시 스레드 시작)")


class FineTuneStatusResponse(BaseModel):
    jobId: str = Field(..., description="작업 ID")
    status: str = Field(..., description="queued | scheduled | running | succeeded | failed")
    learningProgress: int = Field(0, ge=0, le=100, description="학습 진행률 0~100")
    saveProgress: int | None = Field(0, ge=0, le=100, description="저장 단계 진행률 0~100")
    saveStage: str | None = Field(None, description="저장 단계 라벨(adapter/tokenizer/model/done 등)")
    roughScore: int | None = Field(None, description="간이 점수(ROUGE 0~100 환산)")
    rouge1F1: float | None = Field(None, description="최종 ROUGE-1 F1 (0.0~1.0)")
    error: str | None = Field(None, description="에러 메시지(있을 때만)")


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


@router.post(
    "/fine-tuning",
    summary="파인튜닝 실행(텍스트 파라미터 + 학습 파일 업로드)",
    response_model=FineTuneLaunchResponse,
    responses={
        200: {"description": "작업이 생성되었고 즉시 실행 여부가 반환됩니다."},
        400: {"description": "요청 파라미터 오류 또는 중복 실행 락"},
        500: {"description": "서버 내부 오류"},
    },
)
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
    # 문자열로 받아서 빈 문자열("")을 None으로 처리한 뒤, QLORA일 때만 int로 변환
    quantizationBits: str | None = Form(None, description="QLORA 전용: 4 또는 8"),
    startAt: str | None = Form(None, description="예약 시작 ISO8601 (예: 2025-09-19T13:00:00)"),
    startNow: bool | None = Form(None, description="즉시 실행 여부 (True: 바로 시작, False: 예약만 등록)"),
    # 업로드 파일(필수)
    trainSet: UploadFile = File(..., description="학습 CSV 파일"),
):
    # 기본값 채우기
    _category = (category or "qna").strip()
    _base = (baseModelName or "gpt-oss").strip()
    _save = saveModelName or f"fine-tuned-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    _sys = systemPrompt or "당신은 도움이 되는 AI 어시스턴트입니다."
    _bs = int(batchSize) if batchSize is not None else 4
    _epochs = int(epochs) if epochs is not None else 3
    _lr = float(learningRate) if learningRate is not None else 2e-4
    _ofp = True if overfittingPrevention is None else bool(overfittingPrevention)
    _gas = int(gradientAccumulationSteps) if gradientAccumulationSteps is not None else 16
    _tuning = (tuningType or "QLORA").strip().upper() if tuningType else "QLORA"

    # quantizationBits는 QLORA에서만 사용.
    # - 폼에서 ""로 오면 None 처리
    # - QLORA인데 값이 없으면 기본 4, 값이 있으면 int로 변환
    # - FULL/LORA에서는 항상 None
    raw_qbits = (quantizationBits or "").strip() if quantizationBits is not None else None
    if _tuning == "QLORA":
        if not raw_qbits:
            _qbits = 4
        else:
            _qbits = int(raw_qbits)
    else:
        _qbits = None
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


@router.get(
    "/fine-tuning",
    summary="파인튜닝 진행 상태 조회",
    response_model=FineTuneStatusResponse,
    responses={
        200: {"description": "현재 진행 상태 및 진행률을 반환합니다."},
        404: {"description": "해당 작업이 존재하지 않습니다."},
    },
)
async def read_fine_tuning_status(jobId: str = Query(..., description="조회할 작업 ID")):
    return get_fine_tuning_status(job_id=jobId)

@router.get(
    "/feedback-datasets",
    summary="피드백 CSV 목록/다운로드 (단일 API)",
    responses={
        200: {"description": "목록(JSON) 또는 파일 다운로드(csv)"},
        400: {"description": "요청 오류"},
        404: {"description": "파일 없음"},
    },
)
async def feedback_datasets(file: str | None = Query(
    None,
    description="다운로드할 파일명(basename만). 예: feedback_qna_p0.csv",
)):
    # 다운로드 분기
    if file:
        try:
            abs_path, filename = resolve_feedback_download(file)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다.")
        except BadRequestError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return FileResponse(abs_path, media_type="text/csv", filename=filename)

    # 목록 반환
    try:
        return list_feedback_datasets()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"목록 조회 중 오류: {e}")