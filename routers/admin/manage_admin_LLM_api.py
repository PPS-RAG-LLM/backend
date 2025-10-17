# routers/admin/manage_admin_LLM_api.py
from fastapi import APIRouter, Query

from service.admin.manage_admin_LLM import (
    ModelLoadBody,
    get_model_list,
    load_model,
    list_prompts,
    unload_model,
    _infer_category_from_name,
    delete_model, 
    DeleteModelBody,
)

router = APIRouter(
    prefix="/v1/admin/llm",
    tags=["Admin LLM load settings"],
    responses={200: {"description": "Success"}},
)

# === 모델 관련 ===

@router.get(
    "/settings/model-list",
    summary="모델 목록과 로드/활성 상태 조회 (카테고리/서브테스크별)",
)
def model_list(
    category: str = Query(..., description="qa | doc_gen | summary | base | all"),
    subcategory: str | None = Query(
        None, description="doc_gen 서브테스크(=template.name, 예: '출장계획서')"
    ),
):
    return get_model_list(category, subcategory)


@router.post(
    "/settings/model-load",
    summary="모델명을 기준으로 로드 (베이스 모델은 모든 카테고리에 로드로 간주)",
)
def model_load(body: ModelLoadBody = ...):
    cat = _infer_category_from_name(body.modelName)
    return load_model(cat, body.modelName)


@router.post("/settings/model-unload", summary="모델명을 기준으로 명시적 언로드")
def model_unload(body: ModelLoadBody = ...):
    return unload_model(body.modelName)

# === 프롬프트: 목록 조회만 유지 ===

@router.get(
    "/prompts",
    summary="카테고리/서브테스크별 프롬프트 목록 조회(읽기 전용)",
)
def get_prompts(
    category: str = Query(..., description="doc_gen | summary | qa"),
    subtask: str | None = Query(
        None, description="doc_gen 전용: report | travel_plan | meeting_minutes 등"
    ),
):
    return list_prompts(category, subtask)

@router.post("/settings/model-delete", summary="모델 삭제(DB + 디스크). 기본: 모델 디렉터리만 제거, LORA의 base는 보존")
def model_delete(body: DeleteModelBody = ...):
    """
    body 예시:
    {
      "modelName": "gpt-oss-20b",
      "deleteFiles": true,
      "deleteBaseAlso": false,
      "deleteHistory": false
    }
    """
    return delete_model(body)