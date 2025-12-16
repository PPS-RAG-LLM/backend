# routers/admin/manage_admin_LLM_api.py
from typing import Optional
from fastapi import APIRouter, Query

from service.admin.manage_admin_LLM import (
    ModelLoadBody,
    get_model_list,
    load_model,
    list_prompts,
    unload_model,
    delete_model_full, 
    DeleteModelBody,
)

router = APIRouter(
    prefix="/v1/admin/llm",
    tags=["Admin LLM load settings"],
    responses={200: {"description": "Success"}},
)

def _infer_category_from_name(name: str) -> Optional[str]:
    n = (name or "").lower()
    if n.endswith(("_summary", "-summary")):
        return "summary"
    if n.endswith(("_qna", "-qna")):
        return "qna"
    if n.endswith(("_doc_gen", "-doc_gen")):
        return "doc_gen"
    return None
    
# === 모델 관련 ===

@router.get(
    "/settings/model-list",
    summary="모델 목록과 로드/활성 상태 조회 (카테고리/서브테스크별)",
)
def model_list(
    category: str = Query(..., description="qna | doc_gen | summary | base | all"),
    subcategory: str | None = Query(
        None, description="doc_gen 서브테스크(=template.name, 예: '출장계획서')"
    ),
    provider: str | None = Query(
        None, description="openai | anthropic | gemini | huggingface"
    ),
):
    return get_model_list(category, subcategory, provider)


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
    category: str = Query(..., description="doc_gen | summary | qna"),
    subtask: str | None = Query(
        None, description="doc_gen 전용: report | travel_plan | meeting_minutes 등"
    ),
):
    return list_prompts(category, subtask)

@router.post("/settings/model-delete", summary="모델 삭제(DB + 디스크). 기본: 모델 디렉터리만 제거, LORA의 base는 보존")
def model_delete(body: DeleteModelBody):
    """
    Qwen2.5-7B-Instruct-1M
    gpt-oss-20b
    Qwen3-14B
    Qwen3-8B

    bge_m3
    qwen3_0_6b
    qwen3_4b

    body 예:
    {
      "modelName": "gpt-oss-20b"
    }
    """
    return delete_model_full(body.modelName)