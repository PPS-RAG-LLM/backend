# routers/admin/manage_admin_LLM_api.py
from fastapi import APIRouter, Query
from pydantic import BaseModel

from service.admin.manage_admin_LLM import (
    TopKSettingsBody,
    ModelLoadBody,
    CreatePromptBody,
    UpdatePromptBody,
    ActivePromptBody,
    # set_topk_settings,
    get_model_list,
    load_model,
    list_prompts,
    create_prompt,
    get_prompt,
    update_prompt,
    delete_prompt,
    test_prompt,
    # DownloadModelBody,
    # download_model,
    # InsertBaseModelBody,
    # insert_base_model,
    unload_model_for_category,
    get_active_prompt,
    set_active_prompt,
)

router = APIRouter(prefix="/v1/admin/llm", tags=["Admin LLM"], responses={200: {"description": "Success"}})

# === New routes ===

# @router.post("/model/insert-base", summary="오프라인 환경용 베이스 모델 메타 등록(qa/doc_gen/summary/qna)")
# def insert_base_model_route(body: InsertBaseModelBody):
#     return insert_base_model(body)

# 파인튜닝 관련 API는 routers/admin/LLM_finetuning_api.py 에서 제공합니다.

@router.post("/prompt/{prompt_id}", summary="프롬프트 테스트 실행(치환/추론/평가 로그 적재)")
def test_prompt_route(prompt_id: int, body: dict | None = None):
    return test_prompt(prompt_id, body)

# @router.post("/settings", summary="RAG에서 반환할 문서 수(topK) 설정")
# def set_settings(body: TopKSettingsBody):
#     return set_topk_settings(body.topK)

@router.get("/settings/model-list", summary="모델 목록과 로드/활성 상태 조회 (카테고리별 또는 base 전용)")
def model_list(
    category: str = Query(..., description="base | qa | doc_gen | summary | all (또는 'doc_gen:report' 형식 지원)"),
    subcategory: str | None = Query(None, description="서브카테고리. 지정 시 category와 결합하여 'cat:subcat'으로 전달"),
):
    cat_key = f"{category}:{subcategory}" if subcategory else category
    return get_model_list(cat_key)

@router.post("/settings/model-load", summary="모델명을 기준으로 로드 (베이스 모델은 모든 카테고리에 로드로 간주)")
def model_load(body: ModelLoadBody = ...):
    from service.admin import manage_admin_LLM as svc
    cat = svc._infer_category_from_name(body.modelName)
    return svc.load_model(cat, body.modelName)

@router.post("/settings/model-unload", summary="모델명을 기준으로 명시적 언로드")
def model_unload(body: ModelLoadBody = ...):
    from service.admin.manage_admin_LLM import unload_model
    return unload_model(body.modelName)

## moved to /v1/test/llm

# ---------- 프롬프트(6개 구조) ----------
# 카테고리: doc_gen | summary | qna(=qa)
# 서브테스크(doc_gen 전용): report | travel_plan | meeting_minutes 등. 다른 값으로도 OK.

@router.get("/prompts", summary="카테고리/서브테스크별 프롬프트 목록 조회")
def get_prompts(
    category: str = Query(..., description="doc_gen | summary | qa"),
    subtask: str | None = Query(None, description="doc_gen 전용: report | travel_plan | meeting_minutes 등")
):
    return list_prompts(category, subtask)

@router.post("/prompts", summary="프롬프트 생성(카테고리/서브테스크별)")
def create_prompt_route(
    category: str = Query(..., description="doc_gen | summary | qa"),
    subtask: str | None = Query(None, description="doc_gen 전용: report | travel_plan | meeting_minutes 등"),
    body: CreatePromptBody = ...,
):
    return create_prompt(category, subtask, body)

@router.get("/prompt/{prompt_id}", summary="프롬프트 상세 조회")
def get_prompt_route(prompt_id: int):
    return get_prompt(prompt_id)

@router.put("/prompt/{prompt_id}", summary="프롬프트 수정")
def update_prompt_route(prompt_id: int, body: UpdatePromptBody = ...):
    return update_prompt(prompt_id, body)

@router.delete("/prompt/{prompt_id}", summary="프롬프트 삭제")
def delete_prompt_route(prompt_id: int):
    return delete_prompt(prompt_id)

## moved to /v1/test/llm

# ---------- 사용자 선택(활성 프롬프트) ----------
@router.get("/prompt/active", summary="현재 선택된(활성) 프롬프트 조회")
def get_active_prompt_route(
    category: str = Query(..., description="doc_gen | summary | qa"),
    subtask: str | None = Query(None, description="doc_gen 전용"),
):
    return get_active_prompt(category, subtask)

@router.post("/prompt/active", summary="프롬프트 선택(활성화) 저장")
def set_active_prompt_route(body: ActivePromptBody):
    return set_active_prompt(body)

## 기본 모델 관련 엔드포인트는 /v1/test/llm로 이동
