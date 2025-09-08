# routers/admin/manage_admin_LLM_api.py
from fastapi import APIRouter, Query
from pydantic import BaseModel

from service.admin.manage_admin_LLM import (
    TopKSettingsBody,
    ModelLoadBody,
    CreatePromptBody,
    UpdatePromptBody,
    CompareModelsBody,
    set_topk_settings,
    get_model_list,
    load_model,
    compare_models,
    list_prompts,
    create_prompt,
    get_prompt,
    update_prompt,
    delete_prompt,
    test_prompt,
    DownloadModelBody,
    InferBody,
    download_model,
    infer_local,
    InsertBaseModelBody,
    insert_base_model,
    unload_model_for_category,
)

router = APIRouter(prefix="/v1/admin/llm", tags=["Admin LLM"], responses={200: {"description": "Success"}})

# === New routes ===

@router.post("/model/download", summary="HuggingFace 저장소에서 모델 다운로드 (오프라인 환경에서는 미사용)")
def download_model_route(body: DownloadModelBody):
    return download_model(body)

@router.post("/model/insert-base", summary="오프라인 환경용 베이스 모델 메타 등록(qa/doc_gen/summary) | 모든 카테고리 들어가게 해야함 | insert_base_model에서 model_path가 비어 있으면 storage/model/<name>로 자동 설정합니다.")
def insert_base_model_route(body: InsertBaseModelBody):
    return insert_base_model(body)


# 파인튜닝 관련 API는 routers/admin/LLM_finetuning_api.py 에서 제공합니다.


@router.post("/infer", summary="로컬 모델로 단발 추론 실행(테스트용) | 이거 일단 보류 ")
def infer_route(body: InferBody):
    return infer_local(body)

@router.post("/settings", summary="RAG에서 반환할 문서 수(topK) 설정 | 일단 보류")
def set_settings(body: TopKSettingsBody):
    return set_topk_settings(body.topK)

@router.get("/settings/model-list", summary="모델 목록과 로드/활성 상태 조회 (카테고리별 또는 base 전용)")
def model_list(category: str = Query(..., description="base | qa | doc_gen | summary")):
    return get_model_list(category)

@router.post("/settings/model-load", summary="모델명을 기준으로 로드 (베이스 모델은 모든 카테고리에 로드로 간주)")
def model_load(body: ModelLoadBody = ...):
    # 지연 로딩으로 새 함수 유무에 관계없이 동작하도록 처리
    from service.admin import manage_admin_LLM as svc
    load_by_name = getattr(svc, "load_model_by_name", None)
    if callable(load_by_name):
        return load_by_name(body.modelName)
    # 구버전 폴백: 일단 한 카테고리에 로드하면 경로 일치 기준으로 모든 카테고리에서 로드 표시됨
    return svc.load_model("qa", body.modelName)

@router.post("/settings/model-unload", summary="모델명을 기준으로 명시적 언로드 (모든 카테고리에서 내려간 것으로 간주)")
def model_unload(body: ModelLoadBody = ...):
    # 언로드는 카테고리와 무관하게 동작. active 캐시는 카테고리별로 클리어 필요 시 별도 API 사용
    from service.admin.manage_admin_LLM import unload_model
    return unload_model(body.modelName)

@router.get("/compare-models", summary="최근 평가 결과 기준 모델 비교 목록 => 카테고리별 평가 기준 목록. ")
def compare_models_list(category: str = Query(...)):
    # 기존 사양 유지: querystring만 받음
    return compare_models(CompareModelsBody(category=category))

@router.get("/prompts", summary="카테고리별 프롬프트 목록 조회")
def get_prompts(category: str = Query(...)):
    return list_prompts(category)

@router.post("/prompts", summary="프롬프트 생성")
def create_prompt_route(category: str = Query(...), body: CreatePromptBody = ...):
    return create_prompt(category, body)

@router.get("/prompt/{prompt_id}", summary="프롬프트 상세 조회")
def get_prompt_route(prompt_id: int):
    return get_prompt(prompt_id)

@router.put("/prompt/{prompt_id}", summary="프롬프트 수정")
def update_prompt_route(prompt_id: int, body: UpdatePromptBody = ...):
    return update_prompt(prompt_id, body)

@router.delete("/prompt/{prompt_id}", summary="프롬프트 삭제")
def delete_prompt_route(prompt_id: int):
    return delete_prompt(prompt_id)

@router.post("/prompt/{prompt_id}", summary="프롬프트 테스트 실행(치환/추론/평가 로그 적재)")
def test_prompt_route(prompt_id: int, body: dict | None = None):
    return test_prompt(prompt_id, body)
