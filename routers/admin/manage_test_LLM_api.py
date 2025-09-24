from fastapi import APIRouter, Query

from service.admin.manage_test_LLM import (
    InferBody,
    infer_local,
    CompareModelsBody,
    compare_models,
    DefaultModelBody,
    set_default_model,
    get_default_model,
    SelectModelQuery,
    get_selected_model,
)


router = APIRouter(prefix="/v1/test/llm", tags=["Test LLM"], responses={200: {"description": "Success"}})


@router.post("/infer", summary="로컬 모델로 단발 추론 실행(테스트용)")
def infer_route(body: InferBody):
    return infer_local(body)


@router.get("/compare-models", summary="최근 평가 결과 기준 모델 비교 목록")
def compare_models_list(
    category: str = Query(..., description="qa | doc_gen | summary"),
    subcategory: str | None = Query(None, description="세부 테스크 (doc_gen 확장 포함)"),
    modelId: int | None = Query(None),
    promptId: int | None = Query(None),
    prompt: str | None = Query(None, description="치환 완료된 프롬프트 원문(옵션)"),
):
    payload = CompareModelsBody(
        category=category,
        modelId=modelId,
        promptId=promptId,
        prompt=prompt,
    )
    return compare_models(payload)


## moved to /v1/admin/llm


# ===== 기본 모델(테스크/서브테스크) 매핑 =====
@router.get("/settings/default-model", summary="(카테고리/서브테스크) 기본 모델 조회")
def get_default_model_route(
    category: str = Query(..., description="qa | qna | doc_gen | summary"),
    subcategory: str | None = Query(None, description="doc_gen 서브테스크"),
):
    return get_default_model(category, subcategory)


@router.post("/settings/default-model", summary="(카테고리/서브테스크) 기본 모델 지정(단일 보장)")
def set_default_model_route(body: DefaultModelBody):
    return set_default_model(body)


@router.get("/settings/select-model", summary="테스크별 디폴트 모델 확인(프롬프트 테이블 기반)")
def select_model_route(
    category: str = Query(..., description="qa | qna | doc_gen | summary"),
    subcategory: str | None = Query(None, description="doc_gen 서브테스크"),
):
    query = SelectModelQuery(category=category, subcategory=subcategory)
    return get_selected_model(query)


