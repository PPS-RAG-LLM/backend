from fastapi import APIRouter, Query

from service.admin.manage_test_LLM import (
    DefaultModelBody,
    set_default_model,
    get_default_model,
    SelectModelQuery,
    get_selected_model,
    # 신규
    RunEvalBody,
    run_eval_once,
    EvalQuery,
    list_eval_runs,
)


router = APIRouter(prefix="/v1/test/llm", tags=["Test LLM"], responses={200: {"description": "Success"}})


# ====== 신규: 평가 실행 및 조회 ======
@router.post("/run", summary="선택된 템플릿/모델/사용자프롬프트(+RAG)로 단발 평가 실행 및 저장")
def run_eval_route(body: RunEvalBody):
    return run_eval_once(body)

@router.get("/runs", summary="평가 결과 조회(테스크/서브테스크/모델/사용자프롬프트 일치)")
def list_eval_runs_route(
    category: str = Query(..., description="qa | qna | doc_gen | summary"),
    subcategory: str | None = Query(None, description="세부 테스크(=template.name)"),
    modelName: str | None = Query(None, description="llm_models.name"),
    userPrompt: str | None = Query(None, description="사용자 입력 프롬프트(완전일치 비교)"),
    limit: int = Query(50, ge=1, le=200)
):
    q = EvalQuery(category=category, subcategory=subcategory, modelName=modelName, userPrompt=userPrompt, limit=limit)
    return list_eval_runs(q)


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


