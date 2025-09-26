from fastapi import APIRouter, Query, UploadFile, File, Form
from typing import Optional, List

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
    # 보장 실행
    EnsureRunBody,
    ensure_run_if_empty,
)


router = APIRouter(prefix="/v1/test/llm", tags=["Test LLM"], responses={200: {"description": "Success"}})


# ====== 신규: 평가 실행 및 조회 ======
@router.post("/run", summary="선택된 템플릿/모델/사용자프롬프트(+RAG)로 단발 평가 실행 및 저장")
def run_eval_route(body: RunEvalBody):
    return run_eval_once(body)

@router.get("/runs", summary="평가 결과 조회(카테고리/서브테스크/모델/유저프롬프트/프롬프트ID/pdfList 완전일치)")
def list_eval_runs_route(
    category: str = Query(..., description="qa | qna | doc_gen | summary"),
    subcategory: str | None = Query(None, description="세부 테스크(=template.name)"),
    modelName: str | None = Query(None, description="llm_models.name"),
    userPrompt: str | None = Query(None, description="사용자 입력 프롬프트(완전일치 비교)"),
    promptId: int | None = Query(None, description="system_prompt_template.id"),
    pdfList: Optional[List[str]] = Query(None, description="완전일치할 PDF 파일명 리스트")
):
    q = EvalQuery(category=category, subcategory=subcategory, modelName=modelName, userPrompt=userPrompt, promptId=promptId, pdfList=pdfList)
    return list_eval_runs(q)

@router.post("/runs/ensure", summary="조건 일치 레코드가 없으면 LLM+RAG 실행 후 저장, 있으면 기존 결과 반환")
def ensure_run_if_empty_route(body: EnsureRunBody):
    return ensure_run_if_empty(body)


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


# ============================
# Test RAG Sessions
# ============================

@router.post("/rag/session", summary="테스트 세션 생성(임시 컬렉션 + 세션 폴더)")
def create_test_session_route():
    from service.admin.manage_vator_DB import create_test_session
    return create_test_session()

@router.post("/rag/{sid}/ingest", summary="세션 컬렉션으로 PDF 업로드+인제스트")
async def ingest_test_session_files_route(
    sid: str,
    files: List[UploadFile] = File(..., description="테스트용 PDF 파일들"),
    task_types: Optional[str] = Form(None, description="콤마로 구분된 작업유형(doc_gen,summary,qna). 기본: 전부")
):
    from service.admin.manage_vator_DB import VAL_SESSION_ROOT, get_test_session, ingest_test_pdfs
    from pathlib import Path
    import time
    
    # 업로드 파일을 임시 로컬 경로에 저장 후, 인제스트 실행
    pdf_paths: List[str] = []
    meta = get_test_session(sid)
    if not meta:
        return {"error": "invalid sid"}
    sess_dir = Path(meta["dir"])
    sess_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        stem = Path(f.filename).stem
        ext = Path(f.filename).suffix or ".pdf"
        # 동명이인 방지
        dst = sess_dir / f"{stem}_{int(time.time())}{ext}"
        with dst.open("wb") as out:
            out.write(await f.read())
        pdf_paths.append(str(dst))

    ttypes = [t.strip() for t in (task_types or "").split(",") if t.strip()] or None
    return await ingest_test_pdfs(sid, pdf_paths, task_types=ttypes)

@router.get("/rag/{sid}/search", summary="세션 컬렉션에서만 검색(RAG)")
async def search_test_session_route(
    sid: str,
    query: str = Query(...),
    top_k: int = Query(5, ge=1, le=50),
    user_level: int = Query(1, ge=1),
    task_type: str = Query(..., description="doc_gen | summary | qna"),
    model: Optional[str] = Query(None),
    search_type: Optional[str] = Query(None),
):
    from service.admin.manage_vator_DB import RAGSearchRequest, search_documents_test
    req = RAGSearchRequest(query=query, top_k=top_k, user_level=user_level, task_type=task_type, model=model)
    return await search_documents_test(req, sid=sid, search_type_override=search_type)

@router.post("/rag/{sid}/delete-files", summary="세션 컬렉션에서 특정 파일만 삭제")
async def delete_test_files_session_route(
    sid: str,
    file_names: List[str]
):
    from service.admin.manage_vator_DB import delete_test_files_by_names
    return await delete_test_files_by_names(sid, file_names=file_names, task_type=None)

@router.delete("/rag/session/{sid}", summary="세션 정리(컬렉션 드롭 + 폴더 삭제)")
async def drop_test_session_route(sid: str):
    from service.admin.manage_vator_DB import drop_test_session
    return await drop_test_session(sid)


