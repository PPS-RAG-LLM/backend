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
    userPrompt: str | None = Query(None, description="사용자 입력 프롬프트(완전일치)"),
    promptId: int | None = Query(None, description="system_prompt_template.id"),
    pdfList: List[str] | None = Query(None, description="업로드 원본 PDF 파일명 목록(완전일치)"),
):
    q = EvalQuery(
        category=category,
        subcategory=subcategory,
        modelName=modelName,
        userPrompt=userPrompt,
        promptId=promptId,
        pdfList=pdfList,
    )
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

# ============================
# Test RAG Sessions (이미 존재)
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
    from service.admin.manage_vator_DB import get_test_session, ingest_test_pdfs
    from pathlib import Path
    import time
    pdf_paths: List[str] = []
    meta = get_test_session(sid)
    if not meta:
        return {"error": "invalid sid"}
    sess_dir = Path(meta["dir"])
    sess_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        stem = Path(f.filename).stem
        ext = Path(f.filename).suffix or ".pdf"
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

# ============================
# NEW: 업로드와 보장 실행(없으면 실행→저장, 있으면 캐시 반환)
# ============================
@router.post(
    "/runs/ensure-upload",
    summary="PDF 업로드 포함: 동일 조건 run 없으면 테스트 세션에 인제스트→RAG+LLM 실행 후 저장"
)
async def ensure_run_if_empty_upload_route(
    category: str = Form(..., description="qa | qna | doc_gen | summary"),
    subcategory: Optional[str] = Form(None, description="세부 테스크(=template.name)"),
    modelName: Optional[str] = Form(None),
    userPrompt: Optional[str] = Form(None),
    promptId: int = Form(..., description="system_prompt_template.id"),
    files: List[UploadFile] = File(..., description="테스트에 사용할 PDF 파일들(이 파일명 리스트가 pdf_list로 저장/비교됨)"),
    max_tokens: int = Form(512),
    temperature: float = Form(0.7),
    cleanup: bool = Form(True, description="실행 후 세션 정리 여부"),
):
    from pathlib import Path
    import time
    from service.admin.manage_vator_DB import create_test_session, get_test_session, ingest_test_pdfs, drop_test_session
    from service.admin.manage_test_LLM import ensure_run_if_empty_uploaded

    # 1) 테스트 세션 생성
    sess = create_test_session()
    sid = sess["sid"]
    sess_dir = Path(sess["dir"])
    sess_dir.mkdir(parents=True, exist_ok=True)

    # 2) 업로드 저장(내부 파일명은 타임스탬프 부여, pdf_list에는 '원본 파일명'만 보관)
    saved_paths: List[str] = []
    pdf_names: List[str] = []
    for f in files:
        pdf_names.append(Path(f.filename).name)
        stem = Path(f.filename).stem
        ext = Path(f.filename).suffix or ".pdf"
        dst = sess_dir / f"{stem}_{int(time.time())}{ext}"
        with dst.open("wb") as out:
            out.write(await f.read())
        saved_paths.append(str(dst))

    # 3) 인제스트(세션 컬렉션)
    await ingest_test_pdfs(sid, saved_paths, task_types=None)

    # 4) 보장 실행(없으면 실행해서 저장 / 있으면 기존 반환)
    try:
        result = await ensure_run_if_empty_uploaded(
            sid=sid,
            category=category,
            subcategory=subcategory,
            modelName=modelName,
            userPrompt=userPrompt,
            promptId=promptId,
            pdfNames=pdf_names,     # 원본 파일명 기준
            max_tokens=max_tokens,
            temperature=temperature,
        )
    finally:
        if cleanup:
            await drop_test_session(sid)

    return result
