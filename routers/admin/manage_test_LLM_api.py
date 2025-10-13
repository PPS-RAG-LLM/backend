# /home/work/CoreIQ/backend/routers/admin/manage_test_LLM_api.py
from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel
from typing import Optional, List

from service.admin.manage_test_LLM import (
    list_shared_files,
    upload_shared_files,
    delete_shared_files,
    ensure_eval_on_shared_session,
    delete_past_runs,
)

router = APIRouter(
    prefix="/v1/test/llm",
    tags=["Test LLM"],
    responses={200: {"description": "Success"}}
)

# ======= 파일 관리 =======

@router.get("/files", summary="공유 테스트 세션의 원본 파일 목록 조회 (val_data)")
def get_files_route():
    return list_shared_files()

@router.post("/files/upload", summary="공유 테스트 세션에 원본 파일 업로드 + 인제스트")
async def upload_files_route(
    files: List[UploadFile] = File(..., description="PDF 등 원본 파일들")
):
    mem_files: List[tuple[str, bytes]] = []
    for f in files:
        mem_files.append((f.filename, await f.read()))
    return await upload_shared_files(mem_files)

class DeleteFilesBody(BaseModel):
    fileNames: List[str]

@router.delete("/files", summary="공유 테스트 세션에서 원본 파일 삭제(인덱스에서도 제거)")
async def delete_files_route(body: DeleteFilesBody):
    return await delete_shared_files(body.fileNames)

# ======= 평가 실행 (공유 세션 전체를 대상으로 RAG 수행) =======

class EnsureEvalBody(BaseModel):
    category: str
    subcategory: Optional[str] = None
    promptId: int
    modelName: Optional[str] = None
    userPrompt: Optional[str] = None
    top_k: int = 5
    user_level: int = 1
    search_type: Optional[str] = None

@router.post(
    "/runs/ensure-upload",
    summary="공유 세션 전체에서 RAG+LLM 실행 후 저장. 동일 (category, subcategory, promptId, modelName) 이력이 있으면 DB 결과 재사용"
)
async def ensure_eval_route(body: EnsureEvalBody):
    return await ensure_eval_on_shared_session(
        category=body.category,
        subcategory=body.subcategory,
        prompt_id=body.promptId,
        model_name=body.modelName,
        user_prompt=body.userPrompt,
        top_k=body.top_k,
        user_level=body.user_level,
        search_type=body.search_type,
    )

# ======= 과거 답 삭제 =======

class DeleteRunsBody(BaseModel):
    runId: Optional[int] = None
    category: Optional[str] = None
    subcategory: Optional[str] = None
    promptId: Optional[int] = None
    modelName: Optional[str] = None

@router.delete(
    "/runs",
    summary="과거 답 삭제. runId 또는 (category, subcategory, promptId, modelName) 조합으로 삭제"
)
def delete_runs_route(body: DeleteRunsBody):
    return delete_past_runs(
        run_id=body.runId,
        category=body.category,
        subcategory=body.subcategory,
        prompt_id=body.promptId,
        model_name=body.modelName,
    )
