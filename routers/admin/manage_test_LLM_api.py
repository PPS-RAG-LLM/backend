# /home/work/CoreIQ/backend/routers/admin/manage_test_LLM_api.py
from fastapi import APIRouter, UploadFile, File, Form
from pydantic import BaseModel
from typing import Optional, List

from service.admin.manage_test_LLM import (
    list_shared_files,
    upload_shared_files,
    delete_shared_files,
    ensure_eval_on_shared_session,
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
    # 메모리에서 바로 저장/인제스트
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

@router.post(
    "/runs/ensure-upload",
    summary="공유 세션 전체(현재 인제스트된 모든 파일)에서 RAG+LLM 실행 후 llm_eval_runs에 저장(동일키 존재시 기존 run 재사용)"
)
async def ensure_eval_route(
    category: str = Form(..., description="qa | qna | doc_gen | summary"),
    subcategory: Optional[str] = Form(None),
    promptId: int = Form(...),
    modelName: Optional[str] = Form(None),
    userPrompt: Optional[str] = Form(None),
    top_k: int = Form(5),
    user_level: int = Form(1),
    search_type: Optional[str] = Form(None),
):
    return await ensure_eval_on_shared_session(
        category=category,
        subcategory=subcategory,
        prompt_id=promptId,
        model_name=modelName,
        user_prompt=userPrompt,
        top_k=top_k,
        user_level=user_level,
        search_type=search_type,
    )
