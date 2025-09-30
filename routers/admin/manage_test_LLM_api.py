# /home/work/CoreIQ/backend/routers/admin/manage_test_LLM_api.py
from fastapi import APIRouter, UploadFile, File, Form
from typing import List, Optional, Tuple

from service.admin.manage_test_LLM import ensure_run_if_empty_uploaded

router = APIRouter(prefix="/v1/test/llm", tags=["Test LLM"], responses={200: {"description": "Success"}})

@router.post(
    "/runs/ensure-upload",
    summary="업로드한 PDF로 RAG+LLM 실행 후 llm_eval_runs에 저장(동일키가 있으면 기존 결과 반환)",
)
async def ensure_run_if_empty_upload_route(
    category: str = Form(..., description="qa | qna | doc_gen | summary"),
    subcategory: Optional[str] = Form(None, description="세부 테스크(= template.name)"),
    promptId: Optional[int] = Form(None, description="doc_gen일 때 필수 권장"),
    modelName: Optional[str] = Form(None, description="명시 모델명(없으면 디폴트 선택 로직)"),
    userPrompt: Optional[str] = Form(None, description="사용자 추가 프롬프트"),
    files: List[UploadFile] = File(..., description="평가에 사용할 PDF들(이름으로 동일성 판단)"),
    top_k: int = Form(5),
    user_level: int = Form(1),
):
    # 메모리로 읽어서 서비스 레이어에 전달 ([(filename, bytes), ...])
    mem_files: List[Tuple[str, bytes]] = []
    for f in files:
        mem_files.append((f.filename, await f.read()))

    result = await ensure_run_if_empty_uploaded(
        category=category,
        subcategory=subcategory,
        prompt_id=promptId,
        model_name=modelName,
        user_prompt=userPrompt,
        uploaded_files=mem_files,
        top_k=top_k,
        user_level=user_level,
    )
    return result
