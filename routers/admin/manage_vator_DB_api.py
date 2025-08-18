from fastapi import APIRouter, Request, status, Body
from pydantic import BaseModel, Field
from typing import List, Optional
from service.admin.manage_vator_DB import (
    PDFExtractRequest,
    RAGSearchRequest,
    SinglePDFIngestRequest,
    extract_pdfs,
    ingest_embeddings,
    ingest_single_pdf,
    search_documents,
    delete_db,
    set_vector_settings,
    get_vector_settings,
    list_indexed_files,
    delete_files_by_names,
)

router = APIRouter(
    prefix="/v1",
    tags=["Admin Document"],
    responses={
        status.HTTP_200_OK: {"description": "Successful Response"},
        status.HTTP_401_UNAUTHORIZED: {"description": "Unauthorized"},
        status.HTTP_403_FORBIDDEN: {"description": "Forbidden"},
        status.HTTP_404_NOT_FOUND: {"description": "Not found"},
    },
)

class VectorSettingsBody(BaseModel):
    embeddingModel: Optional[str] = Field(
        None,
        description="임베딩 모델 키",
        examples=["bge", "qwen"],
    )
    searchType: Optional[str] = Field(
        None,
        description="검색 타입",
        examples=["hybrid", "bm25"],
    )

class DeleteFilesBody(BaseModel):
    filesToDelete: List[str] = Field(..., description="삭제할 파일 이름 배열", examples=[["회사내규.pdf", "20240835_보고서.pdf"]])

class ExecuteBody(BaseModel):
    question: str
    topK: int = Field(5, gt=0)
    securityLevel: int = Field(1, ge=1)
    sourceFilter: Optional[List[str]] = None
    model: Optional[str] = None


@router.put("/admin/vector/settings", summary="벡터 설정(임베딩 모델/검색 타입) 변경. 임베딩 모델, 검색 방식 등 벡터 DB와 관련된 주요 설정을 업데이트합니다. 설정 변경 시 DB 리셋이 필요할 수 있습니다.")
async def update_vector_settings(body: VectorSettingsBody):
    set_vector_settings(body.embeddingModel, body.searchType)
    return {"message": "updated", **get_vector_settings()}

@router.post("/admin/vector/extract", summary="PDF등 파일 경로를 받아 텍스트와 메타를 추출 [추후 변경 예정=> ]")
async def rag_extract_endpoint(req: PDFExtractRequest, request: Request):
    request.app.extra.get("logger", print)(f"Extract Request from {request.client.host} -> {req.dir_path}")
    return await extract_pdfs(req)

@router.post("/admin/vector/upload-all", summary="추출된 모든 텍스트를 임베딩하여 벡터 DB에 저장 [나중에 extract와 합칠 것]")
async def rag_ingest_endpoint(request: Request):
    body: dict = {}
    try:
        body = await request.json()
    except Exception:
        pass
    model_key = body.get("model") if isinstance(body, dict) else None
    from service.admin.manage_vator_DB import get_vector_settings
    if model_key is None:
        model_key = get_vector_settings()["embeddingModel"]
    request.app.extra.get("logger", print)(f"Bulk Ingest from {request.client.host} (model={model_key})")
    return await ingest_embeddings(model_key=model_key)

@router.post("/admin/vector/upload", summary="하나 이상의 파일 또는 폴더 경로를 받아 벡터 DB에 저장합니다. 경로가 폴더이면 하위 파일들을 재귀적으로 처리하고, 중복된 파일은 최신 데이터로 갱신합니다.")
async def rag_ingest_file_endpoint(req: SinglePDFIngestRequest, request: Request):
    """workspace_id 가 제공되면 SQL(workspace_documents)에 기록됩니다."""
    request.app.extra.get("logger", print)(f"Single Ingest from {request.client.host}: {req.pdf_path} (model={req.model})")
    return await ingest_single_pdf(req)

@router.post("/admin/vector/execute", summary="사용자 질의를 받아 벡터 검색 및 스니펫 반환(관리자 용)")
async def rag_search_endpoint(body: ExecuteBody):
    from service.admin.manage_vator_DB import execute_search
    return await execute_search(
        question=body.question,
        top_k=body.topK,
        security_level=body.securityLevel,
        source_filter=body.sourceFilter,
        model_key=body.model,
    )

@router.post("/user/vector/execute", summary="사용자 질의를 받아 벡터 검색 및 스니펫 반환")
async def user_rag_search_endpoint(body: ExecuteBody):
    from service.admin.manage_vator_DB import execute_search
    return await execute_search(
        question=body.question,
        top_k=body.topK,
        security_level=body.securityLevel,
        source_filter=body.sourceFilter,
        model_key=body.model,
    )

@router.post("/user/vector/execute", summary="사용자 질의를 받아 벡터 검색 및 스니펫 반환")
async def user_rag_search_endpoint(body: ExecuteBody):
    from service.admin.manage_vator_DB import execute_search
    return await execute_search(
        question=body.question,
        top_k=body.topK,
        security_level=body.securityLevel,
        source_filter=body.sourceFilter,
        model_key=body.model,
    )

@router.get("/admin/vector/files", summary="벡터 DB에 저장된 파일 목록을 조회하거나, 파일 이름 또는 보안 레벨로 검색합니다. 파라미터가 없으면 전체 파일 목록을 반환합니다.")
async def list_vector_files(limit: int = 1000, offset: int = 0):
    limit = max(1, min(limit, 16384))
    return await list_indexed_files(limit=limit, offset=offset)

@router.delete("/admin/vector/delete", summary= "파일 이름 목록을 받아 해당하는 파일들을 벡터 DB에서 삭제합니다.")
async def delete_vector_files(body: DeleteFilesBody = Body(...)):
    return await delete_files_by_names(body.filesToDelete)

@router.post("/admin/vector/delete-all", summary="Milvus Lite의 모든 컬렉션 삭제 [임베딩 모델 변경 시 필요]")
async def rag_delete_db_endpoint(request: Request):
    request.app.extra.get("logger", print)(f"Delete DB Request from {request.client.host}")
    return await delete_db()

