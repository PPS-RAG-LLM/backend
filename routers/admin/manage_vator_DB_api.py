from fastapi import APIRouter, Request, status
from pydantic import BaseModel, Field

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


@router.post("/admin/vector/extract", summary="PDF 경로를 받아 텍스트와 메타를 추출")
async def rag_extract_endpoint(req: PDFExtractRequest, request: Request):
    request.app.extra.get("logger", print)(f"Extract Request from {request.client.host} -> {req.dir_path}")
    return await extract_pdfs(req)


class VectorSettingsBody(BaseModel):
    embeddingModel: str | None = Field(None, description="'bge' or 'qwen'")
    searchType: str | None = Field(None, description="'hybrid' or 'bm25'")


@router.put("/admin/vector/settings", summary="벡터 설정(임베딩 모델/검색 타입) 변경")
async def update_vector_settings(body: VectorSettingsBody):
    set_vector_settings(body.embeddingModel, body.searchType)
    return {"message": "updated", **get_vector_settings()}


@router.post("/admin/vector/ingest-file", summary="단일 PDF만 벡터 DB에 반영")
async def rag_ingest_file_endpoint(req: SinglePDFIngestRequest, request: Request):
    request.app.extra.get("logger", print)(f"Single Ingest from {request.client.host}: {req.pdf_path} (model={req.model})")
    return await ingest_single_pdf(req)


@router.post("/admin/vector/ingest", summary="추출된 모든 텍스트를 임베딩하여 벡터 DB에 저장")
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


@router.post("/admin/vector/search", summary="사용자 질의를 받아 벡터 검색 및 스니펫 반환")
async def rag_search_endpoint(req: RAGSearchRequest, request: Request):
    request.app.extra.get("logger", print)(f"Search Request from {request.client.host}: '{req.query}' (level={req.user_level}, model={req.model})")
    return await search_documents(req)


@router.post("/admin/vector/delete-db", summary="Milvus Lite의 모든 컬렉션 삭제")
async def rag_delete_db_endpoint(request: Request):
    request.app.extra.get("logger", print)(f"Delete DB Request from {request.client.host}")
    return await delete_db()
