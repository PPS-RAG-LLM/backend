from fastapi import APIRouter, Request, status, Body
from pydantic import BaseModel, Field
from typing import List, Optional
from service.admin.manage_vator_DB import (
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
    set_security_level_rules,
    get_security_level_rules,
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


class SecurityLevelsBody(BaseModel):
    maxLevel: int = Field(
        ..., ge=1, description="최대 보안 레벨", example=3
    )
    levels: dict[str, str] = Field(
        ...,
        description="각 레벨의 키워드 설정. 키는 '2','3' 또는 'level_2','level_3' 형식. level 1 은 설정 불가",
        example={
            "2": "@연구@윤리@연봉",
            "3": "@부정청탁@퇴직금",
        },
    )


class UploadAllBody(BaseModel):
    chunkSize: Optional[int] = Field(
        None,
        ge=1,
        description="청크 토큰 크기 (기본 512)",
        example=400,
    )
    overlap: Optional[int] = Field(
        None,
        ge=0,
        description="청크 간 오버랩 토큰 수 (기본 64)",
        example=50,
    )


@router.put("/admin/vector/settings", summary="벡터 설정(임베딩 모델/검색 타입) 변경. 임베딩 모델, 검색 방식 등 벡터 DB와 관련된 주요 설정을 업데이트합니다. 설정 변경 시 DB 리셋이 필요할 수 있습니다.")
async def update_vector_settings(body: VectorSettingsBody):
    set_vector_settings(body.embeddingModel, body.searchType)
    return {"message": "updated", **get_vector_settings()}

@router.get(
    "/admin/vector/settings",
    summary="현재 벡터 설정(임베딩 모델/검색 타입) 조회",
)
async def read_vector_settings():
    return get_vector_settings()

@router.post("/admin/vector/extract", summary="row_data의 PDF를 텍스트로 추출하고 보안 레벨 규칙을 적용하여 local_data/securityLevelN 구조로 복사합니다.")
async def rag_extract_endpoint(request: Request):
    request.app.extra.get("logger", print)(f"Extract Request from {request.client.host} (fixed paths)")
    return await extract_pdfs()

@router.post("/admin/vector/security-levels",summary="보안 레벨 규칙 설정 (maxLevel 및 각 레벨의 '@' 구분 키워드 설정)",)
async def set_security_levels(
    body: SecurityLevelsBody = Body(
        ..., 
        example={
            "maxLevel": 3,
            "levels": {
                "2": "@연구@윤리@연봉",
                "3": "@부정청탁@퇴직금"
            }
        }
    )
):
    # Convert keys like 'level_1' or '1' to int -> string mapping
    level_map: dict[int, str] = {}
    for k, v in body.levels.items():
        key = k.strip().lower()
        num = None
        if key.startswith("level_"):
            try:
                num = int(key.replace("level_", ""))
            except ValueError:
                num = None
        else:
            try:
                num = int(key)
            except ValueError:
                num = None
        if num is None:
            continue
        level_map[num] = v
    return set_security_level_rules(max_level=body.maxLevel, levels_map=level_map)

@router.get(
    "/admin/vector/security-levels",
    summary="보안 레벨 규칙 조회 (maxLevel 및 각 레벨의 키워드 목록)",
)
async def get_security_levels():
    return get_security_level_rules()

@router.post(
    "/admin/vector/upload-all",
    summary="추출된 모든 텍스트를 임베딩하여 벡터 DB에 저장 [나중에 extract와 합칠 것]",
)
async def rag_ingest_endpoint(
    request: Request,
    body: UploadAllBody = Body(
        default=UploadAllBody(),
        example={
            "chunkSize": 512,
            "overlap": 64,
        },
    ),
):
    from service.admin.manage_vator_DB import get_vector_settings
    model_key = get_vector_settings()["embeddingModel"]
    chunk_size = body.chunkSize
    overlap = body.overlap
    request.app.extra.get("logger", print)(
        f"Bulk Ingest from {request.client.host} (model={model_key}, chunkSize={chunk_size}, overlap={overlap})"
    )
    return await ingest_embeddings(model_key=model_key, chunk_size=chunk_size, overlap=overlap)

# @router.post("/admin/vector/upload", summary="하나 이상의 파일 또는 폴더 경로를 받아 벡터 DB에 저장합니다. 경로가 폴더이면 하위 파일들을 재귀적으로 처리하고, 중복된 파일은 최신 데이터로 갱신합니다.")
# async def rag_ingest_file_endpoint(req: SinglePDFIngestRequest, request: Request):
#     """workspace_id 가 제공되면 SQL(workspace_documents)에 기록됩니다."""
#     request.app.extra.get("logger", print)(f"Single Ingest from {request.client.host}: {req.pdf_path} (model={req.model})")
#     return await ingest_single_pdf(req)

@router.post("/admin/vector/execute", summary="사용자 질의를 받아 벡터 검색 및 스니펫 반환(관리자 용)")
async def rag_search_endpoint(body: ExecuteBody):
    from service.admin.manage_vator_DB import execute_search, get_vector_settings
    model_key = get_vector_settings()["embeddingModel"]
    return await execute_search(
        question=body.question,
        top_k=body.topK,
        security_level=body.securityLevel,
        source_filter=body.sourceFilter,
        model_key=model_key,
    )

@router.post("/user/vector/execute", summary="사용자 질의를 받아 벡터 검색 및 스니펫 반환")
async def user_rag_search_endpoint(body: ExecuteBody):
    from service.admin.manage_vator_DB import execute_search, get_vector_settings
    model_key = get_vector_settings()["embeddingModel"]
    return await execute_search(
        question=body.question,
        top_k=body.topK,
        security_level=body.securityLevel,
        source_filter=body.sourceFilter,
        model_key=model_key,
    )

@router.post("/user/vector/execute", summary="사용자 질의를 받아 벡터 검색 및 스니펫 반환")
async def user_rag_search_endpoint(body: ExecuteBody):
    from service.admin.manage_vator_DB import execute_search, get_vector_settings
    model_key = get_vector_settings()["embeddingModel"]
    return await execute_search(
        question=body.question,
        top_k=body.topK,
        security_level=body.securityLevel,
        source_filter=body.sourceFilter,
        model_key=model_key,
    )

@router.get(
    "/admin/vector/files",
    summary="벡터 DB에 저장된 파일 목록을 조회. q로 파일명(부분 일치) 검색. 파라미터가 없으면 전체 반환.",
)
async def list_vector_files(limit: int = 1000, offset: int = 0, q: Optional[str] = None):
    limit = max(1, min(limit, 16384))
    return await list_indexed_files(limit=limit, offset=offset, query=q)

@router.delete("/admin/vector/delete", summary= "파일 이름 목록을 받아 해당하는 파일들을 벡터 DB에서 삭제합니다.")
async def delete_vector_files(body: DeleteFilesBody = Body(...)):
    return await delete_files_by_names(body.filesToDelete)

@router.post("/admin/vector/delete-all", summary="Milvus Lite의 모든 컬렉션 삭제 [임베딩 모델 변경 시 필요]")
async def rag_delete_db_endpoint(request: Request):
    request.app.extra.get("logger", print)(f"Delete DB Request from {request.client.host}")
    return await delete_db()

