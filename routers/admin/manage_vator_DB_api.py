from __future__ import annotations

from fastapi import APIRouter, Request, Body, status, Query, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Literal

from service.admin.manage_vator_DB import (
    # 설정
    set_vector_settings,
    get_vector_settings,
    # 인제스트 파라미터(청크/오버랩)  ← 추가
    set_ingest_params,
    get_ingest_params,
    # 보안레벨(작업유형별)
    set_security_level_rules_per_task,
    get_security_level_rules_all,
    upsert_security_level_for_task,
    get_security_level_rules_for_task,
    # 파이프라인
    extract_pdfs,
    ingest_embeddings,
    ingest_single_pdf,
    execute_search,
    # 관리
    list_indexed_files,
    list_indexed_files_overview,
    delete_files_by_names,
    delete_db,
    # 타입
    SinglePDFIngestRequest,
    # 파일 저장
    save_raw_file,
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

# ============================
# Request/Response Models
# ============================

class VectorSettingsBody(BaseModel):
    embeddingModel: Optional[str] = Field(
        None,
        description="임베딩 모델 키 (예: bge, embedding_bge_m3, qwen3_4b 등)"
    )
    searchType: Optional[Literal["hybrid", "semantic", "bm25"]] = Field(
        None,
        description="검색 방식 (hybrid | semantic | bm25)"
    )
    chunkSize: Optional[int] = Field(
        None, ge=256, description="청크 토큰 크기 (기본 512)"
    )
    overlap: Optional[int] = Field(
        None, ge=64, description="청크 간 오버랩 토큰 수 (기본 64)"
    )


class TaskSecurityConfig(BaseModel):
    maxLevel: int = Field(..., ge=1, description="최대 보안 레벨 (>=1)")
    # '@'로 구분된 문자열도 허용(레거시 호환)
    levels: Dict[str, str | List[str]] = Field(
        default_factory=dict,
        description="레벨별 키워드 설정. '@' 문자열 또는 키워드 배열 모두 허용",
        examples=[{"1": "@일반@공개", "2": "@연구@연봉", "3": "@부정"}]
    )


class SecurityLevelsBody(BaseModel):
    service: Optional[str] = Field(default="global", description="서비스 이름(드롭다운)")
    # 작업유형별(doc_gen, summary, qna) 보안설정
    doc_gen: TaskSecurityConfig
    summary: TaskSecurityConfig
    qna: TaskSecurityConfig




class ExecuteBody(BaseModel):
    question: str
    topK: int = Field(5, gt=0)
    securityLevel: int = Field(1, ge=1)
    sourceFilter: Optional[List[str]] = None
    taskType: Literal["doc_gen", "summary", "qna"] = Field(
        ..., description="검색할 작업유형"
    )
    searchMode: Optional[Literal["hybrid", "semantic", "bm25"]] = Field(
        None, description="검색 모드 (기본: hybrid)"
    )


class SingleIngestBody(BaseModel):
    pdfPath: str
    taskTypes: Optional[List[Literal["doc_gen", "summary", "qna"]]] = None
    workspaceId: Optional[int] = None


class DeleteFilesBody(BaseModel):
    filesToDelete: List[str] = Field(
        ...,
        description="삭제할 파일 이름 배열 (예: ['사규.pdf','보도자료_20240101.pdf'])",
        examples=[["회사내규.pdf", "20240835_보고서.pdf"]],
    )
    taskType: Optional[Literal["doc_gen", "summary", "qna"]] = Field(
        None,
        description="지정 시 해당 작업유형 데이터만 삭제. 미지정 시 전체 작업유형에서 삭제"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "filesToDelete": [
                    "81._부정청탁및금품등수수의신고사무처리에관한내규_20191128.pdf"
                ],
                "taskType": "qna"
            }
        }
    }


# ============================
# Vector Settings
# ============================

@router.post(
    "/admin/vector/settings",
    summary="0. 벡터 설정(모델/검색/청크) 업데이트",
)
async def update_vector_settings(body: VectorSettingsBody):
    try:
        ret = set_vector_settings(
            embed_model_key=body.embeddingModel,
            search_type=body.searchType,
            chunk_size=body.chunkSize,
            overlap=body.overlap,
        )
        return {"message": "updated", **ret}

    except Exception as e:
        # 기타 오류 (모델 파일 없음 등)
        return {"error": "백터 DB설정 불가(백터 DB를 전부 삭제)", "detail": str(e)}


@router.get(
    "/admin/vector/settings",
    summary="현재 벡터 설정(임베딩 모델/검색 방식) 조회",
)
async def read_vector_settings():
    return get_vector_settings()


# ============================
# Security Levels (per task type)
# ============================

from typing import List as _ListType, Dict as _DictType
from pydantic import conint

TaskLiteral = Literal["doc_gen", "summary", "qna"]

class SecurityLevelSingleBody(BaseModel):
    maxLevel: conint(ge=1) = Field(..., description="최대 보안 레벨(>=1)")
    levels: _DictType[str, _ListType[str] | str] = Field(default_factory=dict)

@router.post(
    "/admin/vector/security-levels/{taskType}",
    summary="1. 작업유형별 보안레벨 규칙 '개별' 저장(doc_gen/summary/qna 중 하나)",
    status_code=status.HTTP_200_OK,
)
async def set_security_levels_one(taskType: TaskLiteral, body: SecurityLevelSingleBody):
    try:
        res = upsert_security_level_for_task(
            task_type=taskType,
            max_level=int(body.maxLevel),
            levels_raw=body.levels,
        )
        return res
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get(
    "/admin/vector/security-levels",
    summary="보안레벨 규칙 조회(전체 또는 특정 작업유형)"
)
async def get_security_levels(taskType: Optional[TaskLiteral] = None):
    if taskType:
        return get_security_level_rules_for_task(taskType)
    return get_security_level_rules_all()


# ============================
# Pipeline
# ============================

@router.post("/admin/vector/upload-file", summary="2. 파일 업로드(row_data)")
async def upload_raw_file(files: List[UploadFile] = File(...)):
    saved_paths = []
    for file in files:
        content = await file.read()
        saved = save_raw_file(file.filename, content)
        saved_paths.append(saved)
    return {"savedPaths": saved_paths, "count": len(saved_paths)}


@router.post("/admin/vector/extract",summary="3. [전처리 부분] row_data의 다양한 문서를 텍스트/표로 추출 + 작업유형별 보안레벨 산정(meta 반영)")
async def rag_extract_endpoint(request: Request):
    request.app.extra.get("logger", print)(f"[extract] from {request.client.host}")
    return await extract_pdfs()


@router.post("/admin/vector/upload-all",summary="4. (설정된 청크/오버랩으로) 모든 작업유형 인제스트")
async def rag_ingest_endpoint(request: Request):
    s = get_vector_settings()
    request.app.extra.get("logger", print)(
        f"[ingest] from {request.client.host} (model={s['embeddingModel']}, searchType={s['searchType']}, chunkSize={s['chunkSize']}, overlap={s['overlap']})"
    )
    return await ingest_embeddings(
        model_key=s["embeddingModel"],  # 모든 TASK_TYPES 대상으로
    )

@router.post("/admin/vector/upload-one",summary="단일 PDF 인제스트(선택 작업유형 지정 가능)")
async def rag_ingest_one_endpoint(body: SingleIngestBody = Body(...)):
    req = SinglePDFIngestRequest(
        pdf_path=body.pdfPath,
        task_types=body.taskTypes,
        workspace_id=body.workspaceId,
    )
    return await ingest_single_pdf(req)


@router.post("/admin/vector/execute",summary="관리자 검색")
async def rag_search_endpoint(body: ExecuteBody):
    model_key = get_vector_settings()["embeddingModel"]
    return await execute_search(
        question=body.question,
        top_k=body.topK,
        security_level=body.securityLevel,
        source_filter=body.sourceFilter,
        task_type=body.taskType,
        model_key=model_key,
        search_type=body.searchMode,  # ← override
    )


@router.post(
    "/user/vector/execute",
    summary="사용자 검색"
)
async def user_rag_search_endpoint(body: ExecuteBody):
    model_key = get_vector_settings()["embeddingModel"]
    return await execute_search(
        question=body.question,
        top_k=body.topK,
        security_level=body.securityLevel,
        source_filter=body.sourceFilter,
        task_type=body.taskType,
        model_key=model_key,
        search_type=body.searchMode,
    )


# ============================
# Management
# ============================

@router.get(
    "/admin/vector/files",
    summary="인덱싱된 파일 목록(작업유형별 집계) 조회"
)
async def list_vector_files_endpoint(
    limit: int = Query(1000, ge=1, le=16384),
    offset: int = Query(0, ge=0),
    q: Optional[str] = Query(None, description="파일명 부분검색"),
    taskType: Optional[Literal["doc_gen", "summary", "qna"]] = Query(None),
):
    return await list_indexed_files(limit=limit, offset=offset, query=q, task_type=taskType)


@router.get(
    "/admin/vector/files/overview",
    summary="작업유형·보안레벨별 집계 + 파일 리스트"
)
async def list_vector_files_overview():
    return await list_indexed_files_overview()


@router.delete(
    "/admin/vector/delete",
    summary="파일 이름 목록(doc_id 스템) 기반 삭제. taskType 지정 시 해당 작업유형만 삭제"
)
async def delete_vector_files(body: DeleteFilesBody = Body(...)):
    return await delete_files_by_names(body.filesToDelete, task_type=body.taskType)


@router.post(
    "/admin/vector/delete",
    summary="[POST] 파일 이름 목록(doc_id 스템) 기반 삭제. taskType 지정 시 해당 작업유형만 삭제"
)
async def delete_vector_files_post(body: DeleteFilesBody = Body(...)):
    return await delete_files_by_names(body.filesToDelete, task_type=body.taskType)


@router.post(
    "/admin/vector/delete-all",
    summary="Milvus 서버 컬렉션 전체 삭제(초기화)"
)
async def rag_delete_db_endpoint(request: Request):
    request.app.extra.get("logger", print)(f"[delete-all] from {request.client.host}")
    return await delete_db()
 