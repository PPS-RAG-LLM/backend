from __future__ import annotations

from fastapi import APIRouter, Request, Body, status, Query
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
    # 파이프라인
    extract_pdfs,
    ingest_embeddings,
    ingest_single_pdf,
    execute_search,
    # 관리
    list_indexed_files,
    delete_files_by_names,
    delete_db,
    # 타입
    SinglePDFIngestRequest,
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
    searchType: Optional[Literal["hybrid", "bm25"]] = Field(
        None,
        description="검색 방식 (hybrid | bm25)"
    )


class TaskSecurityConfig(BaseModel):
    maxLevel: int = Field(..., ge=1, description="최대 보안 레벨 (>=1)")
    # 레벨별 키워드는 '@'로 구분된 단일 문자열로 받습니다. (예: '2': '@연봉@급여')
    levels: Dict[str, str] = Field(
        default_factory=dict,
        description="레벨별 키워드 설정. 예: {'1': '@일반@공개', '2': '@연구@연봉', '3': '@부정@개인정보'}",
        examples=[{"1": "@일반@공개", "2": "@연구@연봉", "3": "@부정@개인정보"}]
    )


class SecurityLevelsBody(BaseModel):
    # 작업유형별(doc_gen, summary, qna) 보안설정
    doc_gen: TaskSecurityConfig
    summary: TaskSecurityConfig
    qna: TaskSecurityConfig


class UploadAllBody(BaseModel):
    chunkSize: Optional[int] = Field(
        None, ge=1, description="청크 토큰 크기 (기본 512)"
    )
    overlap: Optional[int] = Field(
        None, ge=0, description="청크 간 오버랩 토큰 수 (기본 64)"
    )
    taskTypes: Optional[List[Literal["doc_gen", "summary", "qna"]]] = Field(
        None,
        description="지정 시 해당 작업유형만 인제스트 (미지정 시 모든 작업유형)"
    )


class ExecuteBody(BaseModel):
    question: str
    topK: int = Field(5, gt=0)
    securityLevel: int = Field(1, ge=1)
    sourceFilter: Optional[List[str]] = None
    taskType: Literal["doc_gen", "summary", "qna"] = Field(
        ..., description="검색할 작업유형"
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
            "examples": [
                {
                    "filesToDelete": [
                        "81._부정청탁및금품등수수의신고사무처리에관한내규_20191128.pdf"
                    ],
                    "taskType": "qna"
                }
            ]
        }
    }


# ============================
# Vector Settings
# ============================

@router.put(
    "/admin/vector/settings",
    summary="0. 벡터 설정(임베딩 모델/검색 방식) 업데이트",
)
async def update_vector_settings(body: VectorSettingsBody):
    set_vector_settings(body.embeddingModel, body.searchType)
    return {"message": "updated", **get_vector_settings()}


@router.get(
    "/admin/vector/settings",
    summary="현재 벡터 설정(임베딩 모델/검색 방식) 조회",
)
async def read_vector_settings():
    return get_vector_settings()


# ============================
# Security Levels (per task type)
# ============================

@router.post("/admin/vector/security-levels",summary="1. 작업유형별 보안레벨 규칙 설정(doc_gen/summary/qna 각각)")
async def set_security_levels(body: SecurityLevelsBody = Body(...)):
    # 내부 함수가 요구하는 dict 형태로 변환
    cfg = {
        "doc_gen": body.doc_gen.model_dump(),
        "summary": body.summary.model_dump(),
        "qna": body.qna.model_dump(),
    }
    return set_security_level_rules_per_task(cfg)


@router.get("/admin/vector/security-levels",summary="작업유형별 보안레벨 규칙 조회")
async def get_security_levels():
    return get_security_level_rules_all()


# ============================
# Pipeline
# ============================

@router.post("/admin/vector/extract",summary="2. row_data의 PDF를 텍스트로 추출 + 작업유형별 보안레벨 산정(meta 반영)")
async def rag_extract_endpoint(request: Request):
    request.app.extra.get("logger", print)(f"[extract] from {request.client.host}")
    return await extract_pdfs()


@router.post("/admin/vector/upload-all",summary="3. 추출된 모든 텍스트를 작업유형별로 임베딩 후 Milvus(서버)에 저장")
async def rag_ingest_endpoint(
    request: Request,
    body: UploadAllBody = Body(
        ...,
        examples={  
            "chunkSize": 512,
            "overlap": 64,
            "taskTypes": ["doc_gen", "summary", "qna"]
        }
        # 또는 examples= { ... }  로 여러 예시 제공 가능
    ),
):
    s = get_vector_settings()
    set_ingest_params(body.chunkSize, body.overlap)
    ingest_params = get_ingest_params()
    request.app.extra.get("logger", print)(
        f"[ingest] from {request.client.host} "
        f"(model={s['embeddingModel']}, searchType={s['searchType']}, "
        f"chunkSize={ingest_params['chunkSize']}, overlap={ingest_params['overlap']}, "
        f"tasks={body.taskTypes})"
    )
    return await ingest_embeddings(
        model_key=s["embeddingModel"],
        chunk_size=ingest_params["chunkSize"],
        overlap=ingest_params["overlap"],
        target_tasks=body.taskTypes,
    )

@router.post("/admin/vector/upload-one",summary="단일 PDF 인제스트(선택 작업유형 지정 가능)")
async def rag_ingest_one_endpoint(body: SingleIngestBody = Body(...)):
    req = SinglePDFIngestRequest(
        pdf_path=body.pdfPath,
        task_types=body.taskTypes,
        workspace_id=body.workspaceId,
    )
    return await ingest_single_pdf(req)


@router.post("/admin/vector/execute",summary="4 관리자: 작업유형별 벡터/하이브리드 검색(보안레벨 적용)")
async def rag_search_endpoint(body: ExecuteBody):
    model_key = get_vector_settings()["embeddingModel"]
    return await execute_search(
        question=body.question,
        top_k=body.topK,
        security_level=body.securityLevel,
        source_filter=body.sourceFilter,
        task_type=body.taskType,
        model_key=model_key,
    )


@router.post(
    "/user/vector/execute",
    summary="사용자: 작업유형별 벡터/하이브리드 검색(보안레벨 적용)"
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


@router.delete(
    "/admin/vector/delete",
    summary="파일 이름 목록(doc_id 스템) 기반으로 해당 문서 청크 삭제(작업유형 전체)"
)
async def delete_vector_files(body: DeleteFilesBody = Body(...)):
    return await delete_files_by_names(body.filesToDelete)


@router.post(
    "/admin/vector/delete-all",
    summary="Milvus 서버 컬렉션 전체 삭제(초기화)"
)
async def rag_delete_db_endpoint(request: Request):
    request.app.extra.get("logger", print)(f"[delete-all] from {request.client.host}")
    return await delete_db()
 