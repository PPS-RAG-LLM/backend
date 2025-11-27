from __future__ import annotations

from fastapi import APIRouter, Request, Body, status, Query, UploadFile, File, HTTPException, Form
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Literal, Any
import json as _json

from pymilvus.grpc_gen.common_pb2 import RequestTSO

from service.admin.manage_vator_DB import (
    TASK_TYPES,
    OverrideLevelsRequest,
    delete_collection,
    override_levels_and_ingest,
    # 설정
    set_vector_settings,
    list_available_embedding_models,
    get_security_level_rules_all,
    upsert_security_level_for_task,
    get_security_level_rules_for_task,
    # 파이프라인
    ingest_embeddings,
    execute_search,
    # 관리
    list_indexed_files,
    list_indexed_files_overview,
    delete_files_by_names,
    # 파일 저장
    process_saved_raw_files,
)
from service.preprocessing.rag_preprocessing import extract_documents
from storage.db_models import DocumentType
from utils import logger
from utils.documents import save_raw_file
from utils.model_load import get_vector_settings
router = APIRouter(
    prefix="/v1",
    tags=["Admin Document - RAG"],
    responses={
        status.HTTP_200_OK: {"description": "Successful Response"},
        status.HTTP_401_UNAUTHORIZED: {"description": "Unauthorized"},
        status.HTTP_403_FORBIDDEN: {"description": "Forbidden"},
        status.HTTP_404_NOT_FOUND: {"description": "Not found"},
    },
)
logger = logger(__name__)
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
    question: str = Field(..., examples=["회사의 부정청탁 제도에 대해 알려주세요"])
    topK: int = Field(50, gt=0, description="임베딩 후보 개수")
    rerank_topN: int = Field(5, gt=0, description="리랭크 후 최종 반환 개수")
    securityLevel: int = Field(1, ge=1)
    sourceFilter: Optional[List[str]] = None
    taskType: Literal["doc_gen", "summary", "qna"]
    searchMode: Optional[Literal["hybrid", "semantic", "bm25"]] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "question": "회사의 부정청탁 제도에 대해 알려주세요",
                "topK": 50,
                "rerank_topN": 5,
                "securityLevel": 3,
                "sourceFilter": ["회사규정.pdf", "복리후생안내.pdf"],
                "taskType": "qna",
                "searchMode": "hybrid",
            }
        }
    }


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


@router.get(
    "/admin/vector/embedding-models",
    summary="사용 가능한 임베딩 모델 목록 조회",
)
async def list_embedding_models():
    """
    ./storage/embedding-models 폴더 내의 모델 폴더명들을 반환.
    - embedding_ 접두사가 있으면 제거 (예: embedding_bge_m3 → bge_m3)
    """
    models = list_available_embedding_models()
    return {
        "models": models,
        "count": len(models)
    }


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

@router.post("/admin/vector/full-ingest", summary="전체 파일 추출 및 저장 인제스트") # TODO : MinIO 마이그레이션 필요
async def rag_full_ingest(files: List[UploadFile] = File(...)):
    # 1) RAW 저장
    rel_paths = []
    for f in files:
        rel_paths.append(save_raw_file(f.filename, folder="row_data", content=await f.read()))

    extract_result = await extract_documents(rel_paths)
    # 3) 인제스트
    settings = get_vector_settings()

    # 2) 추출

    ingest_result = await ingest_embeddings(
        model_key=settings["embeddingModel"],
        max_token=int(settings["chunkSize"]),
        overlab=int(settings["overlap"]),
    )
    logger.debug(f"\n\n[API] ingest_result: {ingest_result}\n\n")
    return {
        "uploaded": rel_paths,
        "extract": extract_result,
        "ingest": ingest_result,
    }


@router.post("/admin/vector/execute",summary="관리자 검색")
async def rag_search_endpoint(body: ExecuteBody):
    print(f"🎯 [API] 관리자 검색 엔드포인트 호출: question='{body.question}', topK={body.topK}, rerank_topN={body.rerank_topN}")
    
    model_key = get_vector_settings()["embeddingModel"]
    print(f"🎯 [API] execute_search 호출 시작...")
    
    result = await execute_search(
        question=body.question,
        top_k=body.topK,  # 임베딩 후보 개수
        rerank_top_n=body.rerank_topN,  # 최종 반환 개수
        security_level=body.securityLevel,
        source_filter=body.sourceFilter,
        task_type=body.taskType,
        model_key=model_key,
        search_type=body.searchMode,  # ← override
    )
    
    print(f"🎯 [API] execute_search 호출 완료, 결과 hits={len(result.get('hits', []))}")
    return result


@router.post(
    "/user/vector/execute", summary="사용자 검색"
)
async def user_rag_search_endpoint(body: ExecuteBody):
    model_key = get_vector_settings()["embeddingModel"]
    return await execute_search(
        question=body.question,
        top_k=body.topK,  # 임베딩 후보 개수
        rerank_top_n=body.rerank_topN,  # 최종 반환 개수
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


@router.post("/admin/vector/delete", summary="[POST] 파일 이름 목록(doc_id 스템) 기반 삭제. taskType 지정 시 해당 작업유형만 삭제")
async def delete_vector_files_post(body: DeleteFilesBody = Body(...)):
    return await delete_files_by_names(body.filesToDelete, task_type=body.taskType)


@router.post("/admin/vector/delete-admin-collection", summary="Milvus 서버 Admin 컬렉션 삭제(초기화)")
async def rag_delete_admin_collection_endpoint(request: Request):
    logger.debug(f"[delete-admin-collection] from {request.client.host}")
    return await delete_collection(collection_key=DocumentType.ADMIN.value)

@router.post("/admin/vector/delete-all-collections", summary="Milvus 서버 전체 컬렉션 삭제(초기화)")
async def rag_delete_db_endpoint(request: Request):
    logger.debug(f"[delete-all] from {request.client.host}")
    return await delete_collection(collection_key=None)


# 중복 import 제거 및 상단으로 승격됨

def _parse_level_for_tasks_flex(
    raw: Optional[str],
    qna_level: Optional[str] = None,
    summary_level: Optional[str] = None,
    doc_gen_level: Optional[str] = None,
) -> Dict[str, int]:
    # 1) 개별 필드가 오면 우선 사용 (빈 문자열은 무시)
    def _as_int(x: Optional[str]):
        if x is None:
            return None
        s = str(x).strip()
        if s == "":
            return None
        try:
            return int(s)
        except Exception:
            return None

    lvl_map: Dict[str, int] = {}
    q = _as_int(qna_level); s = _as_int(summary_level); d = _as_int(doc_gen_level)
    if q is not None: lvl_map["qna"] = max(1, q)
    if s is not None: lvl_map["summary"] = max(1, s)
    if d is not None: lvl_map["doc_gen"] = max(1, d)
    if lvl_map:
        return lvl_map

    if raw is None or str(raw).strip() == "":
        raise ValueError("level_for_tasks 값이 비어 있습니다.")

    s = str(raw).strip()

    # 2) 숫자 하나만 오면 모든 task 동일 적용
    if s.isdigit():
        v = max(1, int(s))
        return {"qna": v, "summary": v, "doc_gen": v}

    # 3) JSON 시도
    try:
        obj = _json.loads(s)
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                if k in ("qna", "summary", "doc_gen"):
                    out[k] = max(1, int(v))
            if out:
                return out
    except Exception:
        pass

    # 4) "qna:2,summary:1" / "qna=2&summary=1" 류 파싱
    cand = s.replace("&", ",")
    parts = [p.strip() for p in cand.split(",") if p.strip()]
    out = {}
    for p in parts:
        if ":" in p:
            k, v = p.split(":", 1)
        elif "=" in p:
            k, v = p.split("=", 1)
        else:
            continue
        k = k.strip()
        if k in ("qna", "summary", "doc_gen"):
            try:
                out[k] = max(1, int(v))
            except Exception:
                pass
    if out:
        return out

    raise ValueError('level_for_tasks 파싱 실패. 예) {"qna":2,"summary":1} 또는 "qna:2,summary:1" 또는 "2"')
    
@router.post("/admin/vector/override-levels-upload", 
    summary="-- 단일 파일 올리기 "
    )
async def override_levels_upload_form(
    files: List[UploadFile] = File(...),
    tasks: Optional[str] = Form(None),
    level_for_tasks: Optional[str] = Form(None),
    qna_level: Optional[str] = Form(None),
    summary_level: Optional[str] = Form(None),
    doc_gen_level: Optional[str] = Form(None),
):
    # # 1) 파일 저장
    saved_original_names: List[str] = []
    saved_rel_paths : List[str] = []
    for f in files:
        # save_raw_file이 상대 경로를 돌려주도록 수정, 
        # 단건 전처리/등록을 담당하는 새 헬퍼들을 추가
        content = await f.read()
        rel_path = save_raw_file(f.filename, folder="row_data", content=content)
        saved_original_names.append(f.filename)
        saved_rel_paths.append(rel_path)

    processed_docs = await process_saved_raw_files(saved_rel_paths)
    target_tokens = [doc["doc_id"] for doc in processed_docs] or saved_original_names
    logger.debug(f"🎯 [API] target_tokens: {target_tokens}")

    # 3) task 목록
    tlist = None
    if tasks:
        tlist = [t.strip() for t in tasks.split(",") if t.strip() in TASK_TYPES] or None

    # 4) 레벨 파싱(유연)
    try:
        lvmap = _parse_level_for_tasks_flex(level_for_tasks, qna_level, summary_level, doc_gen_level)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 5) 지정 파일만 레벨 오버라이드 + 해당 파일만 인제스트
    req = OverrideLevelsRequest(files=target_tokens, level_for_tasks=lvmap, tasks=tlist)
    result = await override_levels_and_ingest(req)
    return {"saved": saved_original_names, "ingest_result": result}
