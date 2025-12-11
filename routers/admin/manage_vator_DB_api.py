from __future__ import annotations

from fastapi import APIRouter, Depends, Request, Body, status, Query, UploadFile, File, HTTPException, Form
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Literal
import json as _json
from collections import defaultdict
from pathlib import Path

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
    delete_files_by_names,
    # 파일 저장
    process_saved_raw_files,
)
from service.preprocessing.rag_preprocessing import extract_documents
from storage.db_models import DocumentType
from utils import logger
from utils.auth.session import get_user_id_from_cookie
from utils.documents import save_raw_file
from service.manage_documents.documents import upload_documents # [추가] 통합 업로드 함수

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
from config import config

ADMIN_RAW_DATA_DIR = Path(config.get("admin_raw_data_dir", "storage/raw_files/admin_raw_data"))

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


from repository.rag_settings import get_rag_settings_row

@router.get(
    "/admin/vector/settings",
    summary="현재 벡터 설정(임베딩 모델/검색 방식) 조회",
)
async def read_vector_settings():
    row = get_rag_settings_row()
    return {
        "embeddingModel": row.get("embedding_key"),
        "searchType": row.get("search_type", "hybrid"),
        "chunkSize": int(row.get("chunk_size", 512)),
        "overlap": int(row.get("overlap", 64)),
    }


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
async def rag_full_ingest(
    user_id: int = Depends(get_user_id_from_cookie), 
    files: List[UploadFile] = File(...)
    ):
    # 1) 저장 경로 준비 (기존 로직 유지: securityLevel1)
    target_folder = ADMIN_RAW_DATA_DIR / "securityLevel1"
    target_folder.mkdir(parents=True, exist_ok=True)

    raw_paths = []
    saved_original_names = []

    for f in files:
        filename = f.filename or "unknown"
        # upload_documents 내부에서 파일을 저장하므로 경로만 지정
        file_path = target_folder / filename
        raw_paths.append(str(file_path))
        saved_original_names.append(filename)

    # 2) 통합 업로드 함수 호출
    # 기존 로직이 securityLevel1 폴더에 저장했으므로, 보안 등급을 1로 강제 설정하여 일관성 유지
    default_levels = {"qna": 1, "summary": 1, "doc_gen": 1}

    result = await upload_documents(
        user_id=user_id,
        files=files,
        raw_paths=raw_paths,
        add_to_workspaces=None,
        doc_type=DocumentType.ADMIN,  # 관리자 문서
        override_security_levels=default_levels
    )
    ingest_result = {"save": saved_original_names, "ingest": result}
    logger.info(f"[API] rag_full_ingest 호출 완료, 결과 ingest_result=\n\n{ingest_result}\n")

    return ingest_result


@router.post("/admin/vector/execute",summary="관리자 검색")
async def rag_search_endpoint(body: ExecuteBody):
    print(f"🎯 [API] 관리자 검색 엔드포인트 호출: question='{body.question}', topK={body.topK}, rerank_topN={body.rerank_topN}")
    
    model_key = get_rag_settings_row()["embedding_key"]
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
    model_key = get_rag_settings_row()["embedding_key"]
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
    
    items = await list_indexed_files(limit=16384, offset=0, query=None, task_type=None)
    # agg: task_type -> level -> count
    agg: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for it in items:
        agg[it["taskType"]][int(it["securityLevel"])] += it["chunkCount"]
    # 보기 좋게 변환
    overview = {
        t: {str(lv): agg[t][lv] for lv in sorted(agg[t].keys())} for t in agg.keys()
    }
    return {"overview": overview, "items": items}


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
        elif isinstance(obj, list):
            # 리스트인 경우 [1, 2, 3] -> qna=1, summary=2, doc_gen=3 (순서 가정)
            # 또는 tasks 목록을 알 수 없으므로, 이 함수 단독으로는 처리가 어렵지만
            # 일단 가능한 경우만 처리
            # 여기서는 task 순서를 고정(TASK_TYPES)한다고 가정하거나, 호출처에서 처리해야 함.
            # 하지만 사용자 요청에 따라 [1,2,3] 형태를 지원하기 위해 간단히 매핑
            # (주의: tasks 파라미터와 순서가 일치한다고 가정)
            pass 
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
    summary="파일 업로드 후 레벨 지정하여 바로 전처리 및 인제스트 (통합 업로드 방식)"
    )
async def override_levels_upload_form(
    user_id: int = Depends(get_user_id_from_cookie),
    files: List[UploadFile] = File(...),
    tasks: Optional[str] = Form(None),
    level_for_tasks: Optional[str] = Form(None),
    qna_level: Optional[str] = Form(None),
    summary_level: Optional[str] = Form(None),
    doc_gen_level: Optional[str] = Form(None),
):
    # 1) tasks, levels 파싱 (저장 폴더 결정을 위해 먼저 수행)
    tlist = None
    if tasks:
        tlist = [t.strip() for t in tasks.split(",") if t.strip() in TASK_TYPES] or None

    lvmap = {}
    try:
        # [1,2,3] 형태의 리스트 문자열 처리 시도 (tasks 순서와 매핑 가정)
        s_lvl = str(level_for_tasks).strip()
        if s_lvl.startswith("[") and s_lvl.endswith("]"):
            try:
                lvl_arr = _json.loads(s_lvl)
                if isinstance(lvl_arr, list):
                    # tlist가 있다면 순서대로 매핑, 없으면 TASK_TYPES 순서대로? 
                    # 사용자 요구사항: tasks=[qna, summary, doc_gen], level_for_tasks=[1,2,3]
                    # 따라서 tlist가 있으면 1:1 매핑
                    mapping_target = tlist if tlist else TASK_TYPES
                    for i, t in enumerate(mapping_target):
                        if i < len(lvl_arr):
                             lvmap[t] = max(1, int(lvl_arr[i]))
            except Exception:
                pass

        if not lvmap:
            lvmap = _parse_level_for_tasks_flex(level_for_tasks, qna_level, summary_level, doc_gen_level)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 저장할 폴더 결정: 지정된 레벨 중 가장 높은 레벨 폴더에 저장 (또는 단일 레벨)
    effective_levels = [v for k, v in lvmap.items() if (not tlist) or (k in tlist)]
    max_lvl = max(effective_levels) if effective_levels else 1
    
    # 2) 저장 경로 준비 (documents.py의 upload_documents는 raw_paths를 받음)
    # 기존 로직과 동일한 폴더 구조 유지: ADMIN_RAW_DATA_DIR / securityLevel{max_lvl}
    target_folder = ADMIN_RAW_DATA_DIR / f"securityLevel{max_lvl}"
    target_folder.mkdir(parents=True, exist_ok=True)

    raw_paths = []
    saved_original_names = []

    for f in files:
        filename = f.filename or "unknown"
        # upload_documents 내부에서 파일을 저장하므로 여기서는 경로만 지정해줌
        # 파일명 충돌 방지를 위해 기존 save_raw_file 로직을 따를 수도 있으나,
        # upload_documents는 주어진 경로에 파일을 씀.
        # 여기서는 파일명 그대로 사용하거나 필요시 중복 처리 필요.
        # 일단 파일명 그대로 사용
        file_path = target_folder / filename
        raw_paths.append(str(file_path))
        saved_original_names.append(filename)
        
    # 3) 통합 업로드 함수 호출
    # override_security_levels 파라미터를 통해 강제 레벨 적용
    result = await upload_documents(
        user_id=user_id,
        files=files,
        raw_paths=raw_paths,
        add_to_workspaces=None,
        doc_type=DocumentType.ADMIN,  # 관리자 문서
        override_security_levels=lvmap # 강제 레벨
    )
    
    return {"saved": saved_original_names, "ingest_result": result}
