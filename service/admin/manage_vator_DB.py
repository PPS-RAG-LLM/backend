# === Vector DB Service (Milvus Server, Pro) ===
# - 작업유형(task_type)별 보안레벨 관리: doc_gen | summary | qna
# - Milvus Docker 서버 전용 (Lite 제거)
# - 벡터/하이브리드 검색 지원, 실행 로그 적재

from __future__ import annotations
import asyncio
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel
from config import config as app_config
from repository.rag_settings import get_rag_settings_row, set_rag_settings_row
from repository import security_level as security_repo
from repository.documents import (
    delete_document_vectors,
    delete_documents_by_type_and_ids,
    document_has_vectors,
    get_document_by_source_path,
    get_list_indexed_files,
    insert_document_vectors,
    list_documents_by_type,
    purge_documents_by_collection,
    upsert_document,
    fetch_document_metadata_by_doc_ids,
    bulk_upsert_document_metadata, 
)
from service.preprocessing.rag_preprocessing import ext
from service.retrieval.interface import SearchRequest, retrieval_service
from utils.documents import generate_doc_id
from storage.db_models import DocumentType

from ..vector_db import (
    get_milvus_client,
    milvus_has_data,
)
from service.retrieval.common import (
    parse_doc_version, 
    determine_level_for_task,
)
from utils.model_load import (
    invalidate_embedder_cache,
)
from utils import now_kst, now_kst_string, logger
logger = logger(__name__)


# -------------------------------------------------
# 경로 상수
# -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent  # .../backend/service/admin
PROJECT_ROOT = BASE_DIR.parent.parent  # .../backend
_RETRIEVAL_CFG: Dict[str, Any] = app_config.get("retrieval", {}) or {}
_RETRIEVAL_PATHS: Dict[str, str] = _RETRIEVAL_CFG.get("paths", {}) or {}
_MILVUS_CFG: Dict[str, Any] = _RETRIEVAL_CFG.get("milvus", {}) or {}

MODEL_ROOT_DIR = Path(app_config.get("models_dir", {}).get("embedding_model_path", "storage/embedding-models"))
ADMIN_RAW_DATA_DIR = Path(app_config.get("admin_raw_data_dir", "storage/raw_files/admin_raw_data"))

DATABASE_CFG = app_config.get("database", {}) or {}
SQLITE_DB_PATH = (PROJECT_ROOT / Path(DATABASE_CFG.get("path", "storage/pps_rag.db"))).resolve()
ADMIN_COLLECTION = _MILVUS_CFG.get("ADMIN_DOCS", "admin_docs_collection")
TASK_TYPES = tuple(_RETRIEVAL_CFG.get("task_types") or ("doc_gen", "summary", "qna"))

ZERO_WIDTH_RE = re.compile(r"[\u200B-\u200D\u2060\uFEFF]")
CONTROL_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")
MULTISPACE_LINE_END_RE = re.compile(r"[ \t]+\n")
NEWLINES_RE = re.compile(r"\n{3,}")
ADMIN_DOC_TYPE = DocumentType.ADMIN.value

######


def _max_security_level(sec_map: Dict[str, int]) -> int:
    if not sec_map:
        return 1
    levels = [int(v) for v in sec_map.values() if isinstance(v, (int, float, str))]
    parsed = []
    for lv in levels:
        try:
            parsed.append(int(lv))
        except Exception:
            continue
    return max(parsed or [1])

def register_admin_document(
    *,
    doc_id: str,
    filename: str,
    rel_text_path: str,
    rel_source_path: str,
    sec_map: Dict[str, int],
    version: int,
    preview: str,
    tables: List[Dict[str, Any]],
    total_pages: int,
    pages: Dict[str, Any],
    source_ext: str,
    extraction_info: Dict[str, Any],
) -> None:

    payload = {
        "security_levels": sec_map,
        "version": int(version),
        "preview": preview,
        "tables": tables or [],
        "total_pages": int(total_pages or 0),
        "saved_files": {"text": rel_text_path, "source": rel_source_path},
        "pages": pages or {},
        "source_ext": source_ext,
        "doc_rel_key": rel_source_path,
        "extraction_info": extraction_info,
    }

    upsert_document(
        doc_id=doc_id,
        doc_type=ADMIN_DOC_TYPE,
        filename=filename,
        source_path=rel_source_path,
        security_level=_max_security_level(sec_map),
        payload=payload,
    )


def _doc_matches_tokens(doc: Dict[str, Any], tokens: set[str]) -> bool:
    if not tokens:
        return True
    payload = doc.get("payload") or {}
    candidates = [
        doc.get("doc_id"),
        doc.get("filename"),
        Path(str(doc.get("filename") or "")).stem,
        doc.get("source_path"),
        Path(str(doc.get("source_path") or "")).name,
        payload.get("doc_rel_key"),
    ]
    if payload.get("doc_rel_key"):
        candidates.append(Path(str(payload.get("doc_rel_key"))).name)
        candidates.append(Path(str(payload.get("doc_rel_key"))).stem)
    for value in candidates:
        if value and str(value).lower() in tokens:
            return True
    return False


def _load_admin_documents(file_keys_filter: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    docs = list_documents_by_type(ADMIN_DOC_TYPE)
    if not file_keys_filter:
        return docs
    tokens = {str(f).lower() for f in file_keys_filter if str(f).strip()}
    if not tokens:
        return docs
    return [doc for doc in docs if _doc_matches_tokens(doc, tokens)]


def _doc_name_tokens(doc: Dict[str, Any]) -> set[str]:
    tokens: set[str] = set()

    def _push(value: Any) -> None:
        if not value:
            return
        try:
            s = str(value).strip()
        except Exception:
            return
        if not s:
            return
        tokens.add(s.lower())
        try:
            p = Path(s)
            tokens.add(p.name.lower())
            tokens.add(p.stem.lower())
        except Exception:
            pass

    _push(doc.get("doc_id"))
    _push(doc.get("filename"))
    _push(doc.get("source_path"))
    payload = doc.get("payload") or {}
    _push(payload.get("doc_rel_key"))
    saved = payload.get("saved_files") or {}
    for path in saved.values():
        _push(path)
    return {t for t in tokens if t}


def _build_doc_name_index() -> Dict[str, str]:
    docs = list_documents_by_type(ADMIN_DOC_TYPE)
    index: Dict[str, str] = {}
    for doc in docs:
        doc_id = str(doc.get("doc_id") or "").strip()
        if not doc_id:
            continue
        for token in _doc_name_tokens(doc):
            index.setdefault(token, doc_id)
    return index

# -------------------------------------------------
# SQLite 유틸
# -------------------------------------------------

async def process_saved_raw_files(rel_paths: List[str]) -> List[Dict[str, Any]]:
    if not rel_paths:
        return []
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _process_saved_raw_files_sync, rel_paths)

def _process_saved_raw_files_sync(rel_paths: List[str]) -> List[Dict[str, Any]]:
    level_rules = get_security_level_rules_all()
    results: List[Dict[str, Any]] = []
    for rel in rel_paths:
        info = _process_single_raw_file(rel, level_rules)
        if info:
            results.append(info)
    return results

def _process_single_raw_file(rel_path: str, level_rules: Dict[str, Dict]) -> Optional[Dict[str, Any]]:
    raw_path = (ADMIN_RAW_DATA_DIR / rel_path).resolve()
    if not raw_path.exists():
        logger.warning("[ProcessRaw] RAW 파일을 찾을 수 없습니다: %s", rel_path)
        return None

    try:
        rel_from_raw = raw_path.relative_to(ADMIN_RAW_DATA_DIR)
    except ValueError:
        rel_from_raw = raw_path

    file_ext = ext(raw_path)
    pages_text_dict: Dict[int, str] = {}
    total_pages = 0

    try:
        from service.preprocessing.rag_preprocessing import extract_any
        if file_ext == ".pdf":
            from service.preprocessing.extension.pdf_preprocessing import extract_pdf_with_tables
            text, tables, pages_text_dict, total_pages = extract_pdf_with_tables(raw_path)
        else:
            text, tables = extract_any(raw_path)
    except Exception:
        logger.exception("[ProcessRaw] 추출 실패: %s", raw_path)
        return None

    tables = tables or []
    text = text or ""
    combined_for_level = text + "\n\n" + "\n\n".join(t.get("text", "") for t in tables)
    sec_map = {
        task: determine_level_for_task(
            combined_for_level,
            level_rules.get(task, {"maxLevel": 1, "levels": {}}),
        )
        for task in TASK_TYPES
    }
    max_sec = max(sec_map.values()) if sec_map else 1
    rel_text_path = Path(f"securityLevel{int(max_sec)}") / rel_from_raw.with_suffix(".txt")

    from service.preprocessing.rag_preprocessing import _clean_text as clean_text

    preview = (clean_text(text[:200].replace("\n", " ")) + "…") if text else ""
    rel_source_path = Path(rel_path).as_posix()
    source_entry = str(Path("admin_raw_data") / rel_source_path)
    _, parsed_version = parse_doc_version(raw_path.stem)
    version = int(parsed_version) if parsed_version else 0
    extraction_info = {
        "original_file": raw_path.name,
        "text_length": len(text),
        "table_count": len(tables),
        "extracted_at": now_kst_string(),
    }

    existing = get_document_by_source_path(ADMIN_DOC_TYPE, source_entry)
    doc_id = existing["doc_id"] if existing else generate_doc_id()

    register_admin_document(
        doc_id=doc_id,
        filename=raw_path.name,
        rel_text_path=rel_text_path.as_posix(),
        rel_source_path=source_entry,
        sec_map=sec_map,
        version=int(version),
        preview=preview,
        tables=tables,
        total_pages=total_pages,
        pages=pages_text_dict if pages_text_dict else {},
        source_ext=file_ext,
        extraction_info=extraction_info,
    )

    # DocumentMetadata 저장 (청크/페이지 단위 텍스트)
    metadata_records: List[Dict[str, Any]] = []
    chunk_index = 0

    def _append_record(page: int, chunk_text: str, *, extra_payload: Optional[Dict] = None):
        nonlocal chunk_index
        payload = {"source_file": raw_path.name}
        if extra_payload:
            payload.update(extra_payload)
        metadata_records.append(
            {
                "page": int(page),
                "chunk_index": int(chunk_index),
                "text": chunk_text,
                "payload": payload,
            }
        )
        chunk_index += 1

    pages_tables = defaultdict(list)
    for t in tables:
        page_num = t.get("page", 0)
        if page_num > 0:
            pages_tables[page_num].append(t)

    if pages_text_dict:
        all_page_nums = sorted(set(pages_text_dict) | set(pages_tables))
        for page_num in all_page_nums:
            page_text = pages_text_dict.get(page_num, "")
            if page_text.strip():
                _append_record(page_num, page_text)
            for tbl in pages_tables.get(page_num, []):
                table_text = tbl.get("text", "")
                if table_text.strip():
                    _append_record(
                        page_num,
                        table_text,
                        extra_payload={"table": True, "table_bbox": tbl.get("bbox")}
                    )
    else:
        if text.strip():
            _append_record(1, text)
        for tbl in tables:
            table_text = tbl.get("text", "")
            if table_text.strip():
                _append_record(
                    int(tbl.get("page") or 0),
                    table_text,
                    extra_payload={"table": True, "table_bbox": tbl.get("bbox")}
                )
    
    if metadata_records:
        bulk_upsert_document_metadata(doc_id=doc_id, records=metadata_records)

    return {
        "doc_id": doc_id,
        "filename": raw_path.name,
        "source_path": source_entry,
        "text_path": rel_text_path.as_posix(),
        "security_levels": sec_map,
        "version": int(version),
    }
    

# ---------------- Vector Settings ----------------
def set_vector_settings(embed_model_key: Optional[str] = None,
                        search_type: Optional[str] = None,
                        chunk_size: Optional[int] = None,
                        overlap: Optional[int] = None) -> Dict:
    """
    rag_settings 단일 소스로 설정 저장.
    - 임베딩 모델 변경 시 기존 데이터 존재하면 차단, 활성 모델 갱신 및 캐시 무효화
    - search_type/청크/오버랩은 rag_settings에만 반영
    """
    cur = get_rag_settings_row()
    key_now = cur.get("embedding_key")
    st_now = (cur.get("search_type") or "hybrid").lower()
    cs_now = int(cur.get("chunk_size") or 512)
    ov_now = int(cur.get("overlap") or 64)

    new_key = embed_model_key or key_now
    new_st = (search_type or st_now).lower()
    # DB 제약과 일치(semantic == vector)
    if new_st == "semantic":
        new_st = "vector"
    if new_st not in {"hybrid", "bm25", "vector"}:
        raise ValueError("unsupported searchType; allowed: 'hybrid','bm25','vector'")

    new_cs = int(chunk_size if chunk_size is not None else cs_now)
    new_ov = int(overlap if overlap is not None else ov_now)
    if new_cs <= 0 or new_ov < 0 or new_ov >= new_cs:
        raise ValueError("invalid chunk/overlap (chunk>0, 0 <= overlap < chunk)")

    if embed_model_key is not None:
        client = get_milvus_client()
        if milvus_has_data(client, collection_name=ADMIN_COLLECTION):
            raise RuntimeError("Milvus 컬렉션에 기존 데이터가 남아있습니다. 먼저 /v1/admin/vector/delete-all 을 호출해 초기화하세요.")
        set_rag_settings_row(new_search=new_st, new_chunk=new_cs, new_overlap=new_ov, new_key=new_key)
        invalidate_embedder_cache()

    # Refresh and return in API format
    updated = get_rag_settings_row()
    return {
        "embeddingModel": updated.get("embedding_key"),
        "searchType": updated.get("search_type", "hybrid"),
        "chunkSize": int(updated.get("chunk_size", 512)),
        "overlap": int(updated.get("overlap", 64)),
    }


def list_available_embedding_models() -> List[str]:
    """
    ./storage/embedding-models 폴더 내의 모델 폴더명들을 반환.
    - embedding_ 접두사가 있으면 제거 (예: embedding_bge_m3 → bge_m3)
    - 폴더만 반환 (파일 제외)
    """
    models = []
    if not MODEL_ROOT_DIR.exists():
        return models
    
    for item in MODEL_ROOT_DIR.iterdir():
        if item.is_dir():
            model_name = item.name
            # embedding_ 접두사 제거
            if model_name.startswith("embedding_"):
                model_name = model_name[len("embedding_"):]
            models.append(model_name)
    
    return sorted(models)


# ------------- Security Level (per task) ---------

def _normalize_keywords(val: Any) -> List[str]:
    """
    리스트/튜플/셋: 각 원소를 str로 캐스팅, 공백/해시 제거
    문자열: 콤마(,) 또는 줄바꿈(\n) 기준으로 토큰화
    빈 값 제거 및 중복 제거
    """
    out: List[str] = []
    if isinstance(val, str):
        # 콤마와 줄바꿈을 구분자로 처리
        # 먼저 줄바꿈으로 split하고, 각 줄을 콤마로 split
        lines = val.split('\n')
        toks = []
        for line in lines:
            # 콤마로 split
            parts = line.split(',')
            toks.extend([t.strip() for t in parts])
    elif isinstance(val, (list, tuple, set)):
        toks = [str(t).strip() for t in val]
    else:
        toks = []
    for t in toks:
        if not t:
            continue
        if t.startswith("#"):
            t = t[1:]
        if t and t not in out:
            out.append(t)
    return out


def _normalize_levels(
    levels_raw: Dict[str, Any], max_level: int
) -> Dict[int, List[str]]:
    norm: Dict[int, List[str]] = {}
    for k, v in (levels_raw or {}).items():
        try:
            lv = int(str(k).strip().replace("level_", ""))
        except Exception:
            continue
        if lv < 1 or lv > max_level:
            continue
        kws = _normalize_keywords(v)
        if kws:
            norm[lv] = kws
    return norm


def upsert_security_level_for_task(
    task_type: str, max_level: int, levels_raw: Dict[str, Any]
) -> Dict:
    if task_type not in TASK_TYPES:
        raise ValueError(f"invalid task_type: {task_type}")
    if max_level < 1:
        raise ValueError("maxLevel must be >= 1")

    levels_map = _normalize_levels(levels_raw, max_level)
    
    security_repo.upsert_security_config_and_keywords(task_type, max_level, levels_map)
    return get_security_level_rules_for_task(task_type)


def get_security_level_rules_for_task(task_type: str) -> Dict:
    cfg = security_repo.get_security_config_by_task_type(task_type)
    max_level = int(cfg.max_level) if cfg else 1

    res: Dict[str, Any] = {
        "taskType": task_type,
        "maxLevel": max_level,
        "levels": {str(i): [] for i in range(1, max_level + 1)},
    }
    
    rows = security_repo.get_security_keywords_by_task_type(task_type)
    for lv, kw in rows:
        key = str(int(lv))
        res["levels"].setdefault(key, []).append(str(kw))
    return res


def get_security_level_rules_all() -> Dict:
        # 기본 max_level=1
    max_map = {t: 1 for t in TASK_TYPES}
    
    configs, keywords = security_repo.get_all_security_configs_and_keywords()
    
    for task, max_level in configs:
        max_map[task] = int(max_level)

    res: Dict[str, Dict] = {}
    for task in TASK_TYPES:
        res[task] = {
            "maxLevel": max_map.get(task, 1),
            "levels": {str(i): [] for i in range(1, max_map.get(task, 1) + 1)},
        }

    for task, level, kw in keywords:
        if task in res:
            lv = str(int(level))
            if lv not in res[task]["levels"]:
                res[task]["levels"][lv] = []
            res[task]["levels"][lv].append(str(kw))
    return res


# -------------------------------------------------
# 2) 인제스트 (bulk)
#   - 작업유형별로 동일 청크를 각각 저장(task_type, security_level 분리)
# -------------------------------------------------
async def ingest_embeddings(
    model_key: str | None = None,
    target_tasks: list[str] | None = None,
    # max_token: int = 512,
    # overlab: int = 64,
    collection_name: str = ADMIN_COLLECTION,
    file_keys_filter: list[str] | None = None,
):
    """
    documents 테이블에 저장된 관리자 문서를 기준으로 추출된 텍스트(.txt)를 인제스트한다.
    - VARCHAR(32768 bytes) 초과 방지: split_for_varchar_bytes 로 안전 분할
    - 표는 [[TABLE ...]] 머리글 유지, 이어지는 조각은 [[TABLE_CONT i/n]] 마커로 연속성 표시
    - file_keys_filter 전달 시 doc_id/파일명/스토리지 경로가 일치하는 문서만 인제스트
    """
    tasks = [t for t in (target_tasks or TASK_TYPES) if t in TASK_TYPES]
    if not tasks:
        return {"error": f"유효한 작업유형이 없습니다. 허용: {TASK_TYPES}"}

    documents = _load_admin_documents(file_keys_filter)
    if not documents:
        return {"error": "관리자 문서 메타데이터가 없습니다. 먼저 문서를 추출하세요."}

    doc_ids = [doc["doc_id"] for doc in documents if doc.get("doc_id")]
    metadata_by_doc = fetch_document_metadata_by_doc_ids(doc_ids)
    settings = get_rag_settings_row()
    collection = collection_name

    prepared_inputs: List[Dict[str, Any]] = []
    for doc in documents:
        doc_id = str(doc.get("doc_id") or "").strip()
        if not doc_id:
            continue
        meta_chunks = metadata_by_doc.get(doc_id) or []
        if not meta_chunks:
            logger.warning("[Ingest] metadata missing: doc_id=%s", doc_id)
            continue
        payload = doc.get("payload") or {}
        sec_map = payload.get("security_levels", {}) or {}
        version = int(payload.get("version") or 0)
        filename = doc.get("filename") or doc_id
        saved_files = payload.get("saved_files") or {}
        text_path = saved_files.get("text") or doc.get("source_path") or ""

        chunk_entries = [
            {
                "page": int(entry.get("page") or 0),
                "chunk_idx": int(entry.get("chunk_index") or entry.get("chunk_idx") or 0),
                "text": entry.get("text") or "",
            }
            for entry in meta_chunks
            if entry.get("text")
        ]
        if not chunk_entries:
            continue
        prepared_inputs.append(
            {
                "doc_id": doc_id,
                "version": version,
                "levels": sec_map,
                "chunks": chunk_entries,
                "metadata_records": meta_chunks,
                "source_path": text_path,
                "filename": filename,
            }
        )

    if not prepared_inputs:
        return {"error": "인제스트할 문서 조각을 찾지 못했습니다."}

    def _batch_callback(records: List[Dict[str, Any]], doc_id: str) -> None:
        if not records:
            return
        try:
            insert_document_vectors(
                doc_id=doc_id,
                collection=collection,
                embedding_version=str(model_key or settings["embedding_key"]),
                vectors=records,
            )
        except Exception:
            logger.exception("document_vectors 기록 실패(doc_id=%s)", doc_id)

    ingest_result = await retrieval_service.ingest_documents(
        inputs=prepared_inputs,
        collection_name=collection,
        task_types=tasks,
        batch_callback=_batch_callback,
        upsert_metadata=False,
    )

    ingest_result["message"] = f"Ingest 완료(Milvus Server, collection={collection})"
    return ingest_result

async def search_documents(req: SearchRequest)-> Dict:
    """
    [Legacy/Direct Search]
    검색 -> 리랭킹 -> 중복제거 과정을 모두 수행하여 최종 결과를 반환합니다.
    """
    search_res = await retrieval_service.search(req)
    hits = search_res.get("hits", [])
    context = "\n---\n".join(h["snippet"] for h in hits if h.get("snippet"))
    prompt = f"사용자 질의: {req.query}\n:\n{context}\n\n위 내용을 바탕으로 응답을 생성해 주세요."
    search_res["prompt"] = prompt
    return search_res
    

async def execute_search(
    question: str,
    top_k: int = 20,
    rerank_top_n: int = 5,
    security_level: int = 1,
    source_filter: Optional[List[str]] = None,
    task_type: str = "qna",
    model_key: Optional[str] = None,
    search_type: Optional[str] = None,
) -> Dict:
    print(f"⭐ [ExecuteSearch] 함수 호출: question='{question}', topK={top_k}, rerank_topN={rerank_top_n}")
    req = SearchRequest(
        query=question,
        collection_name = ADMIN_COLLECTION,
        top_k=top_k,
        rerank_top_n=rerank_top_n,
        security_level=security_level,
        search_type=search_type,
        task_type=task_type,
        model_key=model_key,
    )
    res = await search_documents(req)
    
    # ... (이하 소스 필터링 및 체크 파일 생성 로직 유지)
    check_files: List[str] = []
    try:
        for h in res.get("hits", []):
             # ... existing code ...
            doc_id_val = h.get("doc_id")
            if doc_id_val:
                check_files.append(f"{str(doc_id_val)}.pdf")
                continue
            p = Path(h.get("path", ""))
            if str(p):
                check_files.append(p.with_suffix(".pdf").name)
    except Exception:
        pass

    if source_filter and "hits" in res:
        names = {Path(n).stem for n in source_filter}
        res["hits"] = [h for h in res["hits"] if Path(h["path"]).stem in names]

    res["check_file"] = sorted(list(set(check_files)))
    return res


# -------------------------------------------------
# 4) 삭제 관련 함수 (Milvus + RDB)
# -------------------------------------------------

async def delete_collection(collection_key: str | None = None):
    COLLECTIONS = app_config["retrieval"]["milvus"]["collections"]
    invalidate_embedder_cache()
    client = get_milvus_client()
    targets = []

    if collection_key is not None:
        name = COLLECTIONS.get(collection_key)
        targets = [name]
        doc_types = [collection_key]
    else:
        targets = list(COLLECTIONS.values())
        doc_types = list(COLLECTIONS.keys()) 

    dropped = []
    for col in targets:
        if col in client.list_collections():
            client.drop_collection(col)
            dropped.append(col)
    sql_stats = purge_documents_by_collection(doc_types)
    return {"dropped": dropped, "sql": sql_stats}


# -------------------------------------------------
# 5) 검색 관련 함수
# -------------------------------------------------


async def list_indexed_files(
    limit: int = 16384,
    offset: int = 0,
    query: Optional[str] = None,
    task_type: Optional[str] = None,
):
    limit = max(1, min(limit, 16384))
    doc_records = list_documents_by_type(ADMIN_DOC_TYPE)
    doc_map = {doc["doc_id"]: doc for doc in doc_records if doc.get("doc_id")}
    
    rows = get_list_indexed_files(collection_name=ADMIN_COLLECTION, offset=offset, limit=limit, task_type=task_type)

    items: List[Dict[str, Any]] = []
    for doc_id_val, ttype, chunk_count in rows:
        doc_meta = doc_map.get(doc_id_val or "")
        if not doc_meta:
            # doc metadata가 없으면 넘어감
            continue
        file_name = doc_meta.get("filename") or Path(doc_meta.get("source_path") or "").name
        file_path = doc_meta.get("source_path") or ""
        sec_levels = (doc_meta.get("payload") or {}).get("security_levels", {}) or {}
        sec_level = int(sec_levels.get(ttype, doc_meta.get("security_level", 1)))
        items.append(
            {
                "taskType": ttype,
                "fileName": file_name,
                "filePath": file_path,
                "chunkCount": int(chunk_count),
                "indexedAt": doc_meta.get("updated_at"),
                "fileSize": None,
                "securityLevel": sec_level,
            }
        )

    if query:
        q = str(query)
        items = [it for it in items if q in it["fileName"]]
    return items

async def delete_files_by_names(file_names: List[str], task_type: Optional[str] = None):
    """
    파일명(= doc_id stem) 배열을 받아 벡터 DB에서 삭제.
    - task_type 가 None 이면 모든 작업유형(doc_gen/summary/qna)에서 삭제 (기존 동작과 동일)
    - task_type 가 지정되면 해당 작업유형 레코드만 삭제
    """
    if not file_names:
        return {"deleted": 0, "requested": 0}

    try:
        from repository.documents import delete_workspace_documents_by_filenames
    except Exception:
        delete_workspace_documents_by_filenames = None

    client = get_milvus_client()
    milvus_ready = ADMIN_COLLECTION in client.list_collections()

    if milvus_ready:
        try:
            client.load_collection(collection_name=ADMIN_COLLECTION)
        except Exception:
            pass
    else:
        logger.warning("Milvus collection %s not available; skipping vector DB deletion.", ADMIN_COLLECTION)

    # 유효한 task_type 인지 검증
    task_filter = ""
    if task_type:
        if task_type not in TASK_TYPES:
            return {
                "deleted": 0,
                "requested": len(file_names),
                "error": f"invalid taskType: {task_type}",
            }
        task_filter = f" && task_type == '{task_type}'"

    deleted_total = 0
    per_file: dict[str, int] = {}

    doc_ids_to_remove: set[str] = set()
    name_index = _build_doc_name_index()

    for name in file_names:
        raw_name = str(name or "").strip()
        stem = Path(raw_name).stem if raw_name else ""

        doc_id_candidate = None
        for token in filter(None, [raw_name.lower(), stem.lower() if stem else None]):
            doc_id_candidate = name_index.get(token)
            if doc_id_candidate:
                break

        if not doc_id_candidate:
            try:
                base_id, _ver = parse_doc_version(stem or raw_name)
            except Exception:
                base_id = stem or raw_name
            doc_id_candidate = base_id

        if not doc_id_candidate:
            per_file[name] = per_file.get(name, 0)
            continue

        if milvus_ready:
            try:
                filt = f"doc_id == '{doc_id_candidate}'{task_filter}"
                client.delete(collection_name=ADMIN_COLLECTION, filter=filt)
                deleted_total += 1
                per_file[name] = per_file.get(name, 0) + 1
            except Exception:
                logger.exception("Failed to delete from Milvus for file: %s", name)
                per_file[name] = per_file.get(name, 0)
        else:
            per_file[name] = per_file.get(name, 0)

        if task_type:
            delete_document_vectors(doc_id_candidate, task_type)
            if not document_has_vectors(doc_id_candidate):
                if milvus_ready:
                    try:
                        client.delete(collection_name=ADMIN_COLLECTION, filter=f"doc_id == '{doc_id_candidate}'")
                    except Exception:
                        logger.exception("Failed to delete remaining Milvus vectors for doc_id=%s", doc_id_candidate)
                doc_ids_to_remove.add(doc_id_candidate)
        else:
            # 전체 작업유형 삭제 시 SQL/Milvus 모두 제거
            delete_document_vectors(doc_id_candidate, None)
            if milvus_ready:
                try:
                    client.delete(collection_name=ADMIN_COLLECTION, filter=f"doc_id == '{doc_id_candidate}'")
                except Exception:
                    logger.exception("Failed to delete doc_id=%s from Milvus", doc_id_candidate)
            doc_ids_to_remove.add(doc_id_candidate)

    if milvus_ready:
        # Ensure deletion is visible to subsequent queries (file lists/overview)
        try:
            logger.info("flush Milvus after deletion")
            client.flush(ADMIN_COLLECTION)
        except Exception:
            logger.exception("Failed to flush Milvus after deletion")
        # Force reload to avoid any stale cache/state on the server side
        try:
            client.release_collection(collection_name=ADMIN_COLLECTION)
        except Exception:
            pass
        try:
            client.load_collection(collection_name=ADMIN_COLLECTION)
        except Exception:
            logger.exception("Failed to reload collection after deletion")

    deleted_sql = None
    if delete_workspace_documents_by_filenames:
        try:
            # SQL은 작업유형 구분이 없다고 가정(기존 그대로)
            deleted_sql = delete_workspace_documents_by_filenames(file_names)
        except Exception:
            logger.exception("Failed to delete workspace documents in SQL")
            deleted_sql = None

    if doc_ids_to_remove:
        try:
            delete_documents_by_type_and_ids(ADMIN_DOC_TYPE, list(doc_ids_to_remove))
        except Exception:
            logger.exception("Failed to delete admin document metadata for %s", doc_ids_to_remove)

    return {
        "deleted": deleted_total,  # 요청 파일 기준 성공 건수(작업유형 기준 단순 카운트)
        "deleted_sql": deleted_sql,
        "requested": len(file_names),
        "taskType": task_type,
        "perFile": per_file,  # 파일별 처리현황
    }



# === 새 API: 키워드 없이 레벨 오버라이드 후 인제스트 ===
class OverrideLevelsRequest(BaseModel):
    """
    업로드(or 기존) 파일들에 대해 작업유형별 레벨을 강제로 세팅하고 인제스트.
    - files: 대상 파일 이름/경로(비우면 META 전체 대상이지만, 본 엔드포인트에서는 업로드 파일만 전달)
    - level_for_tasks: {"qna":2,"summary":1,"doc_gen":3} (필수)
    - tasks: 작업유형 제한 (미지정 시 모든 TASK_TYPES)
    """
    files: Optional[List[str]] = None
    level_for_tasks: Dict[str, int]
    tasks: Optional[List[str]] = None


async def override_levels_and_ingest(req: OverrideLevelsRequest):
    target_tasks = [t for t in (req.tasks or TASK_TYPES) if t in TASK_TYPES]
    if not target_tasks:
        return {"error": "유효한 작업유형이 없습니다. (허용: doc_gen|summary|qna)"}

    level_map = {t: int(max(1, lv)) for t, lv in (req.level_for_tasks or {}).items() if t in TASK_TYPES}
    if not level_map:
        return {"error": "적용할 보안레벨이 없습니다. level_for_tasks 를 지정하세요."}

    documents = _load_admin_documents(req.files)
    if not documents:
        return {"updated": 0, "ingested": 0, "message": "대상 문서를 찾을 수 없습니다."}

    updated = 0
    target_tokens: List[str] = []
    for doc in documents:
        doc_id = doc.get("doc_id")
        if not doc_id:
            continue
        payload = dict(doc.get("payload") or {})
        sec = payload.get("security_levels") or {}
        for t in target_tasks:
            if t in level_map:
                sec[t] = int(level_map[t])
        payload["security_levels"] = sec
        upsert_document(
            doc_id=doc_id,
            doc_type=ADMIN_DOC_TYPE,
            filename=doc.get("filename") or doc_id,
            source_path=doc.get("source_path"),
            security_level=_max_security_level(sec),
            payload=payload,
        )
        updated += 1
        target_tokens.append(doc_id)

    settings = get_rag_settings_row()
    model_key = settings.get("embedding_key")

    res = await ingest_embeddings(
        model_key=model_key,
        target_tasks=target_tasks,
        collection_name=ADMIN_COLLECTION,
        file_keys_filter=target_tokens,
    )
    return {
        "message": "레벨 오버라이드 후 인제스트 완료",
        "collection": ADMIN_COLLECTION,
        "updated_meta_entries": updated,
        "inserted_chunks": int(res.get("inserted_chunks", 0)),
        "target_count": len(target_tokens),
    }
