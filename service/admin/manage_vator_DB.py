# === Vector DB Service (Milvus Server, Pro) ===
# - 작업유형(task_type)별 보안레벨 관리: doc_gen | summary | qna
# - Milvus Docker 서버 전용 (Lite 제거)
# - 벡터/하이브리드 검색 지원, 실행 로그 적재

from __future__ import annotations
import asyncio
import logging
import re
import shutil
import time
import uuid
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field
from config import config as app_config
from repository.rag_settings import get_rag_settings_row, set_rag_settings_row
from repository.documents import (
    bulk_upsert_document_metadata,
    delete_documents_by_type_and_ids,
    get_document_by_source_path,
    insert_document_vectors,
    list_documents_by_type,
    upsert_document,
)
from utils.database import get_session
from utils.documents import generate_doc_id
from storage.db_models import (
    DocumentType,
    RagSettings,
    SecurityLevelConfigTask,
    SecurityLevelKeywordsTask,
)
from ..vector_db import (
    drop_all_collections,
    ensure_collection_and_index,
    get_milvus_client,
    milvus_has_data,
    run_dense_search,
    run_hybrid_search,
)

def split_for_varchar_bytes(
    text: str,
    hard_max_bytes: int = 32768,
    soft_max_bytes: int = 30000,   # 여유 버퍼
    table_mark: str = "[[TABLE",
) -> list[str]:
    """
    VARCHAR 초과 방지: UTF-8 바이트 기준으로 안전 분할.
    - 표 텍스트는 헤더([[TABLE ...]])를 첫 조각에만 포함.
    - 이후 조각엔 [[TABLE_CONT i/n]] 마커를 부여.
    - 개행 경계 우선(backtrack), 그래도 안되면 하드컷.
    """
    if not text:
        return [""]

    # 표 헤더 분리
    header = ""
    body = text
    if text.startswith(table_mark):
        head_end = text.find("]]")
        if head_end != -1:
            head_end += 2
            if head_end < len(text) and text[head_end] == "\n":
                head_end += 1
            header, body = text[:head_end], text[head_end:]

    def _split_body(b: str) -> list[str]:
        out: list[str] = []
        b_bytes = b.encode("utf-8")
        n = len(b_bytes)
        i = 0
        while i < n:
            j = min(i + soft_max_bytes, n)
            # 개행 경계로 뒤로 물러나기
            k = j
            backtracked = False
            # j부터 i까지 역방향으로 \n 바이트(0x0A) 탐색
            while k > i and (j - k) < 2000:  # 최대 2KB만 백트랙
                if b_bytes[k-1:k] == b"\n":
                    backtracked = True
                    break
                k -= 1
            if backtracked and (k - i) >= int(soft_max_bytes * 0.6):
                cut = k
            else:
                cut = j

            # 하드 컷(멀티바이트 경계 맞추기)
            if cut - i > hard_max_bytes:
                cut = i + hard_max_bytes

            # UTF-8 안전 디코드: 경계가 문자를 반쯤 자를 수 있으니 넉넉히 조정
            chunk = b_bytes[i:cut]
            # 만약 디코드 에러가 나면 한 바이트씩 줄이며 안전 경계 찾기
            while True:
                try:
                    s = chunk.decode("utf-8")
                    break
                except UnicodeDecodeError:
                    cut -= 1
                    if cut <= i:
                        # 최악의 경우 한 글자라도 디코드되게 한 바이트 앞당김
                        cut = i + 1
                    chunk = b_bytes[i:cut]
            out.append(s)
            i = cut
        return out

    if len(text.encode("utf-8")) <= hard_max_bytes:
        return [text]

    parts = _split_body(body)
    if header:
        total = len(parts)
        result = []
        for idx, c in enumerate(parts, start=1):
            if idx == 1:
                # 첫 조각은 헤더 + 본문
                # 전체가 하드맥스를 넘지 않게 헤더와 합친 뒤 한번 더 자르기
                first = header + c
                if len(first.encode("utf-8")) <= hard_max_bytes:
                    result.append(first)
                else:
                    # 너무 크면 헤더는 유지하고 c를 다시 잘라 붙임
                    # (헤더가 길 때 매우 예외적)
                    subparts = _split_body(c)
                    if subparts:
                        # 첫 조각은 헤더 + 첫 sub
                        f = header + subparts[0]
                        if len(f.encode("utf-8")) > hard_max_bytes:
                            # 헤더 자체가 큰 극단: 헤더만 넣고 이후 CONT로 처리
                            result.append(header[:0] + header)  # 그대로
                            # 나머지는 CONT
                            for sidx, sp in enumerate(subparts, start=1):
                                tag = f"[[TABLE_CONT {sidx}/{len(subparts)}]]\n"
                                result.append(tag + sp)
                        else:
                            result.append(f)
                            # 나머지는 CONT
                            for sidx, sp in enumerate(subparts[1:], start=2):
                                tag = f"[[TABLE_CONT {sidx}/{len(subparts)}]]\n"
                                result.append(tag + sp)
                    else:
                        result.append(header)  # 본문이 없으면 헤더만
            else:
                tag = f"[[TABLE_CONT {idx}/{total}]]\n"
                # tag + c 가 하드맥스를 넘지 않도록 재자르기
                rest = tag + c
                if len(rest.encode("utf-8")) <= hard_max_bytes:
                    result.append(rest)
                else:
                    subs = _split_body(c)
                    for sidx, sp in enumerate(subs, start=1):
                        subt = f"[[TABLE_CONT {idx}.{sidx}/{total}]]\n" + sp
                        if len(subt.encode("utf-8")) <= hard_max_bytes:
                            result.append(subt)
                        else:
                            # 그래도 넘으면 하드컷으로 마지막 방어
                            bb = subt.encode("utf-8")[:hard_max_bytes]
                            result.append(bb.decode("utf-8", errors="ignore"))
        return result
    else:
        return parts


# KST 시간 포맷 유틸
from utils.time import now_kst, now_kst_string

from service.retrieval.common import hf_embed_text, chunk_text
from service.retrieval.pipeline import (
    DEFAULT_OUTPUT_FIELDS,
    build_dense_hits,
    # build_rrf_hits,
    build_rerank_payload,
    load_snippet_from_store,
)
from service.retrieval.reranker import rerank_snippets
from utils.model_load import (
    resolve_model_input,
    _get_or_load_embedder,
    _get_or_load_embedder_async,
    _invalidate_embedder_cache,
)

logger = logging.getLogger(__name__)

# -------------------------------------------------
# 경로 상수
# -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent  # .../backend/service/admin
PROJECT_ROOT = BASE_DIR.parent.parent  # .../backend
_RETRIEVAL_CFG: Dict[str, Any] = app_config.get("retrieval", {}) or {}
_RETRIEVAL_PATHS: Dict[str, str] = _RETRIEVAL_CFG.get("paths", {}) or {}
_MILVUS_CFG: Dict[str, Any] = _RETRIEVAL_CFG.get("milvus", {}) or {}


def _cfg_path(key: str, fallback: str) -> Path:
    value = _RETRIEVAL_PATHS.get(key, fallback)
    return (PROJECT_ROOT / Path(value)).resolve()


STORAGE_DIR = _cfg_path("storage_dir", "storage")
USER_DATA_ROOT = _cfg_path("user_data_root", "storage/user_data")
RAW_DATA_DIR = _cfg_path("raw_data_dir", "storage/user_data/row_data")
LOCAL_DATA_ROOT = _cfg_path("local_data_root", "storage/user_data/preprocessed_data")
RESOURCE_DIR = _cfg_path("resources_dir", str(BASE_DIR / "resources"))
EXTRACTED_TEXT_DIR = _cfg_path("extracted_text_dir", "storage/extracted_texts")
MODEL_ROOT_DIR = _cfg_path("model_root_dir", "storage/embedding-models")
RERANK_MODEL_PATH = _cfg_path("rerank_model_path", "storage/rerank_model/Qwen3-Reranker-0.6B")
VAL_SESSION_ROOT = _cfg_path("val_session_root", "storage/val_data")

DATABASE_CFG = app_config.get("database", {}) or {}
SQLITE_DB_PATH = (PROJECT_ROOT / Path(DATABASE_CFG.get("path", "storage/pps_rag.db"))).resolve()

ADMIN_COLLECTION = _MILVUS_CFG.get("ADMIN_DOCS", "admin_docs_collection")

TASK_TYPES = tuple(_RETRIEVAL_CFG.get("task_types") or ("doc_gen", "summary", "qna"))
SUPPORTED_EXTS = set(_RETRIEVAL_CFG.get("supported_extensions"))

ZERO_WIDTH_RE = re.compile(r"[\u200B-\u200D\u2060\uFEFF]")
CONTROL_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")
MULTISPACE_LINE_END_RE = re.compile(r"[ \t]+\n")
NEWLINES_RE = re.compile(r"\n{3,}")


# -------------------------------------------------
# 텍스트 정리 및 다중 확장자 지원
# -------------------------------------------------

# 확장자별 추출 함수들은 service/preprocessing/rag_preprocessing.py와 
# service/preprocessing/extension/ 폴더로 이동했습니다.

# -------------------------------------------------
# Admin 문서 메타데이터 헬퍼
# -------------------------------------------------
ADMIN_DOC_TYPE = DocumentType.ADMIN.value


def _ext(value: Path | str) -> str:
    """Path helper returning lowercase suffix."""
    return Path(value).suffix.lower()


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


def _extract_insert_ids(result: Any) -> List[str]:
    """
    Milvus insert 결과에서 primary key 리스트를 추출한다.
    다양한 리턴 타입(dict/InsertResult 등)을 모두 처리한다.
    """
    ids: Any = None
    if isinstance(result, dict):
        ids = (
            result.get("ids")
            or result.get("primary_keys")
            or result.get("inserted_ids")
        )
    else:
        ids = getattr(result, "primary_keys", None) or getattr(result, "ids", None)
    if not ids:
        return []
    return [str(pk) for pk in ids]


def _build_admin_payload(
    *,
    sec_map: Dict[str, int],
    version: int,
    preview: str,
    tables: List[Dict[str, Any]],
    total_pages: int,
    saved_files: Dict[str, str],
    pages: Dict[str, Any],
    source_ext: str,
    extraction_info: Dict[str, Any],
    rel_key: str,
) -> Dict[str, Any]:
    return {
        "security_levels": sec_map,
        "version": int(version),
        "preview": preview,
        "tables": tables or [],
        "total_pages": int(total_pages or 0),
        "saved_files": saved_files,
        "pages": pages or {},
        "source_ext": source_ext,
        "doc_rel_key": rel_key,
        "extraction_info": extraction_info,
    }


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
    payload = _build_admin_payload(
        sec_map=sec_map,
        version=version,
        preview=preview,
        tables=tables,
        total_pages=total_pages,
        saved_files={"text": rel_text_path, "source": rel_source_path},
        pages=pages,
        source_ext=source_ext,
        extraction_info=extraction_info,
        rel_key=rel_source_path,
    )
    upsert_document(
        doc_id=doc_id,
        doc_type=ADMIN_DOC_TYPE,
        filename=filename,
        storage_path=rel_text_path,
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
        doc.get("storage_path"),
        Path(str(doc.get("storage_path") or "")).name,
        Path(str(doc.get("storage_path") or "")).stem,
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
    _push(doc.get("storage_path"))
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
# Pydantic 스키마
# -------------------------------------------------
class RAGSearchRequest(BaseModel):
    query: str
    top_k: int = Field(5, gt=0)
    user_level: int = Field(1, ge=1)
    task_type: str = Field(..., description="doc_gen | summary | qna")
    model: Optional[str] = None  # 내부적으로 settings에서 로드


class SinglePDFIngestRequest(BaseModel):
    pdf_path: str
    task_types: Optional[List[str]] = None  # 기본은 모든 작업유형
    workspace_id: Optional[int] = None


# -------------------------------------------------
# SQLite 유틸
# -------------------------------------------------


# ====== New helpers ======
def save_raw_file(filename: str, content: bytes) -> str:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    name = Path(filename or "uploaded").name or f"uploaded_{uuid.uuid4().hex}"
    dst = RAW_DATA_DIR / name
    if dst.exists():
        stem, ext = dst.stem, dst.suffix
        dst = RAW_DATA_DIR / f"{stem}_{int(time.time())}{ext}"
    dst.write_bytes(content)
    return str(dst.relative_to(RAW_DATA_DIR).as_posix())


def _write_combined_text_file(
    output_path: Path,
    *,
    text: str,
    tables: List[Dict[str, Any]],
    pages_text_dict: Dict[int, str],
) -> None:
    def _write_tables(handle, items):
        for tbl in items:
            table_text = (tbl.get("text") or "").strip()
            if table_text:
                handle.write(table_text)
                handle.write("\n\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        if pages_text_dict:
            pages_tables: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
            for tbl in tables or []:
                page_num = int(tbl.get("page", 0))
                if page_num > 0:
                    pages_tables[page_num].append(tbl)
            ordered_pages = sorted({*pages_text_dict.keys(), *pages_tables.keys()})
            if ordered_pages:
                for idx, page_num in enumerate(ordered_pages):
                    page_text = pages_text_dict.get(page_num, "")
                    if page_text:
                        handle.write(page_text)
                        handle.write("\n\n")
                    _write_tables(handle, pages_tables.get(page_num, []))
                    if idx < len(ordered_pages) - 1:
                        handle.write("\n---\n\n")
            else:
                if text.strip():
                    handle.write(text)
                    handle.write("\n\n")
                _write_tables(handle, tables or [])
        else:
            if text.strip():
                handle.write(text)
                handle.write("\n\n")
            _write_tables(handle, tables or [])

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
    raw_path = (RAW_DATA_DIR / rel_path).resolve()
    if not raw_path.exists():
        logger.warning("[ProcessRaw] RAW 파일을 찾을 수 없습니다: %s", rel_path)
        return None

    try:
        rel_from_raw = raw_path.relative_to(RAW_DATA_DIR)
    except ValueError:
        rel_from_raw = raw_path

    file_ext = _ext(raw_path)
    pages_text_dict: Dict[int, str] = {}
    total_pages = 0

    try:
        if file_ext == ".pdf":
            from service.preprocessing.extension.pdf_preprocessing import _extract_pdf_with_tables
            text, tables, pages_text_dict, total_pages = _extract_pdf_with_tables(raw_path)
        else:
            from service.preprocessing.rag_preprocessing import extract_any
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

    try:
        _write_combined_text_file(
            EXTRACTED_TEXT_DIR / rel_text_path,
            text=text,
            tables=tables,
            pages_text_dict=pages_text_dict,
        )
    except Exception:
        logger.exception("[ProcessRaw] 통합 텍스트 저장 실패: %s", raw_path)
        return None

    from service.preprocessing.rag_preprocessing import _clean_text as clean_text

    preview = (clean_text(text[:200].replace("\n", " ")) + "…") if text else ""
    rel_source_path = Path(rel_path).as_posix()
    source_entry = str(Path("row_data") / rel_source_path)
    base_name, parsed_version = parse_doc_version(raw_path.stem)
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

    return {
        "doc_id": doc_id,
        "filename": raw_path.name,
        "source_path": source_entry,
        "text_path": rel_text_path.as_posix(),
        "security_levels": sec_map,
    }
    
def save_raw_to_row_data(f):
    """Save FastAPI UploadFile to row_data and return relative path."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    name = Path(getattr(f, "filename", "uploaded"))
    dst = RAW_DATA_DIR / name.name
    if dst.exists():
        stem, ext = name.stem, name.suffix
        dst = RAW_DATA_DIR / f"{stem}_{int(time.time())}{ext}"
    with dst.open("wb") as out:
        data = f.file.read() if hasattr(f, "file") else b""
        out.write(data)
    try:
        return str(dst.relative_to(RAW_DATA_DIR))
    except Exception:
        return dst.name

def warmup_active_embedder(logger_func=print):
    """
    서버 기동 시 호출용(선택). 활성 모델 키를 조회해 캐시를 채움.
    실패해도 서비스는 실제 사용 시 지연 로딩으로 복구됨.
    """
    try:
        key = get_rag_settings_row().get("embedding_key")
        logger_func(f"[warmup] 활성 임베딩 모델: {key}. 로딩 시도...")
        _get_or_load_embedder(key, preload=True)
        logger_func(f"[warmup] 로딩 완료: {key}")
    except Exception as e:
        logger_func(f"[warmup] 로딩 실패(지연 로딩으로 복구 예정): {e}")


def _update_vector_settings(
    search_type: Optional[str] = None,
    chunk_size: Optional[int] = None,
    overlap: Optional[int] = None,
):
    """레거시 API 호환: rag_settings(싱글톤) 업데이트"""
    cur = get_rag_settings_row()
    new_search = (search_type or cur["search_type"]).lower()
    if new_search == "vector":
        new_search = "semantic"
    if new_search not in {"hybrid", "semantic", "bm25"}:
        raise ValueError(
            "unsupported searchType; allowed: 'hybrid','semantic','bm25' (or 'vector' alias)"
        )
    new_chunk = int(chunk_size if chunk_size is not None else cur["chunk_size"])
    new_overlap = int(overlap if overlap is not None else cur["overlap"])
    if new_chunk <= 0 or new_overlap < 0 or new_overlap >= new_chunk:
        raise ValueError("invalid chunk/overlap (chunk>0, 0 <= overlap < chunk)")

    with get_session() as session:
        s = session.query(RagSettings).filter(RagSettings.id == 1).first()
        if not s:
            s = RagSettings(id=1)
            session.add(s)
        s.search_type = new_search
        s.chunk_size = new_chunk
        s.overlap = new_overlap
        s.updated_at = now_kst()
        session.commit()


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
    cur = get_vector_settings()
    key_now = cur.get("embeddingModel")
    st_now = (cur.get("searchType") or "hybrid").lower()
    cs_now = int(cur.get("chunkSize") or 512)
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
        _invalidate_embedder_cache()

    with get_session() as session:
        s = session.query(RagSettings).filter(RagSettings.id == 1).first()
        if not s:
            s = RagSettings(id=1)
            session.add(s)
        s.embedding_key = new_key
        # search_type/chunk/overlap은 _update_vector_settings에서 반영됨. 여기선 존재 시 보존
        if search_type is not None:
            s.search_type = (
                (search_type or "hybrid").lower().replace("vector", "semantic")
            )
        if chunk_size is not None:
            s.chunk_size = int(chunk_size)
        if overlap is not None:
            s.overlap = int(overlap)
        s.updated_at = now_kst()
        session.commit()

    return get_vector_settings()


def get_vector_settings() -> Dict:
    # rag_settings 는 검색 타입/청크/오버랩만 신뢰
    try:
        row = get_rag_settings_row()
    except Exception:
        logger.error("get_rag_settings_row 실패")
        return {
            "embeddingModel": "unknown",
            "searchType": "hybrid",
            "chunkSize": 512,
            "overlap": 64,
        }
    return {
        "embeddingModel": row.get("embedding_key"),                        # ← rag_settings.embedding_key는 무시
        "searchType": row.get("search_type", "hybrid"),
        "chunkSize": int(row.get("chunk_size", 512)),
        "overlap": int(row.get("overlap", 64)),
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
def _parse_at_string_to_keywords(value: str) -> List[str]:
    if not value:
        return []
    toks = [t.strip() for t in value.split("@")]
    return [t for t in toks if t]


def _normalize_keywords(val: Any) -> List[str]:
    """
    리스트/튜플/셋: 각 원소를 str로 캐스팅, 공백/해시 제거
    문자열: '@' 기준으로 토큰화
    빈 값 제거 및 중복 제거
    """
    out: List[str] = []
    if isinstance(val, str):
        toks = [t.strip() for t in val.split("@")]
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

    with get_session() as session:
        # upsert config
        cfg = (
            session.query(SecurityLevelConfigTask)
            .filter(SecurityLevelConfigTask.task_type == task_type)
            .first()
        )
        if not cfg:
            cfg = SecurityLevelConfigTask(task_type=task_type, max_level=int(max_level))
            session.add(cfg)
        else:
            cfg.max_level = int(max_level)
            cfg.updated_at = now_kst()
        # replace keywords
        session.query(SecurityLevelKeywordsTask).filter(
            SecurityLevelKeywordsTask.task_type == task_type
        ).delete()
        for lv, kws in levels_map.items():
            for kw in kws:
                session.add(
                    SecurityLevelKeywordsTask(
                        task_type=task_type, level=int(lv), keyword=str(kw)
                    )
                )
        session.commit()
        return get_security_level_rules_for_task(task_type)


def get_security_level_rules_for_task(task_type: str) -> Dict:
    with get_session() as session:
        cfg = (
            session.query(SecurityLevelConfigTask)
            .filter(SecurityLevelConfigTask.task_type == task_type)
            .first()
        )
        max_level = int(cfg.max_level) if cfg else 1
        res: Dict[str, Any] = {
            "taskType": task_type,
            "maxLevel": max_level,
            "levels": {str(i): [] for i in range(1, max_level + 1)},
        }
        rows = (
            session.query(
                SecurityLevelKeywordsTask.level, SecurityLevelKeywordsTask.keyword
            )
            .filter(SecurityLevelKeywordsTask.task_type == task_type)
            .order_by(
                SecurityLevelKeywordsTask.level.asc(),
                SecurityLevelKeywordsTask.keyword.asc(),
            )
            .all()
        )
        for lv, kw in rows:
            key = str(int(lv))
            res["levels"].setdefault(key, []).append(str(kw))
        return res


def set_security_level_rules_per_task(config: Dict[str, Dict]) -> Dict:
    """
    config = {
      "doc_gen": {"maxLevel": 3, "levels": {"2": "@금액@연봉", "3": "@부정@퇴직금"}},
      "summary": {"maxLevel": 2, "levels": {"2": "@사내비밀"}},
      "qna": {"maxLevel": 3, "levels": {"2": "@연구", "3": "@개인정보"}}
    }
    """
    with get_session() as session:
        # 전체 삭제 후 재삽입(간결/명확)
        session.query(SecurityLevelConfigTask).delete()
        session.query(SecurityLevelKeywordsTask).delete()
        session.flush()

        for task in TASK_TYPES:
            entry = config.get(task) or {}
            max_level = int(entry.get("maxLevel", 1))
            session.add(
                SecurityLevelConfigTask(task_type=task, max_level=max(1, max_level))
            )
            levels = entry.get("levels", {}) or {}
            for lvl_str, at_str in levels.items():
                try:
                    lvl = int(str(lvl_str).strip().replace("level_", ""))
                except Exception:
                    continue
                if lvl <= 1 or lvl > max_level:
                    continue
                for kw in _parse_at_string_to_keywords(str(at_str)):
                    session.add(
                        SecurityLevelKeywordsTask(
                            task_type=task, level=int(lvl), keyword=str(kw)
                        )
                    )
        session.commit()
        return get_security_level_rules_all()


def get_security_level_rules_all() -> Dict:
    with get_session() as session:
        # 기본 max_level=1
        max_map = {t: 1 for t in TASK_TYPES}
        for task, max_level in session.query(
            SecurityLevelConfigTask.task_type, SecurityLevelConfigTask.max_level
        ).all():
            max_map[task] = int(max_level)

        res: Dict[str, Dict] = {}
        for task in TASK_TYPES:
            res[task] = {
                "maxLevel": max_map.get(task, 1),
                "levels": {str(i): [] for i in range(1, max_map.get(task, 1) + 1)},
            }

        rows = (
            session.query(
                SecurityLevelKeywordsTask.task_type,
                SecurityLevelKeywordsTask.level,
                SecurityLevelKeywordsTask.keyword,
            )
            .order_by(
                SecurityLevelKeywordsTask.task_type.asc(),
                SecurityLevelKeywordsTask.level.asc(),
                SecurityLevelKeywordsTask.keyword.asc(),
            )
            .all()
        )
        for task, level, kw in rows:
            if task in res:
                lv = str(int(level))
                if lv not in res[task]["levels"]:
                    res[task]["levels"][lv] = []
                res[task]["levels"][lv].append(str(kw))
        return res


def determine_level_for_task(text: str, task_rules: Dict) -> int:
    max_level = int(task_rules.get("maxLevel", 1))
    levels = task_rules.get("levels", {})
    sel = 1
    # 상위 레벨 우선
    for lvl in range(1, max_level + 1):
        kws = levels.get(str(lvl), [])
        for kw in kws:
            if kw and kw in text:
                sel = max(sel, lvl)
    return sel


# -------------------------------------------------
# 1) PDF → 텍스트 추출 (작업유형별 보안레벨 동시 산정)
# -------------------------------------------------
# extract_pdfs() 함수는 service/preprocessing/pdf/pdf_preprocessing.py로 이동했습니다.


def parse_doc_version(stem: str) -> Tuple[str, int]:
    if "_" in stem:
        base, cand = stem.rsplit("_", 1)
        if cand.isdigit() and len(cand) in (4, 8):
            return base, int(cand)
    return stem, 0


# -------------------------------------------------
# 2) 인제스트 (bulk)
#   - 작업유형별로 동일 청크를 각각 저장(task_type, security_level 분리)
# -------------------------------------------------
async def ingest_embeddings(
    model_key: str | None = None,
    target_tasks: list[str] | None = None,
    max_token: int = 512,
    overlab: int = 64,
    collection_name: str = ADMIN_COLLECTION,
    file_keys_filter: list[str] | None = None,
):
    """
    documents 테이블에 저장된 관리자 문서를 기준으로 추출된 텍스트(.txt)를 인제스트한다.
    - VARCHAR(32768 bytes) 초과 방지: split_for_varchar_bytes 로 안전 분할
    - 표는 [[TABLE ...]] 머리글 유지, 이어지는 조각은 [[TABLE_CONT i/n]] 마커로 연속성 표시
    - file_keys_filter 전달 시 doc_id/파일명/스토리지 경로가 일치하는 문서만 인제스트
    """
    tok, model, device = await _get_or_load_embedder_async(model_key)
    probe_vec = hf_embed_text(tok, model, device, "probe")
    emb_dim = int(probe_vec.shape[0])
    logger.info("[Ingest] 임베딩 모델: %s, 벡터 차원: %s", model_key, emb_dim)

    client = get_milvus_client()
    if collection_name in client.list_collections():
        try:
            desc = client.describe_collection(collection_name)
            existing_dim = None
            for field in desc.get("fields", []):
                if field.get("name") == "embedding":
                    existing_dim = field.get("params", {}).get("dim")
                    break
            if existing_dim and int(existing_dim) != emb_dim:
                logger.warning("[Ingest] 차원 불일치: 기존=%s, 새모델=%s. 컬렉션 재생성.", existing_dim, emb_dim)
                client.drop_collection(collection_name)
        except Exception as exc:
            logger.warning("[Ingest] 컬렉션 정보 확인 실패: %s. 재생성 시도.", exc)
            try:
                client.drop_collection(collection_name)
            except Exception:
                pass

    ensure_collection_and_index(client, emb_dim=emb_dim, metric="IP", collection_name=collection_name)

    tasks = [t for t in (target_tasks or TASK_TYPES) if t in TASK_TYPES]
    if not tasks:
        return {"error": f"유효한 작업유형이 없습니다. 허용: {TASK_TYPES}"}

    documents = _load_admin_documents(file_keys_filter)
    if not documents:
        return {"error": "관리자 문서 메타데이터가 없습니다. 먼저 문서를 추출하세요."}

    total_inserted = 0
    BATCH_SIZE = 128

    for doc in documents:
        rel_txt = str(doc.get("storage_path") or "").strip()
        if not rel_txt:
            continue
        txt_path = EXTRACTED_TEXT_DIR / Path(rel_txt)
        if not txt_path.exists():
            logger.warning("[Ingest] 텍스트 파일 누락: %s", txt_path)
            continue

        payload = doc.get("payload") or {}
        sec_map = payload.get("security_levels", {}) or {}
        doc_id = str(doc.get("doc_id") or "").strip()
        version = int(payload.get("version") or 0)
        if not doc_id:
            doc_id, parsed_version = parse_doc_version(Path(rel_txt).stem)
            version = version or parsed_version
        if version == 0:
            _, version = parse_doc_version(Path(rel_txt).stem)

        try:
            text = txt_path.read_text(encoding="utf-8")
        except Exception:
            text = txt_path.read_text(errors="ignore")

        def _parse_integrated_file(content: str) -> list[tuple[int, str]]:
            page_blocks: list[tuple[int, str]] = []
            lines = content.split("\n")
            current_page = 1
            current_content: List[str] = []
            for line in lines:
                if line.strip() == "---":
                    if current_content:
                        page_text = "\n".join(current_content).strip()
                        if page_text:
                            page_blocks.append((current_page, page_text))
                    current_page += 1
                    current_content = []
                else:
                    current_content.append(line)
            if current_content:
                page_text = "\n".join(current_content).strip()
                if page_text:
                    page_blocks.append((current_page, page_text))
            if not page_blocks and content.strip():
                page_blocks = [(1, content.strip())]
            return page_blocks

        page_blocks = _parse_integrated_file(text)
        logger.info("[Ingest] doc_id=%s → %s개 페이지 블록", doc_id, len(page_blocks))

        chunks_with_page: list[tuple[int, int, str]] = []
        global_chunk_idx = 0
        for page_num, page_text in page_blocks:
            if not page_text:
                continue
            page_chunks = chunk_text(page_text, max_tokens=max_token, overlap=overlab)
            for chunk in page_chunks:
                if chunk.strip():
                    chunks_with_page.append((page_num, global_chunk_idx, chunk))
                    global_chunk_idx += 1

        tables = payload.get("tables", []) or []
        logger.info("[Ingest] doc_id=%s 표 정보: %s개", doc_id, len(tables))

        rel_txt_posix = Path(rel_txt).as_posix()
        chunk_entries: List[Dict[str, Any]] = []
        metadata_records: List[Dict[str, Any]] = []
        metadata_seen: set[int] = set()

        for page_num, idx, chunk_text_val in chunks_with_page:
            for part in split_for_varchar_bytes(chunk_text_val):
                chunk_entries.append(
                    {
                        "page": int(page_num),
                        "chunk_idx": int(idx),
                        "text": part,
                    }
                )
            idx_int = int(idx)
            if idx_int not in metadata_seen:
                metadata_records.append(
                    {
                        "page": int(page_num),
                        "chunk_index": idx_int,
                        "text": chunk_text_val,
                        "payload": {"path": rel_txt_posix},
                    }
                )
                metadata_seen.add(idx_int)
        if metadata_records: 
            bulk_upsert_document_metadata(doc_id=doc_id, records=metadata_records)

        base_idx = len(chunks_with_page)
        for t_i, table in enumerate(tables):
            md = (table.get("text") or "").strip()
            if not md:
                continue
            page = int(table.get("page", 0))
            bbox = table.get("bbox") or []
            bbox_str = ",".join(str(x) for x in bbox) if bbox else ""
            table_text = f"[[TABLE page={page} bbox={bbox_str}]]\n{md}"
            for sub_j, part in enumerate(split_for_varchar_bytes(table_text)):
                chunk_idx = base_idx + t_i * 1000 + sub_j
                chunk_entries.append(
                    {
                        "page": int(page),
                        "chunk_idx": int(chunk_idx),
                        "text": part,
                    }
                )
                if chunk_idx not in metadata_seen:
                    metadata_records.append(
                        {
                            "page": int(page),
                            "chunk_index": int(chunk_idx),
                            "text": part,
                            "payload": {"path": rel_txt_posix, "table": True},
                        }
                    )
                    metadata_seen.add(chunk_idx)

        if metadata_records:
            try:
                bulk_upsert_document_metadata(doc_id=doc_id, records=metadata_records)
            except Exception:
                logger.exception("Failed to upsert metadata for doc_id=%s", doc_id)

        batch: List[Dict[str, Any]] = []
        batch_meta: List[Dict[str, int]] = []
        vector_records: List[Dict[str, Any]] = []

        def flush_batch() -> None:
            nonlocal batch, batch_meta, total_inserted
            if not batch:
                return
            try:
                result = client.insert(collection_name=collection_name, data=batch)
            except Exception:
                logger.exception("Milvus insert 실패(doc_id=%s)", doc_id)
                batch.clear()
                batch_meta.clear()
                return
            ids = _extract_insert_ids(result)
            if ids and len(ids) != len(batch_meta):
                logger.warning(
                    "inserted ids count mismatch doc_id=%s expected=%s got=%s",
                    doc_id,
                    len(batch_meta),
                    len(ids),
                )
            for vec_id, meta in zip(ids or [], batch_meta):
                vector_records.append(
                    {
                        "vector_id": vec_id,
                        "page": meta["page"],
                        "chunk_index": meta["chunk_idx"],
                    }
                )
            total_inserted += len(batch)
            batch.clear()
            batch_meta.clear()
        
        if vector_records:
            insert_document_vectors(
                doc_id=doc_id,
                collection=collection_name,
                embedding_version=str(model_key),
                vectors=vector_records,
            )

        for task in tasks:
            lvl = int(sec_map.get(task, 1))

            for entry_chunk in chunk_entries:
                part = entry_chunk["text"]
                if not part:
                    continue
                vec = hf_embed_text(tok, model, device, part, max_len=max_token)
                if len(vec) != emb_dim:
                    logger.error(
                        "[Ingest] 벡터 차원 불일치: 예상=%s, 실제=%s, doc_id=%s",
                        emb_dim,
                        len(vec),
                        doc_id,
                    )
                    continue
                batch.append(
                    {
                        "embedding": vec.tolist(),
                        "path": rel_txt_posix,
                        "chunk_idx": int(entry_chunk["chunk_idx"]),
                        "task_type": task,
                        "security_level": lvl,
                        "doc_id": doc_id,
                        "version": int(version),
                        "page": int(entry_chunk["page"]),
                        "workspace_id": 0,
                        "text": part,
                    }
                )
                batch_meta.append(
                    {
                        "page": int(entry_chunk["page"]),
                        "chunk_idx": int(entry_chunk["chunk_idx"]),
                    }
                )
                if len(batch) >= BATCH_SIZE:
                    flush_batch()

        flush_batch()

        if vector_records:
            try:
                insert_document_vectors(
                    doc_id=doc_id,
                    collection=collection_name,
                    embedding_version=str(model_key),
                    vectors=vector_records,
                )
            except Exception:
                logger.exception("document_vectors 기록 실패(doc_id=%s)", doc_id)

    try:
        client.flush(collection_name)
    except Exception:
        pass
    ensure_collection_and_index(client, emb_dim=emb_dim, metric="IP", collection_name=collection_name)

    return {
        "message": f"Ingest 완료(Milvus Server, collection={collection_name})",
        "inserted_chunks": int(total_inserted),
    }


# -------------------------------------------------
# 2-1) 단일 파일 인제스트(선택 작업유형)
# -------------------------------------------------
async def ingest_single_pdf(req: SinglePDFIngestRequest):
    try:
        from repository.documents import insert_workspace_document
    except Exception:
        insert_workspace_document = None

    file_path = Path(req.pdf_path)
    if not file_path.exists():
        return {"error": f"파일 경로를 찾을 수 없습니다: {file_path}"}

    if _ext(file_path) not in SUPPORTED_EXTS:
        return {"error": f"지원되지 않는 파일 형식입니다: {_ext(file_path)}"}

    # 추출
    from service.preprocessing.rag_preprocessing import extract_any

    text_all, table_blocks_all = extract_any(file_path)

    # 보안 레벨 판정(본문+표)
    all_rules = get_security_level_rules_all()
    whole_for_level = text_all + "\n\n" + "\n\n".join(t.get("text","") for t in (table_blocks_all or []))
    sec_map = {task: determine_level_for_task(whole_for_level, all_rules.get(task, {"maxLevel": 1, "levels": {}})) for task in TASK_TYPES}
    max_sec = max(sec_map.values()) if sec_map else 1
    sec_folder = f"securityLevel{int(max_sec)}"

    # 보관 및 텍스트 저장
    rel_file = Path(sec_folder) / file_path.name
    (LOCAL_DATA_ROOT / rel_file).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(file_path, LOCAL_DATA_ROOT / rel_file)
    txt_path = EXTRACTED_TEXT_DIR / rel_file.with_suffix(".txt")
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.write_text(text_all, encoding="utf-8")

    from service.preprocessing.rag_preprocessing import _clean_text as clean_text

    doc_id = generate_doc_id()
    _, ver = parse_doc_version(file_path.stem)
    preview = (clean_text(text_all[:200].replace("\n", " ")) + "…") if text_all else ""
    rel_source_path = str(rel_file.as_posix())
    rel_text_path = str(rel_file.with_suffix(".txt").as_posix())
    extraction_info = {
        "original_file": file_path.name,
        "text_length": len(text_all),
        "table_count": len(table_blocks_all or []),
        "extracted_at": now_kst_string(),
    }
    register_admin_document(
        doc_id=doc_id,
        filename=file_path.name,
        rel_text_path=rel_text_path,
        rel_source_path=rel_source_path,
        sec_map=sec_map,
        version=int(ver),
        preview=preview,
        tables=table_blocks_all or [],
        total_pages=0,
        pages={},
        source_ext=_ext(file_path),
        extraction_info=extraction_info,
    )

    # 인제스트
    settings = get_vector_settings()
    tok, model, device = await _get_or_load_embedder_async(settings["embeddingModel"])
    emb_dim = int(hf_embed_text(tok, model, device, "probe").shape[0])
    client = get_milvus_client()
    ensure_collection_and_index(client, emb_dim=emb_dim, metric="IP", collection_name=ADMIN_COLLECTION)

    s = get_vector_settings()
    max_token, overlab = int(s["chunkSize"]), int(s["overlap"])

    # 기존 삭제
    try:
        client.delete(ADMIN_COLLECTION, filter=f"doc_id == '{doc_id}' && version <= {int(ver)}")
    except Exception:
        pass

    tasks = req.task_types or list(TASK_TYPES)
    chunks = chunk_text(text_all, max_tokens=max_token, overlab=overlab)

    chunk_entries: List[Dict[str, Any]] = []
    metadata_records: List[Dict[str, Any]] = []
    metadata_seen: set[int] = set()

    for idx, chunk_text_val in enumerate(chunks):
        for part in split_for_varchar_bytes(chunk_text_val):
            chunk_entries.append(
                {
                    "page": 0,
                    "chunk_idx": int(idx),
                    "text": part,
                }
            )
        if idx not in metadata_seen:
            metadata_records.append(
                {
                    "page": 0,
                    "chunk_index": int(idx),
                    "text": chunk_text_val,
                    "payload": {"path": rel_text_path},
                }
            )
            metadata_seen.add(idx)

    base_idx = len(chunks)
    for t_i, table in enumerate(table_blocks_all or []):
        md = (table.get("text") or "").strip()
        if not md:
            continue
        page = int(table.get("page", 0))
        bbox = table.get("bbox") or []
        bbox_str = ",".join(str(x) for x in bbox) if bbox else ""
        table_text = f"[[TABLE page={page} bbox={bbox_str}]]\n{md}"

        for sub_j, part in enumerate(split_for_varchar_bytes(table_text)):
            chunk_idx = base_idx + t_i * 1000 + sub_j
            chunk_entries.append(
                {
                    "page": int(page),
                    "chunk_idx": int(chunk_idx),
                    "text": part,
                }
            )
            if chunk_idx not in metadata_seen:
                metadata_records.append(
                    {
                        "page": int(page),
                        "chunk_index": int(chunk_idx),
                        "text": part,
                        "payload": {"path": rel_text_path, "table": True},
                    }
                )
                metadata_seen.add(chunk_idx)

    if metadata_records:
        try:
            bulk_upsert_document_metadata(doc_id=doc_id, records=metadata_records)
        except Exception:
            logger.exception("Failed to upsert metadata for doc_id=%s", doc_id)

    batch: List[Dict[str, Any]] = []
    batch_meta: List[Dict[str, int]] = []
    vector_records: List[Dict[str, Any]] = []
    cnt = 0

    def flush_single_batch() -> None:
        nonlocal batch, batch_meta, cnt
        if not batch:
            return
        try:
            result = client.insert(collection_name=ADMIN_COLLECTION, data=batch)
        except Exception:
            logger.exception("Milvus insert 실패(doc_id=%s)", doc_id)
            batch.clear()
            batch_meta.clear()
            return
        ids = _extract_insert_ids(result)
        for vec_id, meta in zip(ids or [], batch_meta):
            vector_records.append(
                {
                    "vector_id": vec_id,
                    "page": meta["page"],
                    "chunk_index": meta["chunk_idx"],
                }
            )
        cnt += len(batch)
        batch.clear()
        batch_meta.clear()

    for task in tasks:
        lvl = int(sec_map.get(task, 1))

        for entry_chunk in chunk_entries:
            part = entry_chunk["text"]
            if not part:
                continue
            vec = hf_embed_text(tok, model, device, part, max_len=max_token)
            batch.append(
                {
                    "embedding": vec.tolist(),
                    "path": rel_text_path,
                    "chunk_idx": int(entry_chunk["chunk_idx"]),
                    "task_type": task,
                    "security_level": lvl,
                    "doc_id": str(doc_id),
                    "version": int(ver),
                    "page": int(entry_chunk["page"]),
                    "workspace_id": 0,
                    "text": part,
                }
            )
            batch_meta.append(
                {
                    "page": int(entry_chunk["page"]),
                    "chunk_idx": int(entry_chunk["chunk_idx"]),
                }
            )
            if len(batch) >= 128:
                flush_single_batch()

    flush_single_batch()

    if vector_records:
        try:
            insert_document_vectors(
                doc_id=doc_id,
                collection=ADMIN_COLLECTION,
                embedding_version=str(settings["embeddingModel"]),
                vectors=vector_records,
            )
        except Exception:
            logger.exception("document_vectors 기록 실패(doc_id=%s)", doc_id)

    try:
        client.flush(ADMIN_COLLECTION)
    except Exception:
        pass
    ensure_collection_and_index(client, emb_dim=emb_dim, metric="IP", collection_name=ADMIN_COLLECTION)

    return {
        "message": f"단일 파일 인제스트 완료(Milvus Server) - {_ext(file_path)}",
        "doc_id": doc_id,
        "version": ver,
        "chunks": cnt,
        "sourceExt": _ext(file_path),
    }

async def ingest_specific_files_with_levels(
    uploads: Optional[List[Any]] = None,          # FastAPI UploadFile 리스트
    paths: Optional[List[str]] = None,            # 로컬 경로 리스트
    tasks: Optional[List[str]] = None,            # 없으면 모든 TASK_TYPES
    level_for_tasks: Optional[Dict[str, int]] = None,  # {"qna":2,"summary":1} 우선
    level: Optional[int] = None,                  # 공통 레벨. 위 map 있으면 무시
    collection_name: Optional[str] = None,
):
    if not uploads and not paths:
        return {"error": "대상 파일이 없습니다. uploads 또는 paths 중 하나는 필요합니다."}

    tasks_eff = [t for t in (tasks or TASK_TYPES) if t in TASK_TYPES]
    if not tasks_eff:
        return {"error": f"유효한 작업유형이 없습니다. 허용: {TASK_TYPES}"}

    lvl_map: Dict[str, int] = {}
    if level_for_tasks:
        for k, v in level_for_tasks.items():
            if k in TASK_TYPES:
                lvl_map[k] = max(1, int(v))
    elif level is not None:
        for t in tasks_eff:
            lvl_map[t] = max(1, int(level))

    # 업로드 저장(임시) + 경로 합치기
    run_id = uuid.uuid4().hex[:8]
    tmp_root = (VAL_SESSION_ROOT / "adhoc" / run_id).resolve()
    tmp_root.mkdir(parents=True, exist_ok=True)

    saved: List[Path] = []
    if uploads:
        for f in uploads:
            fname = Path(getattr(f, "filename", "uploaded")).name
            tmp_path = tmp_root / fname
            try:
                data = await f.read()
            except Exception:
                data = getattr(getattr(f, "file", None), "read", lambda: b"")()
            tmp_path.write_bytes(data or b"")
            saved.append(tmp_path)
    for p in (paths or []):
        pp = Path(str(p)).resolve()
        if pp.exists() and pp.is_file():
            saved.append(pp)

    if not saved:
        return {"error": "저장/유효성 검사 후 남은 파일이 없습니다."}

    # 임베더/컬렉션 준비
    settings = get_vector_settings()
    model_key = settings["embeddingModel"]
    tok, model, device = await _get_or_load_embedder_async(model_key)
    emb_dim = int(hf_embed_text(tok, model, device, "probe").shape[0])

    coll = collection_name or ADMIN_COLLECTION
    client = get_milvus_client()
    ensure_collection_and_index(client, emb_dim=emb_dim, metric="IP", collection_name=coll)

    max_token, overlab = int(settings["chunkSize"]), int(settings["overlap"])

    processed, total = [], 0
    from service.preprocessing.rag_preprocessing import extract_any

    for src in saved:
        try:
            text, tables = extract_any(src)

            # 레벨 결정(강제 > 규칙)
            if lvl_map:
                sec_map = {t: int(lvl_map.get(t, 1)) for t in tasks_eff}
            else:
                all_rules = get_security_level_rules_all()
                whole = text + "\n\n" + "\n\n".join(t.get("text", "") for t in (tables or []))
                sec_map = {
                    t: determine_level_for_task(whole, all_rules.get(t, {"maxLevel": 1, "levels": {}}))
                    for t in tasks_eff
                }
            max_sec = max(sec_map.values()) if sec_map else 1

            # 스니펫 로딩용 텍스트 저장(메인과 분리: __adhoc__)
            rel_txt = Path("__adhoc__") / run_id / f"securityLevel{int(max_sec)}" / src.with_suffix(".txt").name
            abs_txt = EXTRACTED_TEXT_DIR / rel_txt
            abs_txt.parent.mkdir(parents=True, exist_ok=True)
            abs_txt.write_text(text, encoding="utf-8")

            # 문서 ID/버전
            doc_id = generate_doc_id()
            _, ver = parse_doc_version(src.stem)

            # 기존 삭제
            try:
                client.delete(collection_name=coll, filter=f"doc_id == '{doc_id}' && version <= {int(ver)}")
            except Exception:
                pass

            chunks = chunk_text(text, max_tokens=max_token, overlap=overlab)
            chunk_entries: List[Dict[str, Any]] = []
            metadata_records: List[Dict[str, Any]] = []
            metadata_seen: set[int] = set()

            rel_txt_posix = str(rel_txt.as_posix())

            for idx, chunk_text_val in enumerate(chunks):
                for part in split_for_varchar_bytes(chunk_text_val):
                    chunk_entries.append(
                        {
                            "page": 0,
                            "chunk_idx": int(idx),
                            "text": part,
                        }
                    )
                if idx not in metadata_seen:
                    metadata_records.append(
                        {
                            "page": 0,
                            "chunk_index": int(idx),
                            "text": chunk_text_val,
                            "payload": {"path": rel_txt_posix},
                        }
                    )
                    metadata_seen.add(idx)

            base_idx = len(chunks)
            for t_i, tb in enumerate(tables or []):
                md = (tb.get("text") or "").strip()
                if not md:
                    continue
                page = int(tb.get("page", 0))
                bbox = tb.get("bbox") or []
                bbox_str = ",".join(str(x) for x in bbox) if bbox else ""
                table_text = f"[[TABLE page={page} bbox={bbox_str}]]\n{md}"
                for sub_j, part in enumerate(split_for_varchar_bytes(table_text)):
                    chunk_idx = base_idx + t_i * 1000 + sub_j
                    chunk_entries.append(
                        {
                            "page": int(page),
                            "chunk_idx": int(chunk_idx),
                            "text": part,
                        }
                    )
                    if chunk_idx not in metadata_seen:
                        metadata_records.append(
                            {
                                "page": int(page),
                                "chunk_index": int(chunk_idx),
                                "text": part,
                                "payload": {"path": rel_txt_posix, "table": True},
                            }
                        )
                        metadata_seen.add(chunk_idx)

            if metadata_records:
                try:
                    bulk_upsert_document_metadata(doc_id=doc_id, records=metadata_records)
                except Exception:
                    logger.exception("Failed to upsert metadata for doc_id=%s", doc_id)

            batch: List[Dict[str, Any]] = []
            batch_meta: List[Dict[str, int]] = []
            doc_vector_records: List[Dict[str, Any]] = []
            cnt = 0

            def flush_local_batch() -> None:
                nonlocal batch, batch_meta, cnt, total
                if not batch:
                    return
                try:
                    result = client.insert(collection_name=coll, data=batch)
                except Exception:
                    logger.exception("[upload-and-ingest] insert 실패: doc_id=%s", doc_id)
                    batch.clear()
                    batch_meta.clear()
                    return
                ids = _extract_insert_ids(result)
                for vec_id, meta in zip(ids or [], batch_meta):
                    doc_vector_records.append(
                        {
                            "vector_id": vec_id,
                            "page": meta["page"],
                            "chunk_index": meta["chunk_idx"],
                        }
                    )
                cnt += len(batch)
                batch.clear()
                batch_meta.clear()

            for t in tasks_eff:
                lvl = int(sec_map.get(t, 1))

                for entry_chunk in chunk_entries:
                    part = entry_chunk["text"]
                    if not part:
                        continue
                    vec = hf_embed_text(tok, model, device, part, max_len=max_token)
                    batch.append(
                        {
                            "embedding": vec.tolist(),
                            "path": rel_txt_posix,
                            "chunk_idx": int(entry_chunk["chunk_idx"]),
                            "task_type": t,
                            "security_level": lvl,
                            "doc_id": str(doc_id),
                            "version": int(ver),
                            "page": int(entry_chunk["page"]),
                            "workspace_id": 0,
                            "text": part,
                        }
                    )
                    batch_meta.append(
                        {
                            "page": int(entry_chunk["page"]),
                            "chunk_idx": int(entry_chunk["chunk_idx"]),
                        }
                    )
                    if len(batch) >= 128:
                        flush_local_batch()

            flush_local_batch()

            if doc_vector_records:
                try:
                    insert_document_vectors(
                        doc_id=doc_id,
                        collection=coll,
                        embedding_version=str(model_key),
                        vectors=doc_vector_records,
                    )
                except Exception:
                    logger.exception("document_vectors 기록 실패(doc_id=%s)", doc_id)

            processed.append(
                {
                    "file": src.name,
                    "doc_id": doc_id,
                    "version": int(ver),
                    "levels": sec_map,
                    "chunks": cnt,
                }
            )
            total += cnt

        except Exception:
            logger.exception("[upload-and-ingest] failed: %s", src)

    try:
        client.flush(coll)
    except Exception:
        pass
    ensure_collection_and_index(client, emb_dim=emb_dim, metric="IP", collection_name=coll)

    return {
        "message": "Upload & Ingest 완료",
        "collection": coll,
        "runId": run_id,
        "processed": processed,
        "inserted_chunks": int(total),
    }

async def search_documents(req: RAGSearchRequest, 
                            search_type_override: Optional[str] = None,
                            rerank_top_n: Optional[int] = None) -> Dict:
    t0 = time.perf_counter()
    print(f"🔍 [Search] 검색 시작: query='{req.query}', topK={req.top_k}, rerank_topN={rerank_top_n}, task={req.task_type}")
    
    if req.task_type not in TASK_TYPES:
        return {
            "error": f"invalid task_type: {req.task_type}. choose one of {TASK_TYPES}"
        }

    settings = get_vector_settings()
    model_key = req.model or settings["embeddingModel"]
    raw_st = (search_type_override or settings.get("searchType") or "").lower()
    # alias normalization: 'semantic'/'sementic' -> 'vector'; default 'hybrid' if empty
    search_type = (raw_st.replace("semantic", "vector").replace("sementic", "vector") or "hybrid")

    tok, model, device = await _get_or_load_embedder_async(model_key)
    q_emb = hf_embed_text(tok, model, device, req.query)
    client = get_milvus_client()
    ensure_collection_and_index(client, emb_dim=len(q_emb), metric="IP", collection_name=ADMIN_COLLECTION)

    if ADMIN_COLLECTION not in client.list_collections():
        return {"error": "컬렉션이 없습니다. 먼저 데이터 저장(인제스트)을을 수행하세요."}

    # 공통 파라미터
    embedding_candidates = int(req.top_k)  # 임베딩에서 찾을 후보 개수
    final_results = int(rerank_top_n) if rerank_top_n is not None else 5  # 최종 반환 개수
    candidate = max(embedding_candidates, final_results * 2)  # 충분한 후보 확보
    filter_expr = f"task_type == '{req.task_type}' && security_level <= {int(req.user_level)}"
    snippet_loader = partial(
        load_snippet_from_store,
        EXTRACTED_TEXT_DIR,
        max_tokens=512,
        overlap=64,
    )

    # === 분기: 검색 방식 ===
    if search_type == "vector":
        res_dense = run_dense_search(
            client,
            collection_name=ADMIN_COLLECTION,
            query_vector=q_emb.tolist(),
            limit=candidate,
            filter_expr=filter_expr,
            output_fields=DEFAULT_OUTPUT_FIELDS,
        )
        hits_raw = build_dense_hits(res_dense, snippet_loader=snippet_loader)
    else:
        res_hybrid = run_hybrid_search(
            client,
            collection_name=ADMIN_COLLECTION,
            query_vector=q_emb.tolist(),
            query_text=req.query,
            limit=candidate,
            filter_expr=filter_expr,
            output_fields=DEFAULT_OUTPUT_FIELDS,
        )
        hits_raw = build_dense_hits(res_hybrid, snippet_loader=snippet_loader)

    # 검색 결과 상태 로그
    logger.info(f"📊 [Search] 벡터/BM25 검색 완료: 후보 {len(hits_raw)}개 발견")
    if hits_raw:
        logger.info(f"📊 [Search] 첫 번째 후보: doc_id={hits_raw[0].get('doc_id')}, path={hits_raw[0].get('path')}")

    rerank_candidates = build_rerank_payload(hits_raw)

    if rerank_candidates:
        reranked = rerank_snippets(rerank_candidates, query=req.query, top_n=final_results)
        hits_sorted = []
        for res in reranked:
            original = res.metadata or {}
            hits_sorted.append(
                {
                    "score": float(res.score),
                    "path": original.get("path"),
                    "chunk_idx": int(original.get("chunk_idx", 0)),
                    "task_type": original.get("task_type"),
                    "security_level": int(original.get("security_level", 1)),
                    "doc_id": original.get("doc_id"),
                    "page": int(original.get("page", 0)),
                    "snippet": res.text,
                }
            )
    else:
        hits_sorted = sorted(
            hits_raw,
            key=lambda x: x.get("score_fused", x.get("score_vec", x.get("score_sparse", 0.0))),
            reverse=True,
        )[:final_results]
    
    # 리랭크 후 중복 제거
    # 1) snippet_text 기준: 동일한 내용의 스니펫은 하나만 (최고 점수만 유지, doc_id 무관)
    # 2) (doc_id, chunk_idx) 기준: 같은 문서의 같은 청크는 하나만 (chunk_idx 중복 방지)
    # 문서당 제한 없음 - rerank_topN만큼 모두 반환
    seen_by_snippet: dict[str, dict] = {}  # snippet_text -> hit (최고 점수만 유지)
    seen_by_chunk: dict[tuple[str, int], dict] = {}  # (doc_id, chunk_idx) -> hit
    
    original_count = len(hits_sorted)
    
    for hit in hits_sorted:
        doc_id = hit.get("doc_id", "")
        chunk_idx = int(hit.get("chunk_idx", 0))
        snippet = hit.get("snippet", "").strip()
        
        if not snippet:
            continue
        
        chunk_key = (doc_id, chunk_idx)
        
        # 1) snippet_text 중복 체크 - 동일한 내용이면 중복 (다른 문서/청크여도)
        if snippet in seen_by_snippet:
            # 동일한 스니펫이 이미 있으면 더 높은 점수로 교체
            existing = seen_by_snippet[snippet]
            if hit.get("score", 0.0) > existing.get("score", 0.0):
                # 기존 항목의 chunk_key도 제거
                old_doc_id = existing.get("doc_id", "")
                old_chunk_idx = int(existing.get("chunk_idx", 0))
                old_chunk_key = (old_doc_id, old_chunk_idx)
                if old_chunk_key in seen_by_chunk:
                    del seen_by_chunk[old_chunk_key]
                # 새 항목으로 교체
                seen_by_snippet[snippet] = hit
                seen_by_chunk[chunk_key] = hit
            continue  # 중복이므로 스킵
        
        # 2) (doc_id, chunk_idx) 중복 체크 - 같은 문서의 같은 청크는 하나만
        if chunk_key in seen_by_chunk:
            # 같은 (doc_id, chunk_idx)가 이미 있으면 더 높은 점수로 교체
            existing = seen_by_chunk[chunk_key]
            if hit.get("score", 0.0) > existing.get("score", 0.0):
                # 기존 항목의 snippet도 제거
                old_snippet = existing.get("snippet", "").strip()
                if old_snippet in seen_by_snippet and seen_by_snippet[old_snippet] == existing:
                    del seen_by_snippet[old_snippet]
                # 새 항목으로 교체
                seen_by_chunk[chunk_key] = hit
                seen_by_snippet[snippet] = hit
            continue  # 중복이므로 스킵
        
        # 새로운 항목 추가
        seen_by_snippet[snippet] = hit
        seen_by_chunk[chunk_key] = hit
    
    # 중복 제거된 결과를 점수 순으로 정렬하고 rerank_topN만큼만 반환
    deduplicated = sorted(seen_by_snippet.values(), key=lambda x: x.get("score", 0.0), reverse=True)
    hits_sorted = deduplicated[:final_results]
    
    logger.info(f"🔍 [Deduplication] 중복 제거 완료: {len(hits_sorted)}개 결과 (원본: {original_count}개, 제거: {original_count - len(hits_sorted)}개)")

    # 리랭크 결과 로그 출력
    if hits_sorted:
        top_hit = hits_sorted[0]
        logger.info(f"✨ [Rerank] 완료! 최고 점수: {top_hit.get('score', 0):.4f}")
        logger.info(f"🏆 [Rerank] 최고 스니펫 (doc_id: {top_hit.get('doc_id', 'unknown')}): {top_hit.get('snippet', '')[:100]}...")

    # 프롬프트 컨텍스트 생성
    context = "\n---\n".join(h["snippet"] for h in hits_sorted if h.get("snippet"))
    prompt = f"사용자 질의: {req.query}\n:\n{context}\n\n위 내용을 바탕으로 응답을 생성해 주세요."

    elapsed = round(time.perf_counter() - t0, 4)

    # query_logs 삭제: INSERT 제거
    return {
        "elapsed_sec": elapsed,
        "settings_used": {"model": model_key, "searchType": search_type},
        "hits": [
            {
                "score": float(h["score"]),
                "path": h["path"],
                "chunk_idx": int(h["chunk_idx"]),
                "task_type": h["task_type"],
                "security_level": int(h["security_level"]),
                "doc_id": h.get("doc_id"),
                "page": int(h.get("page", 0)),  # 페이지 정보 추가
                "snippet": h["snippet"],
            }
            for h in hits_sorted
        ],
        "prompt": prompt,
    }
    

async def execute_search(
    question: str,
    top_k: int = 20,   # 임베딩 후보 개수
    rerank_top_n: int = 5,    # 최종 반환 개수  
    security_level: int = 1,
    source_filter: Optional[List[str]] = None,
    task_type: str = "qna",
    model_key: Optional[str] = None,
    search_type: Optional[str] = None,
) -> Dict:
    print(f"⭐ [ExecuteSearch] 함수 호출: question='{question}', topK={top_k}, rerank_topN={rerank_top_n}")
    req = RAGSearchRequest(
        query=question,
        top_k=top_k,
        user_level=security_level,
        task_type=task_type,
        model=model_key,
    )
    logger.info(f"📞 [ExecuteSearch] search_documents 호출 전: req 생성 완료")
    res = await search_documents(req, search_type_override=search_type, rerank_top_n=rerank_top_n)
    logger.info(f"📞 [ExecuteSearch] search_documents 호출 후: 결과 hits 수={len(res.get('hits', []))}")
    # Build check_file BEFORE optional source_filter so it reflects original candidates
    check_files: List[str] = []
    logger.debug(f"\n###########################\nres: {res}")
    try:
        for h in res.get("hits", []):
            # Prefer doc_id when available; fallback to path-derived filename
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
# 4) 관리 유틸
# -------------------------------------------------
async def delete_db():
    # 모델 캐시 클리어
    _invalidate_embedder_cache()

    client = get_milvus_client()
    cols = drop_all_collections(client)
    return {"message": "삭제 완료(Milvus Server)", "dropped_collections": cols}

async def list_indexed_files(
    limit: int = 16384,
    offset: int = 0,
    query: Optional[str] = None,
    task_type: Optional[str] = None,
):
    limit = max(1, min(limit, 16384))
    client = get_milvus_client()
    if ADMIN_COLLECTION not in client.list_collections():
        return []

    doc_records = list_documents_by_type(ADMIN_DOC_TYPE)
    doc_map = {doc["doc_id"]: doc for doc in doc_records if doc.get("doc_id")}

    flt = ""
    if task_type and task_type in TASK_TYPES:
        flt = f"task_type == '{task_type}'"
    try:
        rows = client.query(
            collection_name=ADMIN_COLLECTION,
            filter=flt,
            output_fields=["doc_id", "path", "chunk_idx", "security_level", "task_type"],
            limit=limit,
            offset=offset,
            consistency_level="Strong",
        )
    except Exception:
        rows = []

    counts: Dict[Tuple[str, str], int] = defaultdict(int)
    level_map: Dict[Tuple[str, str], int] = {}
    path_map: Dict[Tuple[str, str], str] = {}
    doc_id_map: Dict[Tuple[str, str], Optional[str]] = {}
    for r in rows:
        path = r.get("path") if isinstance(r, dict) else r["path"]
        ttype = r.get("task_type") if isinstance(r, dict) else r["task_type"]
        lvl = int((r.get("security_level") if isinstance(r, dict) else r["security_level"]) or 1)
        doc_id_val = r.get("doc_id") if isinstance(r, dict) else r.get("doc_id")
        key_id = str(doc_id_val).strip() if doc_id_val else ""
        key = (key_id or path, ttype)
        counts[key] += 1
        level_map.setdefault(key, lvl)
        path_map.setdefault(key, path)
        doc_id_map.setdefault(key, key_id or None)

    items = []
    for key, cnt in counts.items():
        ttype = key[1]
        stored_path = path_map.get(key) or ""
        txt_rel = Path(stored_path)
        doc_id_val = doc_id_map.get(key)
        doc_meta = doc_map.get(doc_id_val or "")

        if doc_meta:
            file_name = doc_meta.get("filename") or Path(doc_meta.get("source_path") or stored_path).name
            file_path = doc_meta.get("source_path") or doc_meta.get("storage_path") or stored_path
            sec_levels = (doc_meta.get("payload") or {}).get("security_levels", {}) or {}
            sec_level = int(sec_levels.get(ttype, level_map.get(key, 1)))
        else:
            # fallback to path inference
            file_name = txt_rel.with_suffix(".pdf").name
            file_path = str(txt_rel.with_suffix(".pdf"))
            sec_level = int(level_map.get(key, 1))

        txt_abs = EXTRACTED_TEXT_DIR / txt_rel
        try:
            stat = txt_abs.stat()
            size = stat.st_size
            indexed_at = now_kst_string()
        except FileNotFoundError:
            size = None
            indexed_at = None
        items.append(
            {
                "taskType": ttype,
                "fileName": file_name,
                "filePath": file_path,
                "chunkCount": int(cnt),
                "indexedAt": indexed_at,
                "fileSize": size,
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
    if ADMIN_COLLECTION not in client.list_collections():
        deleted_sql = None
        if delete_workspace_documents_by_filenames:
            deleted_sql = delete_workspace_documents_by_filenames(file_names)
        return {"deleted": 0, "deleted_sql": deleted_sql, "requested": len(file_names)}

    # 로드 보장
    try:
        client.load_collection(collection_name=ADMIN_COLLECTION)
    except Exception:
        pass

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

        doc_ids_to_remove.add(doc_id_candidate)
        try:
            filt = f"doc_id == '{doc_id_candidate}'{task_filter}"
            client.delete(collection_name=ADMIN_COLLECTION, filter=filt)
            deleted_total += 1
            per_file[name] = per_file.get(name, 0) + 1
        except Exception:
            logger.exception("Failed to delete from Milvus for file: %s", name)
            per_file[name] = per_file.get(name, 0)

    # Ensure deletion is visible to subsequent queries (file lists/overview)
    try:
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


async def list_indexed_files_overview():
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
            storage_path=doc.get("storage_path") or "",
            source_path=doc.get("source_path"),
            security_level=_max_security_level(sec),
            payload=payload,
        )
        updated += 1
        target_tokens.append(doc_id)

    settings = get_vector_settings()
    model_key = settings.get("embeddingModel")

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