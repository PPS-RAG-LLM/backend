# === Vector DB Service (Milvus Server, Pro) ===
# - 작업유형(task_type)별 보안레벨 관리: doc_gen | summary | qna
# - Milvus Docker 서버 전용 (Lite 제거)
# - 벡터/하이브리드 검색 지원, 실행 로그 적재

from __future__ import annotations
import asyncio
import json
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
from utils.database import get_session
from storage.db_models import (
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
META_JSON_PATH = _cfg_path("meta_json_path", "storage/extracted_texts/_extraction_meta.json")
MODEL_ROOT_DIR = _cfg_path("model_root_dir", "storage/embedding-models")
RERANK_MODEL_PATH = _cfg_path("rerank_model_path", "storage/rerank_model/Qwen3-Reranker-0.6B")

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
from service.preprocessing.rag_preprocessing import ext, extract_any


# -------------------------------------------------
# 인제스트 파라미터 설정
# -------------------------------------------------
def set_ingest_params(chunk_size: int | None = None, overlap: int | None = None):
    # rag_settings 단일 소스로 저장
    set_vector_settings(chunk_size=chunk_size, overlap=overlap)


def get_ingest_params():
    row = get_rag_settings_row()
    return {"chunkSize": row["chunk_size"], "overlap": row["overlap"]}


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
    out = RAW_DATA_DIR / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(content)
    return str(out)


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
    chunk_size: int | None = None,
    overlap: int | None = None,
    target_tasks: list[str] | None = None,
    collection_name: str = ADMIN_COLLECTION,
    file_keys_filter: list[str] | None = None,  # ★ 추가: 특정 파일만 인제스트
):
    """
    META_JSON을 읽어 추출된 텍스트(.txt)들을 인제스트한다.
    - VARCHAR(32768 bytes) 초과 방지: split_for_varchar_bytes 로 안전 분할
    - 표는 [[TABLE ...]] 머리글 유지, 이어지는 조각은 [[TABLE_CONT i/n]] 마커로 연속성 표시
    - collection_name 파라미터를 끝까지 사용(기본/세션 컬렉션 공용)
    - file_keys_filter 가 주어지면 해당되는 파일(meta key/파일명/스텀)이 '포함'된 항목만 인제스트
    """
    # ==== 설정/모델 ====
    settings = get_vector_settings()
    MAX_TOKENS, OVERLAP = int(settings["chunkSize"]), int(settings["overlap"])

    if not META_JSON_PATH.exists():
        return {"error": "메타 JSON이 없습니다. 먼저 PDF/문서 추출을 수행하세요."}

    eff_model_key = model_key or settings["embeddingModel"]
    tok, model, device = await _get_or_load_embedder_async(eff_model_key)
    
    # 벡터 차원 검증
    probe_vec = hf_embed_text(tok, model, device, "probe")
    emb_dim = int(probe_vec.shape[0])
    logger.info(f"[Ingest] 임베딩 모델: {eff_model_key}, 벡터 차원: {emb_dim}")
    
    client = get_milvus_client()
    
    # 기존 컬렉션이 있으면 차원을 확인하고, 다르면 삭제
    if collection_name in client.list_collections():
        try:
            # 컬렉션 정보 확인
            desc = client.describe_collection(collection_name)
            existing_dim = None
            for field in desc.get("fields", []):
                if field.get("name") == "embedding":
                    existing_dim = field.get("params", {}).get("dim")
                    break
            
            if existing_dim and int(existing_dim) != emb_dim:
                logger.warning(f"[Ingest] 차원 불일치: 기존={existing_dim}, 새모델={emb_dim}. 컬렉션 재생성.")
                client.drop_collection(collection_name)
        except Exception as e:
            logger.warning(f"[Ingest] 컬렉션 정보 확인 실패: {e}. 재생성 시도.")
            try:
                client.drop_collection(collection_name)
            except Exception:
                pass
    
    ensure_collection_and_index(client, emb_dim=emb_dim, metric="IP", collection_name=collection_name)

    # ==== META 로드 및 대상 필터 구성 ====
    meta: dict = json.loads(META_JSON_PATH.read_text(encoding="utf-8"))
    tasks = [t for t in (target_tasks or TASK_TYPES) if t in TASK_TYPES]
    if not tasks:
        return {"error": f"유효한 작업유형이 없습니다. 허용: {TASK_TYPES}"}

    filter_tokens = set()
    if file_keys_filter:
        # meta key / 파일명 / 스템을 모두 매칭할 수 있도록 소문자 토큰화
        for f in file_keys_filter:
            p = Path(str(f))
            filter_tokens.add(str(f).lower())
            filter_tokens.add(p.name.lower())
            filter_tokens.add(p.stem.lower())

    total_inserted = 0
    BATCH_SIZE = 128

    # ==== 인제스트 ====
    # 주의: EXTRACTED_TEXT_DIR 안의 *.txt 를 돌면서, 해당 txt 가 어떤 meta key(원본 확장자)와 매칭되는지 찾는다.
    for txt_path in EXTRACTED_TEXT_DIR.rglob("*.txt"):
        rel_txt = txt_path.relative_to(EXTRACTED_TEXT_DIR)

        # 다양한 확장자 후보로 META key 찾기
        cands = [rel_txt.with_suffix(ext).as_posix() for ext in SUPPORTED_EXTS]
        meta_key = next((k for k in cands if k in meta), None)
        if not meta_key:
            continue

        # ★ 업로드한 것만 인제스트 옵션: meta key / 파일명 / 스템 기준 필터링
        if filter_tokens:
            p = Path(meta_key)
            if (meta_key.lower() not in filter_tokens and
                p.name.lower() not in filter_tokens and
                p.stem.lower() not in filter_tokens):
                continue

        entry = meta.get(meta_key) or {}
        sec_map = entry.get("security_levels", {}) or {}

        # doc_id / version 확보(없으면 파일명에서 유추)
        doc_id = entry.get("doc_id")
        version = int(entry.get("version", 0) or 0)
        if not doc_id or version == 0:
            _id, _ver = parse_doc_version(Path(meta_key).stem)
            doc_id = doc_id or _id
            version = version or _ver
            entry["doc_id"] = doc_id
            entry["version"] = version
            meta[meta_key] = entry  # 변경사항 반영

        # 기존 동일 문서/버전 삭제(작업유형 상관 없이)
        try:
            client.delete(
                collection_name=collection_name,
                filter=f"doc_id == '{doc_id}' && version <= {int(version)}",
            )
        except Exception:
            pass

        # 본문 텍스트 로드 및 청크화
        try:
            text = txt_path.read_text(encoding="utf-8")
        except Exception:
            # 혹시 모를 인코딩 문제 폴백
            text = txt_path.read_text(errors="ignore")
        
        # 통합 파일을 직접 파싱하여 페이지별로 분할 (텍스트와 표가 함께 저장된 파일)
        # 페이지 구분선 "---" 기준으로 페이지 분리
        def _parse_integrated_file(text: str) -> list[tuple[int, str]]:
            """통합 파일을 페이지별로 분할 (페이지 구분선 "---" 기준)"""
            page_blocks: list[tuple[int, str]] = []
            lines = text.split('\n')
            current_page = 1
            current_content = []
            
            for line in lines:
                # 페이지 구분선 확인: "---" (빈 줄로 둘러싸인 경우)
                if line.strip() == "---":
                    # 이전 페이지 저장
                    if current_content:
                        page_text = '\n'.join(current_content).strip()
                        if page_text:
                            page_blocks.append((current_page, page_text))
                    current_page += 1
                    current_content = []
                else:
                    current_content.append(line)
            
            # 마지막 페이지 저장
            if current_content:
                page_text = '\n'.join(current_content).strip()
                if page_text:
                    page_blocks.append((current_page, page_text))
            
            # 페이지 구분선이 없으면 전체를 1페이지로 처리
            if not page_blocks:
                if text.strip():
                    page_blocks = [(1, text.strip())]
            
            return page_blocks
        
        # 통합 파일 파싱
        page_blocks = _parse_integrated_file(text)
        logger.info(f"[Ingest] 통합 파일 파싱: {len(page_blocks)}개 페이지 블록 발견")
        
        # 전체 문서에서 청크 인덱스 누적 (페이지별로 0부터 시작하지 않도록)
        chunks_with_page: list[tuple[int, int, str]] = []  # (page, chunk_idx, chunk_text)
        global_chunk_idx = 0  # 전체 문서에서 누적되는 청크 인덱스
        
        for page_num, page_text in page_blocks:
            if not page_text:
                continue
            page_chunks = chunk_text(page_text, max_tokens=MAX_TOKENS, overlap=OVERLAP)
            for chunk in page_chunks:
                if chunk.strip():  # 빈 청크 제외
                    chunks_with_page.append((page_num, global_chunk_idx, chunk))
                    global_chunk_idx += 1
        
        logger.info(f"[Ingest] 총 {global_chunk_idx}개 청크 생성 (페이지별 청크 인덱스 누적)")
        
        # 표 블록 처리
        # 통합 파일에 이미 표가 포함되어 있으므로, 표를 별도로 인제스트하지 않음
        # (통합 파일을 파싱할 때 표도 함께 청크화되므로 중복 방지)
        tables = entry.get("tables", []) or []
        logger.info(f"[Ingest] 표 정보: {len(tables)}개 (통합 파일에 이미 포함되어 있으므로 별도 인제스트 안 함)")

        batch: list[dict] = []

        for task in tasks:
            lvl = int(sec_map.get(task, 1))

            # 1) 본문 조각 (페이지 정보 포함, 텍스트와 표 모두 포함)
            for page_num, idx, c in chunks_with_page:
                # VARCHAR 한도 안전 분할(바이트 기준)
                for part in split_for_varchar_bytes(c):
                    # 최종 방어(예외적으로 경계 잘림 실패 시)
                    if len(part.encode("utf-8")) > 32768:
                        part = part.encode("utf-8")[:32768].decode("utf-8", errors="ignore")

                    vec = hf_embed_text(tok, model, device, part, max_len=MAX_TOKENS)
                    
                    # 벡터 차원 검증
                    if len(vec) != emb_dim:
                        logger.error(f"[Ingest] 벡터 차원 불일치: 예상={emb_dim}, 실제={len(vec)}, 텍스트='{part[:50]}...'")
                        continue  # 이 벡터는 건너뛰기
                    
                    batch.append({
                        "embedding": vec.tolist(),
                        "path": str(rel_txt.as_posix()),
                        "chunk_idx": int(idx),
                        "task_type": task,
                        "security_level": lvl,
                        "doc_id": str(doc_id),
                        "version": int(version),
                        "page": int(page_num),  # 페이지 번호 추가
                        "workspace_id": 0,
                        "text": part,
                    })
                    if len(batch) >= BATCH_SIZE:                        
                        client.insert(collection_name=collection_name, data=batch)
                        total_inserted += len(batch)
                        batch = []

            # 2) 표 조각은 통합 파일에 이미 포함되어 있으므로 별도 인제스트하지 않음
            # (통합 파일을 파싱할 때 표도 함께 청크화되므로 중복 방지)

        if batch:
            client.insert(collection_name=collection_name, data=batch)
            total_inserted += len(batch)

    # 인덱스/로딩 재보장 및 메타 저장(유추된 doc_id/version 반영)
    try:
        client.flush(collection_name)
    except Exception:
        pass
    ensure_collection_and_index(client, emb_dim=emb_dim, metric="IP", collection_name=collection_name)

    # META에 doc_id/version 보정이 있었다면 저장
    try:
        META_JSON_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

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

    if ext(file_path) not in SUPPORTED_EXTS:
        return {"error": f"지원되지 않는 파일 형식입니다: {ext(file_path)}"}

    # 메타 로드
    if META_JSON_PATH.exists():
        meta = json.loads(META_JSON_PATH.read_text(encoding="utf-8"))
    else:
        meta = {}

    # 추출
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

    doc_id, ver = parse_doc_version(file_path.stem)
    meta[str(rel_file)] = {
        "chars": len(text_all),
        "lines": len(text_all.splitlines()),
        "preview": (clean_text(text_all[:200].replace("\n", " ")) + "…") if text_all else "",
        "security_levels": sec_map,
        "doc_id": doc_id,
        "version": ver,
        "tables": table_blocks_all or [],
        "sourceExt": ext(file_path),
    }
    META_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    META_JSON_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

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
    batch, cnt = [], 0

    for task in tasks:
        lvl = int(sec_map.get(task, 1))

        # 본문: VARCHAR 안전 분할
        for idx, c in enumerate(chunks):
            for part in split_for_varchar_bytes(c):
                if len(part.encode("utf-8")) > 32768:
                    part = part.encode("utf-8")[:32768].decode("utf-8", errors="ignore")
                vec = hf_embed_text(tok, model, device, part, max_len=max_token)
                batch.append({
                    "embedding": vec.tolist(),
                    "path": str(rel_file.with_suffix(".txt")),
                    "chunk_idx": int(idx),
                    "task_type": task,
                    "security_level": lvl,
                    "doc_id": str(doc_id),
                    "version": int(ver),
                    "workspace_id": 0,
                    "text": part,
                })
                if len(batch) >= 128:
                    client.insert(collection_name=ADMIN_COLLECTION, data=batch)
                    cnt += len(batch)
                    batch = []

        # 표: VARCHAR 안전 분할
        base_idx = len(chunks)
        for t_i, t in enumerate(table_blocks_all or []):
            md = (t.get("text") or "").strip()
            if not md:
                continue
            page = int(t.get("page", 0))
            bbox = t.get("bbox") or []
            bbox_str = ",".join(str(x) for x in bbox) if bbox else ""
            table_text = f"[[TABLE page={page} bbox={bbox_str}]]\n{md}"

            for sub_j, part in enumerate(split_for_varchar_bytes(table_text)):
                if len(part.encode("utf-8")) > 32768:
                    part = part.encode("utf-8")[:32768].decode("utf-8", errors="ignore")
                vec = hf_embed_text(tok, model, device, part, max_len=max_token)
                batch.append({
                    "embedding": vec.tolist(),
                    "path": str(rel_file.with_suffix(".txt")),
                    "chunk_idx": int(base_idx + t_i * 1000 + sub_j),
                    "task_type": task,
                    "security_level": lvl,
                    "doc_id": str(doc_id),
                    "version": int(ver),
                    "workspace_id": 0,
                    "text": part,
                })
                if len(batch) >= 128:
                    client.insert(collection_name=ADMIN_COLLECTION, data=batch)
                    cnt += len(batch)
                    batch = []

    if batch:
        client.insert(collection_name=ADMIN_COLLECTION, data=batch)
        cnt += len(batch)

    try:
        client.flush(ADMIN_COLLECTION)
    except Exception:
        pass
    ensure_collection_and_index(client, emb_dim=emb_dim, metric="IP", collection_name=ADMIN_COLLECTION)

    return {
        "message": f"단일 파일 인제스트 완료(Milvus Server) - {ext(file_path)}",
        "doc_id": doc_id,
        "version": ver,
        "chunks": cnt,
        "sourceExt": ext(file_path),
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
    eff_model_key = settings["embeddingModel"]
    tok, model, device = await _get_or_load_embedder_async(eff_model_key)
    emb_dim = int(hf_embed_text(tok, model, device, "probe").shape[0])

    coll = collection_name or ADMIN_COLLECTION
    client = get_milvus_client()
    ensure_collection_and_index(client, emb_dim=emb_dim, metric="IP", collection_name=coll)

    MAX_TOKENS, OVERLAP = int(settings["chunkSize"]), int(settings["overlap"])

    processed, total = [], 0
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
            doc_id, ver = parse_doc_version(src.stem)

            # 기존 삭제
            try:
                client.delete(collection_name=coll, filter=f"doc_id == '{doc_id}' && version <= {int(ver)}")
            except Exception:
                pass

            # 본문
            chunks = chunk_text(text, max_tokens=MAX_TOKENS, overlap=OVERLAP)
            batch, cnt = [], 0
            for t in tasks_eff:
                lvl = int(sec_map.get(t, 1))

                for idx, c in enumerate(chunks):
                    for part in split_for_varchar_bytes(c):
                        if len(part.encode("utf-8")) > 32768:
                            part = part.encode("utf-8")[:32768].decode("utf-8", errors="ignore")
                        vec = hf_embed_text(tok, model, device, part, max_len=MAX_TOKENS)
                        batch.append({
                            "embedding": vec.tolist(),
                            "path": str(rel_txt.as_posix()),
                            "chunk_idx": int(idx),
                            "task_type": t,
                            "security_level": lvl,
                            "doc_id": str(doc_id),
                            "version": int(ver),
                            "workspace_id": 0,
                            "text": part,
                        })
                        if len(batch) >= 128:
                            client.insert(collection_name=coll, data=batch); cnt += len(batch); batch = []

                # 표
                base_idx = len(chunks)
                for t_i, tb in enumerate(tables or []):
                    md = (tb.get("text") or "").strip()
                    if not md:
                        continue
                    page = int(tb.get("page", 0)); bbox = tb.get("bbox") or []
                    bbox_str = ",".join(str(x) for x in bbox) if bbox else ""
                    table_text = f"[[TABLE page={page} bbox={bbox_str}]]\n{md}"
                    for sub_j, part in enumerate(split_for_varchar_bytes(table_text)):
                        if len(part.encode("utf-8")) > 32768:
                            part = part.encode("utf-8")[:32768].decode("utf-8", errors="ignore")
                        vec = hf_embed_text(tok, model, device, part, max_len=MAX_TOKENS)
                        batch.append({
                            "embedding": vec.tolist(),
                            "path": str(rel_txt.as_posix()),
                            "chunk_idx": int(base_idx + t_i * 1000 + sub_j),
                            "task_type": t,
                            "security_level": lvl,
                            "doc_id": str(doc_id),
                            "version": int(ver),
                            "workspace_id": 0,
                            "text": part,
                        })
                        if len(batch) >= 128:
                            client.insert(collection_name=coll, data=batch); cnt += len(batch); batch = []

            if batch:
                client.insert(collection_name=coll, data=batch); cnt += len(batch); batch = []

            processed.append({
                "file": src.name, "doc_id": doc_id, "version": int(ver),
                "levels": sec_map, "chunks": cnt
            })
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

async def search_documents(req: RAGSearchRequest, search_type_override: Optional[str] = None,
                           collection_name: str = ADMIN_COLLECTION, rerank_top_n: Optional[int] = None) -> Dict:
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
        # hits_raw = build_rrf_hits(
        #     res_dense,
        #     res_sparse,
        #     snippet_loader=snippet_loader,
        #     limit=candidate,
        # )

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

    # 메타 로드(원본 확장자 복원용)
    try:
        meta = json.loads(META_JSON_PATH.read_text(encoding="utf-8"))
    except Exception:
        meta = {}

    flt = ""
    if task_type and task_type in TASK_TYPES:
        flt = f"task_type == '{task_type}'"
    try:
        rows = client.query(
            collection_name=ADMIN_COLLECTION,
            filter=flt,
            output_fields=["path", "chunk_idx", "security_level", "task_type"],
            limit=limit,
            offset=offset,
            consistency_level="Strong",
        )
    except Exception:
        rows = []

    counts: Dict[Tuple[str, str], int] = defaultdict(int)
    level_map: Dict[Tuple[str, str], int] = {}
    for r in rows:
        path = r.get("path") if isinstance(r, dict) else r["path"]
        ttype = r.get("task_type") if isinstance(r, dict) else r["task_type"]
        lvl = int((r.get("security_level") if isinstance(r, dict) else r["security_level"]) or 1)
        key = (path, ttype)
        counts[key] += 1
        level_map.setdefault(key, lvl)

    items = []
    for (path, ttype), cnt in counts.items():
        txt_rel = Path(path)

        # 메타에서 원래 확장자를 복원
        cands = [txt_rel.with_suffix(ext).as_posix() for ext in SUPPORTED_EXTS]
        meta_key = next((k for k in cands if k in meta), None)
        if meta_key:
            source_ext = meta.get(meta_key, {}).get("sourceExt") or Path(meta_key).suffix
            orig_rel = txt_rel.with_suffix(source_ext)
        else:
            # 폴백(구버전 데이터): pdf 가정
            orig_rel = txt_rel.with_suffix(".pdf")

        file_name = orig_rel.name
        file_path = str(orig_rel)

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
                "securityLevel": int(level_map.get((path, ttype), 1)),
            }
        )

    if query:
        q = str(query)
        items = [it for it in items if q in it["fileName"]]
    return items

async def delete_files_by_names(file_names: List[str], task_type: Optional[str] = None, collection_name: str = ADMIN_COLLECTION):
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

    for name in file_names:
        stem = Path(name).stem
        # Align fileName -> doc_id by stripping version suffix if present
        try:
            base_id, _ver = parse_doc_version(stem)
        except Exception:
            base_id = stem
        try:
            # doc_id == 'stem' [&& task_type == 'xxx']
            filt = f"doc_id == '{base_id}'{task_filter}"
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

    return {
        "deleted": deleted_total,  # 요청 파일 기준 성공 건수(작업유형 기준 단순 카운트)
        "deleted_sql": deleted_sql,
        "requested": len(file_names),
        "taskType": task_type,
        "perFile": per_file,  # 파일별 처리현황
    }


async def list_indexed_files_overview(collection_name: str = ADMIN_COLLECTION):
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
    if not META_JSON_PATH.exists():
        return {"error": "메타 JSON이 없습니다. 먼저 /v1/admin/vector/extract 를 수행하세요."}

    target_tasks = [t for t in (req.tasks or TASK_TYPES) if t in TASK_TYPES]
    if not target_tasks:
        return {"error": "유효한 작업유형이 없습니다. (허용: doc_gen|summary|qna)"}

    level_map = {t: int(max(1, lv)) for t, lv in (req.level_for_tasks or {}).items() if t in TASK_TYPES}
    if not level_map:
        return {"error": "적용할 보안레벨이 없습니다. level_for_tasks 를 지정하세요."}

    meta = json.loads(META_JSON_PATH.read_text(encoding="utf-8"))

    # 대상 파일 셋(메타키/파일명/스텀 모두 허용)
    def _to_keyset(files: List[str]) -> set:
        out = set()
        for f in files:
            p = Path(f)
            out.update({str(f), p.name, p.stem})
        return out

    all_keys = list(meta.keys())  # "securityLevelX/.../파일명.확장자"
    if req.files:
        ks = _to_keyset(req.files)
        targets = [k for k in all_keys if (k in ks or Path(k).name in ks or Path(k).stem in ks)]
    else:
        targets = all_keys

    if not targets:
        return {"updated": 0, "ingested": 0, "message": "대상 파일이 없습니다."}

    updated = 0
    for k in targets:
        entry = meta.get(k) or {}
        sec = entry.get("security_levels") or {}
        for t in target_tasks:
            if t in level_map:
                sec[t] = int(level_map[t])
        entry["security_levels"] = sec
        meta[k] = entry
        updated += 1

    META_JSON_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # ★ 업로드한(또는 지정한) 파일만 인제스트
    res = await ingest_embeddings(
        model_key=None,
        chunk_size=None,
        overlap=None,
        target_tasks=target_tasks,
        collection_name=ADMIN_COLLECTION,
        file_keys_filter=targets,
    )
    return {
        "message": "레벨 오버라이드 후 인제스트 완료",
        "collection": ADMIN_COLLECTION,
        "updated_meta_entries": updated,
        "inserted_chunks": int(res.get("inserted_chunks", 0)),
        "target_count": len(targets),
    }