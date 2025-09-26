# === Vector DB Service (Milvus Server, Pro) ===
# - 작업유형(task_type)별 보안레벨 관리: doc_gen | summary | qna
# - Milvus Docker 서버 전용 (Lite 제거)
# - 벡터/하이브리드 검색 지원, 실행 로그 적재

from __future__ import annotations

import json
import os
import time
import logging

# sqlite3 제거
import shutil
import threading
import asyncio
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from collections import defaultdict, Counter

import torch
from pydantic import BaseModel, Field
from pymilvus import MilvusClient, DataType
try:
    # Milvus 2.4+ Function/BM25 하이브리드
    from pymilvus import Function, FunctionType
except Exception:
    Function = None
    class FunctionType:
        BM25 = "BM25"
from transformers import AutoModel, AutoTokenizer

# ORM 추가 임포트
from utils.database import get_session
from storage.db_models import (
    EmbeddingModel,
    RagSettings,
    SecurityLevelConfigTask,
    SecurityLevelKeywordsTask,
)

# KST 시간 포맷 유틸
from utils.time import now_kst, now_kst_string

logger = logging.getLogger(__name__)

# -------------------------------------------------
# 경로 상수
# -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent  # .../backend/service/admin
PROJECT_ROOT = BASE_DIR.parent.parent  # .../backend
STORAGE_DIR = PROJECT_ROOT / "storage"
USER_DATA_ROOT = STORAGE_DIR / "user_data"
RAW_DATA_DIR = USER_DATA_ROOT / "row_data"
LOCAL_DATA_ROOT = USER_DATA_ROOT / "preprocessed_data"  # 유지(폴더 구조 호환)
RESOURCE_DIR = (BASE_DIR / "resources").resolve()
EXTRACTED_TEXT_DIR = RESOURCE_DIR / "extracted_texts"
META_JSON_PATH = EXTRACTED_TEXT_DIR / "_extraction_meta.json"
MODEL_ROOT_DIR = (PROJECT_ROOT / "storage" / "embedding-models").resolve()

SQLITE_DB_PATH = (PROJECT_ROOT / "storage" / "pps_rag.db").resolve()

# Milvus Server 접속 정보 (환경변수로 오버라이드 가능)
MILVUS_URI = os.getenv("MILVUS_URI", "http://biz.ppsystem.co.kr:3006")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", None)  # 예: "root:Milvus" (인증 사용 시)
COLLECTION_NAME = "pdf_chunks_pro"

# 작업유형
TASK_TYPES = ("doc_gen", "summary", "qna")

_CURRENT_EMBED_MODEL_KEY = "qwen3_0_6b"
_CURRENT_SEARCH_TYPE = "hybrid"
_CURRENT_CHUNK_SIZE = 512
_CURRENT_OVERLAP = 64


# -------------------------------------------------
# 인제스트 파라미터 설정
# -------------------------------------------------
def set_ingest_params(chunk_size: int | None = None, overlap: int | None = None):
    # rag_settings 단일 소스로 저장
    set_vector_settings(chunk_size=chunk_size, overlap=overlap)


def get_ingest_params():
    row = _get_vector_settings_row()
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


# === Embedding cache(singleton) ===
_EMBED_CACHE: dict[str, tuple[any, any, any]] = {}  # key -> (tok, model, device)
_EMBED_ACTIVE_KEY: Optional[str] = None
_EMBED_LOCK = threading.Lock()


def _invalidate_embedder_cache():
    global _EMBED_CACHE, _EMBED_ACTIVE_KEY
    with _EMBED_LOCK:
        _EMBED_CACHE.clear()
        _EMBED_ACTIVE_KEY = None


def _get_or_load_embedder(model_key: str, preload: bool = False):
    """
    전역 캐시에서 (tok, model, device) 반환.
    - 캐시에 없으면 로드해서 저장(지연 로딩)
    - preload=True는 의미상 웜업 호출일 뿐, 반환 동작은 동일
    """
    global _EMBED_CACHE, _EMBED_ACTIVE_KEY
    if not model_key:
        raise ValueError(
            "활성화된 임베딩 모델이 없습니다. 먼저 /v1/admin/vector/settings에서 모델을 설정하세요."
        )

    with _EMBED_LOCK:
        if _EMBED_ACTIVE_KEY == model_key and model_key in _EMBED_CACHE:
            return _EMBED_CACHE[model_key]
        # 키가 바뀌면 캐시 전체 무효화(동시 2개 방지)
        _EMBED_CACHE.clear()
        tok, model, device = _load_embedder(model_key)
        _EMBED_CACHE[model_key] = (tok, model, device)
        _EMBED_ACTIVE_KEY = model_key
        return _EMBED_CACHE[model_key]


def warmup_active_embedder(logger_func=print):
    """
    서버 기동 시 호출용(선택). 활성 모델 키를 조회해 캐시를 채움.
    실패해도 서비스는 실제 사용 시 지연 로딩으로 복구됨.
    """
    try:
        key = _get_active_embedding_model_name()
        logger_func(f"[warmup] 활성 임베딩 모델: {key}. 로딩 시도...")
        _get_or_load_embedder(key, preload=True)
        logger_func(f"[warmup] 로딩 완료: {key}")
    except Exception as e:
        logger_func(f"[warmup] 로딩 실패(지연 로딩으로 복구 예정): {e}")


async def _get_or_load_embedder_async(model_key: str, preload: bool = False):
    """
    비동기 래퍼: blocking 함수(_get_or_load_embedder)를 스레드풀에서 실행
    이벤트 루프 블로킹 방지
    """
    loop = asyncio.get_running_loop()
    # blocking 함수(_get_or_load_embedder)를 스레드풀에서 실행
    return await loop.run_in_executor(None, _get_or_load_embedder, model_key, preload)


def _get_active_embedding_model_name() -> str:
    """활성화된 임베딩 모델 이름 반환 (없으면 예외)"""
    with get_session() as session:
        row = (
            session.query(EmbeddingModel)
            .filter(EmbeddingModel.is_active == 1)
            .order_by(EmbeddingModel.activated_at.desc().nullslast())
            .first()
        )
        if not row:
            raise ValueError(
                "활성화된 임베딩 모델이 없습니다. 먼저 /v1/admin/vector/settings에서 모델을 설정하세요."
            )
        return str(row.name)


def _set_active_embedding_model(name: str):
    with get_session() as session:
        # 존재하지 않으면 생성
        model = (
            session.query(EmbeddingModel).filter(EmbeddingModel.name == name).first()
        )
        if not model:
            model = EmbeddingModel(name=name, is_active=0)
            session.add(model)
            session.flush()
        # 모두 비활성 → 대상만 활성
        session.query(EmbeddingModel).filter(EmbeddingModel.is_active == 1).update(
            {"is_active": 0, "activated_at": None}
        )
        model.is_active = 1
        model.activated_at = now_kst()
        session.commit()


def _get_vector_settings_row() -> dict:
    """레거시 호환: rag_settings(싱글톤)에서 기본 청크 설정을 읽어온다."""
    with get_session() as session:
        row = session.query(RagSettings).filter(RagSettings.id == 1).first()
        if not row:
            return {"search_type": "hybrid", "chunk_size": 512, "overlap": 64}
        return {
            "search_type": str(row.search_type or "hybrid"),
            "chunk_size": int(row.chunk_size or 512),
            "overlap": int(row.overlap or 64),
        }


def _get_rag_settings_row() -> dict:
    """RAG 전역 설정 로더. 없으면 빈 dict."""
    with get_session() as session:
        row = session.query(RagSettings).filter(RagSettings.id == 1).first()
        if not row:
            return {}
        return {
            "embedding_key": row.embedding_key,
            "search_type": row.search_type,
            "chunk_size": int(row.chunk_size or 512),
            "overlap": int(row.overlap or 64),
        }


def _update_vector_settings(
    search_type: Optional[str] = None,
    chunk_size: Optional[int] = None,
    overlap: Optional[int] = None,
):
    """레거시 API 호환: rag_settings(싱글톤) 업데이트"""
    cur = _get_vector_settings_row()
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


def _milvus_has_data() -> bool:
    client = _client()
    if COLLECTION_NAME not in client.list_collections():
        return False
    try:
        rows = client.query(
            collection_name=COLLECTION_NAME, output_fields=["pk"], limit=1
        )
        return len(rows) > 0
    except Exception:
        # 인덱스/로드 전이라면 컬렉션 있게만 체크
        return True


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
        if _milvus_has_data():
            raise RuntimeError("Milvus 컬렉션에 기존 데이터가 남아있습니다. 먼저 /v1/admin/vector/delete-all 을 호출해 초기화하세요.")
        _set_active_embedding_model(embed_model_key)
        _invalidate_embedder_cache()

    with get_session() as session:
        s = session.query(RagSettings).filter(RagSettings.id == 1).first()
        if not s:
            s = RagSettings(id=1)
            session.add(s)
        s.embedding_key = new_key  # BUGFIX: 'key' -> 'new_key'
        # search_type/chunk/overlap은 _update_vector_settings에서 반영됨. 여기선 존재 시 보존
        if search_type is not None:
            s.search_type = new_st.replace("vector", "semantic")  # 정규화된 값 사용
        if chunk_size is not None:
            s.chunk_size = int(new_cs)
        if overlap is not None:
            s.overlap = int(new_ov)
        s.updated_at = now_kst()
        session.commit()

    return get_vector_settings()


def get_vector_settings() -> Dict:
    rag = _get_rag_settings_row()
    if not rag:
        # 안전 기본값
        try:
            model = _get_active_embedding_model_name()
        except ValueError:
            model = None
        return {
            "embeddingModel": model,
            "searchType": "hybrid",
            "chunkSize": 512,
            "overlap": 64,
        }
    # Fallback (기본값)
    row = _get_vector_settings_row()
    try:
        model = _get_active_embedding_model_name()
    except Exception:
        model = None
    return {
        "embeddingModel": rag.get("embedding_key"),
        "searchType": rag.get("search_type"),
        "chunkSize": rag.get("chunk_size"),
        "overlap": rag.get("overlap"),
    }


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


def _determine_level_for_task(text: str, task_rules: Dict) -> int:
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
# 모델 로딩/임베딩
# -------------------------------------------------
def _resolve_model_input(model_key: Optional[str]) -> Tuple[str, Path]:
    key = (model_key or "bge").lower()
    cands: List[Path] = []
    if MODEL_ROOT_DIR.exists():
        for p in MODEL_ROOT_DIR.iterdir():
            if p.is_dir():
                cands.append(p.resolve())

    def aliases(p: Path) -> List[str]:
        nm = p.name.lower()
        res = [nm]
        if nm.startswith("embedding_"):
            res.append(nm[len("embedding_") :])
        return res

    # 우선 exact/alias
    for p in cands:
        if key in aliases(p):
            return p.name, p
    # 부분일치
    for p in cands:
        if key in p.name.lower():
            return p.name, p
    # fallback: qwen3_0_6b 계열
    for p in cands:
        if "qwen3_0_6b" in p.name.lower():
            return p.name, p
    fb = MODEL_ROOT_DIR / "qwen3_0_6b"
    return fb.name, fb


def _load_embedder(model_key: Optional[str]) -> Tuple[any, any, any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, model_dir = _resolve_model_input(model_key)
    need_files = [
        model_dir / "tokenizer_config.json",
        model_dir / "tokenizer.json",
        model_dir / "config.json",
    ]

    # 모델 파일 누락 빠른 실패
    missing_files = [f for f in need_files if not f.exists()]
    if missing_files:
        logger.error(f"[Embedding Model] 필수 파일 누락: {model_dir}")
        logger.error(
            f"[Embedding Model] 누락된 파일들: {[str(f) for f in missing_files]}"
        )
        raise FileNotFoundError(f"[Embedding Model] 필수 파일 누락: {model_dir}")

    logger.info(f"[Embedding Model] 모델 로딩 시작: {model_key} from {model_dir}")
    tok = AutoTokenizer.from_pretrained(
        str(model_dir), trust_remote_code=True, local_files_only=True
    )
    model = (
        AutoModel.from_pretrained(
            str(model_dir),
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        .to(device)
        .eval()
    )
    logger.info(f"[Embedding Model] 모델 로딩 완료: {model_key}")
    return tok, model, device


def _mean_pooling(outputs, mask):
    token_embeddings = outputs.last_hidden_state
    mask_expanded = mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = torch.sum(token_embeddings * mask_expanded, dim=1)
    counts = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
    return summed / counts


def _embed_text(tok, model, device, text: str, max_len: int = 512):
    inputs = tok(
        text,
        truncation=True,
        padding="longest",
        max_length=max_len,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        outs = model(**inputs)
    vec = (
        _mean_pooling(outs, inputs["attention_mask"]).cpu().numpy()[0].astype("float32")
    )
    return vec


# -------------------------------------------------
# Milvus Client / 컬렉션 스키마
# -------------------------------------------------
def _client() -> MilvusClient:
    kwargs = {"uri": MILVUS_URI}
    if MILVUS_TOKEN:
        kwargs["token"] = MILVUS_TOKEN
    return MilvusClient(**kwargs)


def _ensure_collection_and_index(
    client: MilvusClient, emb_dim: int, metric: str = "IP"
):
    logger.info(f"[Milvus] 컬렉션 및 인덱스 준비 시작: {COLLECTION_NAME}")
    cols = client.list_collections()
    if COLLECTION_NAME not in cols:
        logger.info(f"[Milvus] 컬렉션 생성: {COLLECTION_NAME}")
        schema = client.create_schema(
            auto_id=True, enable_dynamic_field=False, description="PDF chunks (pro)"
        )
        schema.add_field("pk", DataType.INT64, is_primary=True)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=int(emb_dim))
        schema.add_field("path", DataType.VARCHAR, max_length=500)
        schema.add_field("chunk_idx", DataType.INT64)
        schema.add_field(
            "task_type", DataType.VARCHAR, max_length=16
        )  # 'doc_gen'|'summary'|'qna'
        schema.add_field("security_level", DataType.INT64)
        schema.add_field("doc_id", DataType.VARCHAR, max_length=255)
        schema.add_field("version", DataType.INT64)
        # 하이브리드용 텍스트/스파스 필드
        # text: 본문 청크(분석기 활성), text_sparse: BM25 스파스 벡터
        try:
            schema.add_field("text", DataType.VARCHAR, max_length=32768, enable_analyzer=True)
        except TypeError:
            schema.add_field("text", DataType.VARCHAR, max_length=32768)
        try:
            schema.add_field("text_sparse", DataType.SPARSE_FLOAT_VECTOR)
        except Exception:
            logger.warning("[Milvus] SPARSE_FLOAT_VECTOR 미지원 클라이언트입니다. 서버 BM25 하이브리드 사용 불가.")

        # BM25 함수(가능한 경우: text -> text_sparse 자동생성)
        if Function is not None:
            try:
                fn = Function(
                    name="bm25_text2sparse",
                    function_type=FunctionType.BM25,
                    input_field_names=["text"],
                    output_field_names=["text_sparse"],
                )
                schema.add_function(fn)
                logger.info("[Milvus] BM25 Function 연결 완료 (text -> text_sparse)")
            except Exception as e:
                logger.warning(f"[Milvus] BM25 Function 추가 실패: {e}")
        client.create_collection(collection_name=COLLECTION_NAME, schema=schema)
        logger.info(f"[Milvus] 컬렉션 생성 완료: {COLLECTION_NAME}")

    # 1) 덴스 벡터 인덱스(embedding)
    try:
        idx_dense = client.list_indexes(collection_name=COLLECTION_NAME, field_name="embedding")
    except Exception:
        idx_dense = []
    if not idx_dense:
        logger.info(f"[Milvus] (embedding) 인덱스 생성 시작")
        ip = client.prepare_index_params()
        # 덴스 벡터: 기본은 FLAT 유지(환경에 따라 HNSW/IVF 등으로 변경 가능)
        ip.add_index("embedding", "FLAT", metric_type=metric, params={})
        client.create_index(COLLECTION_NAME, ip, timeout=180.0, sync=True)
        logger.info(f"[Milvus] (embedding) 인덱스 생성 완료")

    # 2) 스파스 벡터 인덱스(text_sparse)
    try:
        idx_sparse = client.list_indexes(collection_name=COLLECTION_NAME, field_name="text_sparse")
    except Exception:
        idx_sparse = []
    if not idx_sparse:
        logger.info(f"[Milvus] (text_sparse) 인덱스 생성 시작")
        ip2 = client.prepare_index_params()
        try:
            # 최신 PyMilvus: metric_type 없이도 동작
            ip2.add_index("text_sparse", "SPARSE_INVERTED_INDEX", params={})
        except TypeError:
            # 일부 버전은 metric_type이 필요할 수 있음
            ip2.add_index("text_sparse", "SPARSE_INVERTED_INDEX", metric_type="BM25", params={})
        client.create_index(COLLECTION_NAME, ip2, timeout=180.0, sync=True)
        logger.info(f"[Milvus] (text_sparse) 인덱스 생성 완료")

    # 인덱스 준비 후 로드(이미 로드되어 있으면 내렸다가 다시 올림)
    try:
        client.release_collection(collection_name=COLLECTION_NAME)
    except Exception:
        pass
    client.load_collection(collection_name=COLLECTION_NAME)
    logger.info(f"[Milvus] 컬렉션 로드 완료: {COLLECTION_NAME}")
    
    logger.info(f"[Milvus] 컬렉션 및 인덱스 준비 완료: {COLLECTION_NAME}")


# -------------------------------------------------
# 1) PDF → 텍스트 추출 (작업유형별 보안레벨 동시 산정)
# -------------------------------------------------
async def extract_pdfs():
    import fitz  # type: ignore
    from tqdm import tqdm  # type: ignore

    EXTRACTED_TEXT_DIR.mkdir(parents=True, exist_ok=True)
    LOCAL_DATA_ROOT.mkdir(parents=True, exist_ok=True)
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 규칙 로드
    all_rules = get_security_level_rules_all()  # {task: {"maxLevel":N, "levels":{...}}}

    # 이전 메타 로드
    prev_meta: Dict[str, Dict] = {}
    if META_JSON_PATH.exists():
        try:
            prev_meta = json.loads(META_JSON_PATH.read_text(encoding="utf-8"))
        except Exception:
            logger.exception("Failed to read META JSON; recreating.")

    # 중복 제거 로직: 파일명 마지막 토큰이 날짜/버전 숫자면 최신만 유지
    def _extract_base_and_date(p: Path):
        name = p.stem
        parts = name.split("_")
        date_num = 0
        if len(parts) >= 2:
            cand = parts[-1]
            if cand.isdigit() and len(cand) in (4, 6, 8):
                try:
                    date_num = int(cand)
                except Exception:
                    date_num = 0
        mid_tokens = [t for t in parts[:-1] if t and not t.isdigit()]
        base = max(mid_tokens, key=len) if mid_tokens else parts[0]
        return base, date_num

    pdf_paths = list(RAW_DATA_DIR.rglob("*.pdf"))
    grouped: Dict[str, List[Tuple[Path, int]]] = defaultdict(list)
    for p in pdf_paths:
        base, date_num = _extract_base_and_date(p)
        grouped[base].append((p, date_num))

    kept, removed = [], []
    for base, lst in grouped.items():
        lst_sorted = sorted(lst, key=lambda x: (x[1], len(x[0].name)))
        keep = lst_sorted[-1]
        kept.append(keep[0])
        for old in [p for p, d in lst_sorted[:-1]]:
            try:
                old.unlink(missing_ok=True)
                removed.append(str(old.relative_to(RAW_DATA_DIR)))
            except Exception:
                logger.exception("Failed to remove duplicate: %s", old)

    if not kept:
        return {
            "message": "처리할 PDF가 없습니다.",
            "meta_path": str(META_JSON_PATH),
            "deduplicated": {"removedCount": len(removed), "removed": removed},
        }

    new_meta: Dict[str, Dict] = {}
    for pdf_path in tqdm(kept, desc="PDF 전처리"):
        try:
            doc = fitz.open(pdf_path)
            text_pages = [p.get_text("text").strip() for p in doc]
            pdf_text = "\n\n".join(text_pages)

            # 작업유형별 보안레벨 계산
            sec_map: Dict[str, int] = {}
            for task in TASK_TYPES:
                rules = all_rules.get(task, {"maxLevel": 1, "levels": {}})
                sec_map[task] = _determine_level_for_task(pdf_text, rules)

            # 폴더 배치: 3종 중 최대 레벨 폴더로 복사(기존 구조 유지)
            max_sec = max(sec_map.values()) if sec_map else 1
            sec_folder = f"securityLevel{int(max_sec)}"

            rel_from_raw = pdf_path.relative_to(RAW_DATA_DIR)
            dest_rel_pdf = Path(sec_folder) / rel_from_raw
            dest_pdf_abs = LOCAL_DATA_ROOT / dest_rel_pdf
            dest_pdf_abs.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(pdf_path, dest_pdf_abs)
            except Exception:
                logger.exception("Failed to copy PDF: %s", dest_pdf_abs)

            # 텍스트 저장 (extracted_texts/securityLevelN/...)
            txt_path = EXTRACTED_TEXT_DIR / dest_rel_pdf.with_suffix(".txt")
            txt_path.parent.mkdir(parents=True, exist_ok=True)
            txt_path.write_text(pdf_text, encoding="utf-8")

            # doc_id/version 유추
            stem = rel_from_raw.stem
            doc_id, version = _parse_doc_version(stem)

            info = {
                "chars": len(pdf_text),
                "lines": len(pdf_text.splitlines()),
                "preview": (
                    (pdf_text[:200].replace("\n", " ") + "…") if pdf_text else ""
                ),
                "security_levels": sec_map,  # 작업유형별 보안레벨
                "doc_id": doc_id,
                "version": version,
            }
            new_meta[str(dest_rel_pdf)] = info

        except Exception as e:
            logger.exception("Failed to process: %s", pdf_path)
            try:
                rel_from_raw = pdf_path.relative_to(RAW_DATA_DIR)
                dest_rel_pdf = Path("securityLevel1") / rel_from_raw
                new_meta[str(dest_rel_pdf)] = {"error": str(e)}
            except Exception:
                new_meta[pdf_path.name] = {"error": str(e)}

    META_JSON_PATH.write_text(
        json.dumps(new_meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "message": "PDF 추출 완료",
        "pdf_count": len(kept),
        "meta_path": str(META_JSON_PATH),
        "deduplicated": {"removedCount": len(removed), "removed": removed},
    }


def _parse_doc_version(stem: str) -> Tuple[str, int]:
    if "_" in stem:
        base, cand = stem.rsplit("_", 1)
        if cand.isdigit() and len(cand) in (4, 8):
            return base, int(cand)
    return stem, 0


# -------------------------------------------------
# 2) 인제스트 (bulk)
#   - 작업유형별로 동일 청크를 각각 저장(task_type, security_level 분리)
# -------------------------------------------------
async def ingest_embeddings(model_key: str | None = None, chunk_size: int | None = None, overlap: int | None = None, target_tasks: list[str] | None = None):
    # rag_settings 단일 소스
    s = get_vector_settings()
    MAX_TOKENS = int(s["chunkSize"])
    OVERLAP = int(s["overlap"])

    if not META_JSON_PATH.exists():
        return {"error": "메타 JSON이 없습니다. 먼저 PDF 추출을 수행하세요."}

    # 모델/검색 설정 로드(모델키 우선순위: 인자 > settings)
    settings = get_vector_settings()
    eff_model_key = model_key or settings["embeddingModel"]

    tok, model, device = await _get_or_load_embedder_async(eff_model_key)
    emb_dim = int(_embed_text(tok, model, device, "probe").shape[0])

    client = _client()
    _ensure_collection_and_index(client, emb_dim, metric="IP")

    def chunk_text(text: str, max_tokens: int = MAX_TOKENS, overlap: int = OVERLAP):
        words = text.split()
        chunks: List[str] = []
        start = 0
        while start < len(words):
            end = min(start + max_tokens, len(words))
            chunk = " ".join(words[start:end]).strip()
            if chunk:
                chunks.append(chunk)
            start += max_tokens - overlap
        return chunks

    meta = json.loads(META_JSON_PATH.read_text(encoding="utf-8"))
    # 모든 TASK_TYPES 대상으로 고정
    tasks = list(TASK_TYPES)

    total_inserted = 0
    for txt_path in EXTRACTED_TEXT_DIR.rglob("*.txt"):
        rel_txt = txt_path.relative_to(EXTRACTED_TEXT_DIR)
        rel_pdf_key = rel_txt.with_suffix(".pdf").as_posix()
        if rel_pdf_key not in meta:
            continue
        entry = meta[rel_pdf_key]
        sec_map = entry.get("security_levels", {}) or {}
        doc_id = entry.get("doc_id")
        version = entry.get("version", 0)
        if not doc_id or version == 0:
            _id, _ver = _parse_doc_version(rel_txt.stem)
            doc_id = doc_id or _id
            version = version or _ver
            entry["doc_id"] = doc_id
            entry["version"] = version

        # 이전 동일 doc_id/version 데이터 삭제(작업유형 전체)
        try:
            client.delete(
                COLLECTION_NAME,
                filter=f"doc_id == '{doc_id}' && version <= {int(version)}",
            )
        except Exception:
            pass

        # 텍스트 로드/청크화
        text = txt_path.read_text(encoding="utf-8")
        chunks = chunk_text(text)

        # 작업유형별 삽입
        batch: List[Dict] = []
        for task in tasks:
            lvl = int(sec_map.get(task, 1))
            for idx, c in enumerate(chunks):
                vec = _embed_text(tok, model, device, c, max_len=MAX_TOKENS)
                batch.append({
                    "embedding": vec.tolist(),
                    "path": str(rel_txt),
                    "chunk_idx": int(idx),
                    "task_type": task,
                    "security_level": lvl,
                    "doc_id": str(doc_id),
                    "version": int(version),
                    "text": c,
                })
                if len(batch) >= 128:
                    client.insert(COLLECTION_NAME, batch)
                    total_inserted += len(batch)
                    batch = []
        if batch:
            client.insert(COLLECTION_NAME, batch)
            total_inserted += len(batch)

    try:
        client.flush(COLLECTION_NAME)
    except Exception:
        pass
    _ensure_collection_and_index(client, emb_dim, metric="IP")
    META_JSON_PATH.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"message": "Ingest 완료(Milvus Server)", "inserted_chunks": total_inserted}


# -------------------------------------------------
# 2-1) 단일 파일 인제스트(선택 작업유형)
# -------------------------------------------------
async def ingest_single_pdf(req: SinglePDFIngestRequest):
    import fitz  # type: ignore

    try:
        from repository.documents import insert_workspace_document
    except Exception:
        insert_workspace_document = None

    pdf_path = Path(req.pdf_path)
    if not pdf_path.exists():
        return {"error": f"PDF 경로를 찾을 수 없습니다: {pdf_path}"}

    # 텍스트 생성 및 메타 갱신
    if META_JSON_PATH.exists():
        meta = json.loads(META_JSON_PATH.read_text(encoding="utf-8"))
    else:
        meta = {}

    # securityLevel 폴더는 최대레벨 기준
    with fitz.open(pdf_path) as doc:
        text_all = "\n\n".join(p.get_text("text").strip() for p in doc)

    all_rules = get_security_level_rules_all()
    sec_map = {
        task: _determine_level_for_task(
            text_all, all_rules.get(task, {"maxLevel": 1, "levels": {}})
        )
        for task in TASK_TYPES
    }
    max_sec = max(sec_map.values()) if sec_map else 1
    sec_folder = f"securityLevel{int(max_sec)}"

    rel_pdf = Path(sec_folder) / pdf_path.name
    # 저장 경로: local_data(원본 PDF 복사) + extracted_texts(텍스트)
    (LOCAL_DATA_ROOT / rel_pdf).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pdf_path, LOCAL_DATA_ROOT / rel_pdf)
    txt_path = EXTRACTED_TEXT_DIR / rel_pdf.with_suffix(".txt")
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.write_text(text_all, encoding="utf-8")

    doc_id, ver = _parse_doc_version(pdf_path.stem)
    meta[str(rel_pdf)] = {
        "chars": len(text_all),
        "lines": len(text_all.splitlines()),
        "preview": (text_all[:200].replace("\n", " ") + "…") if text_all else "",
        "security_levels": sec_map,
        "doc_id": doc_id,
        "version": ver,
    }
    META_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    META_JSON_PATH.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 인제스트(선택 작업유형)
    settings = get_vector_settings()
    tok, model, device = _load_embedder(settings["embeddingModel"])
    emb_dim = int(_embed_text(tok, model, device, "probe").shape[0])
    client = _client()
    _ensure_collection_and_index(client, emb_dim, metric="IP")

    s = get_vector_settings()
    MAX_TOKENS, OVERLAP = int(s["chunkSize"]), int(s["overlap"])
    def chunk_text(text: str, max_tokens: int = MAX_TOKENS, overlap: int = OVERLAP):
        words = text.split()
        chunks: List[str] = []
        start = 0
        while start < len(words):
            end = min(start + max_tokens, len(words))
            chunk = " ".join(words[start:end]).strip()
            if chunk:
                chunks.append(chunk)
            start += max_tokens - overlap
        return chunks

    # 기존 삭제
    try:
        client.delete(
            COLLECTION_NAME, filter=f"doc_id == '{doc_id}' && version <= {int(ver)}"
        )
    except Exception:
        pass

    tasks = req.task_types or list(TASK_TYPES)
    chunks = chunk_text(text_all)
    batch, cnt = [], 0
    for task in tasks:
        lvl = int(sec_map.get(task, 1))
        for idx, c in enumerate(chunks):
            vec = _embed_text(tok, model, device, c, max_len=MAX_TOKENS)
            batch.append({
                "embedding": vec.tolist(),
                "path": str(rel_pdf.with_suffix(".txt")),
                "chunk_idx": int(idx),
                "task_type": task,
                "security_level": lvl,
                "doc_id": str(doc_id),
                "version": int(ver),
                "text": c,
            })
            if len(batch) >= 128:
                client.insert(COLLECTION_NAME, batch)
                cnt += len(batch)
                batch = []
    if batch:
        client.insert(COLLECTION_NAME, batch)
        cnt += len(batch)

    if req.workspace_id is not None and insert_workspace_document:
        try:
            insert_workspace_document(
                doc_id=str(doc_id),
                filename=rel_pdf.name,
                docpath=str(rel_pdf),
                workspace_id=int(req.workspace_id),
                metadata={
                    "securityLevels": sec_map,
                    "chunks": int(cnt),
                    "isUserUpload": True,
                },
            )
        except Exception:
            pass

    try:
        client.flush(COLLECTION_NAME)
    except Exception:
        pass
    _ensure_collection_and_index(client, emb_dim, metric="IP")

    return {
        "message": "단일 PDF 인제스트 완료(Milvus Server)",
        "doc_id": doc_id,
        "version": ver,
        "chunks": cnt,
    }


# -------------------------------------------------
# 3) 검색 (vector / hybrid)
#   - task_type 필터 + security_level 제한
#   - hybrid: 벡터 topK*α 후보에 대해 간이 BM25 후처리 리랭크
# -------------------------------------------------
def _bm25_like_score(query: str, doc: str, k1: float = 1.2, b: float = 0.75) -> float:
    # 후보군 소규모 리랭크용 간단 BM25 대용(문서 집합이 작을 때만)
    # 토크나이징 매우 단순화(공백 기준)
    q_terms = [w for w in query.lower().split() if w]
    d_terms = [w for w in doc.lower().split() if w]
    if not q_terms or not d_terms:
        return 0.0
    d_len = len(d_terms)
    tf = Counter(d_terms)
    # IDF는 후보 집합 크기를 사용하기 어려워 고정치에 완화 가중
    score = 0.0
    avgdl = max(1.0, d_len)  # 후보 단일 문서 기준
    for t in set(q_terms):
        f = tf.get(t, 0)
        if f == 0:
            continue
        # 완화 IDF(상수): log(1 + 1/freq) 대신 상수 1.5 사용(경험적)
        idf = 1.5
        denom = f + k1 * (1 - b + b * (d_len / avgdl))
        score += idf * ((f * (k1 + 1)) / (denom if denom != 0 else 1))
    return float(score)


async def search_documents(
    req: RAGSearchRequest, search_type_override: Optional[str] = None
) -> Dict:
    t0 = time.perf_counter()
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
    q_emb = _embed_text(tok, model, device, req.query)
    client = _client()
    _ensure_collection_and_index(client, emb_dim=len(q_emb), metric="IP")

    if COLLECTION_NAME not in client.list_collections():
        return {"error": "컬렉션이 없습니다. 먼저 데이터 저장(인제스트)을을 수행하세요."}

    # 공통 파라미터
    base_limit = int(req.top_k)
    candidate = min(50, max(base_limit, base_limit * 4))
    filter_expr = f"task_type == '{req.task_type}' && security_level <= {int(req.user_level)}"

    def _dense_search(limit=candidate):
        return client.search(
            collection_name=COLLECTION_NAME,
            data=[q_emb.tolist()],
            anns_field="embedding",
            limit=int(limit),
            search_params={"metric_type": "IP", "params": {}},
            output_fields=["path", "chunk_idx", "task_type", "security_level", "doc_id", "text"],
            filter=filter_expr,
        )

    def _sparse_search(query_text: str, limit=candidate):
        """
        BM25 Function(text->text_sparse)이 붙어 있으면 서버가 스파스 점수를 계산한다.
        최신 pymilvus에서는 anns_field='text_sparse', data=['쿼리 문자열'] 형태를 지원.
        일부 버전에서 미지원이면 예외 발생 → 폴백으로 빈 결과 반환.
        """
        try:
            return client.search(
                collection_name=COLLECTION_NAME,
                data=[query_text],
                anns_field="text_sparse",
                limit=int(limit),
                search_params={"params": {}},
                output_fields=["path", "chunk_idx", "task_type", "security_level", "doc_id", "text"],
                filter=filter_expr,
            )
        except Exception as e:
            logger.warning(f"[Milvus] sparse search unavailable: {e}")
            return [[]]

    def _load_snippet(
        path: str, cidx: int, max_tokens: int = 512, overlap: int = 64
    ) -> str:
        try:
            full_txt = (EXTRACTED_TEXT_DIR / path).read_text(encoding="utf-8")
        except Exception:
            return ""
        words = full_txt.split()
        if not words:
            return ""
        start = cidx * (max_tokens - overlap)
        # 보존: 추출 시와 동일 슬라이딩 윈도우는 아니지만 근사 스니펫 제공
        snippet = " ".join(words[start : start + max_tokens]).strip()
        return snippet or " ".join(words[:max_tokens]).strip()

    # === 분기: 검색 방식 ===
    hits_raw = []
    if search_type == "vector":
        results = _dense_search(limit=candidate)
        for hit in results[0]:
            if isinstance(hit, dict):
                ent = hit.get("entity", {})
                path = ent.get("path")
                cidx = int(ent.get("chunk_idx", 0))
                ttype = ent.get("task_type")
                lvl = int(ent.get("security_level", 1))
                doc_id = ent.get("doc_id")
                score_vec = float(hit.get("distance", 0.0))
            else:
                path = hit.entity.get("path")
                cidx = int(hit.entity.get("chunk_idx", 0))
                ttype = hit.entity.get("task_type")
                lvl = int(hit.entity.get("security_level", 1))
                doc_id = hit.entity.get("doc_id")
                score_vec = float(hit.score)
            snippet = _load_snippet(path, cidx)
            hits_raw.append({
                "path": path, "chunk_idx": cidx, "task_type": ttype,
                "security_level": lvl, "doc_id": doc_id,
                "score_vec": score_vec, "score_sparse": 0.0, "snippet": snippet
            })
    else:
        # hybrid / bm25: 덴스 + 스파스 각각 검색
        res_dense = _dense_search(limit=candidate)
        res_sparse = _sparse_search(req.query, limit=candidate)

        def _collect(res, is_dense: bool):
            out = []
            for hit in (res[0] if res and len(res) > 0 else []):
                if isinstance(hit, dict):
                    ent = hit.get("entity", {})
                    path = ent.get("path")
                    cidx = int(ent.get("chunk_idx", 0))
                    ttype = ent.get("task_type")
                    lvl = int(ent.get("security_level", 1))
                    doc_id = ent.get("doc_id")
                    score = float(hit.get("distance", 0.0))
                else:
                    path = hit.entity.get("path")
                    cidx = int(hit.entity.get("chunk_idx", 0))
                    ttype = hit.entity.get("task_type")
                    lvl = int(hit.entity.get("security_level", 1))
                    doc_id = hit.entity.get("doc_id")
                    score = float(hit.score)
                out.append(((path, cidx, ttype, lvl, doc_id), score))
            return out

        dense_list = _collect(res_dense, True)
        sparse_list = _collect(res_sparse, False)

        # RRF 결합 폴백
        rrf: dict[tuple, float] = {}
        K = 60.0
        for rank, (key, _s) in enumerate(dense_list, start=1):
            rrf[key] = rrf.get(key, 0.0) + 1.0 / (K + rank)
        for rank, (key, _s) in enumerate(sparse_list, start=1):
            rrf[key] = rrf.get(key, 0.0) + 1.0 / (K + rank)

        merged = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:candidate]
        for (path, cidx, ttype, lvl, doc_id), fused in merged:
            snippet = _load_snippet(path, cidx)
            s_vec = next((s for (k, s) in dense_list if k == (path, cidx, ttype, lvl, doc_id)), 0.0)
            s_spa = next((s for (k, s) in sparse_list if k == (path, cidx, ttype, lvl, doc_id)), 0.0)
            hits_raw.append({
                "path": path, "chunk_idx": cidx, "task_type": ttype,
                "security_level": lvl, "doc_id": doc_id,
                "score_vec": float(s_vec), "score_sparse": float(s_spa),
                "score_fused": float(fused),
                "snippet": snippet
            })

    # 최종 스코어/정렬
    if search_type == "bm25":
        for h in hits_raw:
            h["score"] = h.get("score_sparse", 0.0) or h.get("score_vec", 0.0)
        hits_sorted = sorted(hits_raw, key=lambda x: x["score"], reverse=True)[:base_limit]
    elif search_type == "hybrid":
        if any("score_fused" in h for h in hits_raw):
            for h in hits_raw:
                h["score"] = h.get("score_fused", 0.0)
        else:
            for h in hits_raw:
                h["score"] = 0.5 * h.get("score_vec", 0.0) + 0.5 * h.get("score_sparse", 0.0)
        hits_sorted = sorted(hits_raw, key=lambda x: x["score"], reverse=True)[:base_limit]
    else:  # vector
        for h in hits_raw:
            h["score"] = h.get("score_vec", 0.0)
        hits_sorted = sorted(hits_raw, key=lambda x: x["score"], reverse=True)[:base_limit]

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
                "snippet": h["snippet"],
            }
            for h in hits_sorted
        ],
        "prompt": prompt,
    }


async def execute_search(
    question: str,
    top_k: int = 5,
    security_level: int = 1,
    source_filter: Optional[List[str]] = None,
    task_type: str = "qna",
    model_key: Optional[str] = None,
    search_type: Optional[str] = None,
) -> Dict:
    req = RAGSearchRequest(
        query=question,
        top_k=top_k,
        user_level=security_level,
        task_type=task_type,
        model=model_key,
    )
    res = await search_documents(req, search_type_override=search_type)
    # Build check_file BEFORE optional source_filter so it reflects original candidates
    check_files: List[str] = []
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

    client = _client()
    cols = client.list_collections()
    for c in cols:
        client.drop_collection(c)
    
    # === 메타 초기화(요구사항: "백터 DB key도 전부 지우게") ===
    try:
        with get_session() as session:
            # 1) 모든 임베딩 모델 비활성화
            session.query(EmbeddingModel).filter(EmbeddingModel.is_active == 1).update(
                {"is_active": 0, "activated_at": None}
            )
            # 2) RagSettings의 임베딩 키 제거
            s = session.query(RagSettings).filter(RagSettings.id == 1).first()
            if s:
                s.embedding_key = None
                s.updated_at = now_kst()
            session.commit()
            logger.info("임베딩 모델 상태 및 RagSettings 키 초기화 완료")
    except Exception as e:
        logger.error(f"메타데이터 초기화 실패: {e}")
    
    return {"message": "삭제 완료(Milvus Server + 메타데이터 초기화)", "dropped_collections": cols}


async def list_indexed_files(
    limit: int = 16384,
    offset: int = 0,
    query: Optional[str] = None,
    task_type: Optional[str] = None,
):
    limit = max(1, min(limit, 16384))
    client = _client()
    if COLLECTION_NAME not in client.list_collections():
        return []

    flt = ""
    if task_type and task_type in TASK_TYPES:
        flt = f"task_type == '{task_type}'"
    try:
        rows = client.query(
            collection_name=COLLECTION_NAME,
            filter=flt,
            output_fields=["path", "chunk_idx", "security_level", "task_type"],
            limit=limit,
            offset=offset,
            consistency_level="Strong",
        )
    except Exception:
        rows = []

    counts: Dict[Tuple[str, str], int] = defaultdict(int)  # (path, task_type) -> chunks
    level_map: Dict[Tuple[str, str], int] = {}
    for r in rows:
        path = r.get("path") if isinstance(r, dict) else r["path"]
        ttype = r.get("task_type") if isinstance(r, dict) else r["task_type"]
        lvl = int(
            (r.get("security_level") if isinstance(r, dict) else r["security_level"])
            or 1
        )
        key = (path, ttype)
        counts[key] += 1
        level_map.setdefault(key, lvl)

    items = []
    for (path, ttype), cnt in counts.items():
        txt_rel = Path(path)
        pdf_rel = txt_rel.with_suffix(".pdf")
        file_name = pdf_rel.name
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
                "filePath": str(pdf_rel),
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

    client = _client()
    if COLLECTION_NAME not in client.list_collections():
        deleted_sql = None
        if delete_workspace_documents_by_filenames:
            deleted_sql = delete_workspace_documents_by_filenames(file_names)
        return {"deleted": 0, "deleted_sql": deleted_sql, "requested": len(file_names)}

    # 로드 보장
    try:
        client.load_collection(collection_name=COLLECTION_NAME)
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
            base_id, _ver = _parse_doc_version(stem)
        except Exception:
            base_id = stem
        try:
            # doc_id == 'stem' [&& task_type == 'xxx']
            filt = f"doc_id == '{base_id}'{task_filter}"
            client.delete(collection_name=COLLECTION_NAME, filter=filt)
            deleted_total += 1
            per_file[name] = per_file.get(name, 0) + 1
        except Exception:
            logger.exception("Failed to delete from Milvus for file: %s", name)
            per_file[name] = per_file.get(name, 0)

    # Ensure deletion is visible to subsequent queries (file lists/overview)
    try:
        client.flush(COLLECTION_NAME)
    except Exception:
        logger.exception("Failed to flush Milvus after deletion")
    # Force reload to avoid any stale cache/state on the server side
    try:
        client.release_collection(collection_name=COLLECTION_NAME)
    except Exception:
        pass
    try:
        client.load_collection(collection_name=COLLECTION_NAME)
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

# ============================
# Test-only RAG Sessions (isolated Milvus collection per session)
# ============================
from pathlib import Path
import uuid
import json
import time

# 세션 루트: ./storage/val_data/sessions/<sid> 에 파일/텍스트를 보관
VAL_SESSION_ROOT = (STORAGE_DIR / "val_data" / "sessions").resolve()

def _session_dir(sid: str) -> Path:
    return VAL_SESSION_ROOT / sid

def _session_extracted_dir(sid: str) -> Path:
    return _session_dir(sid) / "extracted_texts"

def _session_meta_path(sid: str) -> Path:
    return _session_dir(sid) / "session.json"

def _session_collection_name(sid: str) -> str:
    # 프로덕션 컬렉션과 확실히 분리
    return f"pdf_chunks_test_{sid}"

def _write_session_meta(sid: str, data: dict):
    d = _session_dir(sid)
    d.mkdir(parents=True, exist_ok=True)
    _session_meta_path(sid).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def _read_session_meta(sid: str) -> Optional[dict]:
    p = _session_meta_path(sid)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

def create_test_session() -> Dict:
    """
    신규 테스트 세션 생성.
    - 세션 폴더 생성: ./storage/val_data/sessions/<sid>
    - (Milvus 컬렉션은 최초 인제스트 때 생성)
    """
    sid = uuid.uuid4().hex[:12]
    d = _session_dir(sid)
    d.mkdir(parents=True, exist_ok=True)
    meta = {
        "sid": sid,
        "dir": str(d),
        "extracted_dir": str(_session_extracted_dir(sid)),
        "collection": _session_collection_name(sid),
        "created_at": now_kst_string(),
        "pdf_files": [],   # 세션에 업로드/인제스트한 원본 파일명(표시용)
    }
    _write_session_meta(sid, meta)
    return {"sid": sid, "dir": meta["dir"], "collection": meta["collection"]}

def get_test_session(sid: str) -> Optional[dict]:
    return _read_session_meta(sid)

def _ensure_collection_and_index_test(client: MilvusClient, emb_dim: int, collection_name: str, metric: str = "IP"):
    # 프로덕션과 동일 스키마 (필드명/타입 동일)로 만듭니다. 단, 컬렉션명만 세션별로 분리.
    cols = client.list_collections()
    if collection_name not in cols:
        schema = client.create_schema(auto_id=True, enable_dynamic_field=False, description=f"Test PDF chunks ({collection_name})")
        schema.add_field("pk", DataType.INT64, is_primary=True)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=int(emb_dim))
        schema.add_field("path", DataType.VARCHAR, max_length=500)
        schema.add_field("chunk_idx", DataType.INT64)
        schema.add_field("task_type", DataType.VARCHAR, max_length=16)  # 'doc_gen'|'summary'|'qna'
        schema.add_field("security_level", DataType.INT64)
        schema.add_field("doc_id", DataType.VARCHAR, max_length=255)
        schema.add_field("version", DataType.INT64)

        # 텍스트/스파스 (가능 시)
        try:
            schema.add_field("text", DataType.VARCHAR, max_length=32768, enable_analyzer=True)
        except TypeError:
            schema.add_field("text", DataType.VARCHAR, max_length=32768)
        try:
            schema.add_field("text_sparse", DataType.SPARSE_FLOAT_VECTOR)
        except Exception:
            logger.warning("[Milvus][test] SPARSE_FLOAT_VECTOR 미지원 - hybrid시 RRF 폴백 사용")

        if Function is not None:
            try:
                fn = Function(
                    name="bm25_text2sparse",
                    function_type=FunctionType.BM25,
                    input_field_names=["text"],
                    output_field_names=["text_sparse"],
                )
                schema.add_function(fn)
            except Exception as e:
                logger.warning(f"[Milvus][test] BM25 Function 추가 실패: {e}")

        client.create_collection(collection_name=collection_name, schema=schema)

    # 인덱스
    try:
        idx_dense = client.list_indexes(collection_name=collection_name, field_name="embedding")
    except Exception:
        idx_dense = []
    if not idx_dense:
        ip = client.prepare_index_params()
        ip.add_index("embedding", "FLAT", metric_type=metric, params={})
        client.create_index(collection_name, ip, timeout=180.0, sync=True)

    try:
        idx_sparse = client.list_indexes(collection_name=collection_name, field_name="text_sparse")
    except Exception:
        idx_sparse = []
    if not idx_sparse:
        try:
            ip2 = client.prepare_index_params()
            try:
                ip2.add_index("text_sparse", "SPARSE_INVERTED_INDEX", params={})
            except TypeError:
                ip2.add_index("text_sparse", "SPARSE_INVERTED_INDEX", metric_type="BM25", params={})
            client.create_index(collection_name, ip2, timeout=180.0, sync=True)
        except Exception:
            pass

    # 로드
    try:
        client.release_collection(collection_name=collection_name)
    except Exception:
        pass
    client.load_collection(collection_name=collection_name)

async def ingest_test_pdfs(sid: str, pdf_paths: List[str], task_types: Optional[List[str]] = None) -> Dict:
    """
    세션 컬렉션으로만 인제스트. 프로덕션 콜렉션/데이터에 영향 없음.
    """
    meta = get_test_session(sid)
    if not meta:
        return {"error": "invalid sid"}
    coll = meta["collection"]
    sess_extract = Path(meta["extracted_dir"])
    sess_extract.mkdir(parents=True, exist_ok=True)

    # 설정/임베딩 로더
    settings = get_vector_settings()
    tok, model, device = await _get_or_load_embedder_async(settings["embeddingModel"])
    emb_dim = int(_embed_text(tok, model, device, "probe").shape[0])
    client = _client()
    _ensure_collection_and_index_test(client, emb_dim, collection_name=coll, metric="IP")

    # 청크 파라미터
    MAX_TOKENS = int(settings["chunkSize"])
    OVERLAP = int(settings["overlap"])

    def chunk_text(text: str, max_tokens: int = MAX_TOKENS, overlap: int = OVERLAP):
        words = text.split()
        chunks: List[str] = []
        start = 0
        while start < len(words):
            end = min(start + max_tokens, len(words))
            chunk = " ".join(words[start:end]).strip()
            if chunk:
                chunks.append(chunk)
            start += max_tokens - overlap
        return chunks

    # 보안레벨 규칙
    all_rules = get_security_level_rules_all()

    # 실제 처리
    import fitz  # type: ignore
    added_files: List[str] = []
    total_chunks = 0
    tasks = (task_types or list(TASK_TYPES))

    for p in pdf_paths:
        p = str(p)
        src = Path(p)
        if not src.exists():
            continue

        # 텍스트 추출
        with fitz.open(src) as doc:
            text_all = "\n\n".join(pg.get_text("text").strip() for pg in doc)

        # 세션 extracted_texts/<파일명.txt> 로 저장 (중복 방지 타임스탬프)
        ts = int(time.time())
        base_name = f"{src.stem}_{ts}"
        txt_rel = Path(f"{base_name}.txt")
        txt_path = sess_extract / txt_rel
        txt_path.write_text(text_all, encoding="utf-8")

        # 보안 레벨 산정
        sec_map = {task: _determine_level_for_task(text_all, all_rules.get(task, {"maxLevel":1, "levels":{}})) for task in TASK_TYPES}

        # doc_id/version 유추 (세션은 파일명 기반으로 단순화)
        doc_id, version = _parse_doc_version(base_name)

        # 기존 동일 doc_id/version 삭제 (세션 컬렉션 내)
        try:
            client.delete(collection_name=coll, filter=f"doc_id == '{doc_id}' && version <= {int(version)}")
        except Exception:
            pass

        # 청크/임베딩
        chunks = chunk_text(text_all)
        batch = []
        for task in tasks:
            lvl = int(sec_map.get(task, 1))
            for idx, c in enumerate(chunks):
                vec = _embed_text(tok, model, device, c, max_len=MAX_TOKENS)
                batch.append({
                    "embedding": vec.tolist(),
                    "path": str(txt_rel.as_posix()),  # 세션 로컬 텍스트 경로(상대)
                    "chunk_idx": int(idx),
                    "task_type": task,
                    "security_level": lvl,
                    "doc_id": str(doc_id),
                    "version": int(version),
                    "text": c,
                })
                if len(batch) >= 128:
                    client.insert(coll, batch); total_chunks += len(batch); batch = []
        if batch:
            client.insert(coll, batch); total_chunks += len(batch)

        try:
            client.flush(coll)
        except Exception:
            pass

        added_files.append(src.name)

    # 메타 업데이트
    if added_files:
        cur = get_test_session(sid) or {}
        files_prev = list(set(cur.get("pdf_files") or []))
        files_new = sorted(list(set(files_prev + added_files)))
        cur.update({"pdf_files": files_new})
        _write_session_meta(sid, cur)

    return {"sid": sid, "collection": coll, "ingested_files": added_files, "chunks": total_chunks}

async def search_documents_test(req: RAGSearchRequest, sid: str, search_type_override: Optional[str] = None) -> Dict:
    """
    세션 컬렉션에서만 검색. 반환 구조는 프로덕션 search_documents와 최대한 동일.
    """
    meta = get_test_session(sid)
    if not meta:
        return {"error": "invalid sid"}
    coll = meta["collection"]
    sess_extract = Path(meta["extracted_dir"])

    t0 = time.perf_counter()
    if req.task_type not in TASK_TYPES:
        return { "error": f"invalid task_type: {req.task_type}. choose one of {TASK_TYPES}" }

    settings = get_vector_settings()
    model_key = req.model or settings["embeddingModel"]
    raw_st = (search_type_override or settings.get("searchType") or "").lower()
    search_type = (raw_st.replace("semantic", "vector").replace("sementic", "vector") or "hybrid")

    tok, model, device = await _get_or_load_embedder_async(model_key)
    q_emb = _embed_text(tok, model, device, req.query)

    client = _client()
    _ensure_collection_and_index_test(client, emb_dim=len(q_emb), collection_name=coll, metric="IP")
    if coll not in client.list_collections():
        return {"error": "session collection missing"}

    base_limit = int(req.top_k)
    candidate = min(50, max(base_limit, base_limit * 4))
    filter_expr = f"task_type == '{req.task_type}' && security_level <= {int(req.user_level)}"

    def _dense_search(limit=candidate):
        return client.search(
            collection_name=coll,
            data=[q_emb.tolist()],
            anns_field="embedding",
            limit=int(limit),
            search_params={"metric_type": "IP", "params": {}},
            output_fields=["path", "chunk_idx", "task_type", "security_level", "doc_id", "text"],
            filter=filter_expr,
        )

    def _sparse_search(query_text: str, limit=candidate):
        try:
            return client.search(
                collection_name=coll,
                data=[query_text],
                anns_field="text_sparse",
                limit=int(limit),
                search_params={"params": {}},
                output_fields=["path", "chunk_idx", "task_type", "security_level", "doc_id", "text"],
                filter=filter_expr,
            )
        except Exception as e:
            logger.warning(f"[Milvus][test] sparse search unavailable: {e}")
            return [[]]

    def _load_snippet_session(path_rel: str, cidx: int, max_tokens: int = int(settings["chunkSize"]), overlap: int = int(settings["overlap"])) -> str:
        try:
            full_txt = (sess_extract / path_rel).read_text(encoding="utf-8")
        except Exception:
            return ""
        words = full_txt.split()
        if not words:
            return ""
        start = cidx * (max_tokens - overlap)
        snippet = " ".join(words[start: start + max_tokens]).strip()
        return snippet or " ".join(words[:max_tokens]).strip()

    hits_raw = []
    if search_type == "vector":
        res = _dense_search(limit=candidate)
        for hit in res[0]:
            if isinstance(hit, dict):
                ent = hit.get("entity", {})
                path = ent.get("path"); cidx = int(ent.get("chunk_idx", 0))
                ttype = ent.get("task_type"); lvl = int(ent.get("security_level", 1))
                doc_id = ent.get("doc_id"); score_vec = float(hit.get("distance", 0.0))
            else:
                path = hit.entity.get("path"); cidx = int(hit.entity.get("chunk_idx", 0))
                ttype = hit.entity.get("task_type"); lvl = int(hit.entity.get("security_level", 1))
                doc_id = hit.entity.get("doc_id"); score_vec = float(hit.score)
            snippet = _load_snippet_session(path, cidx)
            hits_raw.append({"path": path, "chunk_idx": cidx, "task_type": ttype, "security_level": lvl,
                             "doc_id": doc_id, "score_vec": score_vec, "score_sparse": 0.0, "snippet": snippet})
    else:
        res_dense = _dense_search(limit=candidate)
        res_sparse = _sparse_search(req.query, limit=candidate)

        def _collect(res):
            out = []
            for hit in (res[0] if res and len(res) > 0 else []):
                if isinstance(hit, dict):
                    ent = hit.get("entity", {})
                    path = ent.get("path"); cidx = int(ent.get("chunk_idx", 0))
                    ttype = ent.get("task_type"); lvl = int(ent.get("security_level", 1))
                    doc_id = ent.get("doc_id"); score = float(hit.get("distance", 0.0))
                else:
                    path = hit.entity.get("path"); cidx = int(hit.entity.get("chunk_idx", 0))
                    ttype = hit.entity.get("task_type"); lvl = int(hit.entity.get("security_level", 1))
                    doc_id = hit.entity.get("doc_id"); score = float(hit.score)
                out.append(((path, cidx, ttype, lvl, doc_id), score))
            return out

        dense_list = _collect(res_dense)
        sparse_list = _collect(res_sparse)

        # RRF
        rrf = {}
        K = 60.0
        for rank, (key, _s) in enumerate(dense_list, start=1):
            rrf[key] = rrf.get(key, 0.0) + 1.0 / (K + rank)
        for rank, (key, _s) in enumerate(sparse_list, start=1):
            rrf[key] = rrf.get(key, 0.0) + 1.0 / (K + rank)
        merged = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:candidate]
        for (path, cidx, ttype, lvl, doc_id), fused in merged:
            snippet = _load_snippet_session(path, cidx)
            s_vec = next((s for (k, s) in dense_list if k == (path, cidx, ttype, lvl, doc_id)), 0.0)
            s_spa = next((s for (k, s) in sparse_list if k == (path, cidx, ttype, lvl, doc_id)), 0.0)
            hits_raw.append({"path": path, "chunk_idx": cidx, "task_type": ttype, "security_level": lvl,
                             "doc_id": doc_id, "score_vec": float(s_vec), "score_sparse": float(s_spa),
                             "score_fused": float(fused), "snippet": snippet})

    # 정렬/상위 K
    if search_type == "bm25":
        for h in hits_raw:
            h["score"] = h.get("score_sparse", 0.0) or h.get("score_vec", 0.0)
    elif search_type == "hybrid":
        if any("score_fused" in h for h in hits_raw):
            for h in hits_raw: h["score"] = h.get("score_fused", 0.0)
        else:
            for h in hits_raw: h["score"] = 0.5*h.get("score_vec",0.0) + 0.5*h.get("score_sparse",0.0)
    else:
        for h in hits_raw: h["score"] = h.get("score_vec", 0.0)
    hits_sorted = sorted(hits_raw, key=lambda x: x["score"], reverse=True)[:base_limit]

    # 프롬프트 컨텍스트 + check_file
    context = "\n---\n".join(h["snippet"] for h in hits_sorted if h.get("snippet"))
    prompt = f"사용자 질의: {req.query}\n:\n{context}\n\n위 내용을 바탕으로 응답을 생성해 주세요."
    check_files: List[str] = []
    try:
        for h in hits_sorted:
            # 세션은 원본 파일명이 세션 메타에만 있으므로, path(doc_id)에서 stem을 추출
            did = h.get("doc_id")
            if did:
                check_files.append(f"{did}.pdf")
    except Exception:
        pass
    elapsed = round(time.perf_counter() - t0, 4)

    return {
        "elapsed_sec": elapsed,
        "settings_used": {"model": model_key, "searchType": search_type},
        "hits": [{
            "score": float(h["score"]),
            "path": h["path"],
            "chunk_idx": int(h["chunk_idx"]),
            "task_type": h["task_type"],
            "security_level": int(h["security_level"]),
            "doc_id": h.get("doc_id"),
            "snippet": h["snippet"],
        } for h in hits_sorted],
        "prompt": prompt,
        "check_file": sorted(list(set(check_files))),
    }

async def delete_test_files_by_names(sid: str, file_names: List[str], task_type: Optional[str] = None) -> Dict:
    meta = get_test_session(sid)
    if not meta:
        return {"deleted": 0, "requested": len(file_names), "error": "invalid sid"}
    coll = meta["collection"]
    sess_dir = Path(meta["dir"])
    sess_extract = Path(meta["extracted_dir"])

    if not file_names:
        return {"deleted": 0, "requested": 0}

    client = _client()
    if coll not in client.list_collections():
        return {"deleted": 0, "requested": len(file_names)}

    # task_type 필터
    task_filter = ""
    if task_type:
        if task_type not in TASK_TYPES:
            return {"deleted": 0, "requested": len(file_names), "error": f"invalid taskType: {task_type}"}
        task_filter = f" && task_type == '{task_type}'"

    deleted_total = 0
    per_file: dict[str, int] = {}
    for name in file_names:
        stem = Path(name).stem
        try:
            base_id, _ver = _parse_doc_version(stem)
        except Exception:
            base_id = stem
        try:
            client.delete(collection_name=coll, filter=f"doc_id == '{base_id}'{task_filter}")
            deleted_total += 1
            per_file[name] = per_file.get(name, 0) + 1
        except Exception:
            logger.exception("[test] Milvus delete failed: %s", name)
            per_file[name] = per_file.get(name, 0)

        # 세션 로컬 파일 제거 (txt, 원본 pdf)
        try:
            # extracted_texts/*.txt 중 base_id로 시작하는 것들 삭제
            for t in sess_extract.glob(f"{base_id}_*.txt"):
                t.unlink(missing_ok=True)
        except Exception:
            pass
        try:
            # 업로드 폴더 내 원본 pdf (타임스탬프 붙어서 정확 일치 어렵지만 base_id로 매칭 시도)
            for p in sess_dir.glob(f"{base_id}_*.pdf"):
                p.unlink(missing_ok=True)
        except Exception:
            pass

    try:
        client.flush(coll)
    except Exception:
        pass

    # 메타 업데이트
    cur = get_test_session(sid) or {}
    if cur:
        prev = list(set(cur.get("pdf_files") or []))
        left = [f for f in prev if Path(f).stem not in {Path(n).stem for n in file_names}]
        cur.update({"pdf_files": sorted(left)})
        _write_session_meta(sid, cur)

    return {"deleted": deleted_total, "requested": len(file_names), "taskType": task_type, "perFile": per_file}

async def drop_test_session(sid: str) -> Dict:
    meta = get_test_session(sid)
    if not meta:
        return {"message": "session not found", "sid": sid}

    coll = meta["collection"]
    client = _client()
    try:
        if coll in client.list_collections():
            client.drop_collection(coll)
    except Exception:
        logger.exception("[test] failed to drop collection: %s", coll)

    # 세션 폴더 삭제
    try:
        import shutil
        shutil.rmtree(_session_dir(sid), ignore_errors=True)
    except Exception:
        logger.exception("[test] failed to remove session dir")

    return {"message": "session dropped", "sid": sid}
