# service/admin/manage_admin_LLM.py
from __future__ import annotations

import json
import os
import sqlite3
import time
import gc
import logging
import importlib
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple

from pydantic import BaseModel, Field
from utils.database import get_db as _get_db

# ===== In-memory temp settings (kept for backward-compat) =====
_DEFAULT_TOPK: int = 5

# ===== Constants =====
# Force DB to pps_rag.db by default (can be overridden via COREIQ_DB)
os.environ.setdefault("COREIQ_DB", "/home/work/CoreIQ/backend/storage/pps_rag.db")
DB_PATH = os.getenv("COREIQ_DB", "/home/work/CoreIQ/backend/storage/pps_rag.db")
STORAGE_ROOT = "/home/work/CoreIQ/backend/storage/model"  # train_qwen_rag.py 와 동일
ACTIVE_MODEL_CACHE_KEY_PREFIX = "active_model:"  # e.g. active_model:qa
ACTIVE_PROMPT_CACHE_PREFIX = "active_prompt:"     # active_prompt:qa:report
RAG_TOPK_CACHE_KEY = "rag_topk"

# Backend root for portable paths (same approach as LLM_finetuning.py)
BASE_BACKEND = Path(os.getenv("COREIQ_BACKEND_ROOT", str(Path(__file__).resolve().parents[2])))

# ===== Pydantic models (변수명/스키마 고정) =====
class TopKSettingsBody(BaseModel):
    topK: int = Field(..., gt=0, description="RAG 반환 문서 수")


class ModelLoadBody(BaseModel):
    modelName: str = Field(..., description="로드/언로드할 모델 이름")


class PromptVariable(BaseModel):
    key: str
    value: Optional[str] = None
    type: str = Field(..., description="string | date-time | integer | float | bool")


class CreatePromptBody(BaseModel):
    title: str
    prompt: str
    variables: List[PromptVariable] = []


class UpdatePromptBody(BaseModel):
    title: Optional[str] = None
    prompt: Optional[str] = None
    variables: Optional[List[PromptVariable]] = None


class CompareModelsBody(BaseModel):
    category: str
    modelId: Optional[int] = None
    promptId: Optional[int] = None
    prompt: Optional[str] = None

# === NEW: model download / train / infer request bodies ===

class DownloadModelBody(BaseModel):
    repo: str = Field(..., description="HuggingFace repo id, e.g. Qwen/Qwen2.5-7B-Instruct-1M")
    name: str = Field(..., description="Local folder name to store under STORAGE_ROOT")


class TrainModelBody(BaseModel):
    csv: str = Field(..., description="Path to training csv file")
    base_name: str = Field(..., description="Base model folder name under STORAGE_ROOT")
    ft_name: str = Field(..., description="Fine-tuned model folder name under STORAGE_ROOT")
    epochs: int = 3
    batch_size: int = 4
    lr: float = 2e-4


class InferBody(BaseModel):
    modelName: str = Field(..., description="Model folder name under STORAGE_ROOT or repo id")
    context: str
    question: str
    max_tokens: int = 512
    temperature: float = 0.7


class InsertBaseModelBody(BaseModel):
    name: str = Field(..., description="베이스 모델 표시 이름 (예: gpt-oss-20b)")
    model_path: Optional[str] = Field(
        None, description="옵션. 상대/절대 모두 허용. 미지정 시 STORAGE_ROOT/name 가정"
    )
    provider: str = Field("huggingface", description="모델 제공자")
    category: str = Field("all", description="항상 'all'로 저장(단일 레코드)")

# 활성 프롬프트 선택 저장용
class ActivePromptBody(BaseModel):
    category: str = Field(..., description="doc_gen | summary | qna")
    subtask: Optional[str] = Field(None, description="doc_gen 전용 서브테스크")
    promptId: int = Field(..., description="선택할 프롬프트 ID")

# ===== DB Helpers =====
def _connect() -> sqlite3.Connection:
    # Delegate to shared database connector (respects config/database paths and pragmas)
    return _get_db()
# ===== Migration / helpers =====
def _migrate_llm_models_if_needed():
    conn = _connect()
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    try:
        cur.execute("PRAGMA table_info(llm_models)")
        cols = {r[1] for r in cur.fetchall()}  # name is at index 1
        cur.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='llm_models'")
        row = cur.fetchone()
        ddl = row[0] if row else ""
        need_sub = "subcategory" not in cols
        need_all = ("CHECK" in ddl) and ("'all'" not in ddl)
        if not (need_sub or need_all):
            return
        cur.execute("PRAGMA foreign_keys=off")
        cur.execute(
            """
            CREATE TABLE llm_models__new(
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              provider TEXT NOT NULL,
              name TEXT UNIQUE NOT NULL,
              revision INTEGER,
              model_path TEXT,
              category TEXT NOT NULL CHECK (category IN ('qa','doc_gen','summary','all')),
              subcategory TEXT,
              type TEXT NOT NULL CHECK (type IN ('base','lora','full')) DEFAULT 'base',
              is_default BOOLEAN NOT NULL DEFAULT 0,
              is_active BOOLEAN NOT NULL DEFAULT 1,
              trained_at DATETIME,
              created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        # Copy common columns
        keep = [
            "id","provider","name","revision","model_path","category","type","is_default","is_active","trained_at","created_at"
        ]
        present = []
        for c in keep:
            try:
                cur.execute(f"SELECT 1 FROM llm_models LIMIT 1")
                present.append(c)
            except Exception:
                pass
        sel = ", ".join([c for c in keep if c in cols])
        if sel:
            cur.execute(f"INSERT INTO llm_models__new({sel}) SELECT {sel} FROM llm_models")
        cur.execute("DROP TABLE llm_models")
        cur.execute("ALTER TABLE llm_models__new RENAME TO llm_models")
        cur.execute("PRAGMA foreign_keys=on")
        conn.commit()
    except Exception:
        logging.getLogger(__name__).exception("llm_models migration failed")
    finally:
        conn.close()


def _normalize_model_path_input(p: str) -> str:
    s = (p or "").strip().replace("\\","/")
    if not s:
        return s
    prefixes = (
        "storage/model/",
        "./storage/model/",
        "/home/work/CoreIQ/backend/storage/model/",
    )
    for pref in prefixes:
        if s.startswith(pref):
            s = s[len(pref):]
            break
    if "/storage/model/" in s:
        s = s.split("/storage/model/", 1)[1]
    return s.strip("/")


def _db_set_active_by_path(rel_path: str, active: bool) -> None:
    try:
        conn = _connect()
        cur = conn.cursor()
        cur.execute("UPDATE llm_models SET is_active=? WHERE model_path=?", (1 if active else 0, rel_path))
        conn.commit()
    except Exception:
        logging.getLogger(__name__).exception("failed to sync is_active by path")
    finally:
        try:
            conn.close()
        except Exception:
            pass





# no-op: initialization is handled in main.py via init_db()


# ===== Utilities =====
def _json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


def _now_iso() -> str:
    return datetime.utcnow().isoformat()


def _get_cache(name: str) -> Optional[str]:
    conn = _connect()
    cur = conn.cursor()
    cur.execute("SELECT data FROM cache_data WHERE name=? ORDER BY id DESC LIMIT 1", (name,))
    row = cur.fetchone()
    conn.close()
    return row["data"] if row else None
def _norm_category(category: str) -> str:
    """
    외부 표기는 qna, 내부 스키마/기존 코드는 qa.
    """
    c = (category or "").strip().lower()
    if c == "qna":
        return "qa"
    return c

def _subtask_key(subtask: Optional[str]) -> str:
    return (subtask or "").strip().lower()

def _active_prompt_cache_key(category: str, subtask: Optional[str]) -> str:
    c = _norm_category(category)
    s = _subtask_key(subtask)
    return f"{ACTIVE_PROMPT_CACHE_PREFIX}{c}:{s or '-'}"



def _set_cache(name: str, data: str, belongs_to: str = "global", by_id: Optional[int] = None):
    conn = _connect()
    cur = conn.cursor()
    cur.execute("""
      INSERT INTO cache_data(name, data, belongs_to, by_id, created_at, updated_at)
      VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
    """, (name, data, belongs_to, by_id))
    conn.commit()
    conn.close()


# ---------- 활성 프롬프트: 사용자 선택 저장/조회 ----------
def get_active_prompt(category: str, subtask: Optional[str]) -> Dict[str, Any]:
    key = _active_prompt_cache_key(category, subtask)
    data = _get_cache(key)
    info = json.loads(data) if data else None
    return {"category": _norm_category(category), "subtask": _subtask_key(subtask) or None, "active": info}


def set_active_prompt(body: ActivePromptBody) -> Dict[str, Any]:
    key = _active_prompt_cache_key(body.category, body.subtask)
    payload = {"promptId": body.promptId, "setAt": _now_iso()}
    _set_cache(key, _json(payload), "llm_admin")
    return {"success": True, "category": _norm_category(body.category), "subtask": _subtask_key(body.subtask) or None, "active": payload}


# ===== 새로 추가: 활성 LLM 모델 조회(로깅용) =====
def get_active_llm_models() -> List[Dict[str, Any]]:
    """
    llm_models에서 is_active=1 인 모델 목록을 반환한다.
    로깅/표시 용도로만 사용하며, 모델을 실제로 로드하지 않는다.
    """
    conn = _connect()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, provider, name, category, model_path, type, is_default, is_active, trained_at, created_at
        FROM llm_models
        WHERE is_active=1
        ORDER BY id DESC
        """
    )
    rows = cur.fetchall()
    conn.close()
    out: List[Dict[str, Any]] = []
    for r in rows:
        try:
            keys = r.keys()
            out.append({k: r[k] for k in keys})
        except Exception:
            out.append(dict(r))  # best-effort
    return out


# ===== 새로 추가: 최초 사용 시 지연 로딩 =====
def lazy_load_if_needed(model_name: str) -> Dict[str, Any]:
    """
    주어진 모델이 메모리에 없으면 로드하고, 있으면 아무것도 하지 않는다.
    active cache 설정은 하지 않는다(단순 '최초 사용 시 로드' 용도).
    """
    try:
        if _is_model_loaded(model_name):
            return {"loaded": True, "message": "already loaded", "modelName": model_name}

        lower = (model_name or "").lower()
        if lower.startswith("gpt-oss") or lower.startswith("gpt_oss"):
            # gpt-oss는 어댑터 사전 로더 경로 사용
            if not _preload_via_adapters(model_name):
                return {"loaded": False, "message": "adapter preload failed", "modelName": model_name}
            return {"loaded": True, "message": "adapter preloaded", "modelName": model_name}

        # 일반 모델은 매니저로 로드 시도, 실패 시 어댑터 경로도 시도
        candidate = _resolve_model_path_for_name(model_name) or _resolve_model_fs_path(model_name)
        try:
            _MODEL_MANAGER.load(candidate)
            return {"loaded": True, "message": "loaded", "modelName": model_name}
        except Exception:
            logging.getLogger(__name__).exception("lazy load via manager failed, trying adapter")
            if _preload_via_adapters(model_name):
                return {"loaded": True, "message": "adapter preloaded (fallback)", "modelName": model_name}
            return {"loaded": False, "message": "load failed", "modelName": model_name}
    except Exception:
        logging.getLogger(__name__).exception("lazy_load_if_needed unexpected error")
        return {"loaded": False, "message": "unexpected error", "modelName": model_name}


def _ensure_models_from_fs(category: str) -> None:
    """
    STORAGE_ROOT를 스캔하더라도 DB 스키마 카테고리 제약(qa|doc_gen|summary)과 충돌을 피하기 위해
    여기서는 DB에 쓰지 않는다. 베이스 모델 등록은 insert-base API로만 수행한다.
    """
    return


def _to_rel(p: str) -> str:
    """Return `p` as a path relative to backend root if possible, to keep DB records portable."""
    try:
        return os.path.relpath(p, str(BASE_BACKEND))
    except Exception:
        return p


def _lookup_model_by_name(model_name: str) -> Optional[sqlite3.Row]:
    conn = _connect()
    cur = conn.cursor()
    cur.execute("SELECT * FROM llm_models WHERE name=?", (model_name,))
    row = cur.fetchone()
    conn.close()
    return row


def _active_model_name_for_category(category: str) -> Optional[str]:
    data = _get_cache(ACTIVE_MODEL_CACHE_KEY_PREFIX + category)
    if data:
        try:
            return json.loads(data)["modelName"]
        except Exception:
            return None
    return None


def _set_active_model_for_category(category: str, model_name: str):
    _set_cache(ACTIVE_MODEL_CACHE_KEY_PREFIX + category, _json({"modelName": model_name}), "llm_admin")


# ===== Inference helper (optional) =====
# 가벼운 로컬 추론 실행(테스트용). transformers 미설치 환경을 고려해 예외처리.
_MODEL_CACHE: Dict[str, Any] = {}  # name -> (tokenizer, model)


class _ModelManager:
    """
    Thread-safe in-memory model manager to load/unload models on demand.
    Keeps a process-wide cache of tokenizer/model pairs keyed by resolved path.
    """

    def __init__(self):
        import threading
        self._lock = threading.RLock()
        self._cache: Dict[str, Tuple[Any, Any]] = {}
        self._logger = logging.getLogger(__name__)

    def _resolve_candidate(self, name_or_path: str) -> str:
        # Prefer local storage folder under STORAGE_ROOT if it looks like a model dir
        fs_path = os.path.join(STORAGE_ROOT, name_or_path)
        try:
            if os.path.isfile(os.path.join(fs_path, "config.json")):
                return fs_path
        except Exception:
            # best-effort only
            self._logger.exception("failed while resolving model candidate path")
        return name_or_path

    def is_loaded(self, name_or_path: str) -> bool:
        key = self._resolve_candidate(name_or_path)
        with self._lock:
            return key in self._cache

    def load(self, name_or_path: str) -> Tuple[Any, Any]:
        key = self._resolve_candidate(name_or_path)
        with self._lock:
            if key in self._cache:
                return self._cache[key]
            try:
                from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore
                import torch  # type: ignore

                # gpt-oss는 어댑터 전용 경로로 처리 (HF BitsAndBytes 경로 차단)
                base_key = os.path.basename(key).lower()
                if base_key.startswith("gpt-oss") or base_key.startswith("gpt_oss"):
                    raise RuntimeError("gpt-oss should be loaded via adapter only")

                # Optional quantization (disable via env LLM_DISABLE_BNB=1)
                bnb_config = None
                try:
                    if os.getenv("LLM_DISABLE_BNB", "0") not in ("1", "true", "TRUE", "True"):
                        from transformers import BitsAndBytesConfig  # type: ignore
                        bnb_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_quant_type="nf4",
                            bnb_4bit_compute_dtype=torch.bfloat16,
                        )
                except Exception:
                    bnb_config = None
                tok = AutoTokenizer.from_pretrained(key, trust_remote_code=True, local_files_only=True)
                if tok.pad_token_id is None:
                    if getattr(tok, "eos_token_id", None) is not None:
                        tok.pad_token_id = tok.eos_token_id
                    else:
                        tok.add_special_tokens({"pad_token": "<|pad|>"})
                        tok.pad_token_id = tok.convert_tokens_to_ids("<|pad|>")
                try:
                    mdl = AutoModelForCausalLM.from_pretrained(
                        key,
                        trust_remote_code=True,
                        device_map="auto",
                        quantization_config=bnb_config,
                        local_files_only=True,
                    )
                except Exception:
                    # Fallback: reload without quantization if bnb/triton not available
                    mdl = AutoModelForCausalLM.from_pretrained(
                        key,
                        trust_remote_code=True,
                        device_map="auto",
                        torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
                        local_files_only=True,
                    )
                self._cache[key] = (tok, mdl)
                return tok, mdl
            except Exception:
                # Log and re-raise to let caller decide fallback
                try:
                    self._logger.exception("failed to load model: %s", key)
                except Exception:
                    pass
                raise

    def unload(self, name_or_path: str) -> bool:
        key = self._resolve_candidate(name_or_path)
        with self._lock:
            pair = self._cache.pop(key, None)
        if pair is None:
            return False
        # Best-effort GPU/CPU memory cleanup
        try:
            _, mdl = pair
            try:
                del mdl
            except Exception:
                pass
            try:
                import torch  # type: ignore
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            except Exception:
                logging.getLogger(__name__).exception("torch cuda cleanup failed")
            try:
                gc.collect()
            except Exception:
                logging.getLogger(__name__).exception("gc collect failed during unload")
        except Exception:
            logging.getLogger(__name__).exception("failed during model unload cleanup")
        return True

    def list_loaded(self) -> List[str]:
        with self._lock:
            return list(self._cache.keys())


_MODEL_MANAGER = _ModelManager()

# Adapter preload tracking (for correct toggle/unload)
_ADAPTER_LOADED: set[str] = set()

def _resolve_model_fs_path(name_or_path: str) -> str:
    try:
        if os.path.isabs(name_or_path):
            return name_or_path
        fs_path = os.path.join(STORAGE_ROOT, name_or_path)
        if os.path.isfile(os.path.join(fs_path, "config.json")):
            return fs_path
    except Exception:
        logging.getLogger(__name__).exception("failed to resolve model fs path")
    return name_or_path

def _is_model_loaded(model_name: str) -> bool:
    try:
        # Direct name
        if _MODEL_MANAGER.is_loaded(model_name):
            return True
        # Resolve by DB/storage (so category-suffixed names map to the same folder)
        candidate = _resolve_model_path_for_name(model_name) or _resolve_model_fs_path(model_name)
        if _MODEL_MANAGER.is_loaded(candidate):
            return True
        base = os.path.basename(candidate)
        return (candidate in _ADAPTER_LOADED) or (base in _ADAPTER_LOADED)
    except Exception:
        logging.getLogger(__name__).exception("_is_model_loaded failure")
        return False

def _clear_local_cache_entries(model_name: str) -> None:
    """Best-effort local caches and GPU memory cleanup. Never raises."""
    try:
        _MODEL_CACHE.pop(model_name, None)
    except Exception:
        pass
    try:
        resolved = _resolve_model_fs_path(model_name)
        _MODEL_CACHE.pop(resolved, None)
    except Exception:
        pass
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        try:
            logging.getLogger(__name__).exception("torch cuda cleanup failed in _clear_local_cache_entries")
        except Exception:
            pass
    try:
        gc.collect()
    except Exception:
        try:
            logging.getLogger(__name__).exception("gc collect failed in _clear_local_cache_entries")
        except Exception:
            pass

def _load_local_model(model_name_or_path: str):
    try:
        # Prefer adapter-based for OSS/gpt_oss to avoid HF AutoModel path
        lower = (model_name_or_path or "").lower()
        if lower.startswith("gpt-oss") or lower.startswith("gpt_oss"):
            return None
        # Prefer manager cache
        key = _MODEL_MANAGER._resolve_candidate(model_name_or_path)
        if key in _MODEL_CACHE:
            return _MODEL_CACHE[key]
        try:
            tok, mdl = _MODEL_MANAGER.load(model_name_or_path)
            _MODEL_CACHE[key] = (tok, mdl)
            return tok, mdl
        except Exception:
            logging.getLogger(__name__).exception("_load_local_model failed via manager")
            return None
    except Exception:
        logging.getLogger(__name__).exception("_load_local_model unexpected failure")
        return None


def _simple_generate(prompt_text: str, model_name_or_path: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
    pair = _load_local_model(model_name_or_path)
    if not pair:
        # 추론 환경 없으면 더미 응답
        return "⚠️ 로컬 모델이 로드되지 않아 샘플 응답을 반환합니다. (테스트 전용)"
    tok, mdl = pair
    import torch
    messages = [
        {"role": "system", "content": "You are Qwen, a helpful assistant."},
        {"role": "user", "content": prompt_text},
    ]
    full = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(full, return_tensors="pt").to(mdl.device)
    out = mdl.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=0.9,
        repetition_penalty=1.05,
    )
    gen = out[0][inputs.input_ids.shape[1]:]
    return tok.decode(gen, skip_special_tokens=True)


# ================================================================
# Adapter-based preload/unload (qwen, oss via utils.llms)
# ================================================================

def _db_get_model_path(model_name: str) -> Optional[str]:
    """Resolve model_path for a given model name directly from pps_rag.db (llm_models).

    LLM모델 경로로 모델 켜졌는지 조회
    """
    conn = None
    try:
        conn = _connect()
        cur = conn.cursor()
        cur.execute("SELECT model_path FROM llm_models WHERE name=?", (model_name,))
        row = cur.fetchone()
    except Exception:
        logging.getLogger(__name__).exception("failed to query llm_models for %s", model_name)
        return None
    finally:
        try:
            if conn:
                conn.close()
        except Exception:
            pass

    if not row:
        return None
    path_value = (row["model_path"] or "").strip()
    if not path_value:
        return None
    try:
        # If relative, assume under STORAGE_ROOT
        candidate = path_value
        if not os.path.isabs(candidate):
            candidate = os.path.join(STORAGE_ROOT, candidate)
        if os.path.isfile(os.path.join(candidate, "config.json")):
            return candidate
        # If absolute in DB but not found, try STORAGE_ROOT/name fallback
        fallback = os.path.join(STORAGE_ROOT, path_value)
        if os.path.isfile(os.path.join(fallback, "config.json")):
            return fallback
    except Exception:
        logging.getLogger(__name__).exception("failed while resolving db model_path for %s", model_name)
    return None


def _strip_category_suffix(name: str) -> str:
    # 허용: -qa/-doc_gen/-summary 및 _qa/_qna/_doc_gen/_summary
    for suf in ("-qa", "-doc_gen", "-summary", "_qa", "_qna", "_doc_gen", "_summary"):
        if name.endswith(suf):
            return name[: -len(suf)]
    return name

def _infer_category_from_name(name: str) -> Optional[str]:
    n = (name or "").lower()
    if n.endswith(("_summary", "-summary")):
        return "summary"
    if n.endswith(("_qna", "_qa", "-qa")):
        return "qa"
    if n.endswith(("_doc_gen", "-doc_gen")):
        return "doc_gen"
    return None

def _to_rel_under_storage_root(p: str) -> str:
    """
    storage/model 하위 상대경로(보통 폴더명)로 저장.
    절대경로여도 STORAGE_ROOT 기준 상대경로로 환원.
    """
    try:
        rp = os.path.relpath(p, STORAGE_ROOT)
        if rp in (".", ""):
            return os.path.basename(p.rstrip("/"))
        return rp
    except Exception:
        return os.path.basename(p.rstrip("/"))


def _resolve_model_path_for_name(model_name: str) -> Optional[str]:
    """Resolve a usable local model directory for a given logical name.
    Priority: DB.model_path of exact name → DB.model_path of base-name(카테고리 접미어 제거)
             → STORAGE_ROOT/exact → STORAGE_ROOT/base-name
    Returns absolute-like path or None.
    """
    # 1) exact name in DB
    p = _db_get_model_path(model_name)
    if p and os.path.isfile(os.path.join(p, "config.json")):
        return p
    # 2) base-name in DB
    base = _strip_category_suffix(model_name)
    if base != model_name:
        p2 = _db_get_model_path(base)
        if p2 and os.path.isfile(os.path.join(p2, "config.json")):
            return p2
    # 3) STORAGE_ROOT/exact
    cand = os.path.join(STORAGE_ROOT, model_name)
    if os.path.isfile(os.path.join(cand, "config.json")):
        return cand
    # 4) STORAGE_ROOT/base-name
    cand2 = os.path.join(STORAGE_ROOT, base)
    if os.path.isfile(os.path.join(cand2, "config.json")):
        return cand2
    return None


def _preload_via_adapters(model_name: str) -> bool:
    name = (model_name or "").lower()
    model_path = _db_get_model_path(model_name)
    if not model_path:
        # Fallback to STORAGE_ROOT if present
        cand = os.path.join(STORAGE_ROOT, model_name)
        if os.path.isfile(os.path.join(cand, "config.json")):
            model_path = cand
        else:
            # Also try base-name for suffix forms
            base = _strip_category_suffix(model_name)
            cand2 = os.path.join(STORAGE_ROOT, base)
            if os.path.isfile(os.path.join(cand2, "config.json")):
                model_path = cand2
            else:
                logging.getLogger(__name__).warning("model path not found for %s", model_name)
                return False

    try:
        if name.startswith("gpt_oss") or name.startswith("gpt-oss"):
            mod = importlib.import_module("utils.llms.huggingface.gpt_oss_20b")
            loader = getattr(mod, "load_gpt_oss_20b", None)
            if callable(loader):
                logging.getLogger(__name__).info("[gpt-oss] adapter preload start: %s", model_path)
                loader(model_path)  # lru_cache will retain
                try:
                    _ADAPTER_LOADED.add(model_path)
                    _ADAPTER_LOADED.add(os.path.basename(model_path))
                except Exception:
                    logging.getLogger(__name__).exception("[gpt-oss] adapter tracking add failed")
                logging.getLogger(__name__).info("[gpt-oss] adapter preload done: %s", model_path)
                return True
            return False
        if "qwen" in name:
            mod = importlib.import_module("utils.llms.huggingface.qwen_7b")
            loader = getattr(mod, "load_qwen_instruct_7b", None)
            if callable(loader):
                loader(model_path)  # lru_cache will retain
                try:
                    _ADAPTER_LOADED.add(model_path)
                    _ADAPTER_LOADED.add(os.path.basename(model_path))
                except Exception:
                    pass
                return True
            return False
    except Exception:
        logging.getLogger(__name__).exception("adapter preload failed for %s", model_name)
        return False
    return False


def _unload_via_adapters(model_name: str) -> bool:
    name = (model_name or "").lower()
    ok = False
    try:
        if name.startswith("gpt_oss") or name.startswith("gpt-oss"):
            mod = importlib.import_module("utils.llms.huggingface.gpt_oss_20b")
            loader = getattr(mod, "load_gpt_oss_20b", None)
            if hasattr(loader, "cache_clear"):
                try:
                    loader.cache_clear()  # type: ignore[attr-defined]
                    ok = True
                except Exception:
                    pass
        if "qwen" in name:
            mod = importlib.import_module("utils.llms.huggingface.qwen_7b")
            loader = getattr(mod, "load_qwen_instruct_7b", None)
            if hasattr(loader, "cache_clear"):
                try:
                    loader.cache_clear()  # type: ignore[attr-defined]
                    ok = True
                except Exception:
                    pass
    except Exception:
        logging.getLogger(__name__).exception("adapter unload failed for %s", model_name)
    # remove tracking
    try:
        resolved = _resolve_model_fs_path(model_name)
        _ADAPTER_LOADED.discard(resolved)
        _ADAPTER_LOADED.discard(os.path.basename(resolved))
    except Exception:
        pass
    # Best-effort GPU/CPU cleanup
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        logging.getLogger(__name__).exception("torch cuda cleanup failed after adapter unload")
    try:
        gc.collect()
    except Exception:
        logging.getLogger(__name__).exception("gc collect failed after adapter unload")
    return ok

# ================================================================
# External script helpers (download / training)
# ================================================================

_DOWNLOAD_SCRIPT = "/home/work/CoreIQ/gpu_use/Qwen/custom_scripts/download_qwen_model.py"
_TRAIN_SCRIPT = "/home/work/CoreIQ/gpu_use/Qwen/custom_scripts/train_qwen_rag.py"


def _run_command(cmd: list[str]) -> Tuple[int, str]:
    """Run a shell command and capture (returncode, stdout+stderr)."""
    import subprocess, shlex

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out, _ = proc.communicate()
    return proc.returncode, out


# ===== Public service functions =====

def download_model(body: DownloadModelBody) -> Dict[str, Any]:
    cmd = ["python", _DOWNLOAD_SCRIPT, "--repo", body.repo, "--name", body.name]
    code, out = _run_command(cmd)
    if code != 0:
        return {"success": False, "message": "download failed", "log": out}
    return {"success": True, "message": "download completed", "log": out}


def insert_base_model(body: InsertBaseModelBody) -> Dict[str, Any]:
    """
    기본 모델 등록(단일 레코드):
      - 항상 category='all', subcategory=NULL 로 한 행만 저장
      - model_path는 STORAGE_ROOT 하위 상대경로(폴더명)로 저장
      - 동일 name 존재 시 upsert
    """
    _migrate_llm_models_if_needed()
    base_name = body.name.strip()
    provided = (body.model_path or "").strip()
    rel_folder = _normalize_model_path_input(provided or base_name)
    abs_path = os.path.join(STORAGE_ROOT, rel_folder)
    cfg_ok = os.path.isfile(os.path.join(abs_path, "config.json"))

    conn = _connect(); cur = conn.cursor()
    try:
        cur.execute("SELECT id FROM llm_models WHERE name=?", (base_name,))
        row = cur.fetchone()
        if row:
            cur.execute(
                """
                UPDATE llm_models
                   SET provider=?, model_path=?, category='all', subcategory=NULL, type='base', is_active=1
                 WHERE id=?
                """,
                (body.provider, rel_folder, int(row[0]))
            )
            mdl_id = int(row[0]); existed = True
        else:
            cur.execute(
                """
                INSERT INTO llm_models(provider,name,revision,model_path,category,subcategory,type,is_default,is_active)
                VALUES(?,?,?,?, 'all', NULL, 'base', 0, 1)
                """,
                (body.provider, base_name, 0, rel_folder)
            )
            mdl_id = int(cur.lastrowid); existed = False
        conn.commit()
    finally:
        conn.close()

    note = "ok" if cfg_ok else "경고: 모델 폴더에 config.json이 보이지 않습니다."
    return {"success": True, "inserted": [{"id": mdl_id, "name": base_name, "category":"all", "model_path": rel_folder, "exists": existed}], "pathChecked": cfg_ok, "note": note}


def train_model(body: TrainModelBody) -> Dict[str, Any]:
    output_dir = os.path.join(STORAGE_ROOT, body.ft_name)
    cmd = [
        "python", _TRAIN_SCRIPT,
        "--csv", body.csv,
        "--base_name", body.base_name,
        "--ft_name", body.ft_name,
        "--epochs", str(body.epochs),
        "--batch_size", str(body.batch_size),
        "--lr", str(body.lr),
    ]
    # 비동기로 실행 – 파이썬 백그라운드 프로세스 & job record 추가
    import subprocess, json as _jsonlib
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    # fine_tune_jobs row 생성
    conn = _connect()
    cur = conn.cursor()
    cur.execute("""
      INSERT INTO fine_tune_jobs(provider_job_id, dataset_id, hyperparameters, status, started_at)
      VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
    """, (f"pid:{proc.pid}", None, _json(body.dict()), "running"))
    job_id = cur.lastrowid
    conn.commit()
    conn.close()

    return {"success": True, "jobId": job_id, "pid": proc.pid}


def infer_local(body: InferBody) -> Dict[str, Any]:
    prompt = f"{body.context.strip()}\n위 내용을 참고하여 응답해 주세요\nQuestion: {body.question.strip()}"
    answer = _simple_generate(prompt, body.modelName, body.max_tokens, body.temperature)
    return {"success": True, "answer": answer}


# ===== Service functions (구현) =====
def set_topk_settings(topk: int) -> Dict[str, Any]:
    global _DEFAULT_TOPK  # noqa: PLW0603
    _DEFAULT_TOPK = int(topk)
    _set_cache(RAG_TOPK_CACHE_KEY, _json({"topK": _DEFAULT_TOPK}), "llm_admin")
    return {"success": True}


def get_model_list(category: str) -> Dict[str, Any]:
    # 카테고리 정규화
    category = _norm_category(category)
    _ensure_models_from_fs(category)

    conn = _connect()
    try:
        cur = conn.cursor()
        if category == "all":
            cur.execute(
                """
                SELECT id, name, type, category, model_path
                FROM llm_models
                WHERE is_active=1 AND category='all'
                ORDER BY trained_at DESC, id DESC
                """
            )
        else:
            cur.execute(
                """
                SELECT id, name, type, category, model_path
                FROM llm_models
                WHERE is_active=1 AND (category=? OR category='all')
                ORDER BY trained_at DESC, id DESC
                """,
                (category,)
            )
        rows = cur.fetchall()
    finally:
        conn.close()

    # Compute loaded set by resolved paths and basenames (category-agnostic)
    try:
        loaded_keys = set(_MODEL_MANAGER.list_loaded()) | set(_ADAPTER_LOADED)
        basenames = {os.path.basename(k) for k in loaded_keys}
        loaded_keys |= basenames
    except Exception:
        logging.getLogger(__name__).exception("failed to gather loaded keys")
        loaded_keys = set()

    active_name = _active_model_name_for_category(category)
    models = []
    for r in rows:
        # base 중복 방지: type='base'이며 category!='all'이면 화면에서 숨김
        row_type = str((r["type"] or "")).lower()
        row_cat  = str((r["category"] or "")).lower()
        if (row_type == "base") and (row_cat != "all"):
            continue
        name = r["name"]
        cand = _resolve_model_path_for_name(name) or _resolve_model_fs_path(name) or name
        loaded_flag = (cand in loaded_keys) or (os.path.basename(cand) in loaded_keys) or _is_model_loaded(name)
        models.append({
            "id": r["id"],
            "name": name,
            "loaded": bool(loaded_flag),
            "active": (name == active_name) and bool(loaded_flag),
        })
    if not models:
        models = [{"id": 0, "name": "None", "loaded": False, "active": False}]
    return {"category": category, "models": models}


def load_model(category: str, model_name: str) -> Dict[str, Any]:
    """
    카테고리별 활성 모델을 메모리에 로드하고 active 로 설정한다.
    이미 로드된 경우에도 active 만 갱신한다.
    """
    logger = logging.getLogger(__name__)
    try:
        category = _norm_category(category)
        row = _lookup_model_by_name(model_name)

        # If not loaded, load it
        if not _is_model_loaded(model_name):
            try:
                lower = (model_name or "").lower()
                if lower.startswith("gpt-oss") or lower.startswith("gpt_oss"):
                    # gpt-oss: 어댑터 경로만 사용
                    pre_ok = _preload_via_adapters(model_name)
                    if not pre_ok:
                        raise RuntimeError("adapter preload also failed")
                else:
                    try:
                        # Resolve best candidate path
                        candidate = _resolve_model_path_for_name(model_name) or _resolve_model_fs_path(model_name)
                        _MODEL_MANAGER.load(candidate)
                    except Exception:
                        logging.getLogger(__name__).exception("manager load failed, trying adapter preload")
                        # Prefer central adapters.preload_adapter_model if available
                        try:
                            from utils.llms.adapters import preload_adapter_model  # type: ignore
                            pre_ok = preload_adapter_model(model_name)
                        except Exception:
                            pre_ok = _preload_via_adapters(model_name)
                        if not pre_ok:
                            raise RuntimeError("adapter preload also failed")
            except Exception:
                logger.exception("failed to load model: %s", model_name)
                return {"success": False, "message": f"모델 로드 실패: {model_name}", "category": category, "modelName": model_name}

        # Set active for category
        _set_active_model_for_category(category, model_name)

        # DB is_active 동기화(동일 경로 전체)
        try:
            row = _lookup_model_by_name(model_name)
            rel_path = (row["model_path"] if row else None) or _normalize_model_path_input(os.path.basename((_resolve_model_path_for_name(model_name) or model_name)))
            if rel_path:
                _db_set_active_by_path(rel_path, True)
        except Exception:
            logging.getLogger(__name__).exception("is_active sync on load failed")

        message = "모델 로드 완료"
        if row is None:
            message += " (주의: DB에 모델 메타가 없어 로컬 경로 기준 처리)"
        return {"success": True, "message": message, "category": category, "modelName": model_name, "loaded": True}
    except Exception:
        logger.exception("unexpected error in load_model")
        return {"success": False, "message": "예상치 못한 오류로 작업에 실패했습니다.", "category": category, "modelName": model_name}


def unload_model(model_name: str) -> Dict[str, Any]:
    """Explicitly unload a model from memory (manager + adapters)."""
    logger = logging.getLogger(__name__)
    was_loaded = _is_model_loaded(model_name)
    # 각 단계는 독립적으로 시도하고, 일부 실패해도 최종 상태로 성공 여부 판단
    try:
        try:
            # Try unloading by both name and resolved path to avoid key mismatch
            _MODEL_MANAGER.unload(model_name)
            try:
                candidate = _resolve_model_path_for_name(model_name) or _resolve_model_fs_path(model_name)
                _MODEL_MANAGER.unload(candidate)
            except Exception:
                pass
        except Exception:
            logging.getLogger(__name__).exception("manager unload failed")
        try:
            _unload_via_adapters(model_name)
        except Exception:
            logging.getLogger(__name__).exception("adapter unload failed")
        try:
            _clear_local_cache_entries(model_name)
            try:
                candidate = _resolve_model_path_for_name(model_name) or _resolve_model_fs_path(model_name)
                _clear_local_cache_entries(candidate)
            except Exception:
                pass
        except Exception:
            logging.getLogger(__name__).exception("local cache clear failed")
    finally:
        now_loaded = _is_model_loaded(model_name)
    ok = not now_loaded
    # DB is_active 동기화(동일 경로 전체)
    try:
        row = _lookup_model_by_name(model_name)
        rel_path = (row["model_path"] if row else None)
        if not rel_path:
            resolved = _resolve_model_path_for_name(model_name) or _resolve_model_fs_path(model_name)
            rel_path = _to_rel_under_storage_root(resolved)
        if rel_path:
            _db_set_active_by_path(rel_path, False)
    except Exception:
        logging.getLogger(__name__).exception("is_active sync on unload failed")
    return {"success": bool(ok), "message": ("언로드 완료" if was_loaded and ok else "이미 언로드됨"), "modelName": model_name}


# ===== 기본 모델 매핑(테스크/서브테스크) =====
class DefaultModelBody(BaseModel):
    category: str
    subcategory: Optional[str] = None
    modelName: str


def set_default_model(body: DefaultModelBody) -> Dict[str, Any]:
    _migrate_llm_models_if_needed()
    cat = _norm_category(body.category)
    sub = (body.subcategory or "").strip().lower() or None
    conn = _connect(); cur = conn.cursor()
    try:
        cur.execute("SELECT id FROM llm_models WHERE name=?", (body.modelName,))
        row = cur.fetchone()
        if not row:
            return {"success": False, "message": f"모델이 없습니다: {body.modelName}"}
        model_id = int(row[0])
        cur.execute(
            """
            INSERT INTO llm_task_defaults(category, subcategory, model_id)
            VALUES(?,?,?)
            ON CONFLICT(category, IFNULL(subcategory,'')) DO UPDATE SET model_id=excluded.model_id, updated_at=CURRENT_TIMESTAMP
            """,
            (cat, sub, model_id)
        )
        conn.commit()
        return {"success": True, "category": cat, "subcategory": sub, "modelId": model_id, "modelName": body.modelName}
    finally:
        conn.close()


def get_default_model(category: str, subcategory: Optional[str] = None) -> Dict[str, Any]:
    cat = _norm_category(category)
    sub = (subcategory or "").strip().lower() or None
    conn = _connect(); cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT m.id, m.name FROM llm_task_defaults d
            JOIN llm_models m ON m.id=d.model_id
            WHERE d.category=? AND IFNULL(d.subcategory,'')=IFNULL(?, '')
            LIMIT 1
            """,
            (cat, sub)
        )
        row = cur.fetchone()
        model = {"id": row[0], "name": row[1]} if row else None
        return {"category": cat, "subcategory": sub, "model": model}
    finally:
        conn.close()


def select_model_for_task(category: str, subcategory: Optional[str] = None) -> Optional[str]:
    """
    우선순위:
      1) llm_task_defaults 매핑
      2) 해당 과업(및 서브테스크) 활성 모델
      3) category='all' 활성 베이스
    반환: 모델 name (없으면 None)
    """
    cat = _norm_category(category)
    sub = (subcategory or "").strip().lower() or None
    conn = _connect(); cur = conn.cursor()
    try:
        # 1) explicit default mapping
        cur.execute(
            """
            SELECT m.name
              FROM llm_task_defaults d JOIN llm_models m ON m.id=d.model_id
             WHERE d.category=? AND IFNULL(d.subcategory,'')=IFNULL(?, '')
             LIMIT 1
            """,
            (cat, sub)
        )
        r = cur.fetchone()
        if r:
            return r[0]

        # 2) task-specific active model (doc_gen with subcategory first)
        if cat == "doc_gen" and sub:
            cur.execute(
                """
                SELECT name FROM llm_models
                 WHERE is_active=1 AND category='doc_gen' AND IFNULL(subcategory,'')=?
                 ORDER BY trained_at DESC, id DESC LIMIT 1
                """,
                (sub,)
            )
            r = cur.fetchone()
            if r:
                return r[0]

        cur.execute(
            """
            SELECT name FROM llm_models
             WHERE is_active=1 AND category=?
             ORDER BY trained_at DESC, id DESC LIMIT 1
            """,
            (cat,)
        )
        r = cur.fetchone()
        if r:
            return r[0]

        # 3) fallback base(all)
        cur.execute(
            """
            SELECT name FROM llm_models
             WHERE is_active=1 AND category='all'
             ORDER BY trained_at DESC, id DESC LIMIT 1
            """
        )
        r = cur.fetchone()
        return r[0] if r else None
    finally:
        conn.close()


def unload_model_for_category(category: str, model_name: str) -> Dict[str, Any]:
    """Unload a model and clear active cache if it matches the category."""
    category = _norm_category(category)
    res = unload_model(model_name)
    try:
        active_name = _active_model_name_for_category(category)
        if active_name == model_name:
            _set_cache(ACTIVE_MODEL_CACHE_KEY_PREFIX + category, _json({"modelName": None}), "llm_admin")
    except Exception:
        logging.getLogger(__name__).exception("failed to clear active cache on unload for %s", category)
    res.update({"category": category, "loaded": False})
    return res


def list_loaded_models() -> Dict[str, Any]:
    try:
        return {"loaded": _MODEL_MANAGER.list_loaded()}
    except Exception:
        logging.getLogger(__name__).exception("failed to list loaded models")
        return {"loaded": []}


def compare_models(payload: CompareModelsBody) -> Dict[str, Any]:
    """
    event_logs(event='model_eval')에 저장된 최근 테스트 결과 중,
    요청 category 기준으로 모델별 최신 결과 최대 3개 반환.
    """
    conn = _connect()
    cur = conn.cursor()
    cur.execute("""
      SELECT metadata, occurred_at FROM event_logs
      WHERE event='model_eval'
      ORDER BY occurred_at DESC, id DESC
      LIMIT 200
    """)
    rows = cur.fetchall()
    conn.close()

    results: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        try:
            meta = json.loads(r["metadata"])
        except Exception:
            continue
        if meta.get("category") != payload.category:
            continue
        if payload.modelId and meta.get("modelId") != payload.modelId:
            continue
        if payload.promptId and meta.get("promptId") != payload.promptId:
            continue
        if payload.prompt and meta.get("promptText") != payload.prompt:
            continue

        mname = meta.get("modelName", f"model-{meta.get('modelId','?')}")
        # 모델별 최신 1개만 유지
        if mname not in results:
            results[mname] = {
                "modelId": meta.get("modelId"),
                "modelName": mname,
                "answer": meta.get("answer", ""),
                "rougeScore": meta.get("rougeScore", None) or 0,
                "occurred_at": r["occurred_at"],
            }
        if len(results) >= 3:
            break

    model_list = sorted(results.values(), key=lambda x: x["occurred_at"], reverse=True)[:3]
    return {"modelList": model_list}


def list_prompts(category: str, subtask: Optional[str] = None) -> Dict[str, Any]:
    category = _norm_category(category)
    subtask = _subtask_key(subtask)
    conn = _connect()
    cur = conn.cursor()
    if subtask:
        cur.execute(
            """
          SELECT id, name, content, subtask FROM system_prompt_template
          WHERE category=? AND ifnull(subtask,'')=? AND ifnull(is_active,1)=1
          ORDER BY id DESC
        """,
            (category, subtask),
        )
    else:
        cur.execute(
            """
          SELECT id, name, content, subtask FROM system_prompt_template
          WHERE category=? AND ifnull(is_active,1)=1
          ORDER BY id DESC
        """,
            (category,),
        )
    rows = cur.fetchall()
    conn.close()
    prompt_list = [
        {"promptId": r["id"], "title": r["name"], "prompt": r["content"], "subtask": r["subtask"]}
        for r in rows
    ]
    return {"category": category, "subtask": subtask or None, "promptList": prompt_list}


def create_prompt(category: str, subtask: Optional[str], body: CreatePromptBody) -> Dict[str, Any]:
    start = time.time()
    category = _norm_category(category)
    subtask = _subtask_key(subtask)
    conn = _connect()
    cur = conn.cursor()
    # 템플릿 생성
    required_vars = [v.key for v in body.variables] if body.variables else []
    cur.execute("""
      INSERT INTO system_prompt_template(name, category, content, subtask, required_vars, is_active)
      VALUES(?,?,?,?,?,1)
    """, (body.title, category, body.prompt, (subtask or None), _json(required_vars)))
    template_id = cur.lastrowid

    # 변수/매핑 생성
    if body.variables:
        for v in body.variables:
            cur.execute("""
              INSERT INTO system_prompt_variables(type, key, value, description)
              VALUES(?,?,?,?)
            """, (v.type, v.key, v.value or "", f"Prompt var for {body.title}"))
            var_id = cur.lastrowid
            cur.execute("""
              INSERT INTO prompt_mapping(template_id, variable_id)
              VALUES(?,?)
            """, (template_id, var_id))

    conn.commit()
    conn.close()
    return {"success": True, "durations": int((time.time() - start) * 1000)}


def _fetch_prompt_full(prompt_id: int) -> Tuple[sqlite3.Row, List[Dict[str, Any]]]:
    conn = _connect()
    cur = conn.cursor()
    cur.execute("SELECT * FROM system_prompt_template WHERE id=? AND ifnull(is_active,1)=1", (prompt_id,))
    tmpl = cur.fetchone()
    if not tmpl:
        conn.close()
        raise ValueError("존재하지 않거나 비활성화된 프롬프트입니다.")
    cur.execute("""
      SELECT v.id, v.type, v.key, v.value, v.description
      FROM system_prompt_variables v
      JOIN prompt_mapping m ON m.variable_id=v.id
      WHERE m.template_id=?
    """, (prompt_id,))
    vars_rows = cur.fetchall()
    conn.close()
    variables = [{"id": r["id"], "type": r["type"], "key": r["key"], "value": r["value"], "description": r["description"]}
                 for r in vars_rows]
    return tmpl, variables


def get_prompt(prompt_id: int) -> Dict[str, Any]:
    tmpl, variables = _fetch_prompt_full(prompt_id)
    return {
        "promptId": tmpl["id"],
        "title": tmpl["name"],
        "prompt": tmpl["content"],
        "category": tmpl["category"],
        "subtask": tmpl["subtask"],
        "variables": [{"key": v["key"], "value": v["value"], "type": v["type"]} for v in variables],
    }


def update_prompt(prompt_id: int, body: UpdatePromptBody) -> Dict[str, Any]:
    conn = _connect()
    cur = conn.cursor()

    # 템플릿 업데이트
    if body.title is not None:
        cur.execute("UPDATE system_prompt_template SET name=? WHERE id=?", (body.title, prompt_id))
    if body.prompt is not None:
        cur.execute("UPDATE system_prompt_template SET content=? WHERE id=?", (body.prompt, prompt_id))

    # 변수 업데이트(간단: 모두 재구성)
    if body.variables is not None:
        # 기존 매핑/변수 제거
        cur.execute("SELECT variable_id FROM prompt_mapping WHERE template_id=?", (prompt_id,))
        var_ids = [r["variable_id"] for r in cur.fetchall()]
        cur.execute("DELETE FROM prompt_mapping WHERE template_id=?", (prompt_id,))
        if var_ids:
            cur.execute(f"DELETE FROM system_prompt_variables WHERE id IN ({','.join('?'*len(var_ids))})", var_ids)
        # 신규 삽입
        required_vars = [v.key for v in body.variables]
        cur.execute("UPDATE system_prompt_template SET required_vars=? WHERE id=?", (_json(required_vars), prompt_id))
        for v in body.variables:
            cur.execute("""
              INSERT INTO system_prompt_variables(type, key, value, description)
              VALUES(?,?,?,?)
            """, (v.type, v.key, v.value or "", f"Prompt var for {body.title or ''}".strip()))
            var_id = cur.lastrowid
            cur.execute("INSERT INTO prompt_mapping(template_id, variable_id) VALUES(?,?)", (prompt_id, var_id))

    conn.commit()
    conn.close()
    return {"success": True}


def delete_prompt(prompt_id: int) -> Dict[str, Any]:
    conn = _connect()
    cur = conn.cursor()
    # 하드 삭제(요청 명확성 고려). 필요시 소프트 삭제로 변경 가능.
    cur.execute("DELETE FROM prompt_mapping WHERE template_id=?", (prompt_id,))
    cur.execute("DELETE FROM system_prompt_template WHERE id=?", (prompt_id,))
    conn.commit()
    conn.close()
    return {"success": True}


def _fill_template(content: str, variables: Dict[str, Any]) -> str:
    # 템플릿 내 {{key}} 치환
    out = content
    for k, v in (variables or {}).items():
        out = out.replace("{{" + k + "}}", str(v))
    return out


def test_prompt(prompt_id: int, body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    프롬프트 테스트 실행:
      1) 템플릿/변수 로드 후 치환
      2) 카테고리별 활성모델 선택
      3) 로컬 모델 추론(가능 시) -> 답변 생성
      4) event_logs에 평가 JSON 저장(rougeScore는 제공되면 사용, 아니면 0)
    요청 body 예시:
      {
        "variables": {"date":"2025-01-01", "location":"서울"},
        "category": "summary",
        "modelName": "Qwen2.5-7B-RAG-FT",
        "reference": "정답 텍스트(있을 때만)",
        "max_tokens": 512,
        "temperature": 0.7
      }
    """
    body = body or {}
    variables = body.get("variables", {}) or {}
    category = body.get("category") or "summary"
    model_name = body.get("modelName") or _active_model_name_for_category(category) or "Qwen2.5-7B-Instruct-1M"
    max_tokens = int(body.get("max_tokens", 512))
    temperature = float(body.get("temperature", 0.7))

    tmpl, tmpl_vars = _fetch_prompt_full(prompt_id)
    required = json.loads(tmpl["required_vars"] or "[]")
    # 필수 변수 체크(간단)
    missing = [k for k in required if k not in variables]
    if missing:
        return {"success": False, "error": f"필수 변수 누락: {', '.join(missing)}"}

    prompt_text = _fill_template(tmpl["content"], variables)

    # 간단 추론
    answer = _simple_generate(prompt_text, model_name, max_tokens=max_tokens, temperature=temperature)

    # (선택) ROUGE 점수 산출 – reference 제공 시 1-gram overlap만 간단 계산
    rouge = 0
    ref = body.get("reference")
    if isinstance(ref, str) and ref.strip():
        try:
            ref_tokens = ref.strip().split()
            ans_tokens = (answer or "").strip().split()
            if ref_tokens:
                overlap = len(set(ref_tokens) & set(ans_tokens))
                rouge = int(100 * overlap / len(set(ref_tokens)))
        except Exception:
            rouge = 0

    # 로그 적재
    # 모델 메타(ID) 조회
    row = _lookup_model_by_name(model_name)
    model_id = int(row["id"]) if row else None

    meta = {
        "category": category,
        "promptId": prompt_id,
        "promptText": prompt_text,
        "variables": variables,
        "modelId": model_id,
        "modelName": model_name,
        "answer": answer,
        "rougeScore": rouge,
    }
    conn = _connect()
    cur = conn.cursor()
    cur.execute("INSERT INTO event_logs(event, metadata, user_id, occurred_at) VALUES(?,?,NULL,CURRENT_TIMESTAMP)",
                ("model_eval", _json(meta)))
    conn.commit()
    conn.close()

    return {"success": True, "result": "테스트 실행 완료", "promptId": prompt_id, "answer": answer, "rougeScore": rouge}
