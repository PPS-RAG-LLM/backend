# service/admin/manage_admin_LLM.py
from __future__ import annotations

import json
import os
import sqlite3
import time
import gc
import logging
import importlib
import re
from functools import lru_cache
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from glob import glob

from pydantic import BaseModel, Field
from utils.database import get_db as _get_db

import torch  # device_map='auto' 사용 시 필요
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
try:
    from peft import PeftModel
except Exception:
    PeftModel = None  # peft 미설치 환경 보호
import socket  # ← 추가
WORKER_ID = f"{socket.gethostname()}:{os.getpid()}"
import shutil

# 예외 삼켜서 로깅만 하는 안전 래퍼
def _mark_loaded(model_name: str):
    try:
        _set_cluster_load_state(model_name, True)
    except Exception:
        logging.getLogger(__name__).exception("cluster load state set failed (mark_loaded)")

def _mark_unloaded(model_name: str):
    try:
        _set_cluster_load_state(model_name, False)
    except Exception:
        logging.getLogger(__name__).exception("cluster load state set failed (mark_unloaded)")

def _set_cluster_load_state(model_name: str, loaded: bool):
    payload = {"modelName": model_name, "loaded": loaded, "worker": WORKER_ID, "ts": _now_iso()}
    _set_cache(f"model_loaded:{model_name}", _json(payload), "llm_cluster")

def _get_cluster_load_state(model_name: str) -> bool:
    raw = _get_cache(f"model_loaded:{model_name}")
    if not raw:
        return False
    try:
        data = json.loads(raw)
        # 선택:  N초 이내만 유효로 보는 TTL 로직 추가
        return bool(data.get("loaded"))
    except Exception:
        return False


# ===== In-memory temp settings (kept for backward-compat) =====
_DEFAULT_TOPK: int = 5

# ===== Constants =====
# Backend root (상대 경로 기준 루트)
BASE_BACKEND = Path(__file__).resolve().parents[2]   # .../backend
SERVICE_ROOT = BASE_BACKEND / "service"
# 신규 표준 저장소: ./service/storage/models
STORAGE_ROOT = SERVICE_ROOT / "storage" / "models"

# 레거시 위치(호환용): ./storage/models
LEGACY_STORAGE_ROOT = BASE_BACKEND / "storage" / "models"

# DB 경로(환경변수로 오버라이드 가능)
os.environ.setdefault("COREIQ_DB", str(BASE_BACKEND / "storage" / "pps_rag.db"))
DB_PATH = os.getenv("COREIQ_DB")

ACTIVE_MODEL_CACHE_KEY_PREFIX = "active_model:"  # e.g. active_model:qa

RAG_TOPK_CACHE_KEY = "rag_topk"

# 허용 루트 가드
_ALLOWED_ROOTS = [
    # 서비스 표준
    str((BASE_BACKEND / "service" / "storage" / "models").resolve()),
    # 레거시
    str((BASE_BACKEND / "storage" / "models").resolve()),
    # 컨테이너 마운트(있다면)
    "/storage/models",
]

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
    title: str  # 서브카테고리 표시용 이름(=template.name)
    prompt: str  # system prompt
    userPrompt: Optional[str] = None  # user prompt -> template.sub_content
    variables: List[PromptVariable] = []


class UpdatePromptBody(BaseModel):
    title: Optional[str] = None
    prompt: Optional[str] = None           # system prompt
    userPrompt: Optional[str] = None       # user prompt
    variables: Optional[List[PromptVariable]] = None


# (moved) CompareModelsBody → service/admin/manage_test_LLM.py

# === NEW: model download / train / infer request bodies ===

# class DownloadModelBody(BaseModel):
#     repo: str = Field(..., description="HuggingFace repo id, e.g. Qwen/Qwen2.5-7B-Instruct-1M")
#     name: str = Field(..., description="Local folder name to store under STORAGE_ROOT")


class TrainModelBody(BaseModel):
    csv: str = Field(..., description="Path to training csv file")
    base_name: str = Field(..., description="Base model folder name under STORAGE_ROOT")
    ft_name: str = Field(..., description="Fine-tuned model folder name under STORAGE_ROOT")
    epochs: int = 3
    batch_size: int = 4
    lr: float = 2e-4


# (moved) InferBody → service/admin/manage_test_LLM.py


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

class DeleteModelBody(BaseModel):
    modelName: str


# 공통 정규화 함수 추가
def _canon_storage_path(p: str) -> str:
    """
    같은 모델 디렉터리를 항상 /storage/models/<basename> 로 통일.
    해당 경로에 config.json 이 있으면 그 경로를 반환, 없으면 원본 반환.
    """
    try:
        p = (p or "").strip().replace("\\", "/")
        base = os.path.basename(p.rstrip("/"))
        cand = f"/storage/models/{base}"
        if os.path.isdir(cand) and os.path.isfile(os.path.join(cand, "config.json")):
            return cand
    except Exception:
        pass
    return p


# ===== DB Helpers =====
def _connect() -> sqlite3.Connection:
    # Delegate to shared database connector (respects config/database paths and pragmas)
    return _get_db()
# ===== Migration / helpers =====
# def _migrate_llm_models_if_needed():
#     conn = _connect()
#     conn.row_factory = sqlite3.Row
#     cur = conn.cursor()
#     try:
#         cur.execute("PRAGMA table_info(llm_models)")
#         cols = {r[1] for r in cur.fetchall()}  # name is at index 1
#         cur.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='llm_models'")
#         row = cur.fetchone()
#         ddl = row[0] if row else ""
#         need_sub = "subcategory" not in cols
#         need_all = ("CHECK" in ddl) and ("'all'" not in ddl)
#         if not (need_sub or need_all):
#             return
#         cur.execute("PRAGMA foreign_keys=off")
#         cur.execute(
#             """
#             CREATE TABLE llm_models__new(
#               id INTEGER PRIMARY KEY AUTOINCREMENT,
#               provider TEXT NOT NULL,
#               name TEXT UNIQUE NOT NULL,
#               revision INTEGER,
#               model_path TEXT,
#               category TEXT NOT NULL CHECK (category IN ('qa','doc_gen','summary','all')),
#               subcategory TEXT,
#               type TEXT NOT NULL CHECK (type IN ('base','lora','full')) DEFAULT 'base',
#               is_default BOOLEAN NOT NULL DEFAULT 0,
#               is_active BOOLEAN NOT NULL DEFAULT 1,
#               trained_at DATETIME,
#               created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
#             )
#             """
#         )
#         # Copy common columns
#         keep = [
#             "id","provider","name","revision","model_path","category","type","is_default","is_active","trained_at","created_at"
#         ]
#         present = []
#         for c in keep:
#             try:
#                 cur.execute(f"SELECT 1 FROM llm_models LIMIT 1")
#                 present.append(c)
#             except Exception:
#                 pass
#         sel = ", ".join([c for c in keep if c in cols])
#         if sel:
#             cur.execute(f"INSERT INTO llm_models__new({sel}) SELECT {sel} FROM llm_models")
#         cur.execute("DROP TABLE llm_models")
#         cur.execute("ALTER TABLE llm_models__new RENAME TO llm_models")
#         cur.execute("PRAGMA foreign_keys=on")
#         conn.commit()
#     except Exception:
#         logging.getLogger(__name__).exception("llm_models migration failed")
#     finally:
#         conn.close()


def _normalize_model_path_input(p: str) -> str:
    s = (p or "").strip().replace("\\","/")
    if not s:
        return s
    prefixes = (
        "storage/models/",
        "./storage/models/",
        "/home/work/CoreIQ/backend/storage/models/",
    )
    for pref in prefixes:
        if s.startswith(pref):
            s = s[len(pref):]
            break
    if "/storage/models/" in s:
        s = s.split("/storage/models/", 1)[1]
    return s.strip("/")


def _sanitize_name(name: str) -> str:
    return (name or "").strip().replace("-", "_").replace("/", "_")


def _std_rel_path_for_name(model_name: str) -> str:
    # DB에 저장될 규칙 경로 (반드시 ./ 로 시작)
    return f"./service/storage/models/local_{_sanitize_name(model_name)}"


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


# ===== 새로운 로딩 로직 (일원화) =====
def _is_pathlike(s: str) -> bool:
    if not s: 
        return False
    return os.path.isabs(s) or s.startswith("./") or s.startswith("../") or "/" in s

def _detect_gguf(base: str) -> Tuple[bool, Optional[str]]:
    """base가 .gguf 파일이거나 gguf를 포함한 디렉터리인지 판별."""
    if not base:
        return False, None
    if base.endswith(".gguf") and os.path.exists(base):
        return True, base
    if os.path.isdir(base):
        files = sorted(glob(os.path.join(base, "*.gguf")))
        if files:
            return True, files[0]
    return False, None

def _fetch_llm_and_ft(conn, model_name: str) -> Tuple[Optional[dict], Optional[dict]]:
    cur = conn.cursor()
    cur.execute("SELECT * FROM llm_models WHERE name=?", (model_name,))
    r = cur.fetchone()
    llm = dict(r) if r else None
    if not llm:
        return None, None
    # 최신 활성 FT 1건 우선, 없으면 최신 1건
    cur.execute("""
        SELECT * FROM fine_tuned_models
         WHERE model_id=? AND IFNULL(is_active,1)=1
         ORDER BY id DESC LIMIT 1
    """, (llm["id"],))
    ft = cur.fetchone()
    if not ft:
        cur.execute("""
            SELECT * FROM fine_tuned_models
             WHERE model_id=?
             ORDER BY id DESC LIMIT 1
        """, (llm["id"],))
        ft = cur.fetchone()
    return llm, (dict(ft) if ft else None)

def _choose_artifacts(llm: dict, ft: Optional[dict]) -> Dict[str, Any]:
    """DB 컬럼만 사용하여 로딩 아티팩트 결정"""
    llm_type = str(llm.get("type") or "base").upper()
    base = (llm.get("model_path") or "").strip()
    adapter = (llm.get("mather_path") or "").strip()  # 오탈자 컬럼 그대로 지원

    if ft:
        ft_type = str(ft.get("type") or "").upper()
        ft_base = (ft.get("base_model_path") or "").strip()
        ft_repo = (ft.get("provider_model_id") or "").strip()
        ft_adapter = (ft.get("lora_weights_path") or "").strip()

        if ft_type in ("LORA", "QLORA"):
            base = ft_base or base
            adapter = ft_adapter or adapter
            llm_type = ft_type  # 실행 타입을 FT 타입으로
        elif ft_type in ("FULL", "BASE"):
            # FULL/BASE 결과물 우선순위: provider_model_id(경로/레포) > base_model_path > llm.model_path
            if _is_pathlike(ft_repo) or ("/" in ft_repo and not os.path.exists(ft_repo)):  # HF repo 형태 지원
                base = ft_repo
            else:
                base = ft_base or base
            adapter = ""  # FULL/BASE은 어댑터 불필요

    return {
        "base": base,                      # 로드 대상 (로컬 디렉터리 / .gguf / HF repo id)
        "adapter": adapter or None,        # LoRA/QLoRA 어댑터 경로(있을 때만)
        "exec_type": llm_type,             # 'BASE' | 'LORA' | 'QLORA' | 'FULL'
        "provider": llm.get("provider"),
        "name": llm.get("name"),
    }

def _load_with_transformers(base_ref: str, adapter_path: Optional[str], exec_type: str):
    # 로컬 디렉터리면 config.json 확인, 아니면 HF repo로 간주
    if os.path.isdir(base_ref):
        cfg = os.path.join(base_ref, "config.json")
        if not os.path.exists(cfg):
            raise FileNotFoundError(f"[preload] config.json not found in local dir: {base_ref}")
    else:
        # HF repo 검증(오프라인 환경이면 캐시로 해결)
        AutoConfig.from_pretrained(base_ref)

    tok = AutoTokenizer.from_pretrained(base_ref, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_ref,
        torch_dtype="auto",
        device_map="auto",
    )

    if exec_type in ("LORA", "QLORA"):
        if not adapter_path:
            raise RuntimeError("adapter path missing for LORA/QLORA")
        if PeftModel is None:
            raise RuntimeError("peft not available for LORA/QLORA loading")
        # 디렉터리/파일 유효성
        if not os.path.isdir(adapter_path) and not os.path.exists(adapter_path):
            raise FileNotFoundError(f"adapter not found: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)

    return {"model": model, "tokenizer": tok}

def _load_with_llama_cpp(gguf_path: str):
    # llama.cpp 바인딩 연결 지점
    raise NotImplementedError("llama.cpp loader is not wired here; bind your implementation.")

def load_model_unified(model_name: str):
    """통합 모델 로더: 이름 → 메타 조회 → 경로 결정 → 로드"""
    conn = _connect()
    try:
        llm, ft = _fetch_llm_and_ft(conn, model_name)
    finally:
        conn.close()
    if not llm:
        raise RuntimeError(f"unknown model: {model_name}")

    art = _choose_artifacts(llm, ft)
    base = art["base"]
    adapter = art["adapter"]
    exec_type = art["exec_type"]

    if not base:
        raise RuntimeError(f"model_path is empty for model: {model_name}")

    is_gguf, gguf_path = _detect_gguf(base)
    if is_gguf:
        return _load_with_llama_cpp(gguf_path)

    # Transformers 경로/레포 로드
    return _load_with_transformers(base, adapter, exec_type)

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


def get_prompt(prompt_id: int) -> Dict[str, Any]: 
    tmpl, variables = _fetch_prompt_full(prompt_id) 
    return { "promptId": tmpl["id"], "title": tmpl["name"], "prompt": tmpl["content"], "userPrompt": tmpl.get("sub_content") if isinstance(tmpl, dict) else getattr(tmpl, "sub_content", None), "category": tmpl["category"], "subcategory": tmpl["name"], "variables": [{"key": v["key"], "value": v["value"], "type": v["type"]} for v in variables], }

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
    max_tokens = int(body.get("max_tokens", 512))
    temperature = float(body.get("temperature", 0.7))

    tmpl, tmpl_vars = _fetch_prompt_full(prompt_id)
    subcategory = (tmpl["name"] if isinstance(tmpl, dict) else getattr(tmpl, "name", None)) or None

    if body.get("modelName"):
        model_name = body["modelName"]
    else:
        try:
            # helper는 test 모듈의 것을 재사용
            from service.admin.manage_test_LLM import select_model_for_task
            model_name = select_model_for_task(category, subcategory)
        except Exception:
            logging.getLogger(__name__).exception("select_model_for_task failed in admin test_prompt")
            model_name = None
        if not model_name:
            return {"success": False, "error": "기본/활성/베이스 모델을 찾을 수 없습니다. 먼저 기본 모델을 지정하거나 모델을 로드하세요."}

    required = json.loads(tmpl["required_vars"] or "[]")
    missing = [k for k in required if k not in variables]
    if missing:
        return {"success": False, "error": f"필수 변수 누락: {', '.join(missing)}"}

    system_prompt_text = _fill_template(tmpl["content"], variables)
    user_prompt_raw = (tmpl.get("sub_content") if isinstance(tmpl, dict) else getattr(tmpl, "sub_content", None)) or ""
    user_prompt_text = _fill_template(user_prompt_raw, variables)
    prompt_text = (system_prompt_text + ("\n" + user_prompt_text if user_prompt_text else "")).strip()

    answer = _simple_generate(prompt_text, model_name, max_tokens=max_tokens, temperature=temperature)

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

    row = _lookup_model_by_name(model_name)
    model_id = int(row["id"]) if row else None

    meta = {
        "category": category,
        "subcategory": subcategory,
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
    try:
        cur.execute(
            "INSERT INTO event_logs(event, metadata, user_id, occurred_at) VALUES(?,?,NULL,CURRENT_TIMESTAMP)",
            ("model_eval", _json(meta)),
        )
        conn.commit()
    except Exception:
        logging.getLogger(__name__).exception("failed to insert model_eval event")
    finally:
        conn.close()

    return {"success": True, "result": "테스트 실행 완료", "promptId": prompt_id, "answer": answer, "rougeScore": rouge}


    
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

def _fill_template(content: str, variables: Dict[str, Any]) -> str:
    # 템플릿 내 {{key}} 치환
    out = content
    for k, v in (variables or {}).items():
        out = out.replace("{{" + k + "}}", str(v))
    return out
    

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
    variables = [{"id": r["id"], "type": r["type"], "key": r["key"], "value": r["value"], "description": r["description"]} for r in vars_rows]
    return tmpl, variables

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
        fs_path = str(STORAGE_ROOT / name_or_path)
        try:
            if os.path.isfile(os.path.join(fs_path, "config.json")):
                return fs_path
            # ← 레거시 경로도 체크
            legacy = str(LEGACY_STORAGE_ROOT / name_or_path)
            if os.path.isfile(os.path.join(legacy, "config.json")):
                return legacy
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

                # (변경) gpt-oss도 HF 로더로 직접 로드 허용

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
    """
    절대/상대 무엇이든 실제 폴더(config.json) 있는 경로로 환원.
    우선순위: 규칙 경로 → service root → legacy root → 입력값
    """
    try:
        s = (name_or_path or "").strip().replace("\\", "/")
        # 규칙 경로 후보 (service/local_<name>)
        rule_abs = STORAGE_ROOT / f"local_{_sanitize_name(_strip_category_suffix(os.path.basename(s)))}"
        if (rule_abs / "config.json").is_file():
            return _canon_storage_path(str(rule_abs))

        # ./service/... 상대처리
        if s.startswith("./service/"):
            cand = (BASE_BACKEND / s.lstrip("./"))
            if (cand / "config.json").is_file():
                return _canon_storage_path(str(cand))

        # ./storage/... 상대처리(레거시)
        if s.startswith("./storage/"):
            cand = (BASE_BACKEND / s.lstrip("./"))
            if (cand / "config.json").is_file():
                return _canon_storage_path(str(cand))

        # 이름만 온 경우: service → legacy
        cand = STORAGE_ROOT / os.path.basename(s)
        if (cand / "config.json").is_file():
            return _canon_storage_path(str(cand))
        cand = LEGACY_STORAGE_ROOT / os.path.basename(s)
        if (cand / "config.json").is_file():
            return _canon_storage_path(str(cand))

        # 절대경로 그대로 확인
        if os.path.isabs(s) and os.path.isfile(os.path.join(s, "config.json")):
            return s
    except Exception:
        logging.getLogger(__name__).exception("failed to resolve model fs path")
    return _canon_storage_path(s)

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
        # (변경) gpt-oss도 매니저 경유 HF 로더 사용 허용
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
        {"role": "system", "content": "You are a helpful assistant."},
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
    conn = _connect()
    try:
        cur = conn.cursor()
        cur.execute("SELECT model_path FROM llm_models WHERE name=?", (model_name,))
        row = cur.fetchone()
    except Exception:
        logging.getLogger(__name__).exception("failed to query llm_models for %s", model_name)
        return None
    finally:
        try:
            conn.close()
        except Exception:
            pass

    if not row:
        return None
    val = (row["model_path"] or "").strip().replace("\\", "/")
    if not val:
        return None

    # Handle standardized relative paths like ./service/storage/models/...
    if val.startswith("./"):
        abs_p = (BASE_BACKEND / val.lstrip("./"))
        if (abs_p / "config.json").is_file():
            return _canon_storage_path(str(abs_p))
        std_abs = STORAGE_ROOT / f"local_{_sanitize_name(model_name)}"
        if (std_abs / "config.json").is_file():
            return _canon_storage_path(str(std_abs))
        leg = LEGACY_STORAGE_ROOT / os.path.basename(val)
        if (leg / "config.json").is_file():
            return _canon_storage_path(str(leg))
        return None

    # Absolute or simple name -> resolve via fs helper
    resolved = _resolve_model_fs_path(val)
    return _canon_storage_path(resolved)

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
    """ 상대 경로 반환 """
    try:
        p = str(p)
        # 새 표준(root=SERVICE_ROOT)
        try:
            rp = os.path.relpath(p, str(STORAGE_ROOT))
            if rp not in (".", ""):
                return f"./service/storage/models/{rp}".replace("\\", "/")
        except Exception:
            pass
        # 레거시(root=LEGACY_STORAGE_ROOT)
        try:
            rp = os.path.relpath(p, str(LEGACY_STORAGE_ROOT))
            if rp not in (".", ""):
                return f"./storage/models/{rp}".replace("\\", "/")
        except Exception:
            pass
    except Exception:
        pass
    return (p or "").replace("\\", "/")


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
    cand = STORAGE_ROOT / model_name
    if (cand / "config.json").is_file():
        return str(cand)
    # 4) STORAGE_ROOT/base-name
    cand2 = STORAGE_ROOT / base
    if (cand2 / "config.json").is_file():
        return str(cand2)
    return None

def _preload_via_adapters(model_name: str) -> bool:
    """
    Admin 경로용 프리로드:
      - DB/FS에서 model_path 해석
      - gpt-oss, Qwen 등 utils 전용 로더를 '같은 경로'로 호출해 lru_cache에 올림
      - 그 외는 HF Auto 로더로 로드
      - 성공 시 _ADAPTER_LOADED에 추적 키 추가
    """
    try:
        # 1) 기본 경로 해석 (절대 경로)
        model_path = _db_get_model_path(model_name) or _resolve_model_fs_path(model_name)
        model_path = _canon_storage_path(model_path)  # ← 추가
        raw_path = _db_get_model_path(model_name) or _resolve_model_fs_path(model_path)
        if not (raw_path and os.path.isfile(os.path.join(raw_path, "config.json"))):
            logging.getLogger(__name__).warning("[preload] config.json not found: %s", raw_path)
            return False

        # 2) utils 쪽에서 사용하는 보이는 경로로 매핑 (예: '/storage/models/<basename>')
        def _adapter_visible_path(p: str) -> str:
            try:
                base = os.path.basename(p.rstrip("/"))
                cand = f"/storage/models/{base}"
                return cand if os.path.isdir(cand) and os.path.isfile(os.path.join(cand, "config.json")) else p
            except Exception:
                return p

        adapter_path = _adapter_visible_path(raw_path)

        lower = (model_name or "").lower()

        # 3-a) GPT-OSS는 전용 로더로 (utils의 lru_cache에 적재)
        if lower.startswith(("gpt-oss", "gpt_oss")):
            try:
                from utils.llms.huggingface.gpt_oss_20b import load_gpt_oss_20b
                load_gpt_oss_20b(adapter_path)  # ← utils lru_cache 채움
            except Exception:
                logging.getLogger(__name__).exception("gpt-oss preload failed")
                return False

        # 3-b) Qwen 계열도 전용 로더로 (utils의 lru_cache에 적재)
        elif lower.startswith("qwen"):
            try:
                from utils.llms.huggingface.qwen_7b import load_qwen_instruct_7b
                load_qwen_instruct_7b(adapter_path)  # ← utils lru_cache 채움
            except Exception:
                logging.getLogger(__name__).exception("qwen preload failed")
                return False

        # 3-c) 기타 모델은 HF 로더로 경량 프리로드 (lru_cache는 없지만, 적어도 1회 로컬 캐시/HF weights warmup)
        else:
            try:
                from transformers import AutoTokenizer, AutoModelForCausalLM
                import torch
                tok = AutoTokenizer.from_pretrained(raw_path, trust_remote_code=True, local_files_only=True)
                if tok.pad_token_id is None:
                    if getattr(tok, "eos_token_id", None) is not None:
                        tok.pad_token_id = tok.eos_token_id
                    else:
                        tok.add_special_tokens({"pad_token": "<|pad|>"}); tok.pad_token_id = tok.convert_tokens_to_ids("<|pad|>")
                AutoModelForCausalLM.from_pretrained(
                    raw_path,
                    trust_remote_code=True,
                    device_map="auto",
                    local_files_only=True,
                    torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
                )
            except Exception:
                logging.getLogger(__name__).exception("HF preload failed")
                return False

        # 4) 추적키 추가 (언로드 시 정리 용이)
        _ADAPTER_LOADED.add(raw_path); _ADAPTER_LOADED.add(os.path.basename(raw_path))
        _ADAPTER_LOADED.add(adapter_path); _ADAPTER_LOADED.add(os.path.basename(adapter_path))
        logging.getLogger(__name__).info("[preload] ok: %s (raw=%s, adapter=%s)", model_name, raw_path, adapter_path)
        return True

    except Exception:
        logging.getLogger(__name__).exception("[preload] unexpected failure")
        return False

def _unload_via_adapters(model_name: str) -> bool:
    """
    Admin 경로용 언로드:
      - _ADAPTER_LOADED 추적만 지우고, 메모리 정리만 수행
      - (특정 캐시 클리어 함수 호출 제거)
    """
    try:
        resolved = _resolve_model_fs_path(model_name)
        _ADAPTER_LOADED.discard(resolved)
        _ADAPTER_LOADED.discard(os.path.basename(resolved))
    except Exception:
        pass
    # GPU/CPU 캐시 정리
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        logging.getLogger(__name__).exception("torch cuda cleanup failed after adapter unload")
    try:
        gc.collect()
    except Exception:
        logging.getLogger(__name__).exception("gc collect failed after adapter unload")
    return True

# ================================================================
# External script helpers (download / training)
# ================================================================




def _run_command(cmd: list[str]) -> Tuple[int, str]:
    """Run a shell command and capture (returncode, stdout+stderr)."""
    import subprocess, shlex

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out, _ = proc.communicate()
    return proc.returncode, out


# ===== Public service functions =====

# def insert_base_model(body: InsertBaseModelBody) -> Dict[str, Any]:
#     """
#     기본 모델 등록(단일 레코드):
#       - DB model_path는 항상 './service/storage/models/local_<name>' 로 저장
#       - 폴더 실제 유무는 별도(경고만 표시). 로드는 _resolve 로 판단.
#     """
#     _migrate_llm_models_if_needed()
#     base_name = body.name.strip()
#     rel_folder = _std_rel_path_for_name(base_name)  # 규칙 강제

#     # 파일 존재 점검(둘 다 확인: 신규 표준 / 레거시)
#     abs_new = STORAGE_ROOT / f"local_{_sanitize_name(base_name)}"
#     cfg_ok = (abs_new / "config.json").is_file()
#     if not cfg_ok:
#         abs_old = LEGACY_STORAGE_ROOT / base_name
#         cfg_ok = (abs_old / "config.json").is_file()

#     conn = _connect(); cur = conn.cursor()
#     try:
#         cur.execute("SELECT id FROM llm_models WHERE name=?", (base_name,))
#         row = cur.fetchone()
#         if row:
#             cur.execute(
#                 """
#                 UPDATE llm_models
#                    SET provider=?, model_path=?, category='all', subcategory=NULL, type='base', is_active=1
#                  WHERE id=?
#                 """,
#                 (body.provider, rel_folder, int(row[0]))
#             )
#             mdl_id = int(row[0]); existed = True
#         else:
#             cur.execute(
#                 """
#                 INSERT INTO llm_models(provider,name,revision,model_path,category,subcategory,type,is_default,is_active)
#                 VALUES(?,?,?,?, 'all', NULL, 'base', 0, 1)
#                 """,
#                 (body.provider, base_name, 0, rel_folder)
#             )
#             mdl_id = int(cur.lastrowid); existed = False
#         conn.commit()
#     finally:
#         conn.close()

#     note = "ok" if cfg_ok else "경고: 실제 모델 폴더(config.json) 미확인"
#     return {"success": True, "inserted": [{"id": mdl_id, "name": base_name, "category":"all", "model_path": rel_folder, "exists": existed}], "pathChecked": cfg_ok, "note": note}





# moved to service/admin/manage_test_LLM.py


# ===== Service functions (구현) =====
# def set_topk_settings(topk: int) -> Dict[str, Any]:
#     global _DEFAULT_TOPK  # noqa: PLW0603
#     _DEFAULT_TOPK = int(topk)
#     _set_cache(RAG_TOPK_CACHE_KEY, _json({"topK": _DEFAULT_TOPK}), "llm_admin")
#     return {"success": True}

def get_model_list(category: str, subcategory: Optional[str] = None):
    cat = _norm_category(category)
    sub = (subcategory or "").strip()   # = system_prompt_template.name

    conn = _connect()
    cur = conn.cursor()
    try:
        # --- (ADD) 파인튜닝 모델 id 세트 미리 로딩 ---
        ft_ids: set[int] = set()
        try:
            cur.execute("SELECT DISTINCT model_id FROM fine_tuned_models")
            ft_ids = {int(r["model_id"]) for r in cur.fetchall() if r["model_id"] is not None}
        except Exception:
            ft_ids = set()

        # 1) category=all → 전체(활/비활 포함)
        if cat == "all":
            cur.execute(
                """
                SELECT
                  id, name, provider, category,
                  is_active AS isActive,
                  trained_at, created_at
                  FROM llm_models
                 ORDER BY
                  is_active DESC,
                  trained_at DESC,
                  id DESC
                """
            )
            rows = cur.fetchall()

        # 2) doc_gen + subcategory → 매핑 rouge 점수순
        elif cat == "doc_gen" and sub:
            cur.execute(
                """
                WITH t AS (
                  SELECT id
                    FROM system_prompt_template
                   WHERE category = ?
                     AND name = ?
                     AND IFNULL(is_active,1) = 1
                   ORDER BY IFNULL(is_default,0) DESC, id DESC
                   LIMIT 1
                )
                SELECT
                  m.id,
                  m.name,
                  m.provider,
                  m.category,
                  m.is_active         AS isActive,
                  m.trained_at,
                  m.created_at,
                  IFNULL(pm.rouge_score,-1) AS rougeScore
                FROM llm_models m
                JOIN llm_prompt_mapping pm
                  ON pm.llm_id    = m.id
                 AND pm.prompt_id = (SELECT id FROM t)
                WHERE (m.category = 'doc_gen' OR m.category = 'all')
                ORDER BY IFNULL(pm.rouge_score,-1) DESC,
                         m.trained_at DESC,
                         m.id DESC
                """,
                (cat, sub),
            )
            rows = cur.fetchall()

        # 3) 그 외(qa/summary/doc_gen 전체) → 활성만
        else:
            cur.execute(
                """
                SELECT
                  id, name, provider, category,
                  is_active AS isActive,
                  trained_at, created_at
                  FROM llm_models
                 WHERE is_active = 1
                   AND (category = ? OR category = 'all')
                 ORDER BY trained_at DESC, id DESC
                """,
                (cat,),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    # 현재 카테고리 활성 모델명(캐시)
    try:
        current_active_name = None if cat in ("all", None) else _active_model_name_for_category(cat)
    except Exception:
        current_active_name = None

    models = []
    for i, r in enumerate(rows):
        try:
            cols = r.keys()
        except Exception:
            cols = []
        rouge = r["rougeScore"] if ("rougeScore" in cols) else None

        name = r["name"]

        # ✅ 로드 상태: 클러스터 기준만 사용
        try:
            is_loaded_cluster = _get_cluster_load_state(name)
        except Exception:
            is_loaded_cluster = False

        # active 규칙
        if cat == "doc_gen" and sub:
            active_flag = (i == 0)
        elif cat != "all" and current_active_name:
            active_flag = (name == current_active_name)
        else:
            active_flag = False

        # --- (ADD) fine-tuned 여부 ---
        is_finetuned = bool(r["id"] in ft_ids)

        models.append({
            "id": r["id"],
            "name": name,
            "provider": r["provider"],
            "loaded": bool(is_loaded_cluster),
            "category": r["category"],
            "subcategory": sub if (cat == "doc_gen" and sub) else None,
            "active": bool(active_flag),
            "isActive": bool(r["isActive"]),
            "createdAt": r["created_at"],
            "rougeScore": rouge,
            "isFineTuned": is_finetuned,  # ← 추가 필드 하나만
        })

    return {"category": cat, "models": models}


def lazy_load_if_needed(model_name: str) -> Dict[str, Any]:
    """
    최초 사용시 지연 로딩.
    - FULL은 통짜 로딩, LORA/QLORA는 어댑터 경로 우선 등 기존 규칙 유지
    - 로드 성공 시 클러스터 플래그를 True 로 기록
    """
    try:
        if _is_model_loaded(model_name):
            _mark_loaded(model_name)  # 이미 로컬에 올라와 있어도 클러스터 플래그는 보강
            return {"loaded": True, "message": "already loaded", "modelName": model_name}

        paths = _resolve_paths_for_model(model_name)
        # 우선순위: full -> adapter -> name
        candidate = paths.get("full") or paths.get("adapter") or _resolve_model_fs_path(model_name)

        try:
            _MODEL_MANAGER.load(candidate)
            _mark_loaded(model_name)  # ← 여기 추가
            return {"loaded": True, "message": "loaded", "modelName": model_name}
        except Exception:
            logging.getLogger(__name__).exception("lazy load failed - fallback adapter preload")
            if _preload_via_adapters(model_name):
                _mark_loaded(model_name)  # ← 여기 추가
                return {"loaded": True, "message": "adapter preloaded (fallback)", "modelName": model_name}
            return {"loaded": False, "message": "load failed", "modelName": model_name}
    except Exception:
        logging.getLogger(__name__).exception("lazy_load_if_needed unexpected error")
        return {"loaded": False, "message": "unexpected error", "modelName": model_name}

def load_model(category: str, model_name: str) -> Dict[str, Any]:
    logger = logging.getLogger(__name__)
    try:
        category = _norm_category(category)
        row = _lookup_model_by_name(model_name)

        # Set active for category
        _set_active_model_for_category(category, model_name)

        # DB is_active 동기화
        try:
            row = _lookup_model_by_name(model_name)
            if row and row["model_path"]:
                rel_path = row["model_path"]
            else:
                resolved = _resolve_model_fs_path(model_name)
                rel_path = _to_rel_under_storage_root(resolved)
            if rel_path:
                _db_set_active_by_path(rel_path, True)
        except Exception:
            logging.getLogger(__name__).exception("is_active sync on load failed")

        # ← 반드시 기록 (요청으로 로드될 때만 찍는다: 프리로딩은 여기/지연로딩에서만)
        _mark_loaded(model_name)

        message = "모델 로드 완료"
        if row is None:
            message += " (주의: DB에 모델 메타가 없어 로컬 경로 기준 처리)"
        return {"success": True, "message": message, "category": category, "modelName": model_name, "loaded": True}
    except Exception:
        logger.exception("unexpected error in load_model")
        return {"success": False, "message": "예상치 못한 오류로 작업에 실패했습니다.", "category": category, "modelName": model_name}

def unload_model(model_name: str) -> Dict[str, Any]:
    logger = logging.getLogger(__name__)
    was_loaded = _is_model_loaded(model_name)
    try:
        try:
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

    # DB is_active 동기화(정책에 따라 유지/미유지 선택 가능)
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

    # ← 반드시 기록
    _mark_unloaded(model_name)

    return {"success": bool(ok), "message": ("언로드 완료" if was_loaded and ok else "이미 언로드됨"), "modelName": model_name}



# ===== 기본 모델 매핑(테스크/서브테스크) =====


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

def list_prompts(category: str, subtask: Optional[str] = None) -> Dict[str, Any]:
    category = _norm_category(category)
    subcat = (subtask or "").strip().lower() or None

    conn = _connect()
    cur = conn.cursor()
    try:
        if subcat:
            cur.execute(
                """
                SELECT id, name, system_prompt, user_prompt
                  FROM system_prompt_template
                 WHERE category=? AND lower(name)=? AND ifnull(is_active,1)=1
                 ORDER BY id DESC
                """,
                (category, subcat),
            )
        else:
            cur.execute(
                """
                SELECT id, name, system_prompt, user_prompt
                  FROM system_prompt_template
                 WHERE category=? AND ifnull(is_active,1)=1
                 ORDER BY id DESC
                """,
                (category,),
            )
        rows = cur.fetchall()
    finally:
        conn.close()

    prompt_list = []
    for r in rows:
        prompt_list.append({
            "promptId": r["id"],
            "title": r["name"],                 # 서브카테고리 이름
            "prompt": r["system_prompt"],       # system prompt
            "userPrompt": r["user_prompt"],     # user prompt
            "subcategory": r["name"],           # 명시적으로 내려줌
        })
    return {"category": category, "subcategory": subcat or None, "promptList": prompt_list}





# ==== (추가) LORA/QLORA 베이스/어댑터 경로 인식 강화 ====

def _db_get_model_record(model_name: str) -> Optional[sqlite3.Row]:
    conn = _connect()
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM llm_models WHERE name=?", (model_name,))
        return cur.fetchone()
    finally:
        conn.close()


def _resolve_paths_for_model(model_name: str) -> Dict[str, Optional[str]]:
    """
    모델 이름으로부터 로딩에 필요한 경로를 정리해서 반환.
    - LORA/QLORA: {'base': mather_path(abs), 'adapter': model_path(abs)}
    - FULL:      {'full': model_path(abs)}
    - BASE:      {'base': mather_path(abs)} (실제 로드는 외부 규칙에 따름)
    경로는 가능한 경우 절대경로로 환원.
    """
    row = _db_get_model_record(model_name)
    base_abs = adap_abs = full_abs = None

    def _to_abs_from_rel_or_name(p: Optional[str]) -> Optional[str]:
        if not p:
            return None
        pp = p.replace("\\", "/")
        if pp.startswith("./"):
            cand = (BASE_BACKEND / pp.lstrip("./"))
            return str(cand)
        # 이름/폴더만 온 경우도 지원
        return _resolve_model_fs_path(pp)

    if row:
        t = (row["type"] or "").upper()
        if t in ("LORA", "QLORA"):
            base_abs = _to_abs_from_rel_or_name(row["mather_path"])
            adap_abs = _to_abs_from_rel_or_name(row["model_path"])
            return {"base": base_abs, "adapter": adap_abs, "full": None}
        elif t == "FULL":
            full_abs = _to_abs_from_rel_or_name(row["model_path"]) or _resolve_model_fs_path(model_name)
            return {"base": None, "adapter": None, "full": full_abs}
        elif t == "BASE":
            base_abs = _to_abs_from_rel_or_name(row["mather_path"])
            return {"base": base_abs, "adapter": None, "full": None}

    # DB에 없거나 메타가 빈 경우: 기존 해석으로 폴백
    return {"base": None, "adapter": None, "full": _resolve_model_fs_path(model_name)}


# (기존) _db_get_model_path 는 아래처럼 소폭 조정하여 '단일 경로'가 필요한 코드만 위해 유지
def _db_get_model_path(model_name: str) -> Optional[str]:
    """
    단일 경로가 필요한 호출자 호환용.
    - LORA/QLORA는 어댑터 경로(model_path) 우선 반환
    - FULL은 model_path 또는 규칙 경로 반환
    - BASE는 mather_path 반환(있으면)
    """
    rec = _resolve_paths_for_model(model_name)
    return rec.get("adapter") or rec.get("full") or rec.get("base")




# ===== (추가) 테스크별 디폴트 모델 확인 =====

## moved to service/admin/manage_test_LLM.py: SelectModelQuery


def get_selected_model(q):
    """
    요구사항: 테스크별 default model은 '프롬프트 테이블'에서 확인.
    절차:
      1) system_prompt_template에서 category(및 선택적 name=subcategory) + is_default=1 템플릿 1건을 고른다.
         - 여러 개면 최신 id 우선
      2) 해당 템플릿(prompt_id)에 연결된 llm_prompt_mapping 중 rouge_score가 가장 높은 1건을 선택
      3) 그 llm_id로 llm_models 조회 후 모델 메타 반환
    스키마의 unique 제약(테스크별 default는 1개)은 '확인만' 수행하고, 위반 시에도 가장 높은 rouge_score 1건만 반환.
    """
    category = _norm_category(q.category)
    subcat = (getattr(q, "subcategory", None) or "").strip().lower() or None

    conn = _connect(); cur = conn.cursor()
    try:
        # 1) default 템플릿 선택
        if subcat:
            cur.execute("""
                SELECT id, name FROM system_prompt_template
                 WHERE category=? AND lower(name)=? AND ifnull(is_default,0)=1 AND ifnull(is_active,1)=1
                 ORDER BY id DESC LIMIT 1
            """, (category, subcat))
        else:
            cur.execute("""
                SELECT id, name FROM system_prompt_template
                 WHERE category=? AND ifnull(is_default,0)=1 AND ifnull(is_active,1)=1
                 ORDER BY id DESC LIMIT 1
            """, (category,))
        tmpl = cur.fetchone()
        if not tmpl:
            return {"category": category, "subcategory": subcat, "default": None, "note": "기본 템플릿 없음"}

        prompt_id = int(tmpl["id"])

        # 2) prompt 매핑 중 최고 rouge_score 1건
        cur.execute("""
            SELECT llm_id, prompt_id, rouge_score
              FROM llm_prompt_mapping
             WHERE prompt_id=?
             ORDER BY IFNULL(rouge_score, -1) DESC, llm_id DESC
             LIMIT 1
        """, (prompt_id,))
        mp = cur.fetchone()
        if not mp:
            return {"category": category, "subcategory": tmpl["name"], "default": None, "note": "프롬프트-모델 매핑 없음"}

        # 3) llm_models 메타
        cur.execute("""
            SELECT id, name, provider, type, model_path, mather_path, category, is_active, trained_at, created_at
              FROM llm_models WHERE id=?
        """, (mp["llm_id"],))
        mdl = cur.fetchone()
        if not mdl:
            return {"category": category, "subcategory": tmpl["name"], "default": None, "note": "llm_models에 모델 없음"}

        # 스키마 제약 확인(참고 메시지)
        # 카테고리/서브카테고리별 default가 1개라는 제약을 코드에서는 강제하지 않음(요구대로 '확인만')
        cur.execute("""
            SELECT COUNT(*) AS cnt
              FROM system_prompt_template
             WHERE category=? AND ifnull(is_default,0)=1 AND ifnull(is_active,1)=1
               AND (? IS NULL OR lower(name)=?)
        """, (category, subcat, subcat))
        cnt = (cur.fetchone() or {"cnt": 0})["cnt"]

        return {
            "category": category,
            "subcategory": (tmpl["name"] or None),
            "default": {
                "promptId": prompt_id,
                "llmId": mdl["id"],
                "modelName": mdl["name"],
                "type": mdl["type"],
                "modelPath": mdl["model_path"],
                "matherPath": mdl["mather_path"],
                "rougeScore": mp["rouge_score"],
            },
            "note": ("default 템플릿 다수" if (cnt and cnt > 1) else None)
        }
    finally:
        conn.close()


def _is_under_allowed_roots(path_str: str) -> bool:
    try:
        rp = str(Path(path_str).resolve())
    except Exception:
        return False
    for root in _ALLOWED_ROOTS:
        try:
            if rp == str(Path(root).resolve()) or rp.startswith(str(Path(root).resolve()) + os.sep):
                return True
        except Exception:
            continue
    return False

def _table_exists(conn, table: str) -> bool:
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,))
    return cur.fetchone() is not None
def _collect_fs_delete_targets(model_path_value: str) -> list[str]:
    """
    DB model_path를 기준으로 실제 삭제 후보 경로들을 모두 수집.
    - 상대경로는 backend 루트 기준으로 절대화
    - 베이스네임으로 표준/레거시/마운트 3곳 모두 후보에 포함
    - 중복 제거
    """
    out: list[str] = []
    s = (model_path_value or "").strip().replace("\\", "/")
    if not s:
        return out

    # 1) 입력 자체를 절대화
    try:
        if s.startswith("./"):
            abs1 = str((BASE_BACKEND / s.lstrip("./")).resolve())
        else:
            abs1 = str(Path(s).resolve())
        out.append(abs1)
    except Exception:
        pass

    # 2) 베이스네임 기준 표준/레거시/마운트
    base = os.path.basename(s.rstrip("/"))
    if base:
        std = (BASE_BACKEND / "service" / "storage" / "models" / base).resolve()
        leg = (BASE_BACKEND / "storage" / "models" / base).resolve()
        mnt = Path("/storage/models") / base
        for p in (std, leg, mnt):
            try:
                out.append(str(p.resolve()))
            except Exception:
                out.append(str(p))

    # 3) 중복 제거
    seen = set()
    uniq: list[str] = []
    for p in out:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq

def _purge_dir_contents_safe(dir_path: str, remove_dir_if_empty: bool = False) -> dict:
    """
    dir_path 폴더 '안의 내용'을 전부 삭제(폴더는 기본 유지).
    remove_dir_if_empty=True면 비었을 때 폴더까지 삭제.
    """
    out = {"parent": None, "deleted": [], "skipped": [], "errors": [], "removed_dir": False}
    try:
        if not dir_path:
            out["skipped"].append({"target": dir_path, "reason": "empty-path"})
            return out
        p = Path(dir_path).resolve()
        out["parent"] = str(p)
        if not p.exists() or not p.is_dir():
            out["skipped"].append({"target": str(p), "reason": "not-exists-or-not-dir"})
            return out
        if not _is_under_allowed_roots(str(p)):
            out["skipped"].append({"target": str(p), "reason": "not-under-allowed-roots"})
            return out

        for child in p.iterdir():
            try:
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink(missing_ok=True)
                out["deleted"].append(str(child))
            except Exception as e:
                out["errors"].append({"target": str(child), "error": repr(e)})

        if remove_dir_if_empty:
            try:
                if not any(p.iterdir()):
                    p.rmdir()
                    out["removed_dir"] = True
            except Exception as e:
                out["errors"].append({"target": str(p), "error": f"rmdir-failed: {e}"})
        return out
    except Exception as e:
        logging.getLogger(__name__).exception("purge dir failed: %s", dir_path)
        out["errors"].append({"target": dir_path, "error": repr(e)})
        return out


def _safe_remove_dir_root(dir_path: str, report: dict):
    """
    모델 루트 디렉터리 자체를 삭제한다.
    - 허용 루트 내부만 실행
    - 비어있으면 rmdir, 남아있으면 rmtree
    - report['removed_dir'] 갱신
    """
    try:
        if not dir_path:
            report.setdefault("skipped", []).append({"target": dir_path, "reason": "empty-path"})
            return
        p = Path(dir_path).resolve()
        if not p.exists():
            report.setdefault("skipped", []).append({"target": str(p), "reason": "not-exists"})
            return
        if not p.is_dir():
            report.setdefault("skipped", []).append({"target": str(p), "reason": "not-dir"})
            return
        if not _is_under_allowed_roots(str(p)):
            report.setdefault("skipped", []).append({"target": str(p), "reason": "not-under-allowed-roots"})
            return

        try:
            p.rmdir()                       # 비어있으면 성공
            report["removed_dir"] = True
        except OSError:
            try:
                shutil.rmtree(p)            # 잔여물 있어도 통째 제거
                report["removed_dir"] = True
            except Exception as e:
                report.setdefault("errors", []).append({"target": str(p), "error": repr(e)})
    except Exception as e:
        logging.getLogger(__name__).exception("safe remove dir root failed: %s", dir_path)
        report.setdefault("errors", []).append({"target": dir_path, "error": repr(e)})

def delete_model_full(model_name: str) -> Dict[str, Any]:
    """
    모델 전체 삭제:
      1) 언로드(베스트에포트)
      2) DB에서 model_path 조회
      3) 표준/레거시/마운트 경로 후보들에 대해
         - 디렉터리면: 내부 내용 전체 삭제 후, 디렉터리 자체도 제거
         - 파일이면: 파일 삭제(허용 루트 내에서만)
      4) 연관 레코드 정리 후(llm_prompt_mapping, fine_tuned_models, llm_eval_runs) 최종 llm_models 삭제
    반환: {
      success, modelName,
      fs: [ { parent, deleted[], skipped[], errors[], removed_dir<bool> }, ... ],
      dbDeleted: { llm_prompt_mapping, fine_tuned_models, llm_eval_runs, llm_models }
    }
    """
    name = (model_name or "").strip()
    if not name:
        return {"success": False, "error": "modelName required"}

    # 1) 안전 언로드(실패 무시)
    try:
        unload_model(name)
        _mark_unloaded(name)
    except Exception:
        logging.getLogger(__name__).exception("unload failed (ignored)")

    conn = _connect()
    cur = conn.cursor()
    try:
        # 2) 모델 메타 조회
        cur.execute("SELECT id, model_path FROM llm_models WHERE name=?", (name,))
        row = cur.fetchone()
        if not row:
            return {"success": False, "error": f"unknown model: {name}"}

        mid = int(row["id"])
        model_path_value = (row["model_path"] or "").strip()

        # 3) 파일시스템 정리
        targets = _collect_fs_delete_targets(model_path_value)
        fs_results: list[dict] = []
        for t in targets:
            tp = Path(t)
            if tp.is_dir():
                # 3-1) 디렉터리 내부 전체 삭제
                rep = _purge_dir_contents_safe(str(tp), remove_dir_if_empty=False)
                # 3-2) 디렉터리 자체 제거까지 시도
                _safe_remove_dir_root(str(tp), rep)
                fs_results.append(rep)
            elif tp.is_file():
                info = {
                    "parent": str(tp.parent.resolve()),
                    "deleted": [],
                    "skipped": [],
                    "errors": [],
                    "removed_dir": False,
                }
                try:
                    if _is_under_allowed_roots(str(tp.parent)):
                        tp.unlink(missing_ok=True)
                        info["deleted"].append(str(tp.resolve()))
                    else:
                        info["skipped"].append(
                            {"target": str(tp.resolve()), "reason": "not-under-allowed-roots"}
                        )
                except Exception as e:
                    logging.getLogger(__name__).exception("unlink failed: %s", tp)
                    info["errors"].append({"target": str(tp.resolve()), "error": repr(e)})
                fs_results.append(info)
            else:
                fs_results.append(
                    {
                        "parent": str(tp),
                        "deleted": [],
                        "skipped": [{"target": str(tp), "reason": "not-exists"}],
                        "errors": [],
                        "removed_dir": False,
                    }
                )

        # 4) DB 정리
        deleted = {
            "llm_prompt_mapping": 0,
            "fine_tuned_models": 0,
            "llm_eval_runs": 0,
            "llm_models": 0,
        }

        if _table_exists(conn, "llm_prompt_mapping"):
            cur.execute("DELETE FROM llm_prompt_mapping WHERE llm_id=?", (mid,))
            deleted["llm_prompt_mapping"] = cur.rowcount or 0

        if _table_exists(conn, "fine_tuned_models"):
            cur.execute("DELETE FROM fine_tuned_models WHERE model_id=?", (mid,))
            deleted["fine_tuned_models"] = cur.rowcount or 0

        if _table_exists(conn, "llm_eval_runs"):
            cur.execute("DELETE FROM llm_eval_runs WHERE llm_id=?", (mid,))
            deleted["llm_eval_runs"] = cur.rowcount or 0

        cur.execute("DELETE FROM llm_models WHERE id=?", (mid,))
        deleted["llm_models"] = cur.rowcount or 0

        conn.commit()

        return {
            "success": True,
            "modelName": name,
            "fs": fs_results,
            "dbDeleted": deleted,
        }

    except Exception:
        logging.getLogger(__name__).exception("delete_model_full failed")
        return {"success": False, "error": "delete_model_full failed"}
    finally:
        try:
            conn.close()
        except Exception:
            pass
