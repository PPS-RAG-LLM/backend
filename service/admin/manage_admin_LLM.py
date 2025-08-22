# service/admin/manage_admin_LLM.py
from __future__ import annotations

import json
import os
import sqlite3
import time
import gc
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple

from pydantic import BaseModel, Field

# ===== In-memory temp settings (kept for backward-compat) =====
_DEFAULT_TOPK: int = 5

# ===== Constants =====
DB_PATH = os.getenv("COREIQ_DB", "/home/work/CoreIQ/backend/storage/coreiq.sqlite3")
STORAGE_ROOT = "/home/work/CoreIQ/backend/storage/model"  # train_qwen_rag.py 와 동일
ACTIVE_MODEL_CACHE_KEY_PREFIX = "active_model:"  # e.g. active_model:qa
RAG_TOPK_CACHE_KEY = "rag_topk"

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


# ===== DB Helpers =====
def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db():
    """필요한 테이블이 없으면 생성 (ERD 기반)."""
    conn = _connect()
    cur = conn.cursor()

    # llm_models
    cur.execute("""
    CREATE TABLE IF NOT EXISTS llm_models(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      provider TEXT NOT NULL,
      name TEXT UNIQUE NOT NULL,
      revision INTEGER DEFAULT 0,
      model_path TEXT,
      category TEXT NOT NULL,
      type TEXT NOT NULL DEFAULT 'base',
      is_active BOOLEAN DEFAULT 1,
      trained_at DATETIME,
      created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # system_prompt_template
    cur.execute("""
    CREATE TABLE IF NOT EXISTS system_prompt_template(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      name TEXT NOT NULL,
      category TEXT NOT NULL,
      content TEXT NOT NULL,
      required_vars TEXT,
      is_active BOOLEAN DEFAULT 1
    )
    """)

    # system_prompt_variables
    cur.execute("""
    CREATE TABLE IF NOT EXISTS system_prompt_variables(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      type TEXT NOT NULL,
      key TEXT NOT NULL,
      value TEXT,
      description TEXT NOT NULL
    )
    """)

    # prompt_mapping
    cur.execute("""
    CREATE TABLE IF NOT EXISTS prompt_mapping(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      template_id INTEGER,
      variable_id INTEGER,
      created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
      updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
      FOREIGN KEY(template_id) REFERENCES system_prompt_template(id) ON DELETE CASCADE ON UPDATE CASCADE,
      FOREIGN KEY(variable_id) REFERENCES system_prompt_variables(id) ON DELETE CASCADE ON UPDATE CASCADE
    )
    """)

    # event_logs (모델 평가결과 저장용)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS event_logs(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      event TEXT NOT NULL,
      metadata TEXT,
      user_id INTEGER,
      occurred_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # cache_data (전역설정/활성모델)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS cache_data(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      name TEXT NOT NULL,
      data TEXT NOT NULL,
      belongs_to TEXT,
      by_id INTEGER,
      expires_at DATETIME,
      created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
      updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()


_init_db()


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


def _set_cache(name: str, data: str, belongs_to: str = "global", by_id: Optional[int] = None):
    conn = _connect()
    cur = conn.cursor()
    cur.execute("""
      INSERT INTO cache_data(name, data, belongs_to, by_id, created_at, updated_at)
      VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
    """, (name, data, belongs_to, by_id))
    conn.commit()
    conn.close()


def _ensure_models_from_fs(category: str) -> None:
    """
    STORAGE_ROOT 아래 로컬 디렉터리(학습 산출물 등)에서 config.json이 있는 모델 폴더를 llm_models에 동기화.
    category는 admin UI 구분용이며, 로컬 폴더명에 카테고리가 명시되지 않았다면 기존 category로 등록.
    """
    if not os.path.isdir(STORAGE_ROOT):
        return
    conn = _connect()
    cur = conn.cursor()

    for name in os.listdir(STORAGE_ROOT):
        model_dir = os.path.join(STORAGE_ROOT, name)
        if not os.path.isdir(model_dir):
            continue
        cfg = os.path.join(model_dir, "config.json")
        if os.path.isfile(cfg):
            # DB에 없으면 등록
            cur.execute("SELECT id FROM llm_models WHERE name=?", (name,))
            row = cur.fetchone()
            if not row:
                cur.execute("""
                    INSERT INTO llm_models(provider, name, revision, model_path, category, type, is_active)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, ("hf", name, 0, model_dir, category, "base", 1))
    conn.commit()
    conn.close()


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
                from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig  # type: ignore
                import torch  # type: ignore

                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
                tok = AutoTokenizer.from_pretrained(key, trust_remote_code=True)
                if tok.pad_token_id is None:
                    if getattr(tok, "eos_token_id", None) is not None:
                        tok.pad_token_id = tok.eos_token_id
                    else:
                        tok.add_special_tokens({"pad_token": "<|pad|>"})
                        tok.pad_token_id = tok.convert_tokens_to_ids("<|pad|>")
                mdl = AutoModelForCausalLM.from_pretrained(
                    key,
                    trust_remote_code=True,
                    device_map="auto",
                    quantization_config=bnb_config,
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

def _load_local_model(model_name_or_path: str):
    try:
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
    # 로컬 폴더 동기화 후 DB 조회
    _ensure_models_from_fs(category)
    conn = _connect()
    cur = conn.cursor()
    cur.execute("""
      SELECT id, name FROM llm_models
      WHERE category=? AND is_active=1
      ORDER BY trained_at DESC NULLS LAST, id DESC
    """, (category,))
    rows = cur.fetchall()
    conn.close()

    models = [{"id": r["id"], "name": r["name"]} for r in rows]
    # 없으면 예비 목록(샘플) 제공
    if not models:
        models = [
            {"id": 5, "name": "Qwen-3.3-summary-v3"},
            {"id": 4, "name": "Qwen-3.3-summary-v2"},
            {"id": 3, "name": "Qwen-3.3-summary-v1"},
        ]
    return {"category": category, "models": models}


def load_or_unload_model(category: str, model_name: str) -> Dict[str, Any]:
    """
    카테고리별 '활성 모델'을 지정/해제. 실제 메모리 언로드는 이 계층에서 강제하지 않고,
    추론 계층에서 필요 시 lazy-load 하도록 설계.
    여기에 실제 메모리 로드/언로드 동작을 추가한다.
    """
    logger = logging.getLogger(__name__)
    try:
        row = _lookup_model_by_name(model_name)

        # Determine action: toggle – if loaded, unload; otherwise load
        is_loaded = _MODEL_MANAGER.is_loaded(model_name)
        performed = "unloaded" if is_loaded else "loaded"

        if is_loaded:
            _MODEL_MANAGER.unload(model_name)
            # If unloading the currently active model for this category, clear it
            active_name = _active_model_name_for_category(category)
            if active_name == model_name:
                _set_cache(ACTIVE_MODEL_CACHE_KEY_PREFIX + category, _json({"modelName": None}), "llm_admin")
        else:
            try:
                _MODEL_MANAGER.load(model_name)
            except Exception:
                logger.exception("failed to load model during toggle: %s", model_name)
                return {"success": False, "message": f"모델 로드 실패: {model_name}", "category": category, "modelName": model_name}
            _set_active_model_for_category(category, model_name)

        message = f"모델 {performed} 완료"
        if row is None:
            message += " (주의: DB에 모델 메타가 없어 로컬 경로 기준 처리)"
        return {"success": True, "message": message, "category": category, "modelName": model_name, "loaded": (performed == "loaded")}
    except Exception:
        logger.exception("unexpected error in load_or_unload_model")
        return {"success": False, "message": "예상치 못한 오류로 작업에 실패했습니다.", "category": category, "modelName": model_name}


def unload_model(model_name: str) -> Dict[str, Any]:
    """Explicitly unload a model from memory (does not change active cache unless it matches)."""
    logger = logging.getLogger(__name__)
    try:
        was_loaded = _MODEL_MANAGER.is_loaded(model_name)
        ok = _MODEL_MANAGER.unload(model_name)
        return {"success": bool(ok), "message": ("언로드 완료" if was_loaded and ok else "이미 언로드됨"), "modelName": model_name}
    except Exception:
        logger.exception("failed to unload model: %s", model_name)
        return {"success": False, "message": "언로드 실패", "modelName": model_name}


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


def list_prompts(category: str) -> Dict[str, Any]:
    conn = _connect()
    cur = conn.cursor()
    cur.execute("""
      SELECT id, name, content FROM system_prompt_template
      WHERE category=? AND is_active=1
      ORDER BY id DESC
    """, (category,))
    rows = cur.fetchall()
    conn.close()
    prompt_list = [{"promptId": r["id"], "title": r["name"], "prompt": r["content"]} for r in rows]
    return {"category": category, "promptList": prompt_list}


def create_prompt(category: str, body: CreatePromptBody) -> Dict[str, Any]:
    start = time.time()
    conn = _connect()
    cur = conn.cursor()
    # 템플릿 생성
    required_vars = [v.key for v in body.variables] if body.variables else []
    cur.execute("""
      INSERT INTO system_prompt_template(name, category, content, required_vars, is_active)
      VALUES(?,?,?,?,1)
    """, (body.title, category, body.prompt, _json(required_vars)))
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
    cur.execute("SELECT * FROM system_prompt_template WHERE id=? AND is_active=1", (prompt_id,))
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
