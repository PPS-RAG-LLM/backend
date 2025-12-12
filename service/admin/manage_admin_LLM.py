# service/admin/manage_admin_LLM.py
from __future__ import annotations

import json
import os
import gc
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from glob import glob
from pydantic import BaseModel, Field
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from repository.llm_models import (
    repo_get_llm_model_by_name,
    repo_get_active_llm_models,
    repo_update_llm_model_active_by_path,
    repo_get_llm_model_path_by_name,
    repo_get_llm_model_id_and_path_by_name,
    repo_get_llm_models_by_category_all,
    repo_get_llm_models_by_category_and_subcategory,
    repo_get_llm_models_by_category,
    repo_delete_llm_model,
    repo_get_llm_model_by_id,
    repo_get_fine_tuned_model_by_model_id,
    repo_get_distinct_model_ids_from_fine_tuned_models,
    repo_get_prompt_template_by_id,
    repo_get_prompt_variables_by_template_id,
    repo_get_prompt_templates_by_category_and_name,
    repo_get_default_prompt_template_by_category,
    repo_get_best_llm_prompt_mapping_by_prompt_id,
    repo_count_default_prompt_templates,
)
from repository.event_logs import repo_add_event_log
from repository.llm_deletion import repo_delete_llm_related_data
from repository.cache_data import repo_set_cache, repo_get_cache

try:
    from peft import PeftModel
except Exception:
    PeftModel = None  # peft 미설치 환경 보호
import socket  # ← 추가
WORKER_ID = f"{socket.gethostname()}:{os.getpid()}"
import shutil
from config import config as app_config

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

# 레거시 위치(호환용): ./storage/models/llm
_RETRIEVAL_CFG = app_config.get("models_dir", {}) or {}
LLM_MODEL_DIR = (BASE_BACKEND / Path(_RETRIEVAL_CFG.get("llm_models_path"))).resolve()

LLM_MODEL_DIR.mkdir(parents=True, exist_ok=True)

ACTIVE_MODEL_CACHE_KEY_PREFIX = "active_model:"  # e.g. active_model:qna

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
    같은 모델 디렉터리를 항상 /storage/models/llm/<basename> 로 통일.
    해당 경로에 config.json 이 있으면 그 경로를 반환, 없으면 원본 반환.
    """
    try:
        p = (p or "").strip().replace("\\", "/")
        base = os.path.basename(p.rstrip("/"))
        cand = f"/storage/models/llm/{base}"
        if os.path.isdir(cand) and os.path.isfile(os.path.join(cand, "config.json")):
            return cand
    except Exception:
        pass
    return p


# ===== Migration / helpers =====
def _db_set_active_by_path(rel_path: str, active: bool) -> None:
    try:
        repo_update_llm_model_active_by_path(rel_path, active)
    except Exception:
        logging.getLogger(__name__).exception("failed to sync is_active by path")



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

def _fetch_llm_and_ft(model_name: str) -> Tuple[Optional[dict], Optional[dict]]:
    llm = repo_get_llm_model_by_name(model_name)
    if not llm:
        return None, None
    # 최신 활성 FT 1건 우선, 없으면 최신 1건
    ft = repo_get_fine_tuned_model_by_model_id(llm["id"], active_first=True)
    return llm, ft

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
    llm, ft = _fetch_llm_and_ft(model_name)
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
    """cache_data 테이블에서 캐시 데이터를 조회합니다."""
    return repo_get_cache(name)
    
def _norm_category(category: str) -> str:
    """
    외부 표기는 qna, 내부 스키마/기존 코드는 qna.
    """
    c = (category or "").strip().lower()
    if c == "qna":
        return "qna"
    return c

def _subtask_key(subtask: Optional[str]) -> str:
    return (subtask or "").strip().lower()



def _set_cache(name: str, data: str, belongs_to: str = "global", by_id: Optional[int] = None):
    """cache_data 테이블에 캐시 데이터를 저장합니다."""
    try:
        repo_set_cache(name, data, belongs_to, by_id)
    except Exception:
        logging.getLogger(__name__).exception(f"Failed to set cache: name={name}")


# ---------- 활성 프롬프트: 사용자 선택 저장/조회 ----------

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
    
    # ORM 기반 로그 저장
    repo_add_event_log("model_eval", _json(meta))

    return {"success": True, "result": "테스트 실행 완료", "promptId": prompt_id, "answer": answer, "rougeScore": rouge}


    
# ===== 새로 추가: 활성 LLM 모델 조회(로깅용) =====
def get_active_llm_models() -> List[Dict[str, Any]]:
    """
    llm_models에서 is_active=1 인 모델 목록을 반환한다.
    로깅/표시 용도로만 사용하며, 모델을 실제로 로드하지 않는다.
    """
    return repo_get_active_llm_models()

def _fill_template(content: str, variables: Dict[str, Any]) -> str:
    # 템플릿 내 {{key}} 치환
    out = content
    for k, v in (variables or {}).items():
        out = out.replace("{{" + k + "}}", str(v))
    return out
    

def _fetch_prompt_full(prompt_id: int) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    tmpl = repo_get_prompt_template_by_id(prompt_id)
    if not tmpl:
        raise ValueError("존재하지 않거나 비활성화된 프롬프트입니다.")
    vars_rows = repo_get_prompt_variables_by_template_id(prompt_id)
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


def _lookup_model_by_name(model_name: str) -> Optional[Dict[str, Any]]:
    return repo_get_llm_model_by_name(model_name)


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
        fs_path = str(LLM_MODEL_DIR / name_or_path)
        try:
            if os.path.isfile(os.path.join(fs_path, "config.json")):
                return fs_path
            # ← 레거시 경로도 체크
            legacy = str(LLM_MODEL_DIR / name_or_path)
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
        cand = LLM_MODEL_DIR / os.path.basename(s)
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
    try:
        model_path = repo_get_llm_model_path_by_name(model_name)
        if not model_path:
            return None
        val = (model_path or "").strip().replace("\\", "/")
        if not val:
            return None

        # Handle standardized relative paths like ./service/storage/models/llm/...
        if val.startswith("./"):
            leg = LLM_MODEL_DIR / os.path.basename(val)
            if (leg / "config.json").is_file():
                return _canon_storage_path(str(leg))
            return None

        # Absolute or simple name -> resolve via fs helper
        resolved = _resolve_model_fs_path(val)
        return _canon_storage_path(resolved)
    except Exception:
        logging.getLogger(__name__).exception("failed to query llm_models for %s", model_name)
        return None

def _strip_category_suffix(name: str) -> str:
    # 허용: -qna/-doc_gen/-summary 및 _qna/_doc_gen/_summary
    for suf in ("-qna", "-doc_gen", "-summary", "_qna", "_doc_gen", "_summary"):
        if name.endswith(suf):
            return name[: -len(suf)]
    return name


def _to_rel_under_storage_root(p: str) -> str:
    """ 상대 경로 반환 """
    try:
        p = str(p)
        try:
            rp = os.path.relpath(p, str(LLM_MODEL_DIR))
            if rp not in (".", ""):
                return f"./storage/models/llm/{rp}".replace("\\", "/")
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
    cand =  LLM_MODEL_DIR / model_name
    if (cand / "config.json").is_file():
        return str(cand)
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

        # 2) utils 쪽에서 사용하는 보이는 경로로 매핑 (예: '/storage/models/llm/<basename>')
        def _adapter_visible_path(p: str) -> str:
            try:
                base = os.path.basename(p.rstrip("/"))
                cand = f"/storage/models/llm/{base}"
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
                from utils.llms.huggingface.qwen import load_qwen_instruct_7b
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


def get_model_list(category: str, subcategory: Optional[str] = None):
    cat = _norm_category(category)
    sub = (subcategory or "").strip()   # = system_prompt_template.name

    # --- (ADD) 파인튜닝 모델 id 세트 미리 로딩 ---
    ft_ids: set[int] = set()
    try:
        ft_ids = set(repo_get_distinct_model_ids_from_fine_tuned_models())
    except Exception:
        ft_ids = set()

    # 1) category=all → 전체(활/비활 포함)
    if cat == "all":
        rows = repo_get_llm_models_by_category_all()

    # 2) doc_gen + subcategory → 매핑 rouge 점수순
    elif cat == "doc_gen" and sub:
        rows = repo_get_llm_models_by_category_and_subcategory(cat, sub)

    # 3) 그 외(qna/summary/doc_gen 전체) → 활성만
    else:
        rows = repo_get_llm_models_by_category(cat)

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

    rows = repo_get_prompt_templates_by_category_and_name(category, subcat)

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

def _db_get_model_record(model_name: str) -> Optional[Dict[str, Any]]:
    return repo_get_llm_model_by_name(model_name)


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

    try:
        # 1) default 템플릿 선택
        tmpl = repo_get_default_prompt_template_by_category(category, subcat)
        if not tmpl:
            return {"category": category, "subcategory": subcat, "default": None, "note": "기본 템플릿 없음"}

        prompt_id = int(tmpl["id"])

        # 2) prompt 매핑 중 최고 rouge_score 1건
        mp = repo_get_best_llm_prompt_mapping_by_prompt_id(prompt_id)
        if not mp:
            return {"category": category, "subcategory": tmpl["name"], "default": None, "note": "프롬프트-모델 매핑 없음"}

        # 3) llm_models 메타
        mdl = repo_get_llm_model_by_id(mp["llm_id"])
        if not mdl:
            return {"category": category, "subcategory": tmpl["name"], "default": None, "note": "llm_models에 모델 없음"}

        # 스키마 제약 확인(참고 메시지)
        # 카테고리/서브카테고리별 default가 1개라는 제약을 코드에서는 강제하지 않음(요구대로 '확인만')
        cnt = repo_count_default_prompt_templates(category, subcat)

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
    except Exception:
        logging.getLogger(__name__).exception("get_default_model_for_category failed")
        return {"category": category, "subcategory": subcat, "default": None, "note": "오류 발생"}


def _is_under_allowed_roots(path_str: str) -> bool:
    try:
        # 입력받은 경로와 허용 루트를 모두 절대 경로로 변환
        target = Path(path_str).resolve()
        root = LLM_MODEL_DIR.resolve()
        # target이 root와 같거나, root의 하위 경로인지 확인
        if hasattr(target, "is_relative_to"):
            return target.is_relative_to(root)
        # Python 3.8 이하 호환용 (문자열 비교)
        return str(target).startswith(str(root))
    except Exception:
        return False


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
        std = (LLM_MODEL_DIR / base).resolve()
        for p in std:
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
    try:
        # 2) 모델 메타 조회
        model_info = repo_get_llm_model_id_and_path_by_name(name)
        if not model_info:
            return {"success": False, "error": f"unknown model: {name}"}

        mid = int(model_info["id"])
        model_path_value = (model_info["model_path"] or "").strip()

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
        deleted = repo_delete_llm_related_data(mid)

        return {
            "success": True,
            "modelName": name,
            "fs": fs_results,
            "dbDeleted": deleted,
        }

    except Exception:
        logging.getLogger(__name__).exception("delete_model_full failed")
        return {"success": False, "error": "delete_model_full failed"}
