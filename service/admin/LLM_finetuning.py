# service/admin/LLM_finetuning.py
from __future__ import annotations

# ✅ Unsloth는 transformers/peft 보다 먼저 import
try:
    import unsloth  # noqa: F401
except Exception:
    pass

import os
# 🔧 CUDA 메모리 단편화 완화 (권장)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512,expandable_segments:True,roundup_power2_divisions:16")

import json
import threading
import time
import uuid
import tempfile
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional
from pathlib import Path
from contextlib import contextmanager

from pydantic import BaseModel, Field, field_validator, model_validator
from utils import logger
from errors.exceptions import BadRequestError, InternalServerError

# --- Repository Imports ---
from repository.llm_finetuning import update_job_status, finish_job_success, fail_job
from utils.database import get_session

logger = logger(__name__)

import re
from urllib.parse import quote_plus
_FEEDBACK_FILE_RE = re.compile(r"^feedback_(qna|doc_gen|summary)_p(\d+)\.csv$", re.IGNORECASE)

# ===== 디바이스 유틸 =====
def _get_model_device(model):
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        pass
    try:
        return next(model.parameters()).device
    except Exception:
        pass
    import torch as _torch
    return _torch.device("cuda" if _torch.cuda.is_available() else "cpu")


def _clear_gpu_memory():
    """GPU 메모리 캐시를 완전히 정리하여 단편화를 줄입니다."""
    try:
        import torch
        import gc
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            # 모든 CUDA 캐시 정리
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            # Python GC 실행
            gc.collect()
            # 다시 한번 캐시 정리
            torch.cuda.empty_cache()
            logger.debug("GPU memory cache cleared")
    except Exception as e:
        logger.warning(f"Failed to clear GPU memory: {e}")


# ===== 캐시/임시 디렉토리 관리 =====
@contextmanager
def _ephemeral_cache_env():
    """훈련 중에만 임시 캐시 디렉토리를 사용하고 끝나면 삭제"""
    keys = [
        "HF_HOME", "TRANSFORMERS_CACHE", "HF_DATASETS_CACHE",
        "XDG_CACHE_HOME", "UNSLOTH_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR",
    ]
    tmpdir = tempfile.mkdtemp(prefix="ft_cache_")
    old = {k: os.environ.get(k) for k in keys}
    try:
        for k in keys:
            os.environ[k] = tmpdir
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(tmpdir, ignore_errors=True)

# ===== Progress interval (seconds) =====
FT_PROGRESS_INTERVAL_SEC = int(os.getenv("FT_PROGRESS_INTERVAL_SEC", "2"))
# ===== Heartbeat timeout (seconds) for liveness detection =====
FT_HEARTBEAT_TIMEOUT_SEC = int(os.getenv("FT_HEARTBEAT_TIMEOUT_SEC", "300"))
# ===== OOM retry limit =====
MAX_OOM_RETRIES = int(os.getenv("FT_MAX_OOM_RETRIES", "1"))

# ===== Paths =====
BASE_BACKEND = Path(os.getenv("COREIQ_BACKEND_ROOT", str(Path(__file__).resolve().parents[2])))  # backend/
STORAGE_MODEL_ROOT = os.getenv("STORAGE_MODEL_ROOT", str(BASE_BACKEND / "storage" / "models"))
TRAIN_DATA_ROOT   = os.getenv("TRAIN_DATA_ROOT", str(BASE_BACKEND / "storage" / "train_data"))

# ===== SQLAlchemy ORM (Session) =====
from sqlalchemy import create_engine, select, func
from sqlalchemy.orm import sessionmaker
from storage.db_models import (
    LlmModel, FineTuneDataset, FineTuneJob as ORMJob, FineTunedModel
)

# DB_URL 제거 및 get_session 사용으로 대체됨

# ---- Portable path helpers ----
def _to_rel(p: str) -> str:
    """`p`를 backend 루트 기준 상대 경로로 변환(실패 시 원본 반환)."""
    try:
        return os.path.relpath(p, BASE_BACKEND)
    except Exception:
        return p

def _to_service_rel(p: str) -> str:
    """
    어떤 경로든 최종적으로 './service/<...>' 형태로 표준화.
    예) /.../backend/storage/models/llm/Qwen3-8B  → ./service/storage/models/llm/Qwen3-8B
        storage/models/llm/Qwen3-8B               → ./service/storage/models/llm/Qwen3-8B
        ./service/storage/models/llm/Qwen3-8B     → 그대로 유지
    """
    # 절대경로면 backend 기준 상대경로로
    if os.path.isabs(p):
        p = os.path.relpath(p, BASE_BACKEND)
    s = p.replace("\\", "/").lstrip("./")
    if not s.startswith("service/"):
        s = f"service/{s}"
    # 중복 슬래시 정리
    while "//" in s:
        s = s.replace("//", "/")
    return f"./{s}"

# ===== Helpers: path resolve =====
def _resolve_model_dir(name_or_path: str) -> str:
    """Resolve a local model directory for finetuning.
    Priority: DB.llm_models.model_path (by exact name, then base-name) → STORAGE_MODEL_ROOT/name.
    인터넷 다운로드는 시도하지 않음.
    """
    if os.path.isabs(name_or_path):
        return name_or_path

    # 1) DB에서 llm_models 테이블 조회
    try:
        with get_session() as s:
            # 정확한 이름으로 먼저 찾기
            stmt = select(LlmModel).where(LlmModel.name == name_or_path)
            model = s.execute(stmt).scalars().first()

            if not model:
                # 카테고리 접미사 제거 후 다시 찾기
                def _strip_cat(n: str) -> str:
                    for suf in ("-qna", "-doc_gen", "-summary"):
                        if n.endswith(suf):
                            return n[: -len(suf)]
                    return n

                base_name = _strip_cat(name_or_path)
                if base_name != name_or_path:
                    stmt = select(LlmModel).where(LlmModel.name == base_name)
                    model = s.execute(stmt).scalars().first()

            if model and model.model_path:
                p = model.model_path
                if os.path.isabs(p):
                    return p
                # 상대 경로인 경우 STORAGE_MODEL_ROOT 기준으로 해석
                cand = os.path.join(STORAGE_MODEL_ROOT, p)
                if os.path.isdir(cand):
                    return cand
    except Exception as e:
        logger.warning(f"DB lookup failed for model {name_or_path}: {e}")

    # 2) Fallback to storage root
    return os.path.join(STORAGE_MODEL_ROOT, name_or_path)

def _has_model_signature(dir_path: str) -> bool:
    if not os.path.isdir(dir_path):
        return False
    sigs = [
        "config.json",
        "model.safetensors",
        "pytorch_model.bin",
        "generation_config.json",
    ]
    return any(os.path.isfile(os.path.join(dir_path, s)) for s in sigs)

def _resolve_train_path(p: str) -> str:
    if os.path.isabs(p):
        return p
    cand = os.path.join(TRAIN_DATA_ROOT, p)
    if os.path.isfile(cand):
        return cand
    try:
        for root, _, files in os.walk(TRAIN_DATA_ROOT):
            if p in files:
                return os.path.join(root, p)
    except Exception:
        pass
    return os.path.abspath(p)

# ===== Helpers: detect MXFP4 / gpt-oss =====
def _looks_like_mxfp4_model(dir_or_name: str) -> bool:
    name = (dir_or_name or "").lower()
    if "gpt-oss" in name:
        return True
    # 로컬 폴더일 경우 config에서 힌트 찾기
    try:
        cand = os.path.join(dir_or_name, "config.json")
        if os.path.isfile(cand):
            with open(cand, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            txt = json.dumps(cfg).lower()
            if "mxfp4" in txt or "mx" in txt:
                return True
    except Exception:
        pass
    # 양자화 설정 파일 케이스
    for q in ("quantization_config.json", "quantize_config.json"):
        try:
            cand = os.path.join(dir_or_name, q)
            if os.path.isfile(cand):
                with open(cand, "r", encoding="utf-8") as f:
                    if "mxfp4" in f.read().lower():
                        return True
        except Exception:
            pass
    return False

# ===== Schemas =====
class FineTuneRequest(BaseModel):
    # === 공통 태그(필수) ===
    category: str = Field(..., description="qna | doc_gen | summary")
    subcategory: Optional[str] = Field(None, description="세부 테스크. 현재 주로 doc_gen에서 사용")

    baseModelName: str
    saveModelName: str
    systemPrompt: str
    batchSize: int = 1
    epochs: int = 3
    learningRate: float = 2e-4
    overfittingPrevention: bool = True
    trainSetFile: str
    gradientAccumulationSteps: int = 16
    quantizationBits: Optional[int] = Field(
        default=None, description="QLORA 전용: 양자화 비트 (4 또는 8)",
    )
    tuningType: Optional[str] = Field(
        default="QLORA",
        description="파인튜닝 방식: LORA | QLORA | FULL",
        pattern="^(LORA|QLORA|FULL)$",
    )
    
    @field_validator("tuningType", mode="before")
    @classmethod
    def _v_tuning_type_strip(cls, v):
        """tuningType 값의 앞뒤 공백을 제거합니다."""
        if v is None:
            return None
        if isinstance(v, str):
            return v.strip()
        return v
    startAt: Optional[str] = Field(
        default=None, description="예약 시작 ISO8601 (예: 2025-09-19T13:00:00)"
    )
    startNow: bool = Field(
        default=False, description="즉시 실행 여부 (True: 바로 시작, False: 예약만 등록)"
    )

    @field_validator("category")
    @classmethod
    def _v_category(cls, v: str) -> str:
        allowed = {"qna", "doc_gen", "summary"}
        vv = (v or "").strip().lower()
        if vv not in allowed:
            raise ValueError(f"category must be one of {sorted(allowed)}")
        return vv

    @field_validator("quantizationBits", mode="before")
    @classmethod
    def _v_qbits_empty_to_none(cls, v):
        """
        폼에서 빈 문자열("")로 넘어오는 quantizationBits를 None으로 처리.
        FULL/LORA에서는 quantizationBits를 비워도 되도록 허용하고,
        QLORA인 경우에만 아래 validator/model_validator에서 4 또는 8을 강제한다.
        """
        if v in ("", None):
            return None
        return v

    @field_validator("quantizationBits")
    @classmethod
    def _v_qbits_range(cls, v: Optional[int]):
        if v is None:
            return v
        if v not in (4, 8):
            raise ValueError("quantizationBits must be 4 or 8 when provided")
        return v

    @model_validator(mode="after")
    def _v_qbits_required_for_qlora(self):
        if (self.tuningType or "").upper() == "QLORA":
            if self.quantizationBits not in (4, 8):
                raise ValueError("QLORA requires quantizationBits=4 or 8")
        return self

@dataclass
class FineTuneJob:
    job_id: str
    category: str
    request: Dict[str, Any]
    status: str = "queued"  # queued | running | succeeded | failed

# ===== Common utils =====
def _now_local_str() -> str:
    """현재 시간을 KST로 반환"""
    kst = timezone(timedelta(hours=9))
    return datetime.now(kst).strftime("%Y-%m-%d %H:%M:%S")

def _now_utc() -> datetime:
    """현재 시간을 UTC로 반환"""
    return datetime.now(timezone.utc)

def _to_kst(utc_dt: datetime) -> datetime:
    """UTC datetime을 KST로 변환"""
    if utc_dt.tzinfo is None:
        utc_dt = utc_dt.replace(tzinfo=timezone.utc)
    kst = timezone(timedelta(hours=9))
    return utc_dt.astimezone(kst)

def _ensure_output_dir(model_name: str) -> str:
    try:
        os.makedirs(STORAGE_MODEL_ROOT, exist_ok=True)
    except PermissionError as e:
        raise InternalServerError(f"permission denied creating model root '{STORAGE_MODEL_ROOT}': {e}")
    except OSError as e:
        try:
            if getattr(e, "errno", None) == 13:
                raise InternalServerError(f"permission denied creating model root '{STORAGE_MODEL_ROOT}': {e}")
        except Exception:
            pass
        raise

    out_dir = os.path.join(STORAGE_MODEL_ROOT, model_name)
    try:
        os.makedirs(out_dir, exist_ok=True)
    except PermissionError as e:
        raise InternalServerError(f"permission denied creating output dir '{out_dir}': {e}")
    except OSError as e:
        try:
            if getattr(e, "errno", None) == 13:
                raise InternalServerError(f"permission denied creating output dir '{out_dir}': {e}")
        except Exception:
            pass
        raise
    cfg_path = os.path.join(out_dir, "config.json")
    if not os.path.isfile(cfg_path):
        try:
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump({"created_at": _now_utc().isoformat()}, f, ensure_ascii=False)
        except Exception:
            logger.exception(f"failed to write config.json inside {out_dir}")
    return out_dir

def _ensure_lora_marker(out_dir: str, tuning_type: str):
    if tuning_type.upper() in ("LORA", "QLORA"):
        marker = os.path.join(out_dir, "adapter_config.json")
        if not os.path.isfile(marker):
            try:
                with open(marker, "w", encoding="utf-8") as f:
                    json.dump({"peft_type": "LORA", "tuning": tuning_type.upper()}, f, ensure_ascii=False)
            except Exception:
                logger.exception(f"failed to create LoRA marker {marker}")

def _append_log(log_path: str, line: str):
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line.rstrip("\n") + "\n")
    except Exception:
        # Log but do not interrupt main flow
        logger.exception(f"failed to append log to {log_path}")

# ----- Error helper -----
def _write_error(out_dir: str, message: str):
    """Write error message to a dedicated error.txt inside output dir."""
    try:
        os.makedirs(out_dir, exist_ok=True)
        err_path = os.path.join(out_dir, "error.txt")
        with open(err_path, "a", encoding="utf-8") as f:
            ts = _now_utc().isoformat()
            f.write(f"[{ts}] {message}\n")
    except Exception:
        logger.exception(f"failed to write error file under {out_dir}")


# ===== DB ops =====
def _insert_dataset_if_needed(conn, path: str, category: str) -> int:
    # ==== ORM 버전 ====
    # conn 인자는 호환성 유지용으로 남겨두지만 사용하지 않음
    rel_path = _to_rel(path)
    with get_session() as s:
        stmt = select(FineTuneDataset).where(FineTuneDataset.path == rel_path)
        exist = s.execute(stmt).scalars().first()
        if exist:
            return int(exist.id)
        row = FineTuneDataset(
            name=os.path.basename(path),
            category=category,
            path=rel_path,
            record_count=None,
        )
        s.add(row)
        s.commit()
        s.refresh(row)
        return int(row.id)

def _insert_job(conn, category: str, req: FineTuneRequest, job_id: str, save_name_with_suffix: str, dataset_id: int,
                initial_status: str = "queued", scheduled_at: Optional[str] = None) -> int:
    # ==== ORM 버전 ====
    reserve_now = _now_local_str()
    train_path = _resolve_train_path(req.trainSetFile)
    train_rel_path = _to_rel(train_path)
    hyper = {
        "systemPrompt": req.systemPrompt,
        "batchSize": req.batchSize,
        "epochs": req.epochs,
        "learningRate": req.learningRate,
        "overfittingPrevention": req.overfittingPrevention,
        "tuningType": (req.tuningType or "QLORA").upper(),
        "baseModelName": req.baseModelName,
        "saveModelName": save_name_with_suffix,
        "trainSetFile": train_rel_path,
        "reserveDate": reserve_now,
        "category": category,
        "subcategory": (req.subcategory or None),
        "gradientAccumulationSteps": req.gradientAccumulationSteps,
        "quantizationBits": req.quantizationBits,
    }
    metrics = {"hyperparameters": hyper}
    if scheduled_at:
        metrics["scheduledAt"] = scheduled_at
        metrics["learningProgress"] = 0

    with get_session() as s:
        row = ORMJob(
            provider_job_id=job_id,
            dataset_id=dataset_id,
            status=initial_status,
            started_at=None,  # 실제 시작 시점에 설정
            metrics=json.dumps(metrics, ensure_ascii=False),
        )
        s.add(row)
        s.commit()
        s.refresh(row)
        return int(row.id)

# _update_job_status는 이제 Repository를 사용하거나 직접 구현
# 기존 로직과의 호환성을 위해 래퍼 함수로 유지하되, repository 함수를 호출
def _update_job_status(conn, job_id: str, status: str, progress: int | None = None, rough: int | None = None, extras: dict | None = None, _retries: int = 3):
    """
    repository/llm_finetuning.py의 update_job_status를 호출
    conn 인자는 무시됨 (repository에서 get_session 사용)
    """
    try:
        update_job_status(job_id, status, progress, extras, rough)
        
        # Additional logging
        out_dir = _resolve_out_dir_by_job(job_id)
        if out_dir:
            log_path = os.path.join(out_dir, "train.log")
            msg = f"status={status} progress={progress} rough={rough} extras={extras or {}}"
            _append_log(log_path, f"[{_now_utc().isoformat()}] {msg}")
            
    except Exception as e:
        logger.error(f"Failed to update job status: {e}")

def _resolve_out_dir_by_job(job_id: str) -> Optional[str]:
    # ORM 버전
    with get_session() as s:
        stmt = select(ORMJob).where(ORMJob.provider_job_id == job_id)
        job = s.execute(stmt).scalars().first()
        
        if not job:
            return None
            
        save_name = None
        if job.metrics:
            try:
                mt = json.loads(job.metrics) or {}
                hp = mt.get("hyperparameters") or {}
                save_name = hp.get("saveModelName")
            except Exception:
                pass
        
        if not save_name:
            return None
            
        return os.path.join(STORAGE_MODEL_ROOT, save_name)


# ===== Training (inline, real) =====
def _run_training_inline(job: FineTuneJob, save_name_with_suffix: str):
    """
    Unsloth gpt-oss(20B) 노트북 흐름을 반영한 경량/안정 파이프라인.
    """
    import gc
    try:
        # _update_job_status 호출 (conn은 None으로 전달해도 됨, 내부에서 무시)
        _update_job_status(None, job.job_id, "running", progress=0)

        out_dir = _ensure_output_dir(save_name_with_suffix)
        tuning_type = (job.request.get("tuningType") or "QLORA").upper()
        if tuning_type in ("LORA", "QLORA"):
            _ensure_lora_marker(out_dir, tuning_type)
        log_path = os.path.join(out_dir, "train.log")
        try:
            with open(log_path, "w", encoding="utf-8") as _tmp:
                _tmp.write("")
        except Exception:
            pass
        _append_log(log_path, f"[{_now_utc().isoformat()}] job {job.job_id} started (INLINE)")
        logger.info(
            f"Fine-tuning started jobId={job.job_id} type={tuning_type} base={job.request.get('baseModelName')} "
            f"save={save_name_with_suffix} data={job.request.get('trainSetFile')}"
        )

        # ✅ 임시 캐시 폴더
        tmpdir = tempfile.mkdtemp(prefix="ft_cache_")
        cache_keys = ["HF_HOME", "TRANSFORMERS_CACHE", "HF_DATASETS_CACHE", "XDG_CACHE_HOME", "UNSLOTH_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR"]
        old_cache = {k: os.environ.get(k) for k in cache_keys}
        for k in cache_keys:
            os.environ[k] = tmpdir

        # ===== Imports =====
        try:
            from unsloth import FastLanguageModel  # type: ignore
            import pandas as pd
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                TrainingArguments,
                Trainer,
                BitsAndBytesConfig,
                TrainerCallback,
            )
        except Exception as e:
            _append_log(log_path, f"[{_now_utc().isoformat()}] import error: {e}")
            fail_job(job.job_id, f"import error: {e}")
            logger.error(f"Fine-tuning failed jobId={job.job_id} error=import error: {e}")
            return

        # ===== 데이터 적재/전처리 =====
        system_prompt = job.request.get("systemPrompt") or "You are Qwen, a helpful assistant."
        def build_prompt(context: str, question: str) -> str:
            return f"{context.strip()}\n{system_prompt}\nQuestion: {question.strip()}"

        csv_path = _resolve_train_path(job.request.get("trainSetFile"))
        encodings_to_try = ["utf-8", "utf-8-sig", "cp949", "euc-kr", "latin1"]
        df = None
        for enc in encodings_to_try:
            try:
                df = pd.read_csv(csv_path, encoding=enc, dtype=str).fillna("")
                break
            except Exception:
                continue
        if df is None:
            try:
                import chardet
                with open(csv_path, "rb") as f:
                    raw = f.read(10000)
                enc = chardet.detect(raw).get("encoding") or "utf-8"
                df = pd.read_csv(csv_path, encoding=enc, dtype=str, on_bad_lines="skip").fillna("")
            except Exception as e:
                _append_log(log_path, f"[{_now_utc().isoformat()}] csv load failed: {e}")
                fail_job(job.job_id, f"csv load failed: {e}")
                logger.error(f"Fine-tuning failed jobId={job.job_id} error=csv load failed: {e}")
                return

        conversations = [{
            "conversations": [
                {"from": "user", "value": build_prompt(r.get("ChunkContext",""), r.get("Question",""))},
                {"from": "assistant", "value": r.get("Answer","")},
            ]
        } for _, r in df.iterrows()]

        # ===== 7:3 고정 split (random_state=42) =====
        total = len(conversations)
        eval_size = max(1, int(round(total * 0.3))) if total >= 2 else 0
        if eval_size > 0:
            import random as _R
            idx = list(range(total))
            rng = _R.Random(42)
            rng.shuffle(idx)
            eval_idx = set(idx[:eval_size])
            train_data = [conversations[i] for i in idx[eval_size:]]
            eval_data  = [conversations[i] for i in idx[:eval_size]]
        else:
            train_data, eval_data = conversations, []
        _append_log(log_path, f"[{_now_utc().isoformat()}] split: total={total} eval={eval_size} train={len(train_data)}")

        class RagDataset(torch.utils.data.Dataset):
            def __init__(self, data, tokenizer, max_len=4096):
                self.data = data
                self.tk = tokenizer
                self.max_len = max_len
            def __len__(self): return len(self.data)
            def __getitem__(self, i):
                dialog = self.data[i]["conversations"]
                messages = [
                    {"role": "system", "content": "You are Qwen, a helpful assistant."},
                    {"role": "user", "content": dialog[0]["value"]},
                    {"role": "assistant", "content": dialog[1]["value"]},
                ]
                full = self.tk.apply_chat_template(messages, tokenize=False)
                enc = self.tk(full, max_length=self.max_len, truncation=True, padding="max_length", return_tensors="pt")
                input_ids = enc.input_ids[0]
                assist = self.tk.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
                prefix_len = len(self.tk(assist).input_ids)
                labels = input_ids.clone()
                labels[:prefix_len] = -100
                return {"input_ids": input_ids, "attention_mask": enc.attention_mask[0], "labels": labels}

        base_folder = _resolve_model_dir(job.request.get("baseModelName"))
        if not _has_model_signature(base_folder):
            msg = f"base model not found: {base_folder}"
            _append_log(log_path, f"[{_now_utc().isoformat()}] {msg}")
            fail_job(job.job_id, msg)
            logger.error(f"Fine-tuning failed jobId={job.job_id} error={msg}")
            return

        model_path = base_folder
        output_dir = os.path.join(STORAGE_MODEL_ROOT, save_name_with_suffix)
        is_mxfp4 = _looks_like_mxfp4_model(model_path) or _looks_like_mxfp4_model(job.request.get("baseModelName"))

        # ===== 모델 로딩 전 메모리 정리 =====
        # _clear_gpu_memory()

        # ===== 모델/토크나이저 로드 =====
        # gpt-oss(MXFP4) → Unsloth
        if tuning_type == "QLORA" and is_mxfp4:
            max_len = int(job.request.get("max_len", 3072))  # 💡 기본 3072로 살짝 낮춰 OOM 예방
            # 메모리 단편화 방지를 위해 로딩 전 메모리 정리
            # _clear_gpu_memory()
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_path,
                dtype=None,                    # H100 → bf16 자동
                max_seq_length=max_len,
                load_in_4bit=True,
                full_finetuning=False,
                trust_remote_code=True,
                local_files_only=True,
            )
            # 로딩 후 메모리 정리
            # _clear_gpu_memory()
            # Unsloth 모범사례: 학습 최적화 활성화
            try:
                model = FastLanguageModel.for_training(model)  # 일부 버전에선 in-place. 반환값 호환.
            except Exception:
                pass
            try:
                model = FastLanguageModel.get_peft_model(
                    model,
                    r=64,
                    lora_alpha=16,
                    lora_dropout=0.05,
                    bias="none",
                )
            except Exception:
                from peft import LoraConfig, get_peft_model  # type: ignore
                lora_cfg = LoraConfig(r=64, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
                model = get_peft_model(model, lora_cfg)

        elif tuning_type == "QLORA":
            # 일반 QLORA
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # type: ignore
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
            if tokenizer.pad_token_id is None:
                if getattr(tokenizer, "eos_token_id", None) is not None:
                    tokenizer.pad_token_id = tokenizer.eos_token_id
                else:
                    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
                    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<|pad|>")
            # 메모리 단편화 방지를 위해 로딩 전 메모리 정리
            # _clear_gpu_memory()
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                device_map="auto",
                quantization_config=bnb,
                local_files_only=True,
            )
            model.gradient_checkpointing_enable()
            model = prepare_model_for_kbit_training(model)
            # 로딩 후 메모리 정리
            # _clear_gpu_memory()
            lora_targets = [
                "q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj",
                "down_proj","w1","w2","c_proj","c_attn"
            ]
            from itertools import chain
            target_modules = sorted({
                n.split(".")[-1]
                for n, _ in model.named_modules()
                if any(k in n for k in lora_targets)
            })
            lora_cfg = LoraConfig(
                r=64, lora_alpha=16, target_modules=(target_modules or None),
                lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_cfg)
            max_len = int(job.request.get("max_len", 4096))
        elif tuning_type == "LORA":
            from peft import LoraConfig, get_peft_model  # type: ignore
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
            if tokenizer.pad_token_id is None:
                if getattr(tokenizer, "eos_token_id", None) is not None:
                    tokenizer.pad_token_id = tokenizer.eos_token_id
                else:
                    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
                    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<|pad|>")
            # 메모리 단편화 방지를 위해 로딩 전 메모리 정리
            # _clear_gpu_memory()
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                local_files_only=True,
            )
            model.gradient_checkpointing_enable()
            # 로딩 후 메모리 정리
            # _clear_gpu_memory()
            targets = [
                "q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj",
                "down_proj","w1","w2","c_proj","c_attn"
            ]
            target_modules = sorted({ n.split(".")[-1] for n,_ in model.named_modules() if any(k in n for k in targets) })
            lora_cfg = LoraConfig(r=64, lora_alpha=16, target_modules=(target_modules or None), lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
            model = get_peft_model(model, lora_cfg)
            max_len = int(job.request.get("max_len", 4096))
        else:  # FULL
            # 메모리 단편화 방지를 위해 로딩 전 메모리 정리
            # _clear_gpu_memory()
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
            if tokenizer.pad_token_id is None:
                if getattr(tokenizer, "eos_token_id", None) is not None:
                    tokenizer.pad_token_id = tokenizer.eos_token_id
                else:
                    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
                    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<|pad|>")
            model = AutoModelForCausalLM.from_pretrained(
                model_path, trust_remote_code=True, device_map="auto", torch_dtype=torch.bfloat16, local_files_only=True,
            )
            model.gradient_checkpointing_enable()
            for p in model.parameters(): p.requires_grad = True
            # 로딩 후 메모리 정리
            # _clear_gpu_memory()
            # FULL 파인튜닝은 메모리를 많이 사용하므로 기본 max_len을 더 보수적으로 설정
            max_len = int(job.request.get("max_len", 2048))  # 기본값을 4096에서 2048로 감소

        # ===== 데이터셋 생성 =====
        train_ds = RagDataset(train_data, tokenizer, max_len=max_len)
        eval_ds  = RagDataset(eval_data,  tokenizer, max_len=max_len) if eval_size > 0 else None
        use_eval = eval_ds is not None and len(eval_data) > 0

        # ===== TrainingArguments (버전 호환 키 자동 매핑) =====
        from inspect import signature as _sig
        def _supported_args(cls): return set(_sig(cls.__init__).parameters.keys())
        def _put_kw(supported: set, kw: dict, key: str, value, *aliases: str):
            for k in (key, *aliases):
                if k in supported:
                    kw[k] = value
                    return k
            return None

        optim_name = os.getenv("FT_OPTIM", ("adamw_torch" if tuning_type in ("FULL","LORA") else "paged_adamw_8bit"))
        save_strategy_val = "epoch" if use_eval else "no"
        eval_strategy_val = "epoch" if use_eval else "no"

        _ta = dict(
            output_dir=output_dir,
            num_train_epochs=job.request.get("epochs", 3),
            per_device_train_batch_size=job.request.get("batchSize", 1),
            gradient_accumulation_steps=job.request.get("gradientAccumulationSteps", 8),
            learning_rate=job.request.get("learningRate", 2e-4),
            bf16=True,
            logging_steps=10,
            report_to="none",
            optim=optim_name,
            seed=42,
            data_seed=42,
            save_total_limit=2,
            warmup_ratio=0.05,
        )
        sup = _supported_args(TrainingArguments)
        _put_kw(sup, _ta, "save_strategy", save_strategy_val)
        _put_kw(sup, _ta, "evaluation_strategy", eval_strategy_val, "eval_strategy")
        _put_kw(sup, _ta, "gradient_checkpointing", True)
        _put_kw(sup, _ta, "lr_scheduler_type", "cosine")
        _put_kw(sup, _ta, "save_safetensors", True)
        if use_eval:
            _put_kw(sup, _ta, "load_best_model_at_end", True)
            _put_kw(sup, _ta, "metric_for_best_model", "eval_loss")
            _put_kw(sup, _ta, "greater_is_better", False)
            _put_kw(sup, _ta, "per_device_eval_batch_size", max(1, job.request.get("batchSize", 1)))

        training_args = TrainingArguments(**_ta)

        # ===== 콜백 =====
        class ProgressCallback(TrainerCallback):
            def __init__(self, job_id: str, every_steps: int | None = None):
                self.job_id = job_id
                self.every_steps = every_steps if every_steps else max(1, int(os.getenv("FT_PROGRESS_EVERY_STEPS", "1")))
                self.total_steps = None
            def on_train_begin(self, args, state, control, **kw):
                _update_job_status(None, self.job_id, "running", progress=0)
            def on_train_begin_dataloader(self, args, state, control, **kw):
                try:
                    if state.max_steps and state.max_steps > 0:
                        self.total_steps = int(state.max_steps)
                except Exception:
                    pass
            def on_step_end(self, args, state, control, **kw):
                try:
                    total = self.total_steps or (state.max_steps or 0)
                    if not total or state.global_step % self.every_steps != 0: return
                    pct = int(min(100, max(0, round((state.global_step/total)*100))))
                    _update_job_status(None, self.job_id, "running", progress=pct)
                except Exception:
                    pass
            def on_train_end(self, args, state, control, **kw):
                _update_job_status(None, self.job_id, "running", progress=100)

        class LogCallback(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kw):
                if logs and "loss" in logs:
                    _append_log(log_path, f"[{_now_utc().isoformat()}] step={state.global_step} loss={logs['loss']}")

        def _build_trainer():
            return Trainer(
                model=model,
                args=training_args,
                train_dataset=train_ds,
                eval_dataset=eval_ds,
                tokenizer=tokenizer,  # FutureWarning은 무시 가능 (추후 processing_class로 마이그레이션)
                callbacks=[ProgressCallback(job.job_id), LogCallback()],
            )

        trainer = _build_trainer()

        # ===== 학습 + OOM 세이프 재시도 =====
        import math
        try:
            _append_log(log_path, f"[{_now_utc().isoformat()}] training started...")
            trainer.train()
        except RuntimeError as re:
            if "out of memory" in str(re).lower():
                # 🔻 배치/시퀀스 동시 축소 + **이전 Trainer/그래프 완전 정리**
                old_bs = training_args.per_device_train_batch_size
                new_bs = max(1, old_bs // 2)
                new_len = max(1024, int(max_len * 0.75))
                _append_log(log_path, f"[{_now_utc().isoformat()}] OOM → retry with batch={new_bs}, max_len={new_len}")
                # 메모리 해제
                try:
                    del trainer
                    # _clear_gpu_memory()
                except Exception:
                    pass
                # 재구성
                training_args.per_device_train_batch_size = new_bs
                train_ds = RagDataset(train_data, tokenizer, max_len=new_len)
                eval_ds  = RagDataset(eval_data,  tokenizer, max_len=new_len) if use_eval else None
                trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=train_ds,
                    eval_dataset=eval_ds,
                    tokenizer=tokenizer,
                    callbacks=[ProgressCallback(job.job_id), LogCallback()],
                )
                trainer.train()
            else:
                raise

        # ===== 간단 ROUGE-1 (옵셔널) =====
        def _rouge1_f1(ref: str, cand: str) -> float:
            rw, cw = ref.split(), cand.split()
            if not rw or not cw: return 0.0
            m = sum(1 for w in cw if w in rw)
            p = m/len(cw) if cw else 0.0
            r = m/len(rw) if rw else 0.0
            return 0.0 if (p+r)==0 else 2*(p*r)/(p+r)

        final_rouge = None
        if use_eval:
            model.eval()
            preds, refs = [], []
            device = _get_model_device(model)
            for item in eval_ds:
                inp_ids = item["input_ids"].unsqueeze(0).to(device)
                attn = item["attention_mask"].unsqueeze(0).to(device)
                try:
                    with torch.inference_mode():
                        out_ids = model.generate(
                            inp_ids,
                            attention_mask=attn,
                            max_new_tokens=128,
                            do_sample=False,
                        )[0]
                except Exception:
                    continue
                preds.append(tokenizer.decode(out_ids, skip_special_tokens=True))
                ref_ids = item["labels"]
                ref_ids = torch.where(ref_ids == -100, torch.tensor(tokenizer.pad_token_id), ref_ids)
                refs.append(tokenizer.decode(ref_ids, skip_special_tokens=True))
            scores = [_rouge1_f1(r, p) for r, p in zip(refs, preds)]
            if scores: final_rouge = sum(scores) / len(scores)
            _update_job_status(None, job.job_id, "running", rough=int((final_rouge or 0)*100), extras={"rouge1F1": final_rouge})

        # ===== 저장 =====
        def _save_stage(stage: str, pct: int):
            _append_log(log_path, f"[{_now_utc().isoformat()}] save:{stage} {pct}%")
            _update_job_status(None, job.job_id, "running", extras={"saveStage": stage, "saveProgress": pct})

        _save_stage("start", 5)
        _append_log(log_path, f"[{_now_utc().isoformat()}] saving model...")

        if tuning_type in ("LORA", "QLORA"):
            try:
                model.save_pretrained(output_dir)     # 어댑터만 저장
                _save_stage("adapter", 70)
                tokenizer.save_pretrained(output_dir)
                _save_stage("tokenizer", 90)
                _append_log(log_path, f"[{_now_utc().isoformat()}] saved adapters → {output_dir}")
            except Exception as e:
                trainer.save_model(output_dir)
                _save_stage("model", 70)
                tokenizer.save_pretrained(output_dir)
                _save_stage("tokenizer", 90)
                _append_log(log_path, f"[{_now_utc().isoformat()}] fallback save (trainer.save_model): {e}")
            # MXFP4 병합 저장은 선택(기본 비활성)
            if is_mxfp4 and os.getenv("FT_UNSLOTH_MERGE_SAVE","0") == "1":
                try:
                    model.save_pretrained_merged(output_dir, tokenizer, save_method="mxfp4")
                    _append_log(log_path, f"[{_now_utc().isoformat()}] merged MXFP4 saved → {output_dir}")
                except Exception as e:
                    _append_log(log_path, f"[{_now_utc().isoformat()}] MXFP4 merge save failed: {e}, adapters only.")
        else:
            trainer.save_model(output_dir)
            _save_stage("model", 70)
            tokenizer.save_pretrained(output_dir)
            _save_stage("tokenizer", 90)

        _save_stage("done", 100)

        finish_job_success(
            job.job_id, save_name_with_suffix, job.category, tuning_type, final_rouge, subcategory=job.request.get("subcategory"),
        )
        logger.info(f"Fine-tuning succeeded jobId={job.job_id} save={save_name_with_suffix} type={tuning_type}")
        _append_log(log_path, f"[{_now_utc().isoformat()}] job {job.job_id} succeeded")

    except Exception as e:
        logger.error(f"Fine-tuning failed jobId={job.job_id} error={e}")
        try:
            _append_log(os.path.join(STORAGE_MODEL_ROOT, save_name_with_suffix, "train.log"), f"[{_now_utc().isoformat()}] error: {e}")
        except Exception:
            pass
        
        fail_job(job.job_id, str(e))
        
    finally:
        # 자원 정리
        try:
            del trainer
        except Exception:
            pass
        try:
            del model, tokenizer
        except Exception:
            pass
        try:
            import torch
            gc.collect()
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass
        try:
            for k, v in old_cache.items():
                if v is None: os.environ.pop(k, None)
                else: os.environ[k] = v
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass
        # conn.close()  <-- 제거됨

# ===== Public APIs =====
def _log_to_save_name(save_name_with_suffix: str, message: str):
    try:
        out_dir = _ensure_output_dir(save_name_with_suffix)
        log_path = os.path.join(out_dir, "train.log")
        _append_log(log_path, f"[{_now_utc().isoformat()}] {message}")
    except Exception:
        pass

def start_fine_tuning(category: str, body: FineTuneRequest) -> Dict[str, Any]:
    # startNow와 startAt 동시 지정 방지
    if body.startNow and body.startAt:
        raise BadRequestError("startNow와 startAt은 동시에 사용할 수 없습니다. (둘 중 하나만)")

    # body.category를 최종 신뢰(하위호환을 위해 인수 category는 fallback)
    category = (body.category or category).lower()
    suffix = (body.tuningType or "QLORA").upper()
    # 카테고리까지 포함하여 저장 폴더/모델명을 구분 (예: name-QLORA-qna)
    save_name_with_suffix = f"{body.saveModelName}-{suffix}-{category}"
    _ensure_output_dir(save_name_with_suffix)

    # 중복 실행 차단 (락 파일)
    import fcntl
    out_dir = _ensure_output_dir(save_name_with_suffix)
    lock_path = os.path.join(out_dir, ".run.lock")
    lock_f = open(lock_path, "w")
    try:
        fcntl.flock(lock_f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock_f.write(f"locked at {_now_utc().isoformat()}\n")
        lock_f.flush()
    except BlockingIOError:
        lock_f.close()
        raise BadRequestError(f"another fine-tune is already running for {save_name_with_suffix}")

    base_dir = _resolve_model_dir(body.baseModelName)
    if not _has_model_signature(base_dir):
        msg = f"base model not found: {base_dir}"
        logger.error(msg)
        _log_to_save_name(save_name_with_suffix, msg)
        raise BadRequestError(msg)

    job_id = f"ft-job-{uuid.uuid4().hex[:12]}"

    try:
        train_path = _resolve_train_path(body.trainSetFile)
        dataset_id = _insert_dataset_if_needed(None, train_path, category) # conn 인자 None 전달
        # 예약 시간 파싱
        scheduled_at_iso = None
        delay_sec = 0.0
        if body.startAt:
            try:
                scheduled_dt = datetime.fromisoformat(body.startAt)
                # timezone 정보가 없으면 KST로 가정
                if scheduled_dt.tzinfo is None:
                    kst = timezone(timedelta(hours=9))
                    scheduled_dt = scheduled_dt.replace(tzinfo=kst)
                now_dt = _now_utc()
                delta = (scheduled_dt - now_dt).total_seconds()
                if delta > 1.0:
                    delay_sec = float(delta)
                    scheduled_at_iso = scheduled_dt.isoformat()
            except Exception:
                scheduled_at_iso = None
                delay_sec = 0.0

        _insert_job(
            None, category, body, job_id, save_name_with_suffix, dataset_id, # conn 인자 None
            initial_status=("scheduled" if delay_sec > 1.0 else "queued"),
            scheduled_at=scheduled_at_iso,
        )
    except Exception as e:
        _log_to_save_name(save_name_with_suffix, f"db insert failed: {e}")
        logger.error(f"fine-tuning init failed: {e}")
        raise InternalServerError(f"fine-tuning init failed: {e}")
    finally:
        pass
        # conn.close() <-- 제거됨

    job = FineTuneJob(job_id=job_id, category=category, request=body.model_dump())

    # 🔹 즉시 실행이면 예약(startAt)은 **무시**해서 중복 실행을 원천 차단
    if body.startNow:
        def _launch():
            _run_training_inline(job, save_name_with_suffix)
        t = threading.Thread(target=_launch, daemon=True)
        t.start()
        logger.info(
            f"Fine-tuning started immediately jobId={job.job_id} category={category} base={body.baseModelName} "
            f"save={save_name_with_suffix} tuning={body.tuningType or 'QLORA'}"
        )
        return {"jobId": job_id, "started": True}
    else:
        # 예약 실행 또는 큐에만 등록
        if body.startAt:
            # 예약 실행
            def _launch():
                _run_training_inline(job, save_name_with_suffix)

            # 예약 딜레이 재계산
            delay_sec = 0.0
            try:
                scheduled_dt = datetime.fromisoformat(body.startAt)
                # timezone 정보가 없으면 KST로 가정
                if scheduled_dt.tzinfo is None:
                    kst = timezone(timedelta(hours=9))
                    scheduled_dt = scheduled_dt.replace(tzinfo=kst)
                now_dt = _now_utc()
                delta = (scheduled_dt - now_dt).total_seconds()
                if delta > 1.0:
                    delay_sec = float(delta)
            except Exception:
                delay_sec = 0.0

            if delay_sec > 1.0:
                t = threading.Timer(delay_sec, _launch)
                t.daemon = True
                t.start()
                logger.info(
                    f"Fine-tuning scheduled jobId={job.job_id} at {scheduled_dt.isoformat()} category={category} base={body.baseModelName} save={save_name_with_suffix}"
                )
            else:
                # 예약 시간이 이미 지났으면 즉시 실행
                t = threading.Thread(target=_launch, daemon=True)
                t.start()
                logger.info(
                    f"Fine-tuning started (scheduled time passed) jobId={job.job_id} category={category} base={body.baseModelName} save={save_name_with_suffix}"
                )
        else:
            # 큐에만 등록 (스케줄러가 나중에 실행)
            try:
                _update_job_status(None, job_id, "queued", extras={"reserved": True, "reservedAt": _now_local_str()})
            finally:
                pass
            logger.info(
                f"Fine-tuning queued (not started) jobId={job.job_id} category={category} base={body.baseModelName} save={save_name_with_suffix}"
            )

    return {"jobId": job_id, "started": bool(body.startNow)}

def get_fine_tuning_status(job_id: str) -> Dict[str, Any]:
    with get_session() as s:
        # ORM 버전
        stmt = select(ORMJob).where(ORMJob.provider_job_id == job_id)
        row = s.execute(stmt).scalars().first()
        
        if not row:
            return {"error": "job not found", "jobId": job_id}
            
        metrics = {}
        if row.metrics:
            try:
                metrics = json.loads(row.metrics) or {}
            except Exception:
                metrics = {}

        # Liveness check
        row_status = row.status
        try:
            if row_status == "running":
                hb = metrics.get("heartbeatAt")
                if hb:
                    from dateutil import parser as dtparser
                    last = dtparser.parse(hb)
                    now = _now_utc()
                    if (now - last).total_seconds() > FT_HEARTBEAT_TIMEOUT_SEC:
                        fail_job(job_id, "stale heartbeat")
                        row_status = "failed"
        except Exception:
            pass

        return {
            "jobId": row.provider_job_id,
            "status": row_status,
            "learningProgress": int(metrics.get("learningProgress", 0)),
            "roughScore": int(metrics.get("roughScore", 0)),
            "rouge1F1": metrics.get("rouge1F1"),
            "saveProgress": int(metrics.get("saveProgress", 0)),
            "saveStage": metrics.get("saveStage"),
            "error": metrics.get("error"),
        }

def list_feedback_datasets() -> dict:
    """
    ./storage/train_data 안에서 파일명 패턴
    'feedback_{task}_p{prompt}.csv'에 매칭되는 모든 CSV를 테스크별로 반환.
    반환 경로는 상대경로('./storage/train_data/...')를 제공한다.
    """
    REL_ROOT = "./storage/train_data"

    try:
        names = sorted(os.listdir(TRAIN_DATA_ROOT))
    except FileNotFoundError:
        names = []

    entries = []
    for name in names:
        m = _FEEDBACK_FILE_RE.match(name)
        if not m:
            continue
        task = m.group(1).lower()
        prompt = int(m.group(2))
        abs_path = os.path.join(TRAIN_DATA_ROOT, name)
        if not os.path.isfile(abs_path):
            continue
        st = os.stat(abs_path)
        mtime_dt = datetime.fromtimestamp(st.st_mtime)
        entries.append({
            "task": task,                               # qna | doc_gen | summary
            "file": name,                               # ex) feedback_qna_p0.csv
            "prompt": prompt,                           # 정수 p값
            "bytes": st.st_size,                        # 파일 크기
            "mtime": mtime_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "mtime_iso": mtime_dt.isoformat(),
            "path": f"{REL_ROOT}/{name}",               # 상대경로 (Docker 고려)
            "downloadUrl": f"/v1/admin/llm/feedback-datasets?file={quote_plus(name)}",
        })

    groups = {"qna": [], "doc_gen": [], "summary": []}
    for e in entries:
        groups[e["task"]].append(e)
    for k in groups:
        groups[k].sort(key=lambda x: x["prompt"])

    return {
        "root": REL_ROOT,
        "pattern": "feedback_{task}_p{prompt}.csv",
        "total": len(entries),
        "counts": {k: len(v) for k, v in groups.items()},
        "groups": groups,
    }

def resolve_feedback_download(file: str) -> tuple[str, str]:
    """
    다운로드용 파일 검증/해결:
    - basename만 허용 (경로 탈출 방지)
    - 파일명 패턴 확인
    - ./storage/train_data 내부 존재 확인
    성공 시: (abs_path, filename) 반환
    """
    if os.path.basename(file) != file:
        raise BadRequestError("basename만 허용합니다.")
    m = _FEEDBACK_FILE_RE.match(file)
    if not m:
        raise BadRequestError("잘못된 파일명 형식입니다. (feedback_{task}_p{n}.csv)")
    abs_path = os.path.join(TRAIN_DATA_ROOT, file)
    if not os.path.isfile(abs_path):
        # 라우터에서 404로 매핑하기 위해 표준 예외 사용
        raise FileNotFoundError(f"not found in {TRAIN_DATA_ROOT}: {file}")
    return abs_path, file
