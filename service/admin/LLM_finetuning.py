# service/admin/LLM_finetuning.py
from __future__ import annotations

import json
import os
import threading
import time
import uuid
import sqlite3
import tempfile
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional
from pathlib import Path
from contextlib import contextmanager

from pydantic import BaseModel, Field, field_validator, model_validator
from utils import get_db, logger
from errors.exceptions import BadRequestError, InternalServerError
try:
    import unsloth  # Unsloth는 transformers/peft보다 먼저 import
except Exception:
    pass

logger = logger(__name__)


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
FT_HEARTBEAT_TIMEOUT_SEC = int(os.getenv("FT_HEARTBEAT_TIMEOUT_SEC", "60"))

# ===== Paths =====
BASE_BACKEND = Path(os.getenv("COREIQ_BACKEND_ROOT", str(Path(__file__).resolve().parents[2])))  # backend/
# Force DB to pps_rag.db across the process (can be overridden by env before start)
import os as _os
_os.environ.setdefault("COREIQ_DB", str(BASE_BACKEND / "storage" / "pps_rag.db"))
STORAGE_MODEL_ROOT = os.getenv("STORAGE_MODEL_ROOT", str(BASE_BACKEND / "storage" / "model"))
TRAIN_DATA_ROOT   = os.getenv("TRAIN_DATA_ROOT", str(BASE_BACKEND / "storage" / "train_data"))

# ===== SQLAlchemy ORM (Session) =====
from sqlalchemy import create_engine, select, func
from sqlalchemy.orm import sessionmaker
from storage.db_models import (
    LlmModel, FineTuneDataset, FineTuneJob as ORMJob, FineTunedModel
)
DB_URL = f"sqlite:///{os.environ.get('COREIQ_DB', str(BASE_BACKEND / 'storage' / 'pps_rag.db'))}"
_engine = create_engine(DB_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False)

# ---- Portable path helper ----
def _to_rel(p: str) -> str:
    """Return `p` as a path **relative** to the backend root so DB records do
    not depend on absolute host paths (useful inside Docker). If conversion
    fails, the original path is returned unchanged."""
    try:
        return os.path.relpath(p, BASE_BACKEND)
    except Exception:
        return p

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
        with SessionLocal() as s:
            # 정확한 이름으로 먼저 찾기
            model = s.execute(
                select(LlmModel).where(LlmModel.name == name_or_path)
            ).scalar_one_or_none()
            
            if not model:
                # 카테고리 접미사 제거 후 다시 찾기
                def _strip_cat(n: str) -> str:
                    for suf in ("-qa", "-doc_gen", "-summary"):
                        if n.endswith(suf):
                            return n[: -len(suf)]
                    return n
                
                base_name = _strip_cat(name_or_path)
                if base_name != name_or_path:
                    model = s.execute(
                        select(LlmModel).where(LlmModel.name == base_name)
                    ).scalar_one_or_none()
            
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
    category: str = Field(..., description="qa | doc_gen | summary")
    subcategory: Optional[str] = Field(None, description="세부 테스크. 현재 주로 doc_gen에서 사용")

    baseModelName: str
    saveModelName: str
    systemPrompt: str
    batchSize: int = 4
    epochs: int = 3
    learningRate: float = 2e-4
    overfittingPrevention: bool = True
    trainSetFile: str
    gradientAccumulationSteps: int = 8
    quantizationBits: Optional[int] = Field(
        default=None, description="QLORA 전용: 양자화 비트 (4 또는 8)",
    )
    tuningType: Optional[str] = Field(
        default="QLORA",
        description="파인튜닝 방식: LORA | QLORA | FULL",
        pattern="^(LORA|QLORA|FULL)$",
    )
    startAt: Optional[str] = Field(
        default=None, description="예약 시작 ISO8601 (예: 2025-09-19T13:00:00)"
    )
    startNow: bool = Field(
        default=False, description="즉시 실행 여부 (True: 바로 시작, False: 예약만 등록)"
    )

    @field_validator("category")
    @classmethod
    def _v_category(cls, v: str) -> str:
        allowed = {"qa", "doc_gen", "summary"}
        vv = (v or "").strip().lower()
        if vv not in allowed:
            raise ValueError(f"category must be one of {sorted(allowed)}")
        return vv

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

# ===== List train dirs =====
def list_train_data_dirs() -> Dict[str, Any]:
    items = []
    root_path = TRAIN_DATA_ROOT
    if not os.path.isdir(root_path):
        try:
            logger.warning(f"train data root not found: {root_path}")
        except Exception:
            pass
        return {"root": _to_rel(root_path), "dirs": items}

    # Persist discovered train data directories into DB with relative paths
    conn = get_db()
    try:
        try:
            for entry in os.scandir(root_path):
                if not entry.is_dir():
                    continue
                try:
                    mtime = datetime.fromtimestamp(entry.stat().st_mtime, tz=timezone.utc).isoformat()
                except Exception:
                    logger.exception(f"failed to stat dir: {entry.path}")
                    mtime = None

                file_count = 0
                try:
                    for _, _, files in os.walk(entry.path):
                        file_count += len(files)
                except Exception:
                    logger.exception(f"failed to walk dir: {entry.path}")

                # Only consider directories that actually contain files
                if file_count <= 0:
                    try:
                        logger.info(f"skip empty train dir: {entry.path}")
                    except Exception:
                        pass
                    continue

                abs_dir_path = os.path.join(root_path, entry.name)
                rel_dir_path = _to_rel(abs_dir_path)

                # Insert if not exists
                try:
                    dataset_id = _insert_dataset_if_needed(conn, abs_dir_path, "unknown")
                    # Best-effort update of record_count, if column exists
                    try:
                        cur = conn.cursor()
                        if _has_column(conn, "fine_tune_datasets", "record_count"):
                            cur.execute(
                                "UPDATE fine_tune_datasets SET record_count=?, updated_at=? WHERE id=?",
                                (file_count, _now_utc().isoformat(), dataset_id),
                            )
                            conn.commit()
                    except Exception:
                        logger.exception("failed to update record_count for dataset_id=%s", dataset_id)
                except Exception as e:
                    logger.error(f"failed to upsert train dataset for {abs_dir_path}: {e}")

                items.append({
                    "name": entry.name,
                    "path": rel_dir_path,
                    "fileCount": file_count,
                    "modifiedAt": mtime,
                })
        except Exception:
            logger.exception(f"failed to scan train data root: {root_path}")

        # Fallback: if no items discovered via filesystem, read last known datasets from DB
        if len(items) == 0:
            try:
                cur = conn.cursor()
                rel_root = _to_rel(root_path)
                like_prefix = rel_root if rel_root.endswith(os.sep) else rel_root + os.sep
                cur.execute(
                    "SELECT name, path, record_count, updated_at, created_at FROM fine_tune_datasets WHERE path LIKE ?",
                    (like_prefix + '%',),
                )
                rows = cur.fetchall() or []
                for r in rows:
                    name_val = r[0] if isinstance(r, tuple) else r["name"]
                    rel_path_val = r[1] if isinstance(r, tuple) else r["path"]
                    db_count_val = (r[2] if isinstance(r, tuple) else r.get("record_count") if isinstance(r, dict) else None) or 0
                    modified_val = (r[3] if isinstance(r, tuple) else r.get("updated_at") if isinstance(r, dict) else None) or (r[4] if isinstance(r, tuple) else r.get("created_at") if isinstance(r, dict) else None)

                    # Recompute file count from filesystem for accuracy
                    recomputed_count = db_count_val
                    try:
                        abs_dir = rel_path_val if os.path.isabs(rel_path_val) else os.path.join(str(BASE_BACKEND), rel_path_val)
                        if os.path.isdir(abs_dir):
                            cnt = 0
                            for _, _, files in os.walk(abs_dir):
                                cnt += len(files)
                            recomputed_count = cnt
                            # Update DB with fresh count if column exists
                            try:
                                if _has_column(conn, "fine_tune_datasets", "record_count"):
                                    cur.execute(
                                        "UPDATE fine_tune_datasets SET record_count=?, updated_at=? WHERE name=? AND path=?",
                                        (recomputed_count, _now_utc().isoformat(), name_val, rel_path_val),
                                    )
                                    conn.commit()
                            except Exception:
                                logger.exception("failed to refresh record_count from fallback for path=%s", rel_path_val)
                        else:
                            logger.info(f"fallback path not found: {abs_dir}")
                    except Exception:
                        logger.exception("failed to recompute file count for fallback path=%s", rel_path_val)

                    items.append({
                        "name": name_val,
                        "path": rel_path_val,
                        "fileCount": int(recomputed_count or 0),
                        "modifiedAt": modified_val,
                    })
                if len(items) == 0:
                    logger.info("no train dirs found in filesystem or database")
            except Exception:
                logger.exception("failed to read train datasets from DB for fallback")
    finally:
        try:
            conn.close()
        except Exception:
            logger.exception("failed to close DB connection in list_train_data_dirs")

    items.sort(key=lambda x: (x["modifiedAt"] or ""), reverse=True)
    return {"root": _to_rel(root_path), "dirs": items}

# ===== DB ops =====
def _insert_dataset_if_needed(conn, path: str, category: str) -> int:
    # ==== ORM 버전 ====
    rel_path = _to_rel(path)
    with SessionLocal() as s:
        exist = s.execute(
            select(FineTuneDataset).where(FineTuneDataset.path == rel_path)
        ).scalar_one_or_none()
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

    with SessionLocal() as s:
        row = ORMJob(
            provider_job_id=job_id,
            dataset_id=dataset_id,
            status=initial_status,
            started_at=_now_utc(),
            metrics=json.dumps(metrics, ensure_ascii=False),
        )
        s.add(row)
        s.commit()
        s.refresh(row)
        return int(row.id)

def _update_job_status(conn, job_id: str, status: str, progress: int | None = None, rough: int | None = None, extras: dict | None = None, _retries: int = 3):
    """
    Update job status and optional metrics. Progress percentage is now persisted.
    - If the fine_tune_jobs.metrics column doesn't exist, fall back to updating only status.
    """
    for attempt in range(_retries):
        try:
            cur = conn.cursor()
            has_metrics_col = _has_column(conn, "fine_tune_jobs", "metrics")

            metrics = {}
            if has_metrics_col:
                try:
                    cur.execute("SELECT metrics FROM fine_tune_jobs WHERE provider_job_id=?", (job_id,))
                    row = cur.fetchone()
                    if row and row["metrics"]:
                        try:
                            metrics = json.loads(row["metrics"]) or {}
                        except Exception:
                            metrics = {}
                except Exception:
                    metrics = {}

            # 진행률 반영: 요청값을 그대로 저장(사용자 요청대로 max 적용 제거)
            if progress is not None:
                metrics["learningProgress"] = int(progress)

            if rough is not None:
                metrics["roughScore"] = int(rough)

            if extras:
                try:
                    metrics.update(extras)
                except Exception:
                    pass

            # Update heartbeat timestamp for liveness detection
            try:
                metrics["heartbeatAt"] = _now_utc().isoformat()
            except Exception:
                pass

            if has_metrics_col:
                cur.execute(
                    "UPDATE fine_tune_jobs SET status=?, metrics=? WHERE provider_job_id=?",
                    (status, json.dumps(metrics, ensure_ascii=False), job_id),
                )
            else:
                cur.execute(
                    "UPDATE fine_tune_jobs SET status=? WHERE provider_job_id=?",
                    (status, job_id),
                )
            conn.commit()
            try:
                # Additional: persist status change to train.log for easier tracing
                out_dir = _resolve_out_dir_by_job(conn, job_id)
                if out_dir:
                    log_path = os.path.join(out_dir, "train.log")
                    msg = f"status={status} progress={metrics.get('learningProgress','-')} rough={metrics.get('roughScore','-')} extras={extras or {}}"
                    _append_log(log_path, f"[{_now_utc().isoformat()}] {msg}")
            except Exception:
                pass
            return
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower() and attempt < _retries - 1:
                time.sleep(0.2 * (attempt + 1))
                continue
            try:
                logger.error(f"update_job_status operational error: {e}")
            except Exception:
                pass
            raise

def _has_column(conn, table: str, column: str) -> bool:
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    cols = [r[1] for r in cur.fetchall()]
    return column in cols

def _ensure_ftm_rouge_column(conn):
    try:
        if not _has_column(conn, "fine_tuned_models", "rouge1_f1"):
            cur = conn.cursor()
            cur.execute("ALTER TABLE fine_tuned_models ADD COLUMN rouge1_f1 FLOAT")
            conn.commit()
    except Exception:
        # ignore if cannot add (older sqlite)
        pass

def _finish_job_success(conn, job_id: str, model_name: str, category: str, tuning_type: str, final_rouge: Optional[float] = None, subcategory: Optional[str]=None):
    """
    파인튜닝 완료시 결과를 DB에 반영한다.
    - llm_models:
        - type: FULL | LORA | QLORA (대문자)
        - LORA/QLORA:  model_path=어댑터 폴더(상대), mather_path=베이스 폴더(상대)
        - FULL:        model_path=통짜 저장 폴더(상대), mather_path=NULL
    - fine_tuned_models:
        - type 동일(대문자)
        - rouge1_f1: 최종 ROUGE 저장
    """
    rel_model_path = _to_rel(os.path.join(STORAGE_MODEL_ROOT, model_name))  # 출력(어댑터 or FULL 저장 폴더)
    mdl_type = (tuning_type or "QLORA").upper()

    with SessionLocal() as s:
        # --- job/metrics에서 baseModelName 추출 ---
        job_row = s.execute(select(ORMJob).where(ORMJob.provider_job_id == job_id)).scalar_one_or_none()
        base_model_name = None
        if job_row and job_row.metrics:
            try:
                mt = json.loads(job_row.metrics) or {}
                base_model_name = (mt.get("hyperparameters") or {}).get("baseModelName")
            except Exception:
                base_model_name = None

        # job 상태 업데이트
        if job_row:
            job_row.status = "succeeded"
            job_row.finished_at = _now_utc()
            s.add(job_row)

        # --- 베이스 모델 경로 준비 (상대 저장) ---
        base_model_id = None
        base_model_rel_path = None
        if base_model_name:
            abs_base = _resolve_model_dir(base_model_name)      # 물리 경로
            base_model_rel_path = _to_rel(abs_base)             # 상대 경로
            base_row = s.execute(select(LlmModel).where(LlmModel.name == base_model_name)).scalar_one_or_none()
            if not base_row:
                base_row = LlmModel(
                    provider="hf",
                    name=base_model_name,
                    revision=0,
                    model_path=None,       # BASE는 어댑터 없음
                    mather_path=base_model_rel_path,
                    category="all",
                    # subcategory는 BASE엔 적용하지 않음
                    type="BASE",
                    is_default=False,
                    is_active=True,
                    trained_at=None,
                )
                s.add(base_row)
                s.flush()
            base_model_id = int(base_row.id)

        # --- FT 결과 모델(LlmModel) upsert ---
        m = s.execute(select(LlmModel).where(LlmModel.name == model_name)).scalar_one_or_none()
        if m is None:
            m = LlmModel(
                provider="hf",
                name=model_name,
                revision=0,
                category=category,
                # 서브카테고리 제약 제거(메타만 보유), 과거 데이터 호환을 위해 None으로 저장
                type=mdl_type,
                is_default=False,
                is_active=True,
                trained_at=_now_utc(),
            )

        # 타입별 경로 저장 정책
        if mdl_type in ("LORA", "QLORA"):
            # 어댑터 경로는 model_path, 베이스는 mather_path
            m.model_path = rel_model_path
            m.mather_path = base_model_rel_path
        else:  # FULL
            m.model_path = rel_model_path  # 통짜 저장 폴더
            m.mather_path = None

        m.category = category
        # m.subcategory = subcategory  # <- llm_models에 서브태스크 제약 제거. 필요시 메타로만 유지.
        m.is_active = True
        m.trained_at = _now_utc()
        s.add(m)
        s.flush()  # m.id 확보

        # --- FineTunedModel insert ---
        ftm = FineTunedModel(
            model_id=m.id,
            job_id=job_row.id if job_row else None,
            provider_model_id=model_name,
            lora_weights_path=(rel_model_path if mdl_type in ("LORA", "QLORA") else None),
            type=mdl_type,                    # 대문자 저장
            is_active=True,
            base_model_id=base_model_id,
            base_model_path=base_model_rel_path,
        )
        try:
            setattr(ftm, "rouge1_f1", (final_rouge if final_rouge is not None else None))
        except Exception:
            pass

        s.add(ftm)
        s.commit()

def _resolve_out_dir_by_job(conn, job_id: str) -> Optional[str]:
    cur = conn.cursor()
    # First try portable path: metrics only
    cur.execute("SELECT metrics FROM fine_tune_jobs WHERE provider_job_id=?", (job_id,))
    row = cur.fetchone()
    save_name = None
    if row and row["metrics"]:
        try:
            mt = json.loads(row["metrics"]) or {}
            hp = mt.get("hyperparameters") or {}
            save_name = hp.get("saveModelName")
        except Exception:
            save_name = None
    # If still missing, try optional hyperparameters column
    if not save_name:
        try:
            cur.execute("SELECT hyperparameters FROM fine_tune_jobs WHERE provider_job_id=?", (job_id,))
            r2 = cur.fetchone()
            if r2 and r2["hyperparameters"]:
                try:
                    hp = json.loads(r2["hyperparameters"]) or {}
                    save_name = hp.get("saveModelName")
                except Exception:
                    save_name = None
        except Exception:
            # column may not exist in this DB
            pass
    if not save_name:
        return None
    return os.path.join(STORAGE_MODEL_ROOT, save_name)

def get_fine_tuning_logs(job_id: str, tail: int = 200) -> Dict[str, Any]:
    conn = get_db()
    try:
        out_dir = _resolve_out_dir_by_job(conn, job_id)
        if not out_dir:
            return {"jobId": job_id, "log": "(no output dir yet)", "lines": []}
        log_path = os.path.join(out_dir, "train.log")
        if not os.path.isfile(log_path):
            return {"jobId": job_id, "log": "(log not created yet)", "lines": []}
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        tail = max(1, min(int(tail), 2000))
        return {"jobId": job_id, "lines": [l.rstrip("\n") for l in lines[-tail:]]}
    finally:
        conn.close()

# ===== 시뮬레이터 제거됨 =====

# ===== Training (inline, real) =====
def _run_training_inline(job: FineTuneJob, save_name_with_suffix: str):
    """
    실제 파인튜닝 실행 (CSV -> 회화 프롬프트 -> RagDataset)
    FULL:    bf16 full finetune (no 4-bit)
    LORA:    bf16 + LoRA
    QLORA:   4-bit + LoRA
    """
    conn = get_db()
    try:
        _update_job_status(conn, job.job_id, "running", progress=0)

        out_dir = _ensure_output_dir(save_name_with_suffix)
        tuning_type = (job.request.get("tuningType") or "QLORA").upper()
        if tuning_type in ("LORA", "QLORA"):
            _ensure_lora_marker(out_dir, tuning_type)
        log_path = os.path.join(out_dir, "train.log")
        # Truncate old log to avoid confusion with previous runs having same save name
        try:
            with open(log_path, "w", encoding="utf-8") as _tmp:
                _tmp.write("")
        except Exception:
            pass
        _append_log(log_path, f"[{_now_utc().isoformat()}] job {job.job_id} started (INLINE)")
        # console log
        logger.info(
            f"Fine-tuning started jobId={job.job_id} type={tuning_type} base={job.request.get('baseModelName')} "
            f"save={save_name_with_suffix} data={job.request.get('trainSetFile')}"
        )

        # ✅ 캐시 설정 (임시 디렉토리 사용)
        tmpdir = tempfile.mkdtemp(prefix="ft_cache_")
        cache_keys = ["HF_HOME", "TRANSFORMERS_CACHE", "HF_DATASETS_CACHE", "XDG_CACHE_HOME", "UNSLOTH_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR"]
        old_cache = {k: os.environ.get(k) for k in cache_keys}
        for k in cache_keys:
            os.environ[k] = tmpdir

        # imports
        try:
            try:
                from unsloth import FastLanguageModel  # type: ignore
            except Exception:
                FastLanguageModel = None  # FULL/LORA 경로에서 필요 없음
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
            try:
                _update_job_status(conn, job.job_id, "failed", extras={"error": f"import error: {e}"})
            except Exception:
                cur = conn.cursor()
                cur.execute("UPDATE fine_tune_jobs SET status=? WHERE provider_job_id=?", ("failed", job.job_id))
                conn.commit()
            logger.error(f"Fine-tuning failed jobId={job.job_id} error=import error: {e}")
            return

        # ===== Build dataset from CSV =====
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
                try:
                    _update_job_status(conn, job.job_id, "failed", extras={"error": f"csv load failed: {e}"})
                except Exception:
                    cur = conn.cursor()
                    cur.execute("UPDATE fine_tune_jobs SET status=? WHERE provider_job_id=?", ("failed", job.job_id))
                    conn.commit()
                logger.error(f"Fine-tuning failed jobId={job.job_id} error=csv load failed: {e}")
                return

        conversations = []
        for _, row in df.iterrows():
            conversations.append({
                "conversations": [
                    {"from": "user", "value": build_prompt(row.get("Chunk_Context", ""), row.get("Question", ""))},
                    {"from": "assistant", "value": row.get("Answer", "")},
                ]
            })

        # ===== Deterministic train/test split (random_state=42) =====
        total_examples = len(conversations)
        if total_examples >= 2:
            eval_size = max(1, int(round(total_examples * 0.3)))  # 30% eval
        else:
            eval_size = 0

        if eval_size > 0:
            import random as _py_random
            indices = list(range(total_examples))
            rng = _py_random.Random(42)
            rng.shuffle(indices)
            eval_indices = set(indices[:eval_size])
            train_conversations = [conversations[i] for i in indices[eval_size:]]
            eval_conversations = [conversations[i] for i in indices[:eval_size]]
        else:
            train_conversations = conversations
            eval_conversations = []

        class RagDataset(torch.utils.data.Dataset):
            def __init__(self, data, tokenizer, max_len=4096):
                self.data = data
                self.tokenizer = tokenizer
                self.max_len = max_len
            def __len__(self):
                return len(self.data)
            def __getitem__(self, idx):
                dialog = self.data[idx]["conversations"]
                messages = [
                    {"role": "system", "content": "You are Qwen, a helpful assistant."},
                    {"role": "user", "content": dialog[0]["value"]},
                    {"role": "assistant", "content": dialog[1]["value"]},
                ]
                full_text = self.tokenizer.apply_chat_template(messages, tokenize=False)
                enc = self.tokenizer(
                    full_text,
                    max_length=self.max_len,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )
                input_ids = enc.input_ids[0]
                assist_prompt = self.tokenizer.apply_chat_template(
                    messages[:-1], tokenize=False, add_generation_prompt=True
                )
                prefix_len = len(self.tokenizer(assist_prompt).input_ids)
                labels = input_ids.clone()
                labels[:prefix_len] = -100
                return {"input_ids": input_ids, "attention_mask": enc.attention_mask[0], "labels": labels}

        base_folder = _resolve_model_dir(job.request.get("baseModelName"))
        if not _has_model_signature(base_folder):
            _append_log(log_path, f"[{_now_utc().isoformat()}] base model not found: {base_folder}")
            try:
                _update_job_status(conn, job.job_id, "failed", extras={"error": f"base model not found: {base_folder}"})
            except Exception:
                cur = conn.cursor()
                cur.execute("UPDATE fine_tune_jobs SET status=? WHERE provider_job_id=?", ("failed", job.job_id))
                conn.commit()
            logger.error(f"Fine-tuning failed jobId={job.job_id} error=base model not found: {base_folder}")
            return
        model_path = base_folder
        output_dir = os.path.join(STORAGE_MODEL_ROOT, save_name_with_suffix)

        # ===== Load model/tokenizer per tuning type =====
        if tuning_type == "FULL":
            # Full finetune (no 4-bit)
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
            if tokenizer.pad_token_id is None:
                if getattr(tokenizer, "eos_token_id", None) is not None:
                    tokenizer.pad_token_id = tokenizer.eos_token_id
                else:
                    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
                    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<|pad|>")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                local_files_only=True,
            )
            model.gradient_checkpointing_enable()
            for p in model.parameters():
                p.requires_grad = True

        elif tuning_type == "LORA":
            # LoRA (no 4-bit)
            try:
                from peft import LoraConfig, get_peft_model  # type: ignore
            except Exception as e:
                _append_log(log_path, f"[{_now_utc().isoformat()}] peft not installed for LORA: {e}")
                _update_job_status(conn, job.job_id, "failed", extras={"error": f"peft not installed: {e}"})
                logger.error(f"Fine-tuning failed jobId={job.job_id} error=peft not installed: {e}")
                return
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
            if tokenizer.pad_token_id is None:
                if getattr(tokenizer, "eos_token_id", None) is not None:
                    tokenizer.pad_token_id = tokenizer.eos_token_id
                else:
                    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
                    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<|pad|>")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                local_files_only=True,
            )
            model.gradient_checkpointing_enable()
            # LoRA target discovery
            candidate_keywords = [
                "q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj",
                "down_proj","w1","w2","c_proj","c_attn"
            ]
            lora_target_modules = sorted({
                name.split(".")[-1]
                for name, _ in model.named_modules()
                if any(k in name for k in candidate_keywords)
            })
            lora_config = LoraConfig(
                r=64,
                lora_alpha=16,
                target_modules=lora_target_modules or None,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_config)

        else:
            # QLORA
            is_mxfp4 = _looks_like_mxfp4_model(model_path) or _looks_like_mxfp4_model(job.request.get("baseModelName"))

            if is_mxfp4:
                # ---- MXFP4 (gpt-oss) 전용: Unsloth 로더 사용 ----
                try:
                    from unsloth import FastLanguageModel  # type: ignore
                except Exception as e:
                    _append_log(log_path, f"[{_now_utc().isoformat()}] Unsloth not installed: {e}")
                    _update_job_status(conn, job.job_id, "failed", extras={"error": f"unsloth not installed: {e}"})
                    return

                max_len = int(job.request.get("max_len", 4096))
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=model_path,
                    dtype=None,
                    max_seq_length=max_len,
                    load_in_4bit=True,
                    full_finetuning=False,
                    trust_remote_code=True,
                    local_files_only=True,
                )

                # LoRA 어댑터 추가 (Unsloth 헬퍼 우선)
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
                    candidate_keywords = [
                        "q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj",
                        "down_proj","w1","w2","c_proj","c_attn"
                    ]
                    lora_target_modules = sorted({
                        name.split(".")[-1]
                        for name, _ in model.named_modules()
                        if any(k in name for k in candidate_keywords)
                    })
                    lora_config = LoraConfig(
                        r=64, lora_alpha=16, target_modules=lora_target_modules or None,
                        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                    )
                    model = get_peft_model(model, lora_config)

            else:
                # ---- 일반 QLORA: BitsAndBytes 사용 ----
                try:
                    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # type: ignore
                except Exception as e:
                    _append_log(log_path, f"[{_now_utc().isoformat()}] peft not installed for QLORA: {e}")
                    _update_job_status(conn, job.job_id, "failed", extras={"error": f"peft not installed: {e}"})
                    logger.error(f"Fine-tuning failed jobId={job.job_id} error=peft not installed: {e}")
                    return
                bnb_config = BitsAndBytesConfig(
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
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    device_map="auto",
                    quantization_config=bnb_config,
                    local_files_only=True,
                )
                model.gradient_checkpointing_enable()
                model = prepare_model_for_kbit_training(model)
                candidate_keywords = [
                    "q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj",
                    "down_proj","w1","w2","c_proj","c_attn"
                ]
                lora_target_modules = sorted({
                    name.split(".")[-1]
                    for name, _ in model.named_modules()
                    if any(k in name for k in candidate_keywords)
                })
                lora_config = LoraConfig(
                    r=64,
                    lora_alpha=16,
                    target_modules=lora_target_modules or None,
                    lora_dropout=0.05,
                    bias="none",
                    task_type="CAUSAL_LM",
                )
                model = get_peft_model(model, lora_config)

        # ===== Train =====
        max_len = job.request.get("max_len", 4096)
        train_dataset = RagDataset(train_conversations, tokenizer, max_len=max_len)
        eval_dataset = RagDataset(eval_conversations, tokenizer, max_len=max_len) if len(eval_conversations) > 0 else None

        # ===== Simple ROUGE-1 helper =====
        def _rouge1_f1(reference: str, candidate: str) -> float:
            ref_words = reference.split()
            cand_words = candidate.split()
            if not ref_words or not cand_words:
                return 0.0
            matches = sum(1 for w in cand_words if w in ref_words)
            precision = matches / len(cand_words) if cand_words else 0.0
            recall = matches / len(ref_words) if ref_words else 0.0
            if precision + recall == 0:
                return 0.0
            return 2 * (precision * recall) / (precision + recall)

        import math
        from inspect import signature as _sig

# --- 전략 일치 설정 ---
        has_eval = (eval_dataset is not None) and (len(eval_conversations) > 0)
        save_strategy = "epoch" if has_eval else "no"
        evaluation_strategy = "epoch" if has_eval else "no"

        # Optimizer: FULL/LORA는 기본 adamw_torch, QLORA는 paged_adamw_8bit (환경변수로 오버라이드 가능)
        optim_name = os.getenv(
            "FT_OPTIM",
            ("adamw_torch" if tuning_type in ("FULL", "LORA") else "paged_adamw_8bit"),
        )

        _ta_kwargs = dict(
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
            # 🔹 전략 일치
            save_strategy=save_strategy,
        )

        # transformers 버전 차이를 감안해 signature 검사
        from inspect import signature as _sig
        if "evaluation_strategy" in _sig(TrainingArguments.__init__).parameters:
            _ta_kwargs["evaluation_strategy"] = evaluation_strategy

        # 🔹 평가가 있을 때만 best model 로직 활성화 (전략 일치 필수!)
        if has_eval:
            _ta_kwargs.update(
                dict(
                    load_best_model_at_end=True,
                    metric_for_best_model="eval_loss",
                    greater_is_better=False,
                    per_device_eval_batch_size=max(1, job.request.get("batchSize", 1)),
                    eval_accumulation_steps=None,  # 기본
                )
            )

        training_args = TrainingArguments(**_ta_kwargs)

        # ---- 평가/저장/얼리스톱 설정 (버전 호환 포함) ----
        has_eval_strategy = "evaluation_strategy" in _sig(TrainingArguments.__init__).parameters
        has_save_strategy = "save_strategy" in _sig(TrainingArguments.__init__).parameters
        has_lbm = "load_best_model_at_end" in _sig(TrainingArguments.__init__).parameters
        has_metric_for_best = "metric_for_best_model" in _sig(TrainingArguments.__init__).parameters
        has_greater = "greater_is_better" in _sig(TrainingArguments.__init__).parameters

        use_eval = eval_dataset is not None and len(eval_dataset) > 0
        # 평가/저장 전략은 항상 동일하게 강제하여 충돌 방지
        if has_eval_strategy:
            _ta_kwargs["evaluation_strategy"] = ("epoch" if use_eval else "no")
        if has_save_strategy:
            _ta_kwargs["save_strategy"] = ("epoch" if use_eval else "no")

        # 베스트 모델 로드는 검증셋 있을 때만
        if has_lbm:
            _ta_kwargs["load_best_model_at_end"] = bool(use_eval)
        if has_metric_for_best and use_eval:
            _ta_kwargs["metric_for_best_model"] = "eval_loss"
        if has_greater and use_eval:
            _ta_kwargs["greater_is_better"] = False

        training_args = TrainingArguments(**_ta_kwargs)

        # ===== 콜백들 =====
        class ProgressCallback(TrainerCallback):  # type: ignore[misc]
            def __init__(self, job_id: str, every_steps: int = None):
                self.job_id = job_id
                try:
                    self.every_steps = every_steps if every_steps is not None else max(1, int(os.getenv("FT_PROGRESS_EVERY_STEPS", "1")))
                except Exception:
                    self.every_steps = 1
                self.total_steps = None

            def on_train_begin(self, args, state, control, **kwargs):
                try:
                    c = get_db()
                    _update_job_status(c, self.job_id, "running", progress=0)
                    c.close()
                except Exception:
                    pass

            def on_train_begin_dataloader(self, args, state, control, **kwargs):
                # total steps 계산 (데이터로더 확보 후)
                try:
                    if state.max_steps and state.max_steps > 0:
                        self.total_steps = int(state.max_steps)
                except Exception:
                    pass

            def on_step_end(self, args, state, control, **kwargs):
                try:
                    total = self.total_steps or (state.max_steps or 0)
                    if not total or total <= 0:
                        return
                    if state.global_step % self.every_steps != 0:
                        return
                    pct = int(min(100, max(0, round((state.global_step / total) * 100))))
                    c = get_db()
                    try:
                        _update_job_status(c, self.job_id, "running", progress=pct)
                    finally:
                        c.close()
                except Exception:
                    pass

            def on_train_end(self, args, state, control, **kwargs):
                try:
                    c = get_db()
                    _update_job_status(c, self.job_id, "running", progress=100)
                    c.close()
                except Exception:
                    pass

        class LogCallback(TrainerCallback):  # type: ignore[misc]
            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs and "loss" in logs:
                    _append_log(log_path, f"[{_now_utc().isoformat()}] step={state.global_step} loss={logs['loss']}")

        # 얼리스톱(평가가 있을 때만): 2 에폭 연속 개선 없으면 중단
        extra_callbacks = []
        if use_eval and (job.request.get("overfittingPrevention", True) is True):
            try:
                from transformers import EarlyStoppingCallback  # type: ignore
                extra_callbacks.append(EarlyStoppingCallback(early_stopping_patience=2))
            except Exception:
                pass

        _pc = ProgressCallback(job.job_id)

        def _build_trainer()->Trainer:
            return Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=tokenizer,
                callbacks=[_pc, LogCallback(), *extra_callbacks],
            )

        trainer=_build_trainer()

        try:
            _append_log(log_path, f"[{_now_utc().isoformat()}] training started...")
            trainer.train()
        except RuntimeError as re:
            if "out of memory" in str(re).lower() and training_args.per_device_train_batch_size>1:
                new_bs=max(1,training_args.per_device_train_batch_size//2)
                _append_log(log_path,f"[{_now_utc().isoformat()}] OOM detected, retrying with batch_size={new_bs}")
                training_args.per_device_train_batch_size=new_bs
                trainer=_build_trainer()
                trainer.train()
            else:
                raise

        # Compute final ROUGE-1 on full eval set if available
        final_rouge = None
        if eval_dataset is not None and len(eval_dataset) > 0:
            model.eval()
            preds = []
            refs = []
            for item in eval_dataset:
                inp_ids = item["input_ids"].unsqueeze(0).to(model.device)
                try:
                    out_ids = model.generate(inp_ids, max_new_tokens=128, do_sample=False)[0]
                except Exception:
                    continue
                preds.append(tokenizer.decode(out_ids, skip_special_tokens=True))
                ref_ids = item["labels"]
                ref_ids = torch.where(ref_ids == -100, torch.tensor(tokenizer.pad_token_id), ref_ids)
                refs.append(tokenizer.decode(ref_ids, skip_special_tokens=True))
            scores = [_rouge1_f1(r, p) for r, p in zip(refs, preds)]
            if scores:
                final_rouge = sum(scores) / len(scores)

        _update_job_status(conn, job.job_id, "running", rough=int((final_rouge or 0)*100), extras={"rouge1F1": final_rouge})

        # === 저장(merge) 단계도 사용자에게 보여주기 ===
        def _save_stage(stage: str, pct: int):
            _append_log(log_path, f"[{_now_utc().isoformat()}] save:{stage} {pct}%")
            c = get_db()
            _update_job_status(c, job.job_id, "running", extras={"saveStage": stage, "saveProgress": pct})
            c.close()

        _save_stage("start", 5)
        _append_log(log_path, f"[{_now_utc().isoformat()}] saving model...")
        # ===== Save / Merge =====
        is_mxfp4 = _looks_like_mxfp4_model(model_path) or _looks_like_mxfp4_model(job.request.get("baseModelName"))
        
        if tuning_type in ("LORA", "QLORA"):
            # LoRA/QLoRA: 어댑터만 저장 (베이스는 참조로)
            try:
                # PEFT/Unsloth 모델에서 save_pretrained는 어댑터만 저장
                model.save_pretrained(output_dir)
                _save_stage("adapter", 70)
                tokenizer.save_pretrained(output_dir)
                _save_stage("tokenizer", 90)
                _append_log(log_path, f"[{_now_utc().isoformat()}] saved adapters only → {output_dir}")
            except Exception as e:
                _append_log(log_path, f"[{_now_utc().isoformat()}] adapter save failed: {e}, fallback trainer.save_model")
                trainer.save_model(output_dir)
                _save_stage("model", 70)
                tokenizer.save_pretrained(output_dir)
                _save_stage("tokenizer", 90)
            
            # MXFP4 병합 저장은 환경변수로 제어 (기본 비활성)
            save_merge = os.getenv("FT_UNSLOTH_MERGE_SAVE", "0") == "1"
            if is_mxfp4 and save_merge:
                try:
                    model.save_pretrained_merged(output_dir, tokenizer, save_method="mxfp4")
                    _append_log(log_path, f"[{_now_utc().isoformat()}] merged MXFP4 saved → {output_dir}")
                except Exception as e:
                    _append_log(log_path, f"[{_now_utc().isoformat()}] MXFP4 merge save failed: {e}, using adapters only.")
        else:
            # FULL: 전체 모델 저장
            trainer.save_model(output_dir)
            _save_stage("model", 70)
            tokenizer.save_pretrained(output_dir)
            _save_stage("tokenizer", 90)
        
        # 마무리
        _save_stage("done", 100)

        _finish_job_success(
            conn,
            job.job_id,
            save_name_with_suffix,
            job.category,
            tuning_type,
            final_rouge,
            subcategory=job.request.get("subcategory"),
        )
        logger.info(
            f"Fine-tuning succeeded jobId={job.job_id} save={save_name_with_suffix} type={tuning_type}"
        )
        _append_log(log_path, f"[{_now_utc().isoformat()}] job {job.job_id} succeeded")

    except Exception as e:
        logger.error(f"Fine-tuning failed jobId={job.job_id} error={e}")
        # Log error to file
        try:
            _append_log(os.path.join(STORAGE_MODEL_ROOT, save_name_with_suffix, "train.log"),
                        f"[{_now_utc().isoformat()}] error: {e}")
        except Exception:
            pass

        # Persist failure status & error message to DB (fresh connection to avoid thread issues)
        c = get_db()
        try:
            _update_job_status(c, job.job_id, "failed", progress=0, extras={"error": str(e)})
        finally:
            c.close()
        # Ensure outer scope conn marks failure too (best effort)
        try:
            cur = conn.cursor()
            cur.execute("UPDATE fine_tune_jobs SET status=? WHERE provider_job_id=?", ("failed", job.job_id))
            conn.commit()
        except Exception:
            pass
    finally:
        # ✅ 메모리 정리 강화
        try:
            del trainer
        except Exception:
            pass
        try:
            del model, tokenizer
        except Exception:
            pass
        try:
            import gc, torch
            gc.collect()
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()  # IPC 메모리 정리
        except Exception:
            pass
        # ✅ 캐시 정리 (임시 디렉토리 삭제 및 환경변수 복원)
        try:
            # 환경변수 복원
            for k, v in old_cache.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
            # 임시 디렉토리 삭제
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass
        conn.close()

# ===== Public APIs =====
def _log_to_save_name(save_name_with_suffix: str, message: str):
    try:
        out_dir = _ensure_output_dir(save_name_with_suffix)
        log_path = os.path.join(out_dir, "train.log")
        _append_log(log_path, f"[{_now_utc().isoformat()}] {message}")
    except Exception:
        pass

def start_fine_tuning(category: str, body: FineTuneRequest) -> Dict[str, Any]:
    # body.category를 최종 신뢰(하위호환을 위해 인수 category는 fallback)
    category = (body.category or category).lower()
    suffix = (body.tuningType or "QLORA").upper()
    # 카테고리까지 포함하여 저장 폴더/모델명을 구분 (예: name-QLORA-qa)
    save_name_with_suffix = f"{body.saveModelName}-{suffix}-{category}"
    _ensure_output_dir(save_name_with_suffix)

    base_dir = _resolve_model_dir(body.baseModelName)
    if not _has_model_signature(base_dir):
        msg = f"base model not found: {base_dir}"
        logger.error(msg)
        _log_to_save_name(save_name_with_suffix, msg)
        raise BadRequestError(msg)

    job_id = f"ft-job-{uuid.uuid4().hex[:12]}"

    conn = get_db()
    try:
        train_path = _resolve_train_path(body.trainSetFile)
        dataset_id = _insert_dataset_if_needed(conn, train_path, category)
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
            conn, category, body, job_id, save_name_with_suffix, dataset_id,
            initial_status=("scheduled" if delay_sec > 1.0 else "queued"),
            scheduled_at=scheduled_at_iso,
        )
    except Exception as e:
        _log_to_save_name(save_name_with_suffix, f"db insert failed: {e}")
        logger.error(f"fine-tuning init failed: {e}")
        try:
            conn.close()
        except Exception:
            pass
        raise InternalServerError(f"fine-tuning init failed: {e}")
    finally:
        try:
            conn.close()
        except Exception:
            pass

    job = FineTuneJob(job_id=job_id, category=category, request=body.model_dump())
    
    # 🔹 즉시 실행 여부에 따라 분기
    if body.startNow:
        # 즉시 실행
        def _launch():
            _run_training_inline(job, save_name_with_suffix)
        
        t = threading.Thread(target=_launch, daemon=True)
        t.start()
        logger.info(
            f"Fine-tuning started immediately jobId={job.job_id} category={category} base={body.baseModelName} "
            f"save={save_name_with_suffix} tuning={body.tuningType or 'QLORA'}"
        )
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
                conn2 = get_db()
                _update_job_status(conn2, job_id, "queued", extras={"reserved": True, "reservedAt": _now_local_str()})
            finally:
                try:
                    conn2.close()
                except Exception:
                    pass
            logger.info(
                f"Fine-tuning queued (not started) jobId={job.job_id} category={category} base={body.baseModelName} save={save_name_with_suffix}"
            )

    return {"jobId": job_id, "started": bool(body.startNow)}

def get_fine_tuning_status(job_id: str) -> Dict[str, Any]:
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute("SELECT provider_job_id, status, metrics FROM fine_tune_jobs WHERE provider_job_id=?", (job_id,))
        row = cur.fetchone()
        if not row:
            return {"error": "job not found", "jobId": job_id}
        metrics = {}
        if row["metrics"]:
            try:
                metrics = json.loads(row["metrics"]) or {}
            except Exception:
                metrics = {}
        
        # Liveness: if status is running but heartbeat is stale, flip to failed
        row_status = row["status"]
        try:
            if row_status == "running":
                hb = metrics.get("heartbeatAt")
                if hb:
                    from dateutil import parser as dtparser  # type: ignore
                    last = dtparser.parse(hb)
                    now = _now_utc()
                    if (now - last).total_seconds() > FT_HEARTBEAT_TIMEOUT_SEC:
                        c2 = get_db()
                        try:
                            _update_job_status(c2, job_id, "failed", extras={"error": "stale heartbeat"})
                        finally:
                            c2.close()
                        row_status = "failed"
        except Exception:
            pass

        return {
            "jobId": row["provider_job_id"],
            "status": row_status,
            "learningProgress": int(metrics.get("learningProgress", 0)),
            "roughScore": int(metrics.get("roughScore", 0)),
            "rouge1F1": metrics.get("rouge1F1"),
            "saveProgress": int(metrics.get("saveProgress", 0)),
            "saveStage": metrics.get("saveStage"),
            "error": metrics.get("error"),
        }
    finally:
        conn.close()

# ===== Admin util: wipe fine-tune related tables =====
def reset_fine_tune_tables():
    """Dangerous: delete all fine-tune jobs and model records (use for dev reset)"""
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM fine_tuned_models")
        cur.execute("DELETE FROM fine_tune_jobs")
        cur.execute("DELETE FROM fine_tune_datasets")
        cur.execute("DELETE FROM llm_models")
        conn.commit()
    finally:
        conn.close()
