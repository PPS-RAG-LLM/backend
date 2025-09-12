# service/admin/LLM_finetuning.py
from __future__ import annotations

import json
import os
import threading
import time
import uuid
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional
from pathlib import Path

from pydantic import BaseModel, Field
from utils import get_db, logger
from errors.exceptions import BadRequestError, InternalServerError
logger = logger(__name__)

try:
    from transformers import TrainerCallback  # type: ignore
except Exception:  # transformers may be absent during static analysis
    class TrainerCallback:  # type: ignore
        pass

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
    """
    if os.path.isabs(name_or_path):
        return name_or_path
    # 1) Try DB lookup (huggingface/hf providers)
    try:
        from repository.users.llm_models import get_llm_model_by_provider_and_name as _get
        def _strip_cat(n: str) -> str:
            for suf in ("-qa", "-doc_gen", "-summary"):
                if n.endswith(suf):
                    return n[: -len(suf)]
            return n
        for key in (name_or_path, _strip_cat(name_or_path)):
            for prov in ("huggingface", "hf"):
                row = _get(prov, key)
                if row and row.get("model_path"):
                    p = row["model_path"]
                    if os.path.isabs(p):
                        return p
                    cand = os.path.join(STORAGE_MODEL_ROOT, p)
                    if os.path.isdir(cand):
                        return cand
    except Exception:
        pass
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

# ===== Schemas =====
class FineTuneRequest(BaseModel):
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
        default=None,
        description="QLORA 전용: 양자화 비트 선택 (4 또는 8)",
    )
    tuningType: Optional[str] = Field(
        default="QLORA",
        description="파인튜닝 방식: LORA | QLORA | FULL",
        pattern="^(LORA|QLORA|FULL)$",
    )

@dataclass
class FineTuneJob:
    job_id: str
    category: str
    request: Dict[str, Any]
    status: str = "queued"  # queued | running | succeeded | failed

# ===== Common utils =====
def _now_local_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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
                json.dump({"created_at": datetime.now().isoformat()}, f, ensure_ascii=False)
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
            ts = datetime.now().isoformat()
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
                    mtime = datetime.utcfromtimestamp(entry.stat().st_mtime).isoformat()
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
                                "UPDATE fine_tune_datasets SET record_count=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
                                (file_count, dataset_id),
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
                                        "UPDATE fine_tune_datasets SET record_count=?, updated_at=CURRENT_TIMESTAMP WHERE name=? AND path=?",
                                        (recomputed_count, name_val, rel_path_val),
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
    rel_path = _to_rel(path)
    cur = conn.cursor()
    cur.execute("SELECT id FROM fine_tune_datasets WHERE path=?", (rel_path,))
    row = cur.fetchone()
    if row:
        return int(row["id"])
    try:
        cur.execute("""
            INSERT INTO fine_tune_datasets(name, category, path, record_count)
            VALUES(?, ?, ?, NULL)
        """, (os.path.basename(path), category, rel_path))
    except Exception:
        cur.execute("""
            INSERT INTO fine_tune_datasets(name, path, record_count)
            VALUES(?, ?, NULL)
        """, (os.path.basename(path), rel_path))
    conn.commit()
    return int(cur.lastrowid)

def _insert_job(conn, category: str, req: FineTuneRequest, job_id: str, save_name_with_suffix: str, dataset_id: int) -> int:
    cur = conn.cursor()
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
        "gradientAccumulationSteps": req.gradientAccumulationSteps,
        "quantizationBits": req.quantizationBits,
    }
    try:
        cur.execute("""
            INSERT INTO fine_tune_jobs(provider_job_id, dataset_id, hyperparameters, status, started_at)
            VALUES(?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, (job_id, dataset_id, json.dumps(hyper, ensure_ascii=False), "queued"))
    except Exception:
        cur.execute("""
            INSERT INTO fine_tune_jobs(provider_job_id, dataset_id, metrics, status, started_at)
            VALUES(?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, (job_id, dataset_id, json.dumps({"hyperparameters": hyper}, ensure_ascii=False), "queued"))
    conn.commit()
    return int(cur.lastrowid)

def _update_job_status(conn, job_id: str, status: str, progress: int | None = None, rough: int | None = None, extras: dict | None = None, _retries: int = 3):
    """
    Update job status and optional metrics. Progress percentage is intentionally ignored (not persisted).
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

            # Intentionally do not persist progress percentage
            if rough is not None:
                metrics["roughScore"] = rough
            if extras:
                try:
                    metrics.update(extras)
                except Exception:
                    pass
            # Update heartbeat timestamp for liveness detection
            try:
                metrics["heartbeatAt"] = datetime.now().isoformat()
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
                    msg = f"status={status} progress={progress if progress is not None else '-'} rough={rough if rough is not None else '-'} extras={extras or {}}"
                    _append_log(log_path, f"[{datetime.now().isoformat()}] {msg}")
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

def _finish_job_success(conn, job_id: str, model_name: str, category: str, tuning_type: str, final_rouge: Optional[float] = None):
    cur = conn.cursor()
    cur.execute("UPDATE fine_tune_jobs SET status=?, finished_at=CURRENT_TIMESTAMP WHERE provider_job_id=?",
                ("succeeded", job_id))
    # llm_models upsert-ish
    cur.execute("SELECT id FROM llm_models WHERE name=?", (model_name,))
    row = cur.fetchone()
    if row:
        model_id = int(row["id"])
    else:
        mdl_type = "lora" if tuning_type.upper() in ("LORA", "QLORA") else "full"
        cur.execute("""
            INSERT INTO llm_models(provider, name, revision, model_path, category, type, is_active)
            VALUES(?,?,?,?,?,?,1)
        """, ("hf", model_name, 0, _to_rel(os.path.join(STORAGE_MODEL_ROOT, model_name)), category, mdl_type))
        model_id = int(cur.lastrowid)
    # set trained_at now
    try:
        cur.execute("UPDATE llm_models SET trained_at=CURRENT_TIMESTAMP WHERE id=?", (model_id,))
    except Exception:
        pass

    # fine_tuned_models
    cur.execute("SELECT id FROM fine_tune_jobs WHERE provider_job_id=?", (job_id,))
    job_row = cur.fetchone()
    if job_row:
        ft_job_pk = int(job_row["id"])
        lora_path = _to_rel(os.path.join(STORAGE_MODEL_ROOT, model_name)) if tuning_type.upper() in ("LORA", "QLORA") else None
        # ensure rouge column if available
        _ensure_ftm_rouge_column(conn)
        if _has_column(conn, "fine_tuned_models", "rouge1_f1"):
            cur.execute("""
                INSERT INTO fine_tuned_models(model_id, job_id, provider_model_id, lora_weights_path, type, is_active, rouge1_f1)
                VALUES(?, ?, ?, ?, ?, 1, ?)
            """, (model_id, ft_job_pk, model_name, lora_path, ("lora" if lora_path else "full"), (final_rouge if final_rouge is not None else None)))
        else:
            cur.execute("""
                INSERT INTO fine_tuned_models(model_id, job_id, provider_model_id, lora_weights_path, type, is_active)
                VALUES(?, ?, ?, ?, ?, 1)
            """, (model_id, ft_job_pk, model_name, lora_path, ("lora" if lora_path else "full")))
    conn.commit()

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

# ===== Training (simulated) =====
def _simulate_training(job: FineTuneJob, save_name_with_suffix: str):
    conn = get_db()
    try:
        _update_job_status(conn, job.job_id, "running")
        out_dir = _ensure_output_dir(save_name_with_suffix)
        ttype = (job.request.get("tuningType") or "QLORA").upper()
        if ttype in ("LORA", "QLORA"):
            _ensure_lora_marker(out_dir, ttype)
        log_path = os.path.join(out_dir, "train.log")
        # fresh log
        try:
            with open(log_path, "w", encoding="utf-8") as _tmp:
                _tmp.write("")
        except Exception:
            pass
        # console log
        logger.info(
            f"Fine-tuning started (SIM) jobId={job.job_id} category={job.category} save={save_name_with_suffix}"
        )
        _append_log(log_path, f"[{datetime.now().isoformat()}] job {job.job_id} started (SIMULATED)")
        for p in range(2, 101):
            time.sleep(0.1)
            if p % 5 == 0:
                _append_log(log_path, f"[{datetime.now().isoformat()}] progress {p}%")
        _finish_job_success(conn, job.job_id, save_name_with_suffix, job.category, ttype)
        logger.info(
            f"Fine-tuning succeeded (SIM) jobId={job.job_id} save={save_name_with_suffix} type={ttype}"
        )
        _append_log(log_path, f"[{datetime.now().isoformat()}] job {job.job_id} succeeded")
    except Exception:
        logger.error(f"Fine-tuning failed (SIM) jobId={job.job_id}")
        cur = conn.cursor()
        cur.execute("UPDATE fine_tune_jobs SET status=? WHERE provider_job_id=?", ("failed", job.job_id))
        conn.commit()
        try:
            out_dir = os.path.join(STORAGE_MODEL_ROOT, save_name_with_suffix)
            log_path = os.path.join(out_dir, "train.log")
            _append_log(log_path, f"[{datetime.now().isoformat()}] job {job.job_id} failed")
        except Exception:
            pass
    finally:
        conn.close()

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
        _update_job_status(conn, job.job_id, "running")

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
        _append_log(log_path, f"[{datetime.now().isoformat()}] job {job.job_id} started (INLINE)")
        # console log
        logger.info(
            f"Fine-tuning started jobId={job.job_id} type={tuning_type} base={job.request.get('baseModelName')} "
            f"save={save_name_with_suffix} data={job.request.get('trainSetFile')}"
        )

        # imports
        try:
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
            _append_log(log_path, f"[{datetime.now().isoformat()}] import error: {e}")
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
                _append_log(log_path, f"[{datetime.now().isoformat()}] csv load failed: {e}")
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
        if total_examples >= 10:
            eval_size = max(1, int(round(total_examples * 0.1)))
        elif total_examples >= 2:
            eval_size = 1
        else:
            eval_size = 0

        if eval_size > 0:
            # shuffle deterministically
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
            _append_log(log_path, f"[{datetime.now().isoformat()}] base model not found: {base_folder}")
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
                _append_log(log_path, f"[{datetime.now().isoformat()}] peft not installed for LORA: {e}")
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
            # QLORA: 4-bit + LoRA
            try:
                from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # type: ignore
            except Exception as e:
                _append_log(log_path, f"[{datetime.now().isoformat()}] peft not installed for QLORA: {e}")
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

        # Build kwargs dict first to allow graceful fallback when certain versions
        # of transformers don't support a particular parameter (e.g. evaluation_strategy).
        # Optimizer: avoid bitsandbytes for FULL/LORA by default; keep for QLORA.
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
            save_steps=500,
            save_total_limit=2,
            report_to="none",
            optim=optim_name,
            seed=42,
            data_seed=42,
        )
        # Conditionally include evaluation_strategy only if the current transformers version supports it
        from inspect import signature as _sig
        if "evaluation_strategy" in _sig(TrainingArguments.__init__).parameters:
            _ta_kwargs["evaluation_strategy"] = "epoch" if eval_dataset is not None else "no"

        training_args = TrainingArguments(**_ta_kwargs)

        class LogCallback(TrainerCallback):  # type: ignore[misc]
            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs and "loss" in logs:
                    _append_log(log_path, f"[{datetime.now().isoformat()}] step={state.global_step} loss={logs['loss']}")

        # === Periodic progress callback (interval-based) ===
        # Remove periodic DB progress updates entirely

        def _build_trainer()->Trainer:
            return Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=tokenizer,
                callbacks=[LogCallback()],
            )

        trainer=_build_trainer()

        try:
            trainer.train()
        except RuntimeError as re:
            if "out of memory" in str(re).lower() and training_args.per_device_train_batch_size>1:
                new_bs=max(1,training_args.per_device_train_batch_size//2)
                _append_log(log_path,f"[{datetime.now().isoformat()}] OOM detected, retrying with batch_size={new_bs}")
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
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

        _finish_job_success(conn, job.job_id, save_name_with_suffix, job.category, tuning_type)
        _finish_job_success(conn, job.job_id, save_name_with_suffix, job.category, tuning_type, final_rouge)
        logger.info(
            f"Fine-tuning succeeded jobId={job.job_id} save={save_name_with_suffix} type={tuning_type}"
        )
        _append_log(log_path, f"[{datetime.now().isoformat()}] job {job.job_id} succeeded")

    except Exception as e:
        logger.error(f"Fine-tuning failed jobId={job.job_id} error={e}")
        # Log error to file
        try:
            _append_log(os.path.join(STORAGE_MODEL_ROOT, save_name_with_suffix, "train.log"),
                        f"[{datetime.now().isoformat()}] error: {e}")
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
        conn.close()

# ===== Public APIs =====
def _log_to_save_name(save_name_with_suffix: str, message: str):
    try:
        out_dir = _ensure_output_dir(save_name_with_suffix)
        log_path = os.path.join(out_dir, "train.log")
        _append_log(log_path, f"[{datetime.now().isoformat()}] {message}")
    except Exception:
        pass

def start_fine_tuning(category: str, body: FineTuneRequest) -> Dict[str, Any]:
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
        _insert_job(conn, category, body, job_id, save_name_with_suffix, dataset_id)
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
    # FT_USE_SIM=0 -> 실제 학습
    use_sim = os.getenv("FT_USE_SIM", "0") != "0"
    target = _simulate_training if use_sim else _run_training_inline
    logger.info(
        f"Fine-tuning queued jobId={job.job_id} category={category} base={body.baseModelName} "
        f"save={save_name_with_suffix} tuning={body.tuningType or 'QLORA'}"
    )
    t = threading.Thread(target=target, args=(job, save_name_with_suffix), daemon=True)
    t.start()

    return {"jobId": job_id}

def get_fine_tuning_status(category: str, job_id: str) -> Dict[str, Any]:
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
        try:
            if row["status"] == "running":
                hb = metrics.get("heartbeatAt")
                if hb:
                    from datetime import datetime, timezone
                    from dateutil import parser as dtparser  # type: ignore
                    last = dtparser.parse(hb)
                    now = datetime.now(last.tzinfo or timezone.utc)
                    if (now - last).total_seconds() > FT_HEARTBEAT_TIMEOUT_SEC:
                        c2 = get_db()
                        try:
                            _update_job_status(c2, job_id, "failed", extras={"error": "stale heartbeat"})
                        finally:
                            c2.close()
                        row_status = "failed"
                    else:
                        row_status = row["status"]
                else:
                    row_status = row["status"]
            else:
                row_status = row["status"]
        except Exception:
            row_status = row["status"]

        return {
            "jobId": row["provider_job_id"],
            "status": row_status,
            "learningProgress": int(metrics.get("learningProgress", 0)),
            "roughScore": int(metrics.get("roughScore", 0)),
            "error": metrics.get("error"),
            "rouge1F1": metrics.get("rouge1F1"),
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
