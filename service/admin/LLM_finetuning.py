from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field
from utils.database import get_db

STORAGE_MODEL_ROOT = "/home/work/CoreIQ/backend/storage/model"


class FineTuneRequest(BaseModel):
    baseModelName: str
    saveModelName: str
    systemPrompt: str
    batchSize: int
    epochs: int
    learningRate: float
    overfittingPrevention: bool
    trainSetFile: str
    reserveDate: Optional[str] = None
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
    learning_progress: int = 0  # 1..100
    rough_score: int = 0  # 1..100


def _ensure_output_dir(model_name: str) -> str:
    os.makedirs(STORAGE_MODEL_ROOT, exist_ok=True)
    out_dir = os.path.join(STORAGE_MODEL_ROOT, model_name)
    os.makedirs(out_dir, exist_ok=True)
    # 생성 표식 파일(config.json) – 로컬 로더가 모델 존재를 감지할 수 있게 함
    cfg_path = os.path.join(out_dir, "config.json")
    if not os.path.isfile(cfg_path):
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump({"created_at": datetime.utcnow().isoformat()}, f)
    return out_dir


def _insert_dataset_if_needed(conn, path: str) -> int:
    cur = conn.cursor()
    cur.execute("SELECT id FROM fine_tune_datasets WHERE path=?", (path,))
    row = cur.fetchone()
    if row:
        return int(row["id"])
    cur.execute(
        """
        INSERT INTO fine_tune_datasets(name, category, path, record_count)
        VALUES(?, ?, ?, NULL)
        """,
        (os.path.basename(path), "qa", path),  # category는 UI 구분용, 불명확 시 임시값
    )
    conn.commit()
    return int(cur.lastrowid)


def _insert_job(conn, category: str, req: FineTuneRequest, job_id: str, save_name_with_suffix: str, dataset_id: int) -> int:
    cur = conn.cursor()
    # hyperparameters JSON 저장
    hyper = {
        "systemPrompt": req.systemPrompt,
        "batchSize": req.batchSize,
        "epochs": req.epochs,
        "learningRate": req.learningRate,
        "overfittingPrevention": req.overfittingPrevention,
        "tuningType": req.tuningType or "QLORA",
        "baseModelName": req.baseModelName,
        "saveModelName": save_name_with_suffix,
        "trainSetFile": req.trainSetFile,
        "reserveDate": req.reserveDate,
    }
    cur.execute(
        """
        INSERT INTO fine_tune_jobs(provider_job_id, dataset_id, status, metrics, started_at)
        VALUES(?, ?, ?, ?, CURRENT_TIMESTAMP)
        """,
        (job_id, dataset_id, "queued", json.dumps(hyper, ensure_ascii=False)),
    )
    conn.commit()
    return int(cur.lastrowid)


def _update_job_status(conn, job_id: str, status: str, progress: int | None = None, rough: int | None = None):
    cur = conn.cursor()
    # metrics에 진행도 저장
    cur.execute("SELECT metrics FROM fine_tune_jobs WHERE provider_job_id=?", (job_id,))
    row = cur.fetchone()
    metrics = {}
    if row and row["metrics"]:
        try:
            metrics = json.loads(row["metrics"]) or {}
        except Exception:
            metrics = {}
    if progress is not None:
        metrics["learningProgress"] = progress
    if rough is not None:
        metrics["roughScore"] = rough
    cur.execute(
        "UPDATE fine_tune_jobs SET status=?, metrics=? WHERE provider_job_id=?",
        (status, json.dumps(metrics, ensure_ascii=False), job_id),
    )
    conn.commit()


def _finish_job_success(conn, job_id: str, model_name: str, llm_model_id: int | None = None):
    cur = conn.cursor()
    cur.execute(
        "UPDATE fine_tune_jobs SET status=?, finished_at=CURRENT_TIMESTAMP WHERE provider_job_id=?",
        ("succeeded", job_id),
    )
    # 결과 모델을 llm_models 에도 등록(없으면)
    cur.execute("SELECT id FROM llm_models WHERE name=?", (model_name,))
    row = cur.fetchone()
    if not row:
        cur.execute(
            """
            INSERT INTO llm_models(provider, name, revision, model_path, category, type, is_active)
            VALUES(?,?,?,?,?,?,1)
            """,
            ("hf", model_name, 0, os.path.join(STORAGE_MODEL_ROOT, model_name), "qa", "base"),
        )
    conn.commit()


def _simulate_training(job: FineTuneJob, save_name_with_suffix: str):
    # 매우 단순한 시뮬레이션: 진행도 1~100
    conn = get_db()
    try:
        _update_job_status(conn, job.job_id, "running", progress=1, rough=0)
        _ensure_output_dir(save_name_with_suffix)
        for p in range(2, 101):
            time.sleep(0.1)
            _update_job_status(conn, job.job_id, "running", progress=p)
        _finish_job_success(conn, job.job_id, save_name_with_suffix)
    except Exception:
        cur = conn.cursor()
        cur.execute(
            "UPDATE fine_tune_jobs SET status=? WHERE provider_job_id=?",
            ("failed", job.job_id),
        )
        conn.commit()
    finally:
        conn.close()


def start_fine_tuning(category: str, body: FineTuneRequest) -> Dict[str, Any]:
    # 모델 결과 폴더명 규칙 적용
    suffix = (body.tuningType or "QLORA").upper()
    save_name_with_suffix = f"{body.saveModelName}-{suffix}"

    job_id = f"ft-job-{uuid.uuid4().hex[:12]}"

    conn = get_db()
    try:
        dataset_id = _insert_dataset_if_needed(conn, body.trainSetFile)
        _insert_job(conn, category, body, job_id, save_name_with_suffix, dataset_id)
    finally:
        conn.close()

    # 백그라운드 실행(시뮬레이션)
    job = FineTuneJob(job_id=job_id, category=category, request=body.model_dump())
    t = threading.Thread(target=_simulate_training, args=(job, save_name_with_suffix), daemon=True)
    t.start()

    return {"jobId": job_id}


def get_fine_tuning_status(category: str, job_id: str) -> Dict[str, Any]:
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT provider_job_id, status, metrics FROM fine_tune_jobs WHERE provider_job_id=?",
            (job_id,),
        )
        row = cur.fetchone()
        if not row:
            return {"error": "job not found", "jobId": job_id}
        metrics = {}
        if row["metrics"]:
            try:
                metrics = json.loads(row["metrics"]) or {}
            except Exception:
                metrics = {}
        return {
            "jobId": row["provider_job_id"],
            "status": row["status"],
            "learningProgress": int(metrics.get("learningProgress", 0)),
            "roughScore": int(metrics.get("roughScore", 0)),
        }
    finally:
        conn.close()
