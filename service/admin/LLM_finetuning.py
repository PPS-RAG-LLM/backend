from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, Optional
from pathlib import Path

from pydantic import BaseModel, Field
from utils.database import get_db

BASE_BACKEND = Path(os.getenv("COREIQ_BACKEND_ROOT", str(Path(__file__).resolve().parents[2])))  # backend/
STORAGE_MODEL_ROOT = os.getenv("STORAGE_MODEL_ROOT", str(BASE_BACKEND / "storage" / "model"))
TRAIN_DATA_ROOT   = os.getenv("TRAIN_DATA_ROOT", str(BASE_BACKEND / "storage" / "train_data"))


def _resolve_model_dir(name_or_path: str) -> str:
    if os.path.isabs(name_or_path):
        return name_or_path
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


class FineTuneRequest(BaseModel):
    baseModelName: str
    saveModelName: str
    systemPrompt: str
    batchSize: int = 4
    epochs: int = 3
    learningRate: float = 2e-4
    overfittingPrevention: bool = True
    trainSetFile: str
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


def _now_local_str() -> str:
    # YYYY-MM-DD HH:MM:SS (local)
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _ensure_output_dir(model_name: str) -> str:
    os.makedirs(STORAGE_MODEL_ROOT, exist_ok=True)
    out_dir = os.path.join(STORAGE_MODEL_ROOT, model_name)
    os.makedirs(out_dir, exist_ok=True)
    cfg_path = os.path.join(out_dir, "config.json")
    if not os.path.isfile(cfg_path):
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump({"created_at": datetime.utcnow().isoformat()}, f)
    return out_dir


def _ensure_lora_marker(out_dir: str, tuning_type: str):
    if tuning_type.upper() in ("LORA", "QLORA"):
        marker = os.path.join(out_dir, "adapter_config.json")
        if not os.path.isfile(marker):
            with open(marker, "w", encoding="utf-8") as f:
                json.dump({"peft_type": "LORA", "tuning": tuning_type.upper()}, f)


def list_train_data_dirs() -> Dict[str, Any]:
    items = []
    if not os.path.isdir(TRAIN_DATA_ROOT):
        return {"root": TRAIN_DATA_ROOT, "dirs": items}
    for entry in os.scandir(TRAIN_DATA_ROOT):
        if not entry.is_dir():
            continue
        try:
            mtime = datetime.utcfromtimestamp(entry.stat().st_mtime).isoformat()
        except Exception:
            mtime = None
        file_count = 0
        try:
            for _, _, files in os.walk(entry.path):
                file_count += len(files)
        except Exception:
            pass
        items.append({
            "name": entry.name,
            "path": os.path.join(TRAIN_DATA_ROOT, entry.name),
            "fileCount": file_count,
            "modifiedAt": mtime,
        })
    items.sort(key=lambda x: (x["modifiedAt"] or ""), reverse=True)
    return {"root": TRAIN_DATA_ROOT, "dirs": items}


def _insert_dataset_if_needed(conn, path: str, category: str) -> int:
    cur = conn.cursor()
    cur.execute("SELECT id FROM fine_tune_datasets WHERE path=?", (path,))
    row = cur.fetchone()
    if row:
        return int(row["id"])
    try:
        cur.execute(
            """
            INSERT INTO fine_tune_datasets(name, category, path, record_count)
            VALUES(?, ?, ?, NULL)
            """,
            (os.path.basename(path), category, path),
        )
    except Exception:
        cur.execute(
            """
            INSERT INTO fine_tune_datasets(name, path, record_count)
            VALUES(?, ?, NULL)
            """,
            (os.path.basename(path), path),
        )
    conn.commit()
    return int(cur.lastrowid)


def _insert_job(conn, category: str, req: FineTuneRequest, job_id: str, save_name_with_suffix: str, dataset_id: int) -> int:
    cur = conn.cursor()
    reserve_now = _now_local_str()
    train_path = _resolve_train_path(req.trainSetFile)
    hyper = {
        "systemPrompt": req.systemPrompt,
        "batchSize": req.batchSize,
        "epochs": req.epochs,
        "learningRate": req.learningRate,
        "overfittingPrevention": req.overfittingPrevention,
        "tuningType": req.tuningType or "QLORA",
        "baseModelName": req.baseModelName,
        "saveModelName": save_name_with_suffix,
        "trainSetFile": train_path,
        "reserveDate": reserve_now,
        "category": category,
    }
    try:
        cur.execute(
            """
            INSERT INTO fine_tune_jobs(provider_job_id, dataset_id, hyperparameters, status, started_at)
            VALUES(?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            (job_id, dataset_id, json.dumps(hyper, ensure_ascii=False), "queued"),
        )
    except Exception:
        cur.execute(
            """
            INSERT INTO fine_tune_jobs(provider_job_id, dataset_id, metrics, status, started_at)
            VALUES(?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            (job_id, dataset_id, json.dumps({"hyperparameters": hyper}, ensure_ascii=False), "queued"),
        )
    conn.commit()
    return int(cur.lastrowid)


def _update_job_status(conn, job_id: str, status: str, progress: int | None = None, rough: int | None = None):
    cur = conn.cursor()
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


def _finish_job_success(conn, job_id: str, model_name: str, category: str, tuning_type: str):
    cur = conn.cursor()
    cur.execute(
        "UPDATE fine_tune_jobs SET status=?, finished_at=CURRENT_TIMESTAMP WHERE provider_job_id=?",
        ("succeeded", job_id),
    )
    cur.execute("SELECT id FROM llm_models WHERE name=?", (model_name,))
    row = cur.fetchone()
    model_id = None
    if not row:
        mdl_type = "lora" if tuning_type.upper() in ("LORA", "QLORA") else "full"
        cur.execute(
            """
            INSERT INTO llm_models(provider, name, revision, model_path, category, type, is_active)
            VALUES(?,?,?,?,?,?,1)
            """,
            (
                "hf",
                model_name,
                0,
                os.path.join(STORAGE_MODEL_ROOT, model_name),
                category,
                mdl_type,
            ),
        )
        model_id = int(cur.lastrowid)
    else:
        model_id = int(row["id"])

    cur.execute("SELECT id FROM fine_tune_jobs WHERE provider_job_id=?", (job_id,))
    job_row = cur.fetchone()
    if job_row:
        ft_job_pk = int(job_row["id"])
        lora_path = os.path.join(STORAGE_MODEL_ROOT, model_name) if tuning_type.upper() in ("LORA", "QLORA") else None
        cur.execute(
            """
            INSERT INTO fine_tuned_models(model_id, job_id, provider_model_id, lora_weights_path, type, is_active)
            VALUES(?, ?, ?, ?, ?, 1)
            """,
            (model_id, ft_job_pk, model_name, lora_path, ("lora" if lora_path else "full")),
        )

    conn.commit()


def _resolve_out_dir_by_job(conn, job_id: str) -> Optional[str]:
    cur = conn.cursor()
    cur.execute("SELECT hyperparameters, metrics FROM fine_tune_jobs WHERE provider_job_id=?", (job_id,))
    row = cur.fetchone()
    if not row:
        return None
    save_name = None
    if row["hyperparameters"]:
        try:
            hp = json.loads(row["hyperparameters"]) or {}
            save_name = hp.get("saveModelName")
        except Exception:
            save_name = None
    if not save_name and row["metrics"]:
        try:
            mt = json.loads(row["metrics"]) or {}
            hp = mt.get("hyperparameters") or {}
            save_name = hp.get("saveModelName")
        except Exception:
            save_name = None
    if not save_name:
        return None
    out_dir = os.path.join(STORAGE_MODEL_ROOT, save_name)
    return out_dir


def _append_log(log_path: str, line: str):
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line.rstrip("\n") + "\n")
    except Exception:
        pass


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


def _simulate_training(job: FineTuneJob, save_name_with_suffix: str):
    conn = get_db()
    try:
        _update_job_status(conn, job.job_id, "running", progress=1, rough=0)
        out_dir = _ensure_output_dir(save_name_with_suffix)
        _ensure_lora_marker(out_dir, (job.request.get("tuningType") or "QLORA"))
        log_path = os.path.join(out_dir, "train.log")
        _append_log(log_path, f"[{datetime.utcnow().isoformat()}] job {job.job_id} started (SIMULATED)")
        for p in range(2, 101):
            time.sleep(0.1)
            _update_job_status(conn, job.job_id, "running", progress=p)
            if p % 5 == 0:
                _append_log(log_path, f"[{datetime.utcnow().isoformat()}] progress {p}%")
        _finish_job_success(
            conn,
            job.job_id,
            save_name_with_suffix,
            job.category,
            job.request.get("tuningType") or "QLORA",
        )
        _append_log(log_path, f"[{datetime.utcnow().isoformat()}] job {job.job_id} succeeded")
    except Exception:
        cur = conn.cursor()
        cur.execute(
            "UPDATE fine_tune_jobs SET status=? WHERE provider_job_id=?",
            ("failed", job.job_id),
        )
        conn.commit()
        try:
            out_dir = os.path.join(STORAGE_MODEL_ROOT, save_name_with_suffix)
            log_path = os.path.join(out_dir, "train.log")
            _append_log(log_path, f"[{datetime.utcnow().isoformat()}] job {job.job_id} failed")
        except Exception:
            pass
    finally:
        conn.close()


def _run_training_inline(job: FineTuneJob, save_name_with_suffix: str):
    """Run Qwen RAG-style fine-tuning inline using HF/PEFT, modeled on train_qwen_rag.py."""
    conn = get_db()
    try:
        _update_job_status(conn, job.job_id, "running", progress=1, rough=0)
        out_dir = _ensure_output_dir(save_name_with_suffix)
        tuning_type = (job.request.get("tuningType") or "QLORA").upper()
        if tuning_type in ("LORA", "QLORA"):
            _ensure_lora_marker(out_dir, tuning_type)
        log_path = os.path.join(out_dir, "train.log")
        _append_log(log_path, f"[{datetime.utcnow().isoformat()}] job {job.job_id} started (INLINE)")

        try:
            import pandas as pd
            import torch
            from datasets import Dataset
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                TrainingArguments,
                Trainer,
                BitsAndBytesConfig,
            )
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        except Exception as e:
            _append_log(log_path, f"[{datetime.utcnow().isoformat()}] import error: {e}")
            cur = conn.cursor()
            cur.execute("UPDATE fine_tune_jobs SET status=? WHERE provider_job_id=?", ("failed", job.job_id))
            conn.commit()
            return

        system_prompt = job.request.get("systemPrompt") or "You are Qwen, a helpful assistant."

        def build_prompt(context: str, question: str) -> str:
            return f"{context.strip()}\n{system_prompt}\nQuestion: {question.strip()}"

        # Load CSV with fallbacks
        csv_path = _resolve_train_path(job.request.get("trainSetFile"))
        encodings_to_try = ["utf-8", "utf-8-sig", "cp949", "euc-kr", "latin1"]
        df = None
        for enc in encodings_to_try:
            try:
                import pandas as pd  # ensure injected
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
                _append_log(log_path, f"[{datetime.utcnow().isoformat()}] csv load failed: {e}")
                cur = conn.cursor()
                cur.execute("UPDATE fine_tune_jobs SET status=? WHERE provider_job_id=?", ("failed", job.job_id))
                conn.commit()
                return

        conversations = []
        for _, row in df.iterrows():
            conversations.append({
                "conversations": [
                    {"from": "user", "value": build_prompt(row.get("Chunk_Context", ""), row.get("Question", ""))},
                    {"from": "assistant", "value": row.get("Answer", "")},
                ]
            })

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
                assist_prompt = self.tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
                prefix_len = len(self.tokenizer(assist_prompt).input_ids)
                labels = input_ids.clone()
                labels[:prefix_len] = -100
                return {"input_ids": input_ids, "attention_mask": enc.attention_mask[0], "labels": labels}

        base_folder = _resolve_model_dir(job.request.get("baseModelName"))
        if not _has_model_signature(base_folder):
            _append_log(log_path, f"[{datetime.now().isoformat()}] base model not found: {base_folder}")
            cur = conn.cursor()
            cur.execute("UPDATE fine_tune_jobs SET status=? WHERE provider_job_id=?", ("failed", job.job_id))
            conn.commit()
            return
        model_path = base_folder

        output_dir = os.path.join(STORAGE_MODEL_ROOT, save_name_with_suffix)

        tuning_type = (job.request.get("tuningType") or "QLORA").upper()
        if tuning_type == "FULL":
            # Full fine-tuning without 4-bit
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
        else:
            # LoRA/QLORA path with 4-bit
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
                "q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj","w1","w2","c_proj","c_attn"
            ]
            lora_target_modules = sorted({
                name.split(".")[-1]
                for name, module in model.named_modules()
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

        train_dataset = RagDataset(conversations, tokenizer, max_len=job.request.get("max_len", 4096))

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=job.request.get("epochs", 3),
            per_device_train_batch_size=job.request.get("batchSize", 1),
            gradient_accumulation_steps=8,
            learning_rate=job.request.get("learningRate", 2e-4),
            bf16=True,
            logging_steps=10,
            save_steps=500,
            save_total_limit=2,
            evaluation_strategy="no",
            report_to="none",
            optim="paged_adamw_8bit",
        )

        class LogCallback:
            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs and "loss" in logs:
                    _append_log(log_path, f"[{datetime.utcnow().isoformat()}] step={state.global_step} loss={logs['loss']}")

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            callbacks=[LogCallback()],
        )

        trainer.train()
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        _finish_job_success(
            conn,
            job.job_id,
            save_name_with_suffix,
            job.category,
            tuning_type,
        )
        _append_log(log_path, f"[{datetime.utcnow().isoformat()}] job {job.job_id} succeeded")
    except Exception as e:
        try:
            _append_log(os.path.join(STORAGE_MODEL_ROOT, save_name_with_suffix, "train.log"), f"[{datetime.utcnow().isoformat()}] error: {e}")
        except Exception:
            pass
        cur = conn.cursor()
        cur.execute("UPDATE fine_tune_jobs SET status=? WHERE provider_job_id=?", ("failed", job.job_id))
        conn.commit()
    finally:
        conn.close()


def _log_to_save_name(save_name_with_suffix: str, message: str):
    try:
        out_dir = _ensure_output_dir(save_name_with_suffix)
        log_path = os.path.join(out_dir, "train.log")
        _append_log(log_path, f"[{datetime.utcnow().isoformat()}] {message}")
    except Exception:
        pass


def start_fine_tuning(category: str, body: FineTuneRequest) -> Dict[str, Any]:
    suffix = (body.tuningType or "QLORA").upper()
    save_name_with_suffix = f"{body.saveModelName}-{suffix}"
    _ensure_output_dir(save_name_with_suffix)

    base_dir = _resolve_model_dir(body.baseModelName)
    if not _has_model_signature(base_dir):
        _log_to_save_name(save_name_with_suffix, f"base model not found: {base_dir}")
        return {"error": "base model name ERR"}

    job_id = f"ft-job-{uuid.uuid4().hex[:12]}"

    conn = get_db()
    try:
        train_path = _resolve_train_path(body.trainSetFile)
        dataset_id = _insert_dataset_if_needed(conn, train_path, category)
        _insert_job(conn, category, body, job_id, save_name_with_suffix, dataset_id)
    except Exception as e:
        _log_to_save_name(save_name_with_suffix, f"db insert failed: {e}")
        try:
            conn.close()
        except Exception:
            pass
        return {"error": "fine-tuning init failed"}
    finally:
        try:
            conn.close()
        except Exception:
            pass

    job = FineTuneJob(job_id=job_id, category=category, request=body.model_dump())
    use_sim = os.getenv("FT_USE_SIM", "0") != "0"
    target = _simulate_training if use_sim else _run_training_inline
    t = threading.Thread(target=target, args=(job, save_name_with_suffix), daemon=True)
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
