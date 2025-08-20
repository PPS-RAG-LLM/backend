#!/usr/bin/env python
"""Simple CLI for manually testing LLM fine-tuning workflow.

Usage (from project root):
    python backend/service/admin/test.py \
        --category qa \
        --base-model-name qwen-7b \
        --save-model-name my-ft-model \
        --train-set-file data.csv

Tips:
1. This script automatically sets the environment variable FT_USE_SIM=1 so the
   fine-tuning runs in *simulated* mode (fast, no GPU needed).
2. You can watch the live progress printed to console; the job typically
   completes within a few seconds in simulation mode.
3. Adjust the arguments to point at an existing base model directory and CSV
   file if you wish to test the real trainer (export FT_USE_SIM=0 before run).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Ensure the current directory is on sys.path so local imports resolve when the
# script is executed from project root.
ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.service.admin.LLM_finetuning import (
    FineTuneRequest,
    start_fine_tuning,
    get_fine_tuning_status,
    get_fine_tuning_logs,
)

DEFAULT_TRAIN_FILE = "data.csv"  # searched under storage/train_data via helper


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a fine-tuning job and monitor progress (simulation by default)")
    p.add_argument("--category", default="qa", choices=["qa", "doc_gen", "summary"], help="Job category label")
    p.add_argument("--base-model-name", required=True, help="Base model directory name or absolute path")
    p.add_argument("--save-model-name", required=True, help="Name to save the fine-tuned model under")
    p.add_argument("--train-set-file", default=DEFAULT_TRAIN_FILE, help="CSV file name / path for training data")
    p.add_argument("--tuning-type", default="QLORA", choices=["LORA", "QLORA", "FULL"], help="Finetuning strategy")
    p.add_argument("--batch-size", type=int, default=4, help="Per-device train batch size")
    p.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    p.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps")
    p.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate")
    p.add_argument("--overfitting-prevention", action="store_true", help="Enable overfitting prevention flags")
    p.add_argument("--simulate", action="store_true", help="Run in simulated mode (sets FT_USE_SIM=1)")
    p.add_argument("--poll-interval", type=float, default=1.0, help="Seconds between status polls")
    p.add_argument("--tail", type=int, default=100, help="Log lines to show when finished")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Handle simulation flag
    if args.simulate:
        os.environ["FT_USE_SIM"] = "1"
    else:
        # Ensure real mode unless user already set the env var externally
        os.environ.setdefault("FT_USE_SIM", "0")

    body = FineTuneRequest(
        baseModelName=args.base_model_name,
        saveModelName=args.save_model_name,
        systemPrompt="You are Qwen, a helpful assistant.",
        batchSize=args.batch_size,
        epochs=args.epochs,
        learningRate=args.learning_rate,
        gradientAccumulationSteps=args.grad_accum,
        overfittingPrevention=args.overfitting_prevention,
        trainSetFile=args.train_set_file,
        tuningType=args.tuning_type,
    )

    resp = start_fine_tuning(args.category, body)
    if "error" in resp:
        print("[ERROR]", resp["error"], file=sys.stderr)
        sys.exit(1)

    job_id = resp["jobId"]
    print("Started job:", job_id)

    last_dump = None
    while True:
        st = get_fine_tuning_status(args.category, job_id)
        dump = json.dumps(st, ensure_ascii=False)
        if dump != last_dump:
            print("Status:", dump)
            last_dump = dump
        if st.get("status") in ("succeeded", "failed"):
            break
        time.sleep(max(0.2, args.poll_interval))

    print("\n===== Tail logs =====")
    logs = get_fine_tuning_logs(job_id, tail=args.tail)
    for line in logs.get("lines", []):
        print(line)

    if st.get("status") == "succeeded":
        print("\nJob completed successfully ✅")
    else:
        print("\nJob failed ❌")
        # Save logs to a local error file for inspection
        err_file = f"ft_error_{job_id}.txt"
        with open(err_file, "w", encoding="utf-8") as f:
            for line in logs.get("lines", []):
                f.write(line + "\n")
        print(f"Error log saved to {err_file}")


if __name__ == "__main__":
    main()
