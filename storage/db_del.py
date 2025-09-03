#!/usr/bin/env python
"""Dangerous helper: wipe fine-tuning related tables in pps_rag.db

Usage:
    python backend/storage/db_del.py    # deletes records and VACUUMs DB

This will DELETE all rows from:
  • fine_tuned_models
  • fine_tune_jobs
  • fine_tune_datasets
  • llm_models

Only use in development environments!"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root on path when executed directly
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from service.admin.LLM_finetuning import reset_fine_tune_tables
from utils.database import get_db


def main() -> None:
    reset_fine_tune_tables()
    # Optional VACUUM to reclaim space
    conn = get_db()
    try:
        conn.execute("VACUUM;")
        conn.commit()
    finally:
        conn.close()
    print("[db_del] fine-tune tables cleared and database vacuumed.")


if __name__ == "__main__":
    main()
