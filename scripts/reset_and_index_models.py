#!/usr/bin/env python3
"""
Reset and re-index LLM model metadata in the local SQLite database.

What this script does (by default):
1) Reset model-related tables (wipe):
   - fine_tuned_models
   - fine_tune_jobs
   - fine_tune_datasets
   - chat_feedback
   - llm_models
   - cache_data rows related to active models (name LIKE 'active_model:%')

2) (Optional) Scan filesystem under STORAGE_ROOT (default: backend/storage/models):
   - Count discoverable base folders (with config.json) for your reference.
   - NOTE: Does NOT insert rows into llm_models (schema only allows qa|doc_gen|summary).

Usage:
python scripts/reset_and_index_models.py --reset --index
python scripts/reset_and_index_models.py --reset            # only wipe
python scripts/reset_and_index_models.py --index            # only re-index

Environment variables:
  COREIQ_DB: path to sqlite3 DB (defaults to backend/storage/coreiq.sqlite3)
  STORAGE_MODEL_ROOT: path to model storage (defaults to backend/storage/models)
"""

from __future__ import annotations

import argparse
import os
import sqlite3
from pathlib import Path


def _backend_root() -> Path:
    return Path(__file__).resolve().parents[1]


# Default DB to pps_rag.db (can be overridden with COREIQ_DB)
DB_PATH = os.getenv("COREIQ_DB", "/home/work/CoreIQ/backend/storage/pps_rag.db")
STORAGE_MODEL_ROOT = os.getenv("STORAGE_MODEL_ROOT", str(_backend_root() / "storage" / "models"))


def connect_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def wipe_model_tables(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    # Child tables first to avoid FK issues (if any)
    tables = [
        "fine_tuned_models",
        "fine_tune_jobs",
        "fine_tune_datasets",
        "chat_feedback",
        "llm_models",
    ]
    for t in tables:
        try:
            cur.execute(f"DELETE FROM {t}")
        except Exception as e:
            print(f"[warn] skipping wipe for table={t}: {e}")
    # Clear active model cache entries
    try:
        cur.execute("DELETE FROM cache_data WHERE name LIKE 'active_model:%'")
    except Exception as e:
        print(f"[warn] skipping cache wipe: {e}")
    conn.commit()


def _has_config_json(dir_path: str) -> bool:
    return os.path.isfile(os.path.join(dir_path, "config.json"))


def reindex_from_fs(conn: sqlite3.Connection, storage_root: str) -> int:
    if not os.path.isdir(storage_root):
        print(f"[info] storage root not found: {storage_root}")
        return 0
    inserted = 0  # kept for backward-compat; we no longer insert into DB here
    for entry in os.scandir(storage_root):
        if not entry.is_dir():
            continue
        model_dir = entry.path
        if not _has_config_json(model_dir):
            continue
        # No DB insert here by design; base models should be registered via API or during fine-tuning
    return inserted


def main():
    parser = argparse.ArgumentParser(description="Reset and re-index LLM model metadata")
    parser.add_argument("--reset", action="store_true", help="Wipe model-related tables")
    parser.add_argument("--index", action="store_true", help="Index base models from filesystem")
    parser.add_argument("--model-root", default=STORAGE_MODEL_ROOT, help="Model storage root directory")
    args = parser.parse_args()

    # Default: do both if neither specified
    do_reset = args.reset or (not args.reset and not args.index)
    do_index = args.index or (not args.reset and not args.index)

    print(f"[info] DB_PATH={DB_PATH}")
    print(f"[info] STORAGE_MODEL_ROOT={args.model_root}")

    conn = connect_db()
    try:
        if do_reset:
            print("[info] wiping model-related tables ...")
            wipe_model_tables(conn)
            print("[info] wipe completed")
        if do_index:
            print("[info] indexing base models from filesystem ...")
            n = reindex_from_fs(conn, args.model_root)
            print(f"[info] indexing completed, inserted={n}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()


