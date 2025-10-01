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
from sqlalchemy import text
from utils.database import get_session
from storage.db_models import (
    FineTunedModel,
    FineTuneJob, 
    FineTuneDataset,
    LlmModel
)

# Ensure project root on path when executed directly
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

    
def main() -> None:
    session = get_session()
    try:
        # ORM을 사용하여 레코드 삭제
        session.query(FineTunedModel).delete()
        session.query(FineTuneJob).delete()
        session.query(FineTuneDataset).delete()
        session.query(LlmModel).delete()
        session.commit()
        
        # VACUUM으로 공간 확보
        session.execute(text("VACUUM;"))
        session.commit()
        
        print("[db_del] fine-tune tables cleared and database vacuumed.")
    except Exception as e:
        session.rollback()
        print(f"[db_del] Error: {e}")
        raise
    finally:
        session.close()

if __name__ == "__main__":
    main()
