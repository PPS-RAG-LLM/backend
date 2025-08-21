from typing import Any, Optional, Dict
from utils import get_db, logger

logger = logger(__name__)

def get_llm_model_by_provider_and_name(provider: str, name: str) -> Optional[Dict[str, Any]]:
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT provider, name, model_path
            FROM llm_models
            WHERE name= ? AND is_active=1 AND provider=?
            """,
            (name, provider)
        )
        row = cur.fetchone()
        logger.info(f"DB에서 가져온 모델 정보: {dict(row)}")
        return dict(row) if row else None
    finally:
        conn.close()