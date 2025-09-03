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
        if row is None:
            logger.info("DB에서 가져온 모델 정보: None")
            return None
        try:
            data = dict(row)
        except Exception:
            logger.exception("row dict 변환 실패")
            return None
        logger.info(f"DB에서 가져온 모델 정보: {data}")
        return data
    finally:
        conn.close()