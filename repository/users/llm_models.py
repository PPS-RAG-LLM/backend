from typing import Any, Optional, Dict
from utils import get_db, logger

logger = logger(__name__)

def get_llm_model_by_provider_and_name(provider: str, name: str) -> Optional[Dict[str, Any]]:
    """Retrieve active LLM model metadata by provider/name.

    Returns dict with at least {provider, name, model_path} or None if not found.
    """
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT provider, name, model_path
            FROM llm_models
            WHERE name=? AND provider=? AND is_active=1
            LIMIT 1
            """,
            (name, provider),
        )
        row = cur.fetchone()
        if row:
            logger.debug({"llm_model": dict(row)})
            return dict(row)
        logger.warning(f"LLM model not found provider={provider} name={name}")
        return None
    finally:
        conn.close()
