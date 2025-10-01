from typing import Any, Optional, Dict
from utils import logger
from utils.database import get_session
from sqlalchemy import select
from storage.db_models import LlmModel

logger = logger(__name__)


def get_llm_model_by_provider_and_name(provider: str, name: str) -> Optional[Dict[str, Any]]:
    """활성화된 LLM 모델을 provider와 name으로 조회한다."""
    logger.info(f"provider : {provider}\nmodelName : {name}")
    with get_session() as session:
        stmt = (
            select(LlmModel.provider, LlmModel.name, LlmModel.model_path)
            .where(
                LlmModel.name == name,
                LlmModel.provider == provider,
                LlmModel.is_active == True,
            )
            .limit(1)
        )
        row = session.execute(stmt).first()
        if row is None:
            logger.info("DB에서 가져온 모델 정보: None")
            return None
        try:
            data = dict(row._mapping)
        except Exception:
            logger.exception("row dict 변환 실패")
            return None
        logger.info(f"DB에서 가져온 모델 정보: {data}")
        return data