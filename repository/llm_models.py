from typing import Any, Optional, Dict
from utils import logger
from utils.database import get_session
from sqlalchemy import select, or_, desc, func
from storage.db_models import LlmModel, LlmPromptMapping

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


def repo_find_fallback_model(category: str) -> Optional[str]:
    """
    해당 카테고리(또는 all)의 활성화된 최신 모델 이름을 조회합니다.
    """
    with get_session() as session:
        stmt = select(LlmModel.name).where(
            LlmModel.is_active == True,
            or_(LlmModel.category == category, LlmModel.category == 'all')
        ).order_by(
            desc(LlmModel.trained_at), desc(LlmModel.id)
        ).limit(1)
        
        return session.execute(stmt).scalar()


def repo_find_best_mapped_model(prompt_id: int) -> Optional[str]:
    """
    템플릿(prompt_id)에 매핑된 모델 중 Rouge Score가 가장 높은(또는 최신) 모델 이름을 조회합니다.
    """
    with get_session() as session:
        stmt = select(LlmModel.name).join(
            LlmPromptMapping, LlmPromptMapping.llm_id == LlmModel.id
        ).where(
            LlmPromptMapping.prompt_id == prompt_id
        ).order_by(
            desc(func.coalesce(LlmPromptMapping.rouge_score, -1)),
            desc(LlmPromptMapping.llm_id)
        ).limit(1)
        
        return session.execute(stmt).scalar()