from typing import Optional, List, Dict, Any
from utils import logger
from utils.database import get_session
from sqlalchemy import select, delete
from storage.db_models import (
    LlmModel, LlmPromptMapping, FineTunedModel, LlmEvalRun
)

logger = logger(__name__)

def repo_delete_llm_related_data(model_id: int) -> Dict[str, int]:
    """
    LLM 모델 ID와 관련된 모든 데이터(매핑, 파인튜닝 모델, 평가 실행 이력)를 삭제하고,
    최종적으로 LLM 모델 자체를 삭제합니다.
    삭제된 레코드 수를 반환합니다.
    """
    deleted_counts = {
        "llm_prompt_mapping": 0,
        "fine_tuned_models": 0,
        "llm_eval_runs": 0,
        "llm_models": 0,
    }
    
    with get_session() as session:
        try:
            # 1. llm_prompt_mapping 삭제
            stmt_mapping = delete(LlmPromptMapping).where(LlmPromptMapping.llm_id == model_id)
            result_mapping = session.execute(stmt_mapping)
            deleted_counts["llm_prompt_mapping"] = result_mapping.rowcount

            # 2. fine_tuned_models 삭제
            stmt_ft = delete(FineTunedModel).where(FineTunedModel.model_id == model_id)
            result_ft = session.execute(stmt_ft)
            deleted_counts["fine_tuned_models"] = result_ft.rowcount

            # 3. llm_eval_runs 삭제
            stmt_eval = delete(LlmEvalRun).where(LlmEvalRun.llm_id == model_id)
            result_eval = session.execute(stmt_eval)
            deleted_counts["llm_eval_runs"] = result_eval.rowcount

            # 4. llm_models 삭제
            stmt_model = delete(LlmModel).where(LlmModel.id == model_id)
            result_model = session.execute(stmt_model)
            deleted_counts["llm_models"] = result_model.rowcount

            session.commit()
            return deleted_counts
            
        except Exception:
            session.rollback()
            logger.exception(f"Failed to delete related data for model_id: {model_id}")
            raise

