from __future__ import annotations
from typing import Optional

from utils import logger
from utils.database import get_session
from storage.db_models import EmbeddingModel
from sqlalchemy.dialects.sqlite import insert 

logger = logger(__name__)


def get_active_embedding_model_name() -> str:
    """활성화된 임베딩 모델 이름 반환 (없으면 예외)"""
    with get_session() as session:
        row = (
            session.query(EmbeddingModel)
            .filter(EmbeddingModel.is_active == 1)
            .order_by(EmbeddingModel.activated_at.desc().nullslast())
            .first()
        )
        if not row:
            raise ValueError(
                "활성화된 임베딩 모델이 없습니다. 먼저 /v1/admin/vector/settings에서 모델을 설정하세요."
            )
        return str(row.name)


def get_embedding_model_path_by_name(model_key: str) -> Optional[str]:
    """
    활성화된 임베딩 모델의 model_path를 DB에서 조회.
    model_key와 일치하고 is_active=1인 행의 model_path를 반환.
    없으면 None.
    """
    with get_session() as session:
        row = (
            session.query(EmbeddingModel)
            .filter(EmbeddingModel.is_active == 1, EmbeddingModel.name == model_key)
            .order_by(EmbeddingModel.activated_at.desc().nullslast())
            .first()
        )
        if row and row.model_path:
            return str(row.model_path)
        return None