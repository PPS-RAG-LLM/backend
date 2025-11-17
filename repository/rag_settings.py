from __future__ import annotations

from utils import logger
from utils.database import get_session
from storage.db_models import  RagSettings
from sqlalchemy.dialects.sqlite import insert 

logger = logger(__name__)


def get_rag_settings_row() -> dict:
    """RAG 전역 설정 로더. 없으면 빈 dict."""
    with get_session() as session:
        row = session.query(RagSettings).filter(RagSettings.id == 1).first()
        if not row:
            return {}
        return {
            "embedding_key": row.embedding_key,
            "search_type": row.search_type,
            "chunk_size": int(row.chunk_size or 512),
            "overlap": int(row.overlap or 64),
        }


def get_vector_settings_row() -> dict:
    """레거시 호환: rag_settings(싱글톤)에서 기본 청크 설정을 읽어온다."""
    with get_session() as session:
        row = session.query(RagSettings).filter(RagSettings.id == 1).first()
        if not row:
            return {"search_type": "hybrid", "chunk_size": 512, "overlap": 64}
        return {
            "search_type": str(row.search_type or "hybrid"),
            "chunk_size": int(row.chunk_size or 512),
            "overlap": int(row.overlap or 64),
        }
