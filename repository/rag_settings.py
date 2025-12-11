from __future__ import annotations

from utils import logger, now_kst
from utils.database import get_session
from storage.db_models import  RagSettings

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


def set_rag_settings_row(new_search: str, new_chunk: int, new_overlap: int, new_key: str):
    with get_session() as session:
        settings = session.query(RagSettings).get(1)
        if not settings:
            settings = RagSettings(id=1)
            session.add(settings)

        settings.embedding_key = new_key
        settings.search_type = new_search
        settings.chunk_size = new_chunk
        settings.overlap = new_overlap
        settings.updated_at = now_kst()
        session.commit()

