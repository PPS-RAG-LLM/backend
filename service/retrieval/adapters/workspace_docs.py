"""워크스페이스 문서 검색 어댑터."""

from __future__ import annotations

from typing import List

from utils import logger

from repository.documents import list_doc_ids_by_workspace
from service.retrieval.adapters.base import RetrievalResult
from service.retrieval.adapters.temp_attatchments import TempAttachmentsVectorAdapter
from service.vector_db.milvus_store import resolve_collection
from storage.db_models import DocumentType

logger = logger(__name__)


class WorkspaceDocsAdapter(TempAttachmentsVectorAdapter):
    """workspace_documents 테이블에 등록된 문서를 Milvus에서 검색한다."""

    def __init__(self) -> None:
        super().__init__(source="workspace")
        self.collection_name = resolve_collection(DocumentType.WORKSPACE.value)

    def search(
        self,
        query: str,
        top_k: int,
        *,
        workspace_id: int,
        threshold: float = 0.0,
        mode: str = "hybrid",
    ) -> List[RetrievalResult]:
        if not workspace_id:
            return []

        rows = list_doc_ids_by_workspace(int(workspace_id)) or []
        doc_ids = [
            str(row["doc_id"]) if isinstance(row, dict) else str(row)
            for row in rows
            if row
        ]
        if not doc_ids:
            logger.info("[WorkspaceAdapter] no documents for workspace_id=%s", workspace_id)
            return []

        return super().search(
            query,
            top_k,
            doc_ids=doc_ids,
            threshold=threshold,
            mode=mode,
            workspace_id=workspace_id,
        )