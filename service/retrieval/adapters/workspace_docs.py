"""워크스페이스 문서 검색 어댑터."""

from __future__ import annotations

from typing import List
from utils import logger
from repository.documents import list_doc_ids_by_workspace

from service.retrieval.adapters.base import RetrievalResult
from service.retrieval.adapters.temp_attatchments import TempAttachmentsVectorAdapter

logger = logger(__name__)
class WorkspaceDocsAdapter(TempAttachmentsVectorAdapter):
    """workspace_documents 테이블에 등록된 문서를 검색한다."""

    def __init__(self) -> None:
        super().__init__(source="workspace")

    def search(
        self,
        query: str,
        top_k: int,
        *,
        workspace_id: int,
        threshold: float = 0.0,
    ) -> List[RetrievalResult]:
        """
        특정 워크스페이스에 연결된 문서에서 스니펫 검색.

        Args:
            query: 사용자 질문
            top_k: 반환할 결과 수
            workspace_id: workspace_documents.workspace_id
            threshold: 코사인 유사도 하한
        """
        if not workspace_id:
            return []

        rows = list_doc_ids_by_workspace(int(workspace_id)) or []
        doc_ids = [
            str(row["doc_id"]) if isinstance(row, dict) else str(row)
            for row in rows
            if row
        ]
        logger.info("[WorkspaceAdapter] workspace_id=%s doc_ids=%s", workspace_id, doc_ids)
        if not doc_ids:
            return []

        return super().search(
            query,
            top_k,
            doc_ids=doc_ids,
            threshold=threshold,
        )