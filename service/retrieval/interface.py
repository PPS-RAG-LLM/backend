from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from pathlib import Path

from repository.rag_settings import get_rag_settings_row
from utils import logger


LOGGER = logger(__name__)


@dataclass(slots=True)
class SearchRequest:
    """Retrieval entrypoint request."""

    query: str
    collection_name: str
    task_type: str = "qna"
    security_level: int = 1
    top_k: int = 20
    rerank_top_n: Optional[int] = 5
    search_type: Optional[str] = None
    model_key: Optional[str] = None
    filters: Dict[str, Any] = field(default_factory=dict)
    extra_options: Dict[str, Any] = field(default_factory=dict)


class RetrievalService:
    """Single entrypoint for RAG search & ingestion."""

    async def search(self, req: SearchRequest) -> Dict[str, Any]:
        """
        Execute retrieval with unified pipeline.
        defers heavy imports to keep interface lightweight.
        """

        if not req.query.strip():
            raise ValueError("query must not be empty")
        if req.top_k <= 0:
            raise ValueError("top_k must be positive")

        # Lazy import to avoid circular deps during bootstrap
        from .search_pipeline import run_search_pipeline

        settings = get_rag_settings_row()
        model_key = req.model_key or settings.get("embedding_key")

        LOGGER.debug(
            "[RetrievalService] search query=%s collection=%s task=%s top_k=%s rerank=%s",
            req.query,
            req.collection_name,
            req.task_type,
            req.top_k,
            req.rerank_top_n,
        )

        return await run_search_pipeline(
            query=req.query,
            collection=req.collection_name,
            task_type=req.task_type,
            security_level=req.security_level,
            top_k=req.top_k,
            rerank_top_n=req.rerank_top_n,
            search_type=req.search_type,
            model_key=model_key,
            filters=req.filters,
            extra_options=req.extra_options,
        )

    async def ingest_documents(
        self,
        *,
        inputs: Sequence[Union[str, Path, Dict[str, Any]]],
        collection_name: str,
        task_types: Sequence[str],
        security_level_config: Optional[Dict[str, Dict[str, Any]]] = None,
        override_level_map: Optional[Dict[str, int]] = None,
        doc_id_generator: Optional[Callable[[Any], str]] = None,
        doc_id_version_parser: Optional[Callable[[str], Tuple[str, int]]] = None,
        metadata_extras: Optional[Dict[str, Any]] = None,
        pre_ingest_callback: Optional[Callable[[Dict[str, Any]], Any]] = None,
        post_ingest_callback: Optional[Callable[[Dict[str, Any]], Any]] = None,
        batch_callback: Optional[Callable[[List[Dict[str, Any]], str], Any]] = None,
        upsert_metadata: bool = True,
    ) -> Dict[str, Any]:
        """Run ingestion pipeline with shared settings from DB."""

        if not inputs:
            raise ValueError("inputs must not be empty")
        if not task_types:
            raise ValueError("task_types must not be empty")

        from .ingestion import ingest_common

        settings = get_rag_settings_row()
        normalized_inputs: List[Union[str, Path, Dict[str, Any]]] = [
            Path(item) if isinstance(item, str) else item for item in inputs
        ]

        LOGGER.info(
            "[RetrievalService] ingest collection=%s tasks=%s count=%s",
            collection_name,
            list(task_types),
            len(normalized_inputs),
        )

        return await ingest_common(
            inputs=normalized_inputs,
            collection_name=collection_name,
            task_types=list(task_types),
            settings=settings,
            security_level_config=security_level_config,
            override_level_map=override_level_map,
            doc_id_generator=doc_id_generator,
            doc_id_version_parser=doc_id_version_parser,
            metadata_extras=metadata_extras,
            pre_ingest_callback=pre_ingest_callback,
            post_ingest_callback=post_ingest_callback,
            batch_callback=batch_callback,
            upsert_metadata=upsert_metadata,
        )


# Singleton-style helper
retrieval_service = RetrievalService()


