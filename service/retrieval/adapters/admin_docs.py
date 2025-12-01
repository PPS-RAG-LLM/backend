"""Milvus 기반 글로벌 검색 어댑터."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from config import config as app_config
from service.retrieval.adapters.base import BaseRetrievalAdapter, RetrievalResult
from service.retrieval.interface import SearchRequest, retrieval_service

_RETRIEVAL_CFG = app_config.get("retrieval", {}) or {}
_MILVUS_CFG = _RETRIEVAL_CFG.get("milvus", {}) or {}
_ADMIN_COLLECTION = _MILVUS_CFG.get("ADMIN_DOCS", "admin_docs_collection")


def _run_search_candidates(**kwargs: Any) -> Dict[str, Any]:
    """동기 환경에서 RetrievalService.search 실행."""

    request = SearchRequest(
        query=kwargs["question"],
        collection_name=kwargs.get("collection_name", _ADMIN_COLLECTION),
        task_type=kwargs.get("task_type", "qna"),
        security_level=int(kwargs.get("security_level", 1)),
        top_k=int(kwargs.get("top_k", 5)),
        rerank_top_n=kwargs.get("rerank_top_n"),
        search_type=kwargs.get("search_type"),
        model_key=kwargs.get("model_key"),
    )

    async def _runner() -> Dict[str, Any]:
        return await retrieval_service.search(request)

    try:
        return asyncio.run(_runner())
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(_runner())
        finally:
            loop.close()

class AdminDocsAdapter(BaseRetrievalAdapter):
    """Milvus Server 검색 래퍼."""

    def __init__(self) -> None:
        super().__init__(source="milvus")

    def search(
        self,
        query: str,
        top_k: int,
        *,
        security_level: int,
        task_type: str = "qna",
        search_type: Optional[str] = None,
        model_key: Optional[str] = None,
        rerank_top_n: Optional[int] = None,
        source_filter: Optional[List[str]] = None,
    ) -> List[RetrievalResult]:
        response = _run_search_candidates(
            question=query,
            top_k=top_k,
            security_level=int(security_level), 
            task_type=task_type,
            model_key=model_key,
            search_type=search_type,
            rerank_top_n=0,
            collection_name=_ADMIN_COLLECTION,
        )
        hits = response.get("hits", []) or []
        results: List[RetrievalResult] = []
        for hit in hits:
            snippet = str(hit.get("snippet") or "").strip()
            if not snippet:
                continue
            results.append(
                self._build_result(
                    doc_id=hit.get("doc_id"),
                    title=str(hit.get("doc_id") or hit.get("path") or "Milvus"),
                    text=snippet,
                    score=float(hit.get("score", hit.get("score_vec", 0.0))),
                    chunk_index=hit.get("chunk_idx"),
                    page=hit.get("page"),
                    metadata={
                        "path": hit.get("path"),
                        "task_type": hit.get("task_type"),
                        "security_level": hit.get("security_level"),
                    },
                )
            )
        # 리랭킹은 unified.py에서 수행되므로 여기서는 후보군 그대로 반환
        return results 