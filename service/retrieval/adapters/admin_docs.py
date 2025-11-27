"""Milvus 기반 글로벌 검색 어댑터."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from service.admin.manage_vator_DB import search_vector_candidates # type: ignore
from service.retrieval.admin_search import RAGSearchRequest
from service.retrieval.adapters.base import BaseRetrievalAdapter, RetrievalResult


def _run_search_candidates(**kwargs: Any) -> Dict[str, Any]:
    """async search_vector_candidates를 동기적으로 실행"""
    req = RAGSearchRequest(
        query=kwargs["question"],
        top_k=kwargs["top_k"],
        user_level=kwargs["security_level"],
        task_type=kwargs["task_type"],
        model=kwargs.get("model_key"),
    )
    search_type = kwargs.get("search_type")
    
    try:
        return asyncio.run(search_vector_candidates(req, search_type_override=search_type))
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(search_vector_candidates(req, search_type_override=search_type))
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
        # [변경] search_vector_candidates 호출 (리랭킹 X)
        response = _run_search_candidates(
            question=query,
            top_k=top_k, # 후보군 넉넉히
            security_level=int(security_level), 
            task_type=task_type,
            model_key=model_key,
            search_type=search_type,
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
                    score=float(hit.get("score_fused", hit.get("score_vec", 0.0))), # 리랭크 전 점수 사용
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