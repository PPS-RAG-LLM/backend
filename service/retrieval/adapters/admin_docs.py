"""Milvus 기반 글로벌 검색 어댑터."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from service.admin.manage_vator_DB import execute_search  # type: ignore
from service.retrieval.adapters.base import BaseRetrievalAdapter, RetrievalResult


def _run_execute_search(**kwargs: Any) -> Dict[str, Any]:
    """manage_vator_DB.execute_search 는 async이므로 동기 코드에서 호출할 수 있도록 helper 제공."""
    try:
        return asyncio.run(execute_search(**kwargs))
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(execute_search(**kwargs))
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
        """
        Milvus 글로벌 검색 실행.

        Args:
            query: 사용자 질문
            top_k: embedding 후보 개수
            security_level: 사용자 보안 레벨
            task_type: 작업 유형(doc_gen|summary|qna)
            search_type: vector/hybrid/bm25
            model_key: 임베딩 모델 키
            rerank_top_n: 최종 반환 개수 (없으면 top_k 사용)
            source_filter: 파일명 필터
        """
        response = _run_execute_search(
            question=query,
            top_k=10,
            rerank_top_n=rerank_top_n or top_k, # TODO : 사용자 부분 리랭킹 중복사용 제거 (속도최적화)
            security_level=int(security_level), 
            source_filter=source_filter,
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
                    score=float(hit.get("score", 0.0)),
                    chunk_index=hit.get("chunk_idx"),
                    page=hit.get("page"),
                    metadata={
                        "path": hit.get("path"),
                        "task_type": hit.get("task_type"),
                        "security_level": hit.get("security_level"),
                    },
                )
            )
        return results[: top_k or len(results)]

