"""검색 어댑터 공통 인터페이스."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(slots=True)
class RetrievalResult:
    """통일된 RAG 스니펫 스키마."""
    doc_id: Optional[str]
    title: str
    text: str
    score: float
    source: str
    chunk_index: Optional[int] = None
    page: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseRetrievalAdapter(ABC):
    """검색 어댑터 기반 클래스."""

    source: str
    def __init__(self, source: str) -> None:
        self.source = source

    @abstractmethod
    def search(self, query: str, top_k: int, **kwargs: Any) -> List[RetrievalResult]:
        """
        검색 실행.

        Args:
            query: 사용자 질문
            top_k: 최고 결과 개수
            **kwargs: 어댑터별 추가 파라미터
        """

    def _build_result(
        self,
        *,
        doc_id: Optional[str],
        title: str,
        text: str,
        score: float,
        chunk_index: Optional[int] = None,
        page: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RetrievalResult:
        """하위 클래스에서 통일된 포맷으로 쉽게 변환하도록 도우미 제공."""
        return RetrievalResult(
            doc_id=doc_id,
            title=title,
            text=text,
            score=score,
            source=self.source,
            chunk_index=chunk_index,
            page=page,
            metadata=metadata or {},
        )

