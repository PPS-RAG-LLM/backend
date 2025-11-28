"""
도메인 중립 검색 서비스 패키지.

ADMIN과 USER 영역 모두에서 공통으로 사용할 검색/리랭크 유틸리티를 제공합니다.
"""

from service.retrieval.unified import unified_search  # noqa: F401
from service.retrieval.common import (  # noqa: F401
    embed_text,
    cosine_similarity,
    # get_document_title,
    # load_document_vectors,
)

__all__ = [
    "unified_search",
    "embed_text",
    "cosine_similarity",
    # "get_document_title",
    # "load_document_vectors",
]

