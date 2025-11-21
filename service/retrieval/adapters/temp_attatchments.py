"""로컬 벡터 캐시 검색 어댑터."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from utils import logger
from service.retrieval.adapters.base import BaseRetrievalAdapter, RetrievalResult
from service.retrieval.common import (
    cosine_similarity,
    embed_text,
    get_document_title,
    load_document_vectors,
)

logger = logger(__name__)


class TempAttachmentsVectorAdapter(BaseRetrievalAdapter):
    """doc_id 목록을 입력받아 로컬 벡터 캐시에서 스니펫을 찾는다."""

    def __init__(self, source: str = "local") -> None:
        super().__init__(source=source)

    def search(
        self,
        query: str,
        top_k: int,
        *,
        doc_ids: List[str],
        threshold: float = 0.0,
    ) -> List[RetrievalResult]:
        """
        로컬 문서(doc_id 기반)에서 검색.

        Args:
            query: 사용자 질문
            top_k: 반환할 최대 결과 수
            doc_ids: 검색 대상 문서 ID 목록
            threshold: 코사인 유사도 하한선
        """
        if not doc_ids:
            return []

        query_vec = embed_text(query)
        title_cache: Dict[str, str] = {}
        hits: List[RetrievalResult] = []
        logger.debug("[LocalAdapter] doc_ids=%s", doc_ids)

        for doc_id in doc_ids:
            logger.debug("[LocalAdapter] doc_id=%s", doc_id)
            vectors = load_document_vectors(doc_id)
            if not vectors:
                logger.warning("[LocalAdapter] no vectors for doc_id=%s", doc_id)
                continue
            if doc_id not in title_cache:
                title_cache[doc_id] = get_document_title(doc_id)
            title = title_cache[doc_id]

            for idx, chunk in enumerate(vectors):
                vec = chunk.get("values")
                meta = chunk.get("metadata") or {}
                text = str(meta.get("text") or "")
                if not text:
                    continue

                if not isinstance(vec, list):
                    continue
                try:
                    chunk_vec = np.asarray(vec, dtype=float)
                except Exception:  # pragma: no cover - 잘못된 벡터 방어
                    continue

                score = cosine_similarity(query_vec, chunk_vec)
                if score < threshold:
                    continue

                hits.append(
                    self._build_result(
                        doc_id=str(doc_id),
                        title=title,
                        text=text,
                        score=score,
                        chunk_index=idx,
                        page=meta.get("page"),
                        metadata={"chunk_index": idx, **meta},
                    )
                )

        hits.sort(key=lambda r: r.score, reverse=True)
        if top_k <= 0:
            return hits
        return hits[: max(1, int(top_k))]

