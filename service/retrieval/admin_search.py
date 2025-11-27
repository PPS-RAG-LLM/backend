from __future__ import annotations
import asyncio
import time
from typing import Any, Dict, List, Optional
from pathlib import Path
from pydantic import BaseModel, Field

from config import config as app_config
from repository.documents import fetch_metadata_by_vector_ids
from repository.rag_settings import get_rag_settings_row
from service.vector_db import (
    ensure_collection_and_index,
    get_milvus_client,
    run_dense_search,
    run_hybrid_search,
)
from service.retrieval.common import hf_embed_text
from service.retrieval.pipeline import (
    DEFAULT_OUTPUT_FIELDS,
    build_dense_hits,
    build_rerank_payload,
)
from service.retrieval.reranker import rerank_snippets
from utils.model_load import get_or_load_embedder_async
from utils import logger

logger = logger(__name__)

# --- Config & Constants ---
_RETRIEVAL_CFG = app_config.get("retrieval", {}) or {}
_MILVUS_CFG = _RETRIEVAL_CFG.get("milvus", {}) or {}
ADMIN_COLLECTION = _MILVUS_CFG.get("ADMIN_DOCS", "admin_docs_collection")
TASK_TYPES = tuple(_RETRIEVAL_CFG.get("task_types") or ("doc_gen", "summary", "qna"))

# --- Models ---
class RAGSearchRequest(BaseModel):
    query: str
    top_k: int = Field(5, gt=0)
    user_level: int = Field(1, ge=1)
    task_type: str = Field(..., description="doc_gen | summary | qna")
    model: Optional[str] = None 



# -------------------------------------------------
# 검색 / 리랭킹 / 중복제거 분리 함수
# -------------------------------------------------

def apply_reranking(hits: List[Dict[str, Any]], query: str, top_n: int = 5) -> List[Dict[str, Any]]:
    """
    검색된 후보군(hits)에 대해 Reranking을 수행하고 점수 순으로 정렬하여 반환합니다.
    """
    if not hits:
        return []
    rerank_candidates = build_rerank_payload(hits)
    # 후보가 있으면 리랭킹 수행
    if rerank_candidates:
        reranked = rerank_snippets(rerank_candidates, query=query, top_n=top_n)
        hits_sorted = []
        for res in reranked:
            original = res.metadata or {}
            hits_sorted.append(
                {
                    "score": float(res.score),
                    "path": original.get("path"),
                    "chunk_idx": int(original.get("chunk_idx", 0)),
                    "task_type": original.get("task_type"),
                    "security_level": int(original.get("security_level", 1)),
                    "doc_id": original.get("doc_id"),
                    "page": int(original.get("page", 0)),
                    "snippet": res.text,
                }
            )
        return hits_sorted
    # 리랭크 후보가 없거나 실패 시 기존 점수 정렬 (Fallback)
    return sorted(
        hits,
        key=lambda x: x.get("score_fused", x.get("score_vec", x.get("score_sparse", 0.0))),
        reverse=True,
    )[:top_n]


def deduplicate_hits(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    리랭크된 결과에서 스니펫 텍스트 및 문서 청크 기준 중복을 제거합니다.
    """
    seen_by_snippet: dict[str, dict] = {}
    seen_by_chunk: dict[tuple[str, int], dict] = {}
    
    for hit in hits:
        doc_id = hit.get("doc_id", "")
        chunk_idx = int(hit.get("chunk_idx", 0))
        snippet = hit.get("snippet", "").strip()
        if not snippet:
            continue
        chunk_key = (doc_id, chunk_idx)

        # 1) snippet_text 중복 체크
        if snippet in seen_by_snippet:
            existing = seen_by_snippet[snippet]
            # 점수가 더 높으면 교체
            if hit.get("score", 0.0) > existing.get("score", 0.0):
                old_key = (existing.get("doc_id", ""), int(existing.get("chunk_idx", 0)))
                if old_key in seen_by_chunk:
                    del seen_by_chunk[old_key]
                seen_by_snippet[snippet] = hit
                seen_by_chunk[chunk_key] = hit
            continue

        # 2) chunk_key 중복 체크
        if chunk_key in seen_by_chunk:
            existing = seen_by_chunk[chunk_key]
            if hit.get("score", 0.0) > existing.get("score", 0.0):
                old_snippet = existing.get("snippet", "").strip()
                if old_snippet in seen_by_snippet:
                    del seen_by_snippet[old_snippet]
                seen_by_chunk[chunk_key] = hit
                seen_by_snippet[snippet] = hit
            continue
        
        # 중복 아님 -> 등록
        seen_by_snippet[snippet] = hit
        seen_by_chunk[chunk_key] = hit
        
    # 점수 내림차순 정렬 반환
    return sorted(seen_by_snippet.values(), key=lambda x: x.get("score", 0.0), reverse=True)

