from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence

from repository.documents import fetch_metadata_by_vector_ids
from service.retrieval.reranker import rerank_snippets
from service.retrieval.common import hf_embed_text
from service.vector_db import (
    ensure_collection_and_index,
    get_milvus_client,
    run_dense_search,
    run_hybrid_search,
    DEFAULT_OUTPUT_FIELDS,
    build_dense_hits,
    build_rerank_payload,
)
from utils import logger
from utils.model_load import get_or_load_embedder_async


LOGGER = logger(__name__)


async def run_search_pipeline(
    *,
    query: str,
    collection: str,
    task_type: str,
    security_level: int,
    top_k: int,
    rerank_top_n: Optional[int],
    search_type: Optional[str],
    model_key: str,
    filters: Optional[Dict[str, Any]] = None,
    extra_options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Unified Milvus search pipeline (dense + hybrid + rerank)."""

    t0 = time.perf_counter()
    top_k = max(1, int(top_k))
    if rerank_top_n is None:
        rerank_limit = min(top_k, 5)
        rerank_enabled = True
    else:
        rerank_limit = max(0, int(rerank_top_n))
        rerank_enabled = rerank_limit > 0

    tok, model, device = await get_or_load_embedder_async(model_key)
    query_vector = hf_embed_text(tok, model, device, query)

    client = get_milvus_client()
    ensure_collection_and_index(
        client,
        emb_dim=len(query_vector),
        metric="IP",
        collection_name=collection,
    )
    if collection not in client.list_collections():
        return _empty_result(
            elapsed=time.perf_counter() - t0,
            model_key=model_key,
            search_type=search_type,
            collection=collection,
        )

    candidate_limit = _determine_candidate_limit(
        requested=top_k,
        extra_options=extra_options,
    )
    filter_expr = _build_filter_expression(task_type, security_level, filters)

    resolved_search_type = (search_type or "").lower().replace("semantic", "vector")
    if resolved_search_type not in {"vector", "hybrid", "bm25"}:
        resolved_search_type = "hybrid"

    if resolved_search_type == "vector":
        raw = run_dense_search(
            client,
            collection_name=collection,
            query_vector=query_vector.tolist(),
            limit=candidate_limit,
            filter_expr=filter_expr,
            output_fields=DEFAULT_OUTPUT_FIELDS,
        )
    else:
        raw = run_hybrid_search(
            client,
            collection_name=collection,
            query_vector=query_vector.tolist(),
            query_text=query,
            limit=candidate_limit,
            filter_expr=filter_expr,
            output_fields=DEFAULT_OUTPUT_FIELDS,
        )

    dense_hits = build_dense_hits(raw)
    if not dense_hits:
        return _empty_result(
            elapsed=time.perf_counter() - t0,
            model_key=model_key,
            search_type=resolved_search_type,
            collection=collection,
        )

    meta_map = _fetch_snippets(dense_hits)
    for hit in dense_hits:
        vector_id = str(hit.get("vector_id") or "")
        meta = meta_map.get(vector_id)
        if meta:
            hit["doc_id"] = hit.get("doc_id") or meta.get("doc_id")
            hit["chunk_idx"] = meta.get("chunk_index", hit.get("chunk_idx"))
            hit["snippet"] = meta.get("text", hit.get("snippet"))
            hit["path"] = meta.get("source_path", hit.get("path"))
            hit["page"] = meta.get("page", hit.get("page", 0))

    if rerank_enabled:
        hits_reranked = apply_reranking(dense_hits, query=query, top_n=rerank_limit)
    else:
        hits_reranked = dense_hits
    hits_final = deduplicate_hits(hits_reranked)
    hits_final = hits_final[:top_k]
    for hit in hits_final:
        if "score" not in hit:
            hit["score"] = float(
                hit.get("score_fused", hit.get("score_vec", hit.get("score_sparse", 0.0)))
            )

    elapsed = time.perf_counter() - t0
    return {
        "elapsed_sec": round(elapsed, 4),
        "settings_used": {
            "model": model_key,
            "searchType": resolved_search_type,
            "collection": collection,
        },
        "hits": hits_final,
    }


def apply_reranking(
    hits: Sequence[Dict[str, Any]],
    *,
    query: str,
    top_n: int,
) -> List[Dict[str, Any]]:
    """Apply reranker to hits (fallback to vector score sorting)."""

    if not hits:
        return []

    rerank_candidates = build_rerank_payload(hits)
    if not rerank_candidates:
        return sorted(
            hits,
            key=lambda x: x.get("score_fused", x.get("score_vec", 0.0)),
            reverse=True,
        )[:top_n]

    reranked = rerank_snippets(rerank_candidates, query=query, top_n=top_n)
    transformed: List[Dict[str, Any]] = []
    for item in reranked:
        meta = item.metadata or {}
        transformed.append(
            {
                "score": float(item.score),
                "doc_id": meta.get("doc_id"),
                "path": meta.get("path"),
                "vector_id": meta.get("vector_id"),
                "chunk_idx": int(meta.get("chunk_idx", 0)),
                "task_type": meta.get("task_type"),
                "security_level": int(meta.get("security_level", 1)),
                "page": int(meta.get("page", 0)),
                "snippet": item.text,
            }
        )
    return transformed


def deduplicate_hits(hits: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicates by snippet text and (doc_id, chunk_idx)."""

    seen_snippet: Dict[str, Dict[str, Any]] = {}
    seen_chunk: Dict[tuple[str, int], Dict[str, Any]] = {}

    for hit in hits:
        doc_id = str(hit.get("doc_id") or "")
        chunk_idx = int(hit.get("chunk_idx", 0))
        snippet = (hit.get("snippet") or "").strip()
        if not snippet:
            continue
        chunk_key = (doc_id, chunk_idx)

        existing_snippet = seen_snippet.get(snippet)
        if existing_snippet:
            if hit.get("score", 0.0) > existing_snippet.get("score", 0.0):
                _replace_chunk_ref(seen_chunk, existing_snippet, chunk_key, hit)
                seen_snippet[snippet] = hit
            continue

        existing_chunk = seen_chunk.get(chunk_key)
        if existing_chunk:
            if hit.get("score", 0.0) > existing_chunk.get("score", 0.0):
                prev_snippet = (existing_chunk.get("snippet") or "").strip()
                if prev_snippet in seen_snippet:
                    del seen_snippet[prev_snippet]
                seen_chunk[chunk_key] = hit
                seen_snippet[snippet] = hit
            continue

        seen_snippet[snippet] = hit
        seen_chunk[chunk_key] = hit

    return sorted(
        seen_snippet.values(),
        key=lambda x: x.get("score", x.get("score_fused", 0.0)),
        reverse=True,
    )


def _replace_chunk_ref(
    seen_chunk: Dict[tuple[str, int], Dict[str, Any]],
    previous_hit: Dict[str, Any],
    chunk_key: tuple[str, int],
    new_hit: Dict[str, Any],
) -> None:
    prev_key = (previous_hit.get("doc_id", ""), int(previous_hit.get("chunk_idx", 0)))
    if prev_key in seen_chunk:
        del seen_chunk[prev_key]
    seen_chunk[chunk_key] = new_hit


def _determine_candidate_limit(
    *,
    requested: int,
    extra_options: Optional[Dict[str, Any]],
) -> int:
    multiplier = 2
    if extra_options:
        multiplier = int(extra_options.get("candidate_multiplier") or multiplier)
    limit = requested * multiplier
    return max(limit, requested, 10)


def _build_filter_expression(
    task_type: str,
    security_level: int,
    filters: Optional[Dict[str, Any]],
) -> str:
    clauses = [f"task_type == '{task_type}'", f"security_level <= {int(security_level)}"]
    if not filters:
        return " && ".join(clauses)

    for key, value in filters.items():
        if value is None:
            continue
        if isinstance(value, str):
            clauses.append(f"{key} == '{value}'")
        elif isinstance(value, (int, float)):
            clauses.append(f"{key} == {value}")
        elif isinstance(value, Sequence):
            sanitized = ", ".join(
                f"'{v}'" if isinstance(v, str) else str(v) for v in value if v is not None
            )
            if sanitized:
                clauses.append(f"{key} in [{sanitized}]")
    return " && ".join(clauses)


def _fetch_snippets(hits: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    vector_ids = [str(hit.get("vector_id") or "") for hit in hits if hit.get("vector_id")]
    if not vector_ids:
        return {}
    return fetch_metadata_by_vector_ids(vector_ids)


def _empty_result(
    *,
    elapsed: float,
    model_key: str,
    search_type: Optional[str],
    collection: str,
) -> Dict[str, Any]:
    return {
        "elapsed_sec": round(elapsed, 4),
        "settings_used": {
            "model": model_key,
            "searchType": (search_type or "hybrid"),
            "collection": collection,
        },
        "hits": [],
    }


