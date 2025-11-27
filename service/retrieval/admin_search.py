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
from utils.model_load import _get_or_load_embedder_async
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
    collection_name: Optional[str] = None

# --- Helpers ---
def get_vector_settings() -> Dict:
    try:
        row = get_rag_settings_row()
    except Exception:
        logger.error("get_rag_settings_row 실패 qwen3_4b 모델을 기본값으로 리턴합니다.")
        return {
            "embeddingModel": "qwen3_4b",
            "searchType": "hybrid",
            "chunkSize": 512,
            "overlap": 64,
        }
    return {
        "embeddingModel": row.get("embedding_key"),
        "searchType": row.get("search_type", "hybrid"),
        "chunkSize": int(row.get("chunk_size", 512)),
        "overlap": int(row.get("overlap", 64)),
    }

# --- Search Logic ---

async def search_vector_candidates(
    req: RAGSearchRequest, 
    search_type_override: Optional[str] = None,
) -> Dict[str, Any]:
    """
    [Pure Search] 
    Milvus에서 벡터/하이브리드 검색만 수행하고 메타데이터를 매핑하여 반환합니다.
    (Reranking 및 Deduplication을 수행하지 않음 -> 통합 검색 등에서 활용)
    """
    t0 = time.perf_counter()
    
    if req.task_type not in TASK_TYPES:
        return {"hits": [], "error": f"invalid task_type: {req.task_type}"}

    settings = get_vector_settings()
    model_key = req.model or settings["embeddingModel"]
    raw_st = (search_type_override or settings.get("searchType") or "").lower()
    search_type = (raw_st.replace("semantic", "vector").replace("sementic", "vector") or "hybrid")

    # Embedder 로드 (async)
    tok, model, device = await _get_or_load_embedder_async(model_key)
    
    # Query Embedding (Sync/CPU bound - may block loop briefly)
    q_emb = hf_embed_text(tok, model, device, req.query)
    
    client = get_milvus_client()
    ensure_collection_and_index(client, emb_dim=len(q_emb), metric="IP", collection_name=req.collection_name)

    if req.collection_name not in client.list_collections():
        return {"hits": [], "settings_used": {"model": model_key, "searchType": search_type}, "elapsed_sec": 0.0}

    # 후보군 검색 (Rerank 전이므로 top_k보다 넉넉하게 가져옴)
    candidate_limit = int(req.top_k) * 2 
    if candidate_limit < 10: candidate_limit = 10

    filter_expr = f"task_type == '{req.task_type}' && security_level <= {int(req.user_level)}"

    if search_type == "vector":
        raw_results = run_dense_search(
            client,
            collection_name=req.collection_name,
            query_vector=q_emb.tolist(),
            limit=candidate_limit,
            filter_expr=filter_expr,
            output_fields=DEFAULT_OUTPUT_FIELDS,
        )
    else:
        raw_results = run_hybrid_search(
            client,
            collection_name=req.collection_name,
            query_vector=q_emb.tolist(),
            query_text=req.query,
            limit=candidate_limit,
            filter_expr=filter_expr,
            output_fields=DEFAULT_OUTPUT_FIELDS,
        )
    hits_raw = build_dense_hits(raw_results)
    
    # 메타데이터(텍스트 등) Fetch
    vector_ids = [str(h["vector_id"]) for h in hits_raw if h.get("vector_id")]
    meta_map = fetch_metadata_by_vector_ids(vector_ids)
    
    valid_hits = []
    for hit in hits_raw:
        vid = str(hit.get("vector_id") or "")
        meta = meta_map.get(vid)
        if meta:
            hit["doc_id"] = hit.get("doc_id") or meta.get("doc_id")
            hit["chunk_idx"] = meta.get("chunk_index")
            hit["snippet"] = meta.get("text")
            hit["path"] = meta.get("source_path", hit.get("path"))
            valid_hits.append(hit)

    elapsed = round(time.perf_counter() - t0, 4)
    return {
        "hits": valid_hits,
        "settings_used": {"model": model_key, "searchType": search_type},
        "elapsed_sec": elapsed
    }


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


async def search_documents(req: RAGSearchRequest, 
                            search_type_override: Optional[str] = None,
                            rerank_top_n: Optional[int] = None,
                            collection_name: Optional[str] = None) -> Dict:
    """
    [Legacy/Direct Search]
    검색 -> 리랭킹 -> 중복제거 과정을 모두 수행하여 최종 결과를 반환합니다.
    """
    # 1. 순수 검색 (Candidates)
    search_res = await search_vector_candidates(req, search_type_override)
    hits_raw = search_res.get("hits", [])
    
    # 2. 리랭킹 (Rerank)
    final_results = int(rerank_top_n) if rerank_top_n is not None else 5
    hits_reranked = apply_reranking(hits_raw, req.query, top_n=final_results)
    
    # 3. 중복 제거 (Dedup)
    hits_sorted = deduplicate_hits(hits_reranked)
    
    # 결과 포맷팅 (Prompt 생성 등)
    context = "\n---\n".join(h["snippet"] for h in hits_sorted if h.get("snippet"))
    prompt = f"사용자 질의: {req.query}\n:\n{context}\n\n위 내용을 바탕으로 응답을 생성해 주세요."

    return {
        "elapsed_sec": search_res["elapsed_sec"],
        "settings_used": search_res["settings_used"],
        "hits": [
            {
                "score": float(h["score"]),
                "path": h.get("path"),
                "chunk_idx": int(h["chunk_idx"]),
                "task_type": h["task_type"],
                "security_level": int(h["security_level"]),
                "doc_id": h.get("doc_id"),
                "page": int(h.get("page", 0)),
                "snippet": h["snippet"],
            }
            for h in hits_sorted
        ],
        "prompt": prompt,
    }


async def execute_search(
    question: str,
    top_k: int = 20,
    rerank_top_n: int = 5,
    security_level: int = 1,
    source_filter: Optional[List[str]] = None,
    task_type: str = "qna",
    model_key: Optional[str] = None,
    search_type: Optional[str] = None,
    collection_name: Optional[str] = None,
) -> Dict:
    print(f"⭐ [ExecuteSearch] 함수 호출: question='{question}', topK={top_k}, rerank_topN={rerank_top_n}")
    req = RAGSearchRequest(
        query=question,
        top_k=top_k,
        user_level=security_level,
        task_type=task_type,
        model=model_key,
    )
    # search_documents 내부에서 search_vector_candidates -> apply_reranking -> deduplicate_hits 순차 실행
    res = await search_documents(req, search_type_override=search_type, rerank_top_n=rerank_top_n, collection_name=collection_name)
    
    check_files: List[str] = []
    try:
        for h in res.get("hits", []):
            doc_id_val = h.get("doc_id")
            if doc_id_val:
                check_files.append(f"{str(doc_id_val)}.pdf")
                continue
            p = Path(h.get("path", ""))
            if str(p):
                check_files.append(p.with_suffix(".pdf").name)
    except Exception:
        pass

    if source_filter and "hits" in res:
        names = {Path(n).stem for n in source_filter}
        res["hits"] = [h for h in res["hits"] if Path(h["path"]).stem in names]

    res["check_file"] = sorted(list(set(check_files)))
    return res

