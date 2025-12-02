"""통합 검색 엔진."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from service.retrieval.adapters.base import RetrievalResult
from service.retrieval.adapters.temp_attatchments import TempAttachmentsVectorAdapter
from service.retrieval.adapters.admin_docs import AdminDocsAdapter
from service.retrieval.adapters.workspace_docs import WorkspaceDocsAdapter
from service.retrieval.reranker import rerank_snippets
from utils import logger


logger = logger(__name__)

def unified_search(query: str, config: Dict[str, Any]) -> List[RetrievalResult]:
    """
    여러 검색 소스를 통합 호출.
    Args:
        query: 사용자 질문
        config: 검색 설정 (workspace_id, attachments, security_level 등)
    """
    logger.info(f"[UnifiedSearch] config: {config}")

    top_k = int(config.get("top_k"))
    threshold = float(config.get("threshold") or 0.0)
    sources = tuple(config.get("sources"))
    enable_rerank = bool(config.get("enable_rerank", False))
    rerank_top_n = int(config.get("rerank_top_n"))
    attachments = config.get("attachments") or []
    

    logger.info(
        "[UnifiedSearch] query='%s' sources=%s top_k=%s workspace_id=%s attachments=%s",
        query,
        sources,
        top_k,
        config.get("workspace_id"),
        len(attachments),
    )

    results: List[RetrievalResult] = []

    with ThreadPoolExecutor() as executor: # 비동기 실행을 위해 ThreadPoolExecutor 사용
        future_to_source = {}

        # 1. Workspace Docs
        if "WS_DOCS" in sources and config.get("workspace_id"):
            logger.info(f"[UnifiedSearch] WS_DOCS source enabled and workspace_id={config.get('workspace_id')}")
            adapter = WorkspaceDocsAdapter()
            future = executor.submit(
                adapter.search,
                query,
                top_k,
                workspace_id=int(config["workspace_id"]),
                threshold=threshold,
                mode=str(config.get("search_type") or "hybrid"),
                model_key=config.get("model_key"),
            )
            future_to_source[future] = "WS_DOCS"
        elif "workspace" in sources:
            logger.info("WS_DOCS source enabled but workspace_id missing")

        # 2. Temp Attachments
        if "TEMP_ATTACH" in sources:
            logger.info(f"TEMP_ATTACH source enabled and attachments={config.get('attachments')}")
            attachment_doc_ids = extract_doc_ids_from_attachments(config.get("attachments"))
            if attachment_doc_ids:
                adapter = TempAttachmentsVectorAdapter()
                future = executor.submit(
                    adapter.search,
                    query,
                    top_k,
                    doc_ids=attachment_doc_ids,
                    threshold=threshold,
                    mode=str(config.get("search_type") or "hybrid"),
                    model_key=config.get("model_key"),
                )
                future_to_source[future] = "TEMP_ATTACH"
            else:
                logger.info(f"TEMP_ATTACH source enabled but no attachment doc_ids")

        # 3. Admin Docs
        if "ADMIN_DOCS" in sources:
            logger.info(f"ADMIN_DOCS source enabled and security_level={config.get('security_level')}")
            sec_level = int(config.get("security_level") or 1)
            adapter = AdminDocsAdapter()
            future = executor.submit(
                adapter.search,
                query,
                top_k,
                security_level=sec_level,
                task_type=str(config.get("task_type") or "qna"),
                search_type=config.get("search_type"),
                model_key=config.get("model_key"),
                rerank_top_n=rerank_top_n,
            )
            future_to_source[future] = "ADMIN_DOCS"

        # # 4. LLM Test (Admin Docs reuse)
        # if "LLM_TEST" in sources:
        #     logger.info(f"LLM_TEST source enabled and security_level={config.get('security_level')}")
        #     sec_level = int(config.get("security_level") or 1)
        #     adapter = AdminDocsAdapter()
        #     future = executor.submit(
        #         adapter.search,
        #         query,
        #         top_k,
        #         security_level=sec_level,
        #         task_type=str(config.get("task_type") or "qna"),
        #         search_type=config.get("search_type"),
        #         model_key=config.get("model_key"),
        #         rerank_top_n=rerank_top_n,
        #     )
        #     future_to_source[future] = "LLM_TEST"

        # Collect results
        for future in as_completed(future_to_source):
            source_name = future_to_source[future]
            try:
                hits = future.result()
                logger.info(f"[UnifiedSearch] {source_name} hits={len(hits)}")
                results.extend(hits)
            except Exception as exc:
                logger.error(f"[UnifiedSearch] {source_name} search failed: {exc}")
                
    if not results:
        logger.info("[UnifiedSearch] no hits from any sources")
        return []

    merged = _deduplicate(results)
    logger.info("[UnifiedSearch] merged hits=%s", len(merged))

    # Global Reranking (Single Pass)
    if enable_rerank and len(merged) > 1:
        reranked = rerank_snippets(merged, query=query, top_n=rerank_top_n)
        return reranked

    merged.sort(key=lambda r: r.score, reverse=True)
    return merged[: max(1, rerank_top_n if enable_rerank else top_k)]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _deduplicate(results: Iterable[RetrievalResult]) -> List[RetrievalResult]:
    """
    doc_id + chunk_index 기준으로 dedup.
    점수가 더 높은 항목을 유지한다.
    """
    dedup: Dict[Tuple[str, int], RetrievalResult] = {}
    for item in results:
        key = (str(item.doc_id or item.title), int(item.chunk_index or 0))
        if key not in dedup or item.score > dedup[key].score:
            dedup[key] = item
    return list(dedup.values())


def extract_doc_ids_from_attachments(attachments: Any) -> List[str]:
    """
    첨부 파일 메타 정보에서 doc_id 추출.
    기존 service.users.chat.retrieval.retrieval.extract_doc_ids_from_attachments 와 동일한 로직.
    """
    doc_ids: List[str] = []

    for att in attachments or []:
        if isinstance(att, dict):
            location = str(att.get("docId") or "").strip()
        else:
            location = str(getattr(att, "docId", "")).strip()
        if not location:
            continue
        doc_ids.append(location)
        continue

    seen = set()
    unique: List[str] = []
    for doc_id in doc_ids:
        if doc_id in seen:
            continue
        seen.add(doc_id)
        unique.append(doc_id)

    return unique
