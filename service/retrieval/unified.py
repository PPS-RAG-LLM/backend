"""통합 검색 엔진."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from service.retrieval.adapters.base import RetrievalResult
from service.retrieval.adapters.local import LocalVectorAdapter
from service.retrieval.adapters.milvus import MilvusAdapter
from service.retrieval.adapters.workspace import WorkspaceAdapter
from service.retrieval.common import get_document_dirs
from service.retrieval.reranker import rerank_snippets
from utils import logger

DEFAULT_SOURCES = ("workspace", "local", "milvus")
LOGGER = logger(__name__)


def unified_search(query: str, config: Dict[str, Any]) -> List[RetrievalResult]:
    """
    여러 검색 소스를 통합 호출.
    Args:
        query: 사용자 질문
        config: 검색 설정 (workspace_id, attachments, security_level 등)
    """

    top_k = int(config.get("top_k") or 13)
    threshold = float(config.get("threshold") or 0.0)
    sources = tuple(config.get("sources") or DEFAULT_SOURCES)
    enable_rerank = bool(config.get("enable_rerank", False))
    rerank_top_n = int(config.get("rerank_top_n"))
    attachments = config.get("attachments") or []

    LOGGER.info(
        "[UnifiedSearch] query='%s' sources=%s top_k=%s workspace_id=%s attachments=%s",
        query,
        sources,
        top_k,
        config.get("workspace_id"),
        len(attachments),
    )

    results: List[RetrievalResult] = []

    if "workspace" in sources and config.get("workspace_id"):
        adapter = WorkspaceAdapter()
        workspace_hits = adapter.search(
            query,
            top_k,
            workspace_id=int(config["workspace_id"]),
            threshold=threshold,
        )
        LOGGER.info("[UnifiedSearch] workspace hits=%s", len(workspace_hits))
        results.extend(workspace_hits)
    elif "workspace" in sources:
        LOGGER.info("[UnifiedSearch] workspace source enabled but workspace_id missing")

    if "local" in sources:
        attachment_doc_ids = extract_doc_ids_from_attachments(config.get("attachments"))
        if attachment_doc_ids:
            adapter = LocalVectorAdapter()
            local_hits = adapter.search(
                query,
                top_k,
                doc_ids=attachment_doc_ids,
                threshold=threshold,
            )
            LOGGER.info(
                "[UnifiedSearch] local hits=%s (doc_ids=%s)",
                len(local_hits),
                attachment_doc_ids,
            )
            results.extend(local_hits)
        else:
            LOGGER.info("[UnifiedSearch] local source enabled but no attachment doc_ids")

    if "milvus" in sources:
        sec_level = int(config.get("security_level") or 1)
        adapter = MilvusAdapter()
        milvus_hits = adapter.search(
            query,
            top_k,
            security_level=sec_level,
            task_type=str(config.get("task_type") or "qna"),
            search_type=config.get("search_type"),
            model_key=config.get("model_key"),
            rerank_top_n=rerank_top_n,
        )
        LOGGER.info("[UnifiedSearch] milvus hits=%s", len(milvus_hits))
        results.extend(milvus_hits)

    if not results:
        LOGGER.info("[UnifiedSearch] no hits from any sources")
        return []

    merged = _deduplicate(results)
    LOGGER.info("[UnifiedSearch] merged hits=%s", len(merged))
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
    doc_info_dir, _ = get_document_dirs()
    doc_ids: List[str] = []

    for att in attachments or []:
        if isinstance(att, dict):
            location = str(att.get("contentString") or "").strip()
        else:
            location = str(getattr(att, "contentString", "")).strip()
        if not location:
            continue
        basename = location.split("/")[-1].split("?")[0]
        if not basename.endswith(".json"):
            continue

        base = basename[:-5].strip()

        # 1) 파일명 끝에 UUID가 붙어 있으면 doc_id로 사용
        maybe_uuid = base.rsplit("-", 1)[-1].strip()
        if maybe_uuid and maybe_uuid.count("-") == 4:
            doc_ids.append(maybe_uuid)
            continue

        # 2) documents-info/<파일명>.json 에서 id를 읽어본다
        info_path = doc_info_dir / basename
        if info_path.exists():
            try:
                data = json.loads(info_path.read_text(encoding="utf-8"))
                doc_id = str(data.get("id") or "").strip()
                if doc_id:
                    doc_ids.append(doc_id)
            except Exception:
                continue

    seen = set()
    unique: List[str] = []
    for doc_id in doc_ids:
        if doc_id in seen:
            continue
        seen.add(doc_id)
        unique.append(doc_id)
    return unique

