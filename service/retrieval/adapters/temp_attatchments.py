"""Milvus 기반 임시 첨부 검색 어댑터."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from service.retrieval.pipeline import DEFAULT_OUTPUT_FIELDS
from utils import logger

from repository.documents import get_documents_by_ids
from service.retrieval.adapters.base import BaseRetrievalAdapter, RetrievalResult
from service.retrieval.common import embed_text, hf_embed_text
from service.vector_db.milvus_store import (
    get_milvus_client,
    resolve_collection,
    run_dense_search,
    run_hybrid_search,
)
from storage.db_models import DocumentType
from utils.model_load import _get_or_load_embedder

logger = logger(__name__)


class TempAttachmentsVectorAdapter(BaseRetrievalAdapter):
    """doc_id 목록을 입력받아 Milvus에서 검색."""

    def __init__(self, source: str = "milvus") -> None:
        super().__init__(source=source)
        self.collection_name = resolve_collection(DocumentType.TEMP.value)

    def search(
        self,
        query: str,
        top_k: int,
        *,
        doc_ids: List[str],
        threshold: float = 0.0,
        mode: str = "hybrid",
        workspace_id: Optional[int] = None,
        model_key: Optional[str] = None,
    ) -> List[RetrievalResult]:
        if not doc_ids:
            return []
        if model_key:
            try:
                tok, model, device = _get_or_load_embedder(model_key)
                query_vec = hf_embed_text(tok, model, device, query).tolist()
            except Exception as e:
                logger.warning(f"[TempAttachments] Failed to load model {model_key}, fallback to default: {e}")
                query_vec = embed_text(query).tolist()
        else:
            query_vec = embed_text(query).tolist()

        filter_expr = self._build_filter_expr(doc_ids=doc_ids, workspace_id=workspace_id)
        client = get_milvus_client()
        if workspace_id:
            output_fields = list(DEFAULT_OUTPUT_FIELDS) + ["workspace_id"]
        else:
            output_fields = list(DEFAULT_OUTPUT_FIELDS)

        if mode == "hybrid":
            raw_hits = run_hybrid_search(
                client,
                collection_name=self.collection_name,
                query_vector=query_vec,
                query_text=query,
                limit=top_k,
                filter_expr=filter_expr,
                output_fields=output_fields,
            )
        else:
            raw_hits = run_dense_search(
                client,
                collection_name=self.collection_name,
                query_vector=query_vec,
                limit=top_k,
                filter_expr=filter_expr,
                output_fields=output_fields,
            )

        return self._normalize_hits(raw_hits, threshold=threshold)[: max(1, top_k)]

    def _build_filter_expr(
        self, *, doc_ids: List[str], workspace_id: Optional[int]
    ) -> str:
        clauses = []
        if doc_ids:
            quoted = ", ".join(f'"{doc_id}"' for doc_id in doc_ids)
            clauses.append(f"doc_id in [{quoted}]")
        if workspace_id is not None:
            clauses.append(f"workspace_id == {int(workspace_id)}")
        return " and ".join(clauses) if clauses else ""

    def _normalize_hits(
        self, raw_hits: Iterable, *, threshold: float
    ) -> List[RetrievalResult]:
        flattened = list(self._iter_hits(raw_hits))
        doc_ids = {self._extract_field(hit, "doc_id") for hit in flattened}
        doc_map = get_documents_by_ids([d for d in doc_ids if d])

        results: List[RetrievalResult] = []
        for hit in flattened:
            score = self._extract_score(hit)
            if score is None or score < threshold:
                continue
            doc_id = str(self._extract_field(hit, "doc_id") or "")
            if not doc_id:
                continue
            chunk_idx = self._extract_field(hit, "chunk_idx")
            page = self._extract_field(hit, "page")
            text = str(self._extract_field(hit, "text") or "").strip()
            if not text:
                continue
            doc_meta = doc_map.get(doc_id, {})
            title = doc_meta.get("filename") or doc_id
            metadata = {
                "workspace_id": doc_meta.get("workspace_id"),
                "doc_type": doc_meta.get("doc_type"),
            }
            results.append(
                self._build_result(
                    doc_id=doc_id,
                    title=title,
                    text=text,
                    score=float(score),
                    chunk_index=int(chunk_idx or 0),
                    page=int(page) if page is not None else None,
                    metadata=metadata,
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)
        return results

    @staticmethod
    def _iter_hits(raw_hits: Iterable) -> Iterable:
        for batch in raw_hits or []:
            if batch is None:
                continue
            for hit in batch:
                yield hit

    @staticmethod
    def _extract_field(hit, field: str):
        entity = getattr(hit, "entity", None)
        if entity is not None:
            getter = getattr(entity, "get", None)
            if callable(getter):
                value = getter(field)
                if value is not None:
                    return value
            if hasattr(entity, field):
                return getattr(entity, field)
        if isinstance(hit, dict):
            if field in hit:
                return hit[field]
            entity = hit.get("entity")
            if isinstance(entity, dict) and field in entity:
                return entity[field]
        return getattr(hit, field, None)

    @staticmethod
    def _extract_score(hit) -> Optional[float]:
        if hasattr(hit, "score"):
            return float(getattr(hit, "score"))
        if hasattr(hit, "distance"):
            return float(getattr(hit, "distance"))
        if isinstance(hit, dict):
            if "score" in hit:
                return float(hit["score"])
            if "distance" in hit:
                return float(hit["distance"])
        return None

