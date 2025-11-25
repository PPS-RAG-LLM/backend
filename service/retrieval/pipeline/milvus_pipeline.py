"""Milvus 검색 파이프라인 공용 유틸리티."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
from utils import logger

logger = logger(__name__)

TABLE_MARK = "[[TABLE"
SnippetLoader = Callable[[str, int], str]
DEFAULT_OUTPUT_FIELDS: Tuple[str, ...] = (
    "pk",  # = vector_id
    "path",
    "chunk_idx",
    "task_type",
    "security_level", 
    "doc_id",
    "text",
    "page",
)

def _iter_hits(raw_results: Sequence[Any]) -> Iterable[Tuple[Dict[str, Any], float, Optional[str]]]:
    """Milvus search 결과를 통합된 형태로 순회."""
    if not raw_results:
        return
    hits = raw_results[0]
    for hit in hits:
        if isinstance(hit, dict):
            entity = hit.get("entity", {}) or {}
            score = float(hit.get("distance", hit.get("score", 0.0)) or 0.0)
        else:
            entity = getattr(hit, "entity", {}) or {}
            score = float(getattr(hit, "score", 0.0) or 0.0)
        ent_text = entity.get("text") if isinstance(entity, dict) else None
        yield entity, score, ent_text


def build_dense_hits(
    raw_results: Sequence[Any],
    *,
    table_mark: str = TABLE_MARK,
) -> List[Dict[str, Any]]:
    """덴스 검색 결과 리스트를 표준 dict 형태로 변환."""
    hits: List[Dict[str, Any]] = []
    for entity, score, ent_text in _iter_hits(raw_results) or []:
        doc_id = entity.get("doc_id")
        if not doc_id:
            continue
        vector_id = entity.get("pk") or entity.get("vector_id")
        chunk_idx = int(entity.get("chunk_idx", 0) or 0)
        task_type = entity.get("task_type")
        security_level = int(entity.get("security_level", 1) or 1)
        doc_id = entity.get("doc_id")
        page = int(entity.get("page", 0) or 0)
        snippet_source = "entity"
        snippet = str(ent_text or "").strip()
        if snippet and snippet.startswith(table_mark):
            snippet_source = "entity"
        else:
            snippet_source = "entity" if snippet else "empty"
        logger.debug(
            "[Snippet] source=%s doc_id=%s  vector_id=%s chunk=%s len=%s",
            snippet_source, doc_id, vector_id, chunk_idx, len(snippet or ""),
        )
        hits.append(
            {
                "vector_id": vector_id,
                "doc_id": doc_id,
                "chunk_idx": chunk_idx,
                "task_type": task_type,
                "security_level": security_level,
                "page": page,
                "score_vec": float(score),
                "score_sparse": 0.0,
                "score_fused": float(score),
                "snippet": snippet,
            }
        )
    return hits

def build_rerank_payload(
    hits: Sequence[Dict[str, Any]],
    *,
    source: str = "milvus",
) -> List[Dict[str, Any]]:
    """리랭커 입력 페이로드 구성."""
    payload: List[Dict[str, Any]] = []
    for hit in hits:
        snippet = str(hit.get("snippet") or "").strip()
        if not snippet:
            continue
        fused = hit.get("score_fused")
        vec = hit.get("score_vec")
        spa = hit.get("score_sparse")
        score = float(fused if fused is not None else vec if vec is not None else spa or 0.0)
        payload.append(
            {
                "text": snippet,
                "score": score,
                "doc_id": hit.get("doc_id"),
                "title": hit.get("doc_id") or hit.get("path") or "snippet",
                "source": source,
                "metadata": hit,
            }
        )
    return payload


def load_snippet_from_store(
    base_dir: Path,
    path: str,
    chunk_idx: int,
    *,
    max_tokens: int,
    overlap: int,
) -> str:
    """추출된 텍스트 저장소에서 스니펫을 읽어오는 공용 함수."""
    file_path = base_dir / path
    try:
        full_text = file_path.read_text(encoding="utf-8")
    except Exception as exc:  # pragma: no cover - 파일 누락 대비
        logger.warning("[MilvusSnippet] load failed (%s): %s", file_path, exc)
        full_text = ""
    if not full_text:
        return ""

    words = full_text.split()
    if not words:
        return ""

    window = max_tokens - overlap
    if window <= 0:
        window = max_tokens
    start = max(0, chunk_idx * window)
    snippet = " ".join(words[start : start + max_tokens]).strip()
    if snippet:
        return snippet
    return " ".join(words[:max_tokens]).strip()

