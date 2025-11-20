"""Milvus 검색 파이프라인 공용 유틸리티."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
from pymilvus import AnnSearchRequest, RRFRanker
from utils import logger

logger = logger(__name__)

TABLE_MARK = "[[TABLE"
SnippetLoader = Callable[[str, int], str]

DEFAULT_OUTPUT_FIELDS: Tuple[str, ...] = (
    "path",
    "chunk_idx",
    "task_type",
    "security_level",
    "doc_id",
    "text",
    "page",
)


@dataclass(frozen=True)
class _RRFEntry:
    key: Tuple[str, int, str, int, Optional[str]]
    score: float
    ent_text: Optional[str]
    page: int


def run_dense_search(
    client: Any,
    *,
    collection_name: str,
    query_vector: Sequence[float],
    limit: int,
    filter_expr: str,
    output_fields: Sequence[str] = DEFAULT_OUTPUT_FIELDS,
) -> List[Any]:
    """Milvus 덴스 검색 실행."""
    logger.info(f"유사도 검색 실행")
    logger.debug(f"유사도 검색: \n\n<{collection_name}>\n\n<{limit}>\n\n<{filter_expr}>\n\n<{output_fields}>")

    return client.search(
        collection_name=collection_name,
        data=[list(query_vector)],
        anns_field="embedding",
        limit=int(limit),
        search_params={"metric_type": "IP", "params": {}},
        output_fields=list(output_fields),
        filter=filter_expr,
    )


def run_hybrid_search(
    client: Any,
    *,
    collection_name: str,
    query_vector: Sequence[float],
    query_text: str,
    limit: int,
    filter_expr: str,
    output_fields: Sequence[str] = DEFAULT_OUTPUT_FIELDS,
) -> List[Any]:
    """Milvus 스파스 검색 실행(BM25 유사)."""
    try:
        logger.info(f"하이브리드 검색 실행")
        logger.debug(f"하이브리드 검색: {collection_name}, {limit}, {filter_expr}, {output_fields}")

        dense_req = AnnSearchRequest(
            data=[list(query_vector)],
            anns_field="embedding",                  # 컬렉션에 저장한 dense 필드명
            param={"metric_type": "IP", "params": {}},
            limit=int(limit),
        )
        sparse_req = AnnSearchRequest(
            data=[query_text],
            anns_field="text_sparse",                # BM25 Function 결과 필드
            param={"metric_type": "BM25", "params": {}},
            limit=int(limit),
        )
        return client.hybrid_search(
            collection_name=collection_name,
            reqs=[dense_req, sparse_req],
            ranker=RRFRanker(k=60),                  # 필요하면 WeightedRanker 로 변경 가능
            limit=int(limit),
            filter=filter_expr,
            output_fields=list(output_fields),
        )

    except Exception as exc:  # pragma: no cover - pymilvus 버전 차이 대응
        logger.warning("[MilvusSparse] search unavailable: %s", exc)
        return [[]]


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


def _resolve_snippet(
    ent_text: Optional[str],
    snippet_loader: SnippetLoader,
    *,
    path: str,
    chunk_idx: int,
    table_mark: str = TABLE_MARK,
) -> str:
    if isinstance(ent_text, str) and ent_text.startswith(table_mark):
        return ent_text
    return snippet_loader(path, chunk_idx)


def build_dense_hits(
    raw_results: Sequence[Any],
    *,
    snippet_loader: SnippetLoader,
    table_mark: str = TABLE_MARK,
) -> List[Dict[str, Any]]:
    """덴스 검색 결과 리스트를 표준 dict 형태로 변환."""
    hits: List[Dict[str, Any]] = []
    for entity, score, ent_text in _iter_hits(raw_results) or []:
        path = entity.get("path")
        if not path:
            continue
        chunk_idx = int(entity.get("chunk_idx", 0) or 0)
        task_type = entity.get("task_type")
        security_level = int(entity.get("security_level", 1) or 1)
        doc_id = entity.get("doc_id")
        page = int(entity.get("page", 0) or 0)
        snippet_source = "entity"
        snippet = str(ent_text or "").strip()
        if snippet and snippet.startswith(table_mark):
            # table markdown 그대로 사용
            pass
        else:
            if not snippet:
                snippet_source = "store"
                snippet = _resolve_snippet(
                    ent_text,
                    snippet_loader,
                    path=str(path),
                    chunk_idx=chunk_idx,
                    table_mark=table_mark,
                )
            else:
                snippet_source = "entity"
        logger.debug(
            "[Snippet] source=%s path=%s chunk=%s len=%s",
            snippet_source,
            path,
            chunk_idx,
            len(snippet or ""),
        )
        hits.append(
            {
                "path": path,
                "chunk_idx": chunk_idx,
                "task_type": task_type,
                "security_level": security_level,
                "doc_id": doc_id,
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



# def _collect_for_rrf(raw_results: Sequence[Any]) -> List[_RRFEntry]:
#     entries: List[_RRFEntry] = []
#     for entity, score, ent_text in _iter_hits(raw_results) or []:
#         path = entity.get("path")
#         if not path:
#             continue
#         chunk_idx = int(entity.get("chunk_idx", 0) or 0)
#         task_type = entity.get("task_type")
#         security_level = int(entity.get("security_level", 1) or 1)
#         doc_id = entity.get("doc_id")
#         page = int(entity.get("page", 0) or 0)
#         key = (str(path), chunk_idx, task_type, security_level, doc_id)
#         entries.append(_RRFEntry(key=key, score=float(score), ent_text=ent_text, page=page))
#     return entries


# def build_rrf_hits(
#     dense_results: Sequence[Any],
#     sparse_results: Sequence[Any],
#     *,
#     snippet_loader: SnippetLoader,
#     limit: int,
#     table_mark: str = TABLE_MARK,
#     rrf_k: float = 60.0,
# ) -> List[Dict[str, Any]]:
#     """덴스/스파스 결과를 RRF 방식으로 결합."""
#     dense_entries = _collect_for_rrf(dense_results)
#     sparse_entries = _collect_for_rrf(sparse_results)

#     rrf_scores: Dict[Tuple[str, int, str, int, Optional[str]], float] = {}
#     text_map: Dict[Tuple[str, int, str, int, Optional[str]], Optional[str]] = {}
#     page_map: Dict[Tuple[str, int, str, int, Optional[str]], int] = {}
#     score_map: Dict[Tuple[str, int, str, int, Optional[str]], Tuple[float, float]] = {}

#     for rank, entry in enumerate(dense_entries, start=1):
#         key = entry.key
#         rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
#         if entry.ent_text is not None:
#             text_map[key] = entry.ent_text
#         if entry.page and key not in page_map:
#             page_map[key] = entry.page
#         prev_dense, prev_sparse = score_map.get(key, (0.0, 0.0))
#         score_map[key] = (max(prev_dense, entry.score), prev_sparse)

#     for rank, entry in enumerate(sparse_entries, start=1):
#         key = entry.key
#         rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
#         if entry.ent_text is not None:
#             text_map[key] = entry.ent_text
#         if entry.page and key not in page_map:
#             page_map[key] = entry.page
#         prev_dense, prev_sparse = score_map.get(key, (0.0, 0.0))
#         score_map[key] = (prev_dense, max(prev_sparse, entry.score))

#     merged = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)[: int(limit)]
#     hits: List[Dict[str, Any]] = []
#     for key, fused in merged:
#         path, chunk_idx, task_type, security_level, doc_id = key
#         snippet = _resolve_snippet(
#             text_map.get(key),
#             snippet_loader,
#             path=path,
#             chunk_idx=chunk_idx,
#             table_mark=table_mark,
#         )
#         dense_score, sparse_score = score_map.get(key, (0.0, 0.0))
#         hits.append(
#             {
#                 "path": path,
#                 "chunk_idx": chunk_idx,
#                 "task_type": task_type,
#                 "security_level": security_level,
#                 "doc_id": doc_id,
#                 "page": page_map.get(key, 0),
#                 "score_vec": dense_score,
#                 "score_sparse": sparse_score,
#                 "score_fused": float(fused),
#                 "snippet": snippet,
#             }
#         )
#     return hits

