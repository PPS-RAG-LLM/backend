from __future__ import annotations

import logging
import os
from typing import Any, List, Optional, Sequence

from pymilvus import AnnSearchRequest, DataType, MilvusClient, RRFRanker

from config import config as app_config

try:
    from pymilvus import Function, FunctionType
except Exception:  # pragma: no cover - pymilvus 구버전 대응
    Function = None

    class FunctionType:  # type: ignore
        BM25 = "BM25"

logger = logging.getLogger(__name__)

_RETRIEVAL_CFG = app_config.get("retrieval", {}) or {}
_MILVUS_CFG = _RETRIEVAL_CFG.get("milvus", {}) or {}

MILVUS_URI = _MILVUS_CFG.get("uri")
MILVUS_TOKEN = _MILVUS_CFG.get("token") or None
_COLLECTIONS = _MILVUS_CFG.get("collections", {})

def resolve_collection(doc_type: str) -> str:
    try:
        return _COLLECTIONS[doc_type]
    except KeyError:
        raise ValueError(f"Unknown doc_type={doc_type}")

def get_milvus_client() -> MilvusClient:
    """MilvusClient 인스턴스를 생성한다."""
    kwargs = {"uri": MILVUS_URI, "alias": "default"}
    if MILVUS_TOKEN:
        kwargs["token"] = MILVUS_TOKEN
    return MilvusClient(**kwargs)


def ensure_collection_and_index(
    client: MilvusClient,
    *,
    emb_dim: int,
    metric: str = "IP",
    collection_name: str,
    description: Optional[str] = None,
) -> None:
    """컬렉션/인덱스를 보장하고 로드한다."""
    description = description or f"PDF chunks ({collection_name})"
    logger.info(f"[Milvus] 컬렉션 및 인덱스 준비 시작: {collection_name}")
    _create_collection_if_needed(client, collection_name, emb_dim, description)
    _ensure_dense_index(client, collection_name, metric)
    _ensure_sparse_index(client, collection_name)
    reload_collection(client, collection_name)


def _create_collection_if_needed(
    client: MilvusClient,
    collection_name: str,
    emb_dim: int,
    description: str,
) -> None:
    cols = client.list_collections()
    if collection_name in cols:
        return

    logger.info(f"[Milvus] 컬렉션 생성: {collection_name}")
    schema = client.create_schema(
        auto_id=True,
        enable_dynamic_field=False,
        description=description,
    )
    schema.add_field("pk", DataType.INT64, is_primary=True)
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=int(emb_dim))
    schema.add_field("path", DataType.VARCHAR, max_length=500)
    schema.add_field("chunk_idx", DataType.INT64)
    schema.add_field("task_type", DataType.VARCHAR, max_length=16)
    schema.add_field("security_level", DataType.INT64)
    schema.add_field("doc_id", DataType.VARCHAR, max_length=255)
    schema.add_field("workspace_id", DataType.INT64)
    schema.add_field("version", DataType.INT64)
    schema.add_field("page", DataType.INT64)
    try:
        schema.add_field("text", DataType.VARCHAR, max_length=32768, enable_analyzer=True)
    except TypeError:
        schema.add_field("text", DataType.VARCHAR, max_length=32768)
    try:
        schema.add_field("text_sparse", DataType.SPARSE_FLOAT_VECTOR)
    except Exception:
        logger.warning("[Milvus] SPARSE_FLOAT_VECTOR 미지원 클라이언트입니다. 서버 BM25 하이브리드 사용 불가.")

    if Function is not None:
        try:
            fn = Function(
                name="bm25_text2sparse",
                function_type=FunctionType.BM25,
                input_field_names=["text"],
                output_field_names=["text_sparse"],
            )
            schema.add_function(fn)
            logger.info("[Milvus] BM25 Function 연결 완료 (text -> text_sparse)")
        except Exception as exc:
            logger.warning(f"[Milvus] BM25 Function 추가 실패: {exc}")

    client.create_collection(collection_name=collection_name, schema=schema)
    logger.info(f"[Milvus] 컬렉션 생성 완료: {collection_name}")


def _ensure_dense_index(client: MilvusClient, collection_name: str, metric: str) -> None:
    try:
        idx_dense = client.list_indexes(collection_name=collection_name, field_name="embedding")
    except Exception:
        idx_dense = []
    if idx_dense:
        return
    logger.info(f"[Milvus] (embedding) 인덱스 생성 시작 @ {collection_name}")
    params = client.prepare_index_params()
    params.add_index("embedding", "FLAT", metric_type=metric, params={})
    client.create_index(collection_name, params, timeout=180.0, sync=True)
    logger.info(f"[Milvus] (embedding) 인덱스 생성 완료 @ {collection_name}")


def _ensure_sparse_index(client: MilvusClient, collection_name: str) -> None:
    try:
        idx_sparse = client.list_indexes(collection_name=collection_name, field_name="text_sparse")
    except Exception:
        idx_sparse = []
    if idx_sparse:
        return
    logger.info(f"[Milvus] (text_sparse) 인덱스 생성 시작 @ {collection_name}")
    params = client.prepare_index_params()
    try:
        params.add_index("text_sparse", "SPARSE_INVERTED_INDEX", params={})
    except TypeError:
        params.add_index("text_sparse", "SPARSE_INVERTED_INDEX", metric_type="BM25", params={})
    client.create_index(collection_name, params, timeout=180.0, sync=True)
    logger.info(f"[Milvus] (text_sparse) 인덱스 생성 완료 @ {collection_name}")


def reload_collection(client: MilvusClient, collection_name: str) -> None:
    """지정 컬렉션을 재로드한다."""
    try:
        client.release_collection(collection_name=collection_name)
    except Exception:
        pass
    client.load_collection(collection_name=collection_name)
    logger.info(f"[Milvus] 컬렉션 로드 완료: {collection_name}")


def milvus_has_data(client: MilvusClient, collection_name: str) -> bool:
    """컬렉션에 데이터가 존재하는지 확인한다."""
    if collection_name not in client.list_collections():
        return False
    try:
        rows = client.query(
            collection_name=collection_name,
            output_fields=["pk"],
            limit=1,
        )
        return len(rows) > 0
    except Exception:
        return True


def drop_all_collections(client: MilvusClient) -> List[str]:
    """모든 컬렉션을 삭제하고, 삭제한 목록을 반환한다."""
    cols = client.list_collections()
    for name in cols:
        client.drop_collection(name)
    return cols


def run_dense_search(
    client: Any,
    *,
    collection_name: str,
    query_vector: Sequence[float],
    limit: int,
    filter_expr: str,
    output_fields: Sequence[str],
) -> List[Any]:
    """Milvus 덴스 검색 실행."""
    logger.info("[Milvus] 유사도 검색 실행")
    logger.debug(
        "dense search -> collection=%s limit=%s filter=%s fields=%s",
        collection_name,
        limit,
        filter_expr,
        output_fields,
    )
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
    output_fields: Sequence[str],
) -> List[Any]:
    """Milvus 하이브리드 검색 실행(BM25 + 벡터)."""
    try:
        logger.info("[Milvus] 하이브리드 검색 실행")
        logger.debug(
            "hybrid search -> collection=%s limit=%s filter=%s fields=%s",
            collection_name,
            limit,
            filter_expr,
            output_fields,
        )
        dense_req = AnnSearchRequest(
            data=[list(query_vector)],
            anns_field="embedding",
            param={"metric_type": "IP", "params": {}},
            limit=int(limit),
        )
        sparse_req = AnnSearchRequest(
            data=[query_text],
            anns_field="text_sparse",
            param={"metric_type": "BM25", "params": {}},
            limit=int(limit),
        )
        return client.hybrid_search(
            collection_name=collection_name,
            reqs=[dense_req, sparse_req],
            ranker=RRFRanker(k=60),
            limit=int(limit),
            filter=filter_expr,
            output_fields=list(output_fields),
        )
    except Exception as exc:  # pragma: no cover
        logger.warning("[Milvus] 하이브리드 검색 실패: %s", exc)
        return [[]]

