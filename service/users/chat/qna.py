"""QA 카테고리 스트리밍 로직"""
import asyncio
from typing import Any, Dict, Generator, List, Optional

from errors import BadRequestError
from repository.rag_settings import get_rag_settings_row
from repository.workspace_chat import get_chat_history_by_thread_id
from utils import logger
from service.retrieval.unified import (
    extract_doc_ids_from_attachments,
    unified_search,
)
from storage.db_models import DocumentType
from service.retrieval.adapters.base import RetrievalResult
from service.retrieval.interface import SearchRequest, retrieval_service
from service.vector_db.milvus_store import resolve_collection
from .common import (
    build_user_message_with_context,
    preflight_stream_chat_for_workspace,
    resolve_runner,
    stream_and_persist,
)
import json

logger = logger(__name__)

def _insert_rag_context(
    security_level: int, # 보안레벨
    ws            : Dict[str, Any], # 워크스페이스
    body          : Dict[str, Any], # 요청 본문
    ) -> tuple[List[Dict[str, Any]], List[str]]:
    """
    RAG 컨텍스트 검색
    Returns:
        (snippets, temp_doc_ids)
    """
    top_k = int(ws.get("top_n") or 4)
    threshold = float(ws.get("similarity_threshold") or 0.0)
    attachments = body.get("attachments")
    temp_doc_ids = extract_doc_ids_from_attachments(attachments)
    logger.debug(f"TEMP_DOC_IDS: {temp_doc_ids}")

    try:
        rag_settings = get_rag_settings_row()
        model_key = rag_settings.get("embedding_key")
    except Exception as exc:  # pragma: no cover - 안전장치
        logger.warning("활성 임베딩 모델 조회 실패: %s", exc)
        model_key = None

    sources_config = _resolve_rag_sources(ws.get("rag_sources"))
    logger.info("[RAG] resolved sources=%s", sources_config)

    workspace_only = (
        len(sources_config) == 1 and sources_config[0] == DocumentType.WORKSPACE.value
    )
    if workspace_only:
        rerank_top_n = top_k if bool(ws.get("enable_rerank", True)) else 0
        try:
            workspace_results = _search_workspace_sources(
                query=body["message"],
                workspace_id=int(ws.get("id") or 0),
                security_level=security_level,
                top_k=top_k + 10,
                rerank_top_n=rerank_top_n,
                search_type=ws.get("vector_search_mode"),
                model_key=model_key,
            )
            snippets = [_result_to_legacy_dict(res) for res in workspace_results]
            return snippets, temp_doc_ids
        except Exception as exc:  # pragma: no cover - 로깅
            logger.error("Workspace RAG search failed: %s", exc)
            return [], temp_doc_ids

    config = {
        "workspace_id": ws.get("id"),
        "attachments": attachments,
        "security_level": security_level,
        "top_k": top_k + 10,
        "threshold": threshold,
        "sources": sources_config,
        "enable_rerank": bool(ws.get("enable_rerank", True)),
        "rerank_top_n": top_k,
        "task_type": "qna",
        "search_type": ws.get("vector_search_mode"),
        "model_key": model_key,
    }
    logger.debug("CONFIG: %s", config)

    try:
        results = unified_search(body["message"], config)
        snippets = [_result_to_legacy_dict(res) for res in results]
        return snippets, temp_doc_ids
    except Exception as exc:  # pragma: no cover - 로깅
        logger.error("RAG context build failed: %s", exc)
        return [], temp_doc_ids


def _result_to_legacy_dict(result: RetrievalResult) -> Dict[str, Any]:
    """기존 dict 기반 컨텍스트 포맷으로 변환."""
    logger.debug(f"\n\nRESULT: \n\n{result}\n\n")
    return {
        "title": result.title,
        "score": result.score,
        "doc_id": result.doc_id,
        "text": result.text,
        "source": result.source,
        "page": result.page,
    }


def _search_workspace_sources(
    *,
    query: str,
    workspace_id: int,
    security_level: int,
    top_k: int,
    rerank_top_n: int,
    search_type: Optional[str],
    model_key: Optional[str],
) -> List[RetrievalResult]:
    """RetrievalService를 통해 워크스페이스 문서를 직접 검색."""
    if workspace_id <= 0:
        return []

    collection = resolve_collection(DocumentType.WORKSPACE.value)
    request = SearchRequest(
        query=query,
        collection_name=collection,
        task_type="qna",
        security_level=security_level,
        top_k=top_k,
        rerank_top_n=rerank_top_n,
        search_type=search_type,
        model_key=model_key,
        filters={"workspace_id": workspace_id},
    )
    raw = _run_retrieval_search(request)
    hits = raw.get("hits", [])

    results: List[RetrievalResult] = []
    for hit in hits:
        snippet = str(hit.get("snippet") or "").strip()
        if not snippet:
            continue
        results.append(
            RetrievalResult(
                doc_id=hit.get("doc_id"),
                title=str(hit.get("doc_id") or hit.get("path") or "workspace"),
                text=snippet,
                score=float(hit.get("score", 0.0)),
                source=DocumentType.WORKSPACE.value,
                chunk_index=int(hit.get("chunk_idx", 0)),
                page=int(hit.get("page", 0)) if hit.get("page") is not None else None,
                metadata={
                    "path": hit.get("path"),
                    "collection": collection,
                    "workspace_id": workspace_id,
                },
            )
        )
    return results


def _run_retrieval_search(request: SearchRequest) -> Dict[str, Any]:
    """동기 컨텍스트에서 RetrievalService 호출."""

    async def _runner() -> Dict[str, Any]:
        return await retrieval_service.search(request)

    try:
        return asyncio.run(_runner())
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(_runner())
        finally:
            loop.close()


def stream_chat_for_qna(
    user_id         : int, # 사용자 ID
    security_level  : int, # 보안레벨
    slug            : str, # 워크스페이스 슬러그
    category        : str, # 카테고리
    body            : Dict[str, Any], # 요청 본문
    thread_slug     : str | None = None, # 스레드 슬러그
) -> Generator[str, None, None]:
    """QA 카테고리 스트리밍
    특징:
    - RAG 검색으로 관련 문서 청크 검색
    - Chat history 포함
    - 사용자 질문에 대한 답변
    """
    # 1. Preflight 검증
    pre = preflight_stream_chat_for_workspace(user_id, slug, category, body, thread_slug)
    ws = pre["ws"]
    thread_id = pre["thread_id"]

    # 2. 메시지 검증
    body = dict(body)
    body["message"] = str(body.get("message") or "").strip()
    if not body["message"]:
        raise BadRequestError("message is required")
    logger.debug(f"BODY : {body}")

    # 3. LLM runner 준비
    runner = resolve_runner(body["provider"], body["model"])
    messages: List[Dict[str, Any]] = []
    # 3.1 system prompt 준비
    from .common.message_builder import build_system_message
    system_prompt = ws.get("system_prompt") or ""
    if system_prompt:
        messages.append(build_system_message(system_prompt, category, body))

    # 4. Chat history 로드
    if ws["chat_history"] > 0 and thread_id is not None:
        history = get_chat_history_by_thread_id(user_id, thread_id, ws["chat_history"])
        for chat in history[::-1]:
            messages.append({"role": "user", "content": chat["prompt"]})
            assistant_text = chat["response"]
            try:
                assistant_text = json.loads(assistant_text).get("text", assistant_text)
            except Exception:
                pass
            messages.append({"role": "assistant", "content": assistant_text})

    # 5. RAG context 검색
    snippets : List[Dict[str, Any]] = []
    temp_doc_ids: List[str] = []

    rag_flag = body.get("rag_flag") 
    if isinstance(rag_flag, str):               # 프런트에서 true/false 문자열이나 실제 bool을 보내도 문제 없이 처리되고,
        rag_flag = rag_flag.lower() == "true"   # snippets 와 temp_doc_ids 는 항상 정의된 상태로 아래 로직에 전달
    rag_flag = True if rag_flag is None else bool(rag_flag) # rag_flag 가 None 이면 True, 그 외는 bool 값으로 변환

    if rag_flag:
        snippets, temp_doc_ids = _insert_rag_context(security_level, ws, body) # 보안레벨, 워크스페이스, 정보 전달
        # logger.debug(f"\n## SEARCHED SNIPPETS from RAG: \n{snippets[:100]}...\n")
    else:
        logger.info("RAG disabled for this request; skipping context retrieval.")

    # 6. User message에 context 포함
    user_message = build_user_message_with_context(
        body["message"], 
        snippets, 
        ws.get("query_refusal_response", "")
    )
    messages.append({"role": "user", "content": user_message})
    logger.debug(f"\nMESSAGES:\n{messages}")

    # 7. 스트리밍 및 저장
    yield from stream_and_persist(
        user_id, 
        category, 
        ws, 
        body, 
        runner, 
        messages, 
        snippets, 
        temp_doc_ids, 
        thread_id
    )
def _resolve_rag_sources(raw_sources: Any) -> tuple:
    """워크스페이스 설정에서 RAG 소스 배열을 안전하게 파싱."""
    DEFAULT_SOURCES = [DocumentType.WORKSPACE.value, DocumentType.TEMP.value, DocumentType.ADMIN.value, DocumentType.LLM_TEST.value]
    if not raw_sources:
        return DEFAULT_SOURCES

    parsed: List[str] = []

    if isinstance(raw_sources, str):
        try:
            loaded = json.loads(raw_sources)
        except json.JSONDecodeError:
            loaded = raw_sources
        if isinstance(loaded, (list, tuple, set)):
            parsed = [str(s).strip().lower() for s in loaded if str(s).strip()]
        elif isinstance(loaded, str):
            parsed = [loaded.strip().lower()]
    elif isinstance(raw_sources, (list, tuple, set)):
        parsed = [str(s).strip().lower() for s in raw_sources if str(s).strip()]
    else:
        return DEFAULT_SOURCES

    normalized = []
    for src in parsed:
        if src in DEFAULT_SOURCES and src not in normalized:
            normalized.append(src)
    return tuple(normalized or DEFAULT_SOURCES)

