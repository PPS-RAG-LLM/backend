"""QA 카테고리 스트리밍 로직"""
from typing import Dict, Any, Generator, List
from errors import BadRequestError
from utils import logger
from repository.workspace_chat import get_chat_history_by_thread_id
from ..common import (
    preflight_stream_chat_for_workspace,
    build_user_message_with_context,
    resolve_runner,
    stream_and_persist,
)
from ..retrieval import (
    retrieve_contexts_local,
    extract_doc_ids_from_attachments,
)
from repository.documents import list_doc_ids_by_workspace
import json

logger = logger(__name__)


def _insert_rag_context(ws: Dict[str, Any], body: Dict[str, Any]) -> tuple[List[Dict[str, Any]], List[str]]:
    """
    RAG 컨텍스트 검색
    
    Returns:
        (snippets, temp_doc_ids)
    """
    try:
        candidate_doc_ids = []
        
        # 후보 문서: 워크스페이스 전역 + 첨부 임시 문서
        try:
            if ws.get("id"):
                ws_docs = list_doc_ids_by_workspace(ws["id"]) or []
                logger.info(f"\n## 워크스페이스 문서 목록: \n{ws_docs}\n")
                candidate_doc_ids.extend([str(d["doc_id"]) if isinstance(d, dict) else str(d) for d in ws_docs])
        except Exception:
            pass
        
        temp_doc_ids = extract_doc_ids_from_attachments(body.get("attachments"))
        logger.info(f"\n## 스레드 임시 첨부 문서 목록: \n{temp_doc_ids}\n")
        
        # 첨부에서 온 임시 문서 Retrieval 추가
        candidate_doc_ids.extend(temp_doc_ids)
        # 중복 제거
        candidate_doc_ids = list(dict.fromkeys(candidate_doc_ids))
        logger.info(f"\n## 후보 문서 목록: \n{candidate_doc_ids}\n")

        snippets = []
        if candidate_doc_ids:
            top_k = int(ws.get("top_n") or 4)
            thr = float(ws.get("similarity_threshold") or 0.0)
            snippets = retrieve_contexts_local(body["message"], candidate_doc_ids, top_k=top_k, threshold=thr)
            logger.info(f"\n## 검색된 SNIPPETS: {len(snippets)}개\n")
        else:
            logger.info(f"\n## 문서 없음 - RAG 스킵\n")
        
        return snippets, temp_doc_ids
        
    except Exception as e:
        logger.error(f"RAG context build failed: {e}")
        return [], []


def stream_chat_for_qa(
    user_id: int,
    slug: str,
    category: str,
    body: Dict[str, Any],
    thread_slug: str | None = None,
) -> Generator[str, None, None]:
    """
    QA 카테고리 스트리밍
    
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
    logger.info(f"BODY : {body}")

    # 3. LLM runner 준비
    runner = resolve_runner(body["provider"], body["model"])

    # 4. Chat history 로드
    messages: List[Dict[str, Any]] = []
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
    snippets, temp_doc_ids = _insert_rag_context(ws, body)
    logger.info(f"\n## 검색된 SNIPPETS 목록: \n{snippets}\n")

    # 6. User message에 context 포함
    user_message = build_user_message_with_context(
        body["message"], 
        snippets, 
        ws.get("query_refusal_response", "")
    )
    messages.append({"role": "user", "content": user_message})
    logger.info(f"\nMESSAGES:\n{messages}")

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

