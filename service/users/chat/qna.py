"""QA 카테고리 스트리밍 로직"""
from typing import Dict, Any, Generator, List
from errors import BadRequestError
from utils import logger
from repository.workspace_chat import get_chat_history_by_thread_id
from .common import (
    preflight_stream_chat_for_workspace,
    build_user_message_with_context,
    resolve_runner,
    stream_and_persist,
)
from .retrieval import (
    retrieve_contexts_local,
    extract_doc_ids_from_attachments,
)
from service.admin.manage_vator_DB import execute_search, get_vector_settings # milvusDB 검색 함수
from repository.documents import list_doc_ids_by_workspace
import json
import asyncio

logger = logger(__name__)

def _fetch_milvus_snippets(
    question        : str, # 질문
    security_level  : int, # 보안레벨
    ws              : Dict[str, Any], # 워크스페이스
    top_k           : int, # 상위 K개
    ) -> List[Dict[str, Any]]:
    try:
        settings = get_vector_settings() 
    except Exception as exc:
        logger.warning(f"[Milvus] 설정 조회 실패: {exc}")
        return []
    model_key = settings.get("embeddingModel")
    if not model_key:
        logger.info("[Milvus] 활성화된 임베딩 모델이 없어 글로벌 검색을 건너뜁니다.")
        return []
    search_type = ws.get("vector_search_mode") # workspace내에 미리 저장된 검색 타입 사용

    logger.debug(f"###### 모델 키 model_key: {model_key}")
    logger.debug(f"###### 검색 타입 search_type: {search_type}")
    logger.debug(f"###### 보안 레벨 security_level: {security_level}")

    try:
        result = _run_execute_search(
            question        =question,
            # top_k=max(top_k, 5),
            rerank_top_n    =min(top_k, 2), # WORKSPACE TOP-K
            security_level  =security_level, # USER
            task_type       ="qna", # 작업 유형
            model_key       =model_key, # 모델 키
            search_type     =search_type, # 검색 타입
        )
        logger.debug(f"\n###########################\nresult: {result[200:]}...")
    except Exception as exc:
        logger.exception(f"[Milvus] 검색 실패: {exc}")
        return []

    snippets: List[Dict[str, Any]] = []

    for hit in result.get("hits", []):
        snippet_text = str(hit.get("snippet") or "").strip()
        if not snippet_text:
            continue
        snippets.append(
            {
                "title": hit.get("doc_id") or hit.get("path") or "Milvus",
                "score": float(hit.get("score", 0.0)),
                "doc_id": hit.get("doc_id"),
                "text": snippet_text,
                "source": "milvus",
            }
        )
        if len(snippets) >= top_k:
            break
    return snippets


def _run_execute_search(**kwargs):
    try:
        return asyncio.run(execute_search(**kwargs))
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(execute_search(**kwargs))
        finally:
            loop.close()

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
    try:
        candidate_doc_ids = []
        # 후보 문서: 워크스페이스 전역 + 첨부 임시 문서 + Milvus DB 검색
        # ---------------------------------- 워크스페이스 전역 문서 ----------------------------------
        try:
            if ws.get("id"):
                ws_docs = list_doc_ids_by_workspace(ws["id"]) or []
                logger.info(f"\n## 워크스페이스 문서 목록: \n{ws_docs}\n")
                candidate_doc_ids.extend([str(d["doc_id"]) if isinstance(d, dict) else str(d) for d in ws_docs])
        except Exception:
            pass
        # ---------------------------------- 첨부 임시 문서 ----------------------------------
        temp_doc_ids = extract_doc_ids_from_attachments(body.get("attachments"))
        logger.info(f"\n## Temporary document list from attachments: \n{temp_doc_ids}\n")

        candidate_doc_ids.extend(temp_doc_ids)      # 첨부에서 온 임시 문서를 Retrieval 추가
        candidate_doc_ids = list(dict.fromkeys(candidate_doc_ids)) # 중복 제거
        logger.info(f"\n## Candidate document list: \n{candidate_doc_ids}\n")

        snippets = []
        if candidate_doc_ids:
            top_k = int(ws.get("top_n") or 4)
            thr = float(ws.get("similarity_threshold") or 0.0)
            snippets = retrieve_contexts_local(body["message"], candidate_doc_ids, top_k=top_k, threshold=thr)
            logger.info(f"\n## Searched SNIPPETS from temporary documents: {len(snippets)}개\n")
        else:
            logger.info(f"\n## No documents found - Skipping RAG\n")
        #  ---------------------------------- Milvus DB 검색 ----------------------------------
        top_k = int(ws.get("top_n") or 4)
        milvus_snippets = _fetch_milvus_snippets(body["message"], security_level, ws, top_k)
        if milvus_snippets:
            logger.debug(f"\n## Milvus 글로벌 스니펫: {len(milvus_snippets)}개\n## Searched SNIPPETS from Milvus DB: \n{milvus_snippets[100:]}...\n")
            snippets.extend(milvus_snippets)
        else:
            logger.info("\n## No snippets found from Milvus DB\n")

        return snippets, temp_doc_ids
    except Exception as e:
        logger.error(f"RAG context build failed: {e}")
        return [], []


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
        logger.debug(f"\n## SEARCHED SNIPPETS from RAG: \n{snippets}\n")
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

