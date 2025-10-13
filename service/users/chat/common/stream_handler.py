"""스트리밍 및 DB 저장 로직"""
from typing import Dict, Any, List, Generator
from utils import logger
from utils.llms.registry import Streamer
from repository.workspace_chat import insert_chat_history
from repository.documents import delete_document_vectors_by_doc_ids
import json
import time

logger = logger(__name__)


def stream_and_persist(
    user_id: int,
    category: str,
    ws: Dict[str, Any],
    body: Dict[str, Any],
    runner: Streamer,
    messages: List[Dict[str, Any]],
    snippets: List[Dict[str, Any]],
    temp_doc_ids: List[str],
    thread_id: int | None = None,
) -> Generator[str, None, None]:
    """
    스트리밍 응답 생성 및 DB 저장
    
    Args:
        user_id: 사용자 ID
        category: 카테고리 (qa, summary, doc_gen)
        ws: 워크스페이스 정보
        body: 요청 본문
        runner: LLM streamer
        messages: 메시지 목록
        snippets: RAG 검색 결과 (sources에 저장될 내용)
        temp_doc_ids: 임시 문서 ID 목록 (정리용)
        thread_id: 스레드 ID (QA만 해당)
    """
    temperature = ws.get("temperature")
    acc_text: List[str] = []
    t0 = time.perf_counter()
    
    # 스트리밍 응답 생성
    for chunk in runner.stream(messages, temperature=temperature):
        if chunk:
            acc_text.append(chunk)
            yield chunk
    
    duration = max(time.perf_counter() - t0, 0.0)

    # snippets를 sources 형식으로 변환
    sources = []
    for snippet in snippets:
        sources.append({
            "doc_id": snippet.get("doc_id"),
            "title": snippet.get("title"),
            "text": snippet.get("text"),
            "score": round(snippet.get("score", 0.0), 5),
            "page": snippet.get("page"),
            "chunk_index": snippet.get("chunk_index"),
        })
    
    # 응답 JSON 구성
    response_json = {
        "text": "".join(acc_text),
        "sources": sources,
        "type": "chat",  # 일단 chat으로 고정 query 모드는 사용하지 않음
        "attachments": body.get("attachments") or [],
        "metrics": {
            "completion_tokens": 0,
            "prompt_tokens": 0,
            "total_tokens": 0,
            "outputTps": 0.0 if duration == 0 else len("".join(acc_text)) / max(duration, 1e-6),
            "duration": round(duration, 3),
        },
    }
    
    # DB 저장
    insert_chat_history(
        user_id=user_id,
        category=category,
        workspace_id=ws["id"],
        prompt=body["message"],
        response=json.dumps(response_json, ensure_ascii=False),
        thread_id=thread_id,
    )
    
    # 임시 벡터 정리
    try:
        if temp_doc_ids:
            delete_document_vectors_by_doc_ids(temp_doc_ids)
    except Exception as exc:
        logger.error(f"vector cleanup failed: {exc}")

