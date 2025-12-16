"""스트리밍 및 DB 저장 로직"""
from typing import Dict, Any, List, Generator
from utils import logger
from utils.llms.registry import Streamer
from repository.workspace_chat import insert_chat_history
from service.manage_documents.documents import delete_documents_by_ids
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
        category: 카테고리 (qna, summary, doc_gen)
        ws: 워크스페이스 정보
        body: 요청 본문
        runner: LLM streamer
        messages: 메시지 목록
        snippets: RAG 검색 결과 (sources에 저장될 내용)
        temp_doc_ids: 임시 문서 ID 목록 (정리용)
        thread_id: 스레드 ID (QA만 해당)
    """
    temperature = ws.get("temperature")
    # [추가] 워크스페이스의 provider에 맞는 API 키 추출
    provider = ws.get("provider", "openai")
    # 키가 필요 없는 provider 목록
    NO_KEY_PROVIDERS = ("huggingface", "ollama", "local")
    
    api_key = None
    if provider not in NO_KEY_PROVIDERS:
        # openai -> openai_api_key, anthropic -> anthropic_api_key
        api_key_name = f"{provider}_api_key"
        api_key = ws.get(api_key_name)
        
        # 상용 API인데 키가 없는 경우 로그 경고 (실제 에러는 runner에서 발생시킴)
        if not api_key:
             logger.warning(f"No API key found for provider: {provider}")

    acc_text: List[str] = []
    t0 = time.perf_counter()


    sources = []
    for snippet in snippets:
        sources.append({
            "doc_id": snippet.get("doc_id"),
            "title": snippet.get("title"),
            "text": snippet.get("text"),
            "score": round(snippet.get("score", 0.0), 5),
            "page": snippet.get("page"),
            "chunk_index": snippet.get("chunk_index"),
            "source": snippet.get("source"),  # "milvus" 또는 "local"
        })
    # 🔥 스트리밍 시작 전에 소스 먼저 전송
    if sources:
        payload = json.dumps(sources, ensure_ascii=False)
        yield f"__SOURCES__:{payload}"
        logger.debug("__SOURCES__ (first 100 chars): %s", payload[:100])
    else:
        logger.debug("__SOURCES__: []")

    # 스트리밍 응답 생성
    try:
        # stream 함수가 api_key 인자를 지원하도록 수정되었다고 가정
        stream_gen = runner.stream(messages, temperature=temperature, api_key=api_key)
    except TypeError:
        # 만약 runner.stream이 api_key를 안 받는 구버전이면 없이 호출
        stream_gen = runner.stream(messages, temperature=temperature)
    for chunk in stream_gen:
        if chunk:
            acc_text.append(chunk)
            yield chunk
    duration = max(time.perf_counter() - t0, 0.0)
    
    # TODO : 응답 JSON 구성 (TOKEN 카운트 추가)
    response_json = {
        "text": "".join(acc_text),
        "sources": sources,
        "type": "chat",  # 일단 chat으로 고정 query 모드는 사용하지 않음
        "attachments": body.get("attachments") or [],
        "metrics": {
            "completion_tokens": 0,
            "prompt_tokens": 0,
            "total_tokens": 0,
            "output_tps": 0.0 if duration == 0 else len("".join(acc_text)) / max(duration, 1e-6),
            "duration": round(duration, 3),
        },
    }
    # DB 저장
    chat_id = insert_chat_history(
        user_id=user_id,
        category=category,
        workspace_id=ws["id"],
        prompt=body["message"],
        response=json.dumps(response_json, ensure_ascii=False),
        thread_id=thread_id,
        model=body["model"],
    )
    logger.debug(f"CHAT_ID : {chat_id}")
    try:
        if temp_doc_ids:
            delete_documents_by_ids(temp_doc_ids)
    except Exception as exc:
        logger.error(f"temp doc cleanup failed: {exc}")
        
    yield f"__CHAT_ID__: {chat_id}"


