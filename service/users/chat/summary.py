"""Summary 카테고리 스트리밍 로직"""
from typing import Dict, Any, Generator, List
from errors import BadRequestError
from utils import logger
from .common import (
    preflight_stream_chat_for_workspace,
    build_system_message,
    build_user_message_with_context,
    resolve_runner,
    stream_and_persist,
)
from .summary_doc.document_loader import get_full_documents_for_summary

logger = logger(__name__)


def _compose_summary_message(
    user_prompt: str = None,
    original_text: str = None, 
    parsed_documents: List[Dict[str, Any]] = None
) -> str:
    """
    Summary용 메시지 구성
    
    특징:
    - originalText 또는 parsed_documents 중 최소 하나는 필수
    - user_prompt는 선택사항 (추가 요청사항)
    - 둘 다 있으면 모두 CONTEXTS로 포함
    """
    contexts = []
    
    # 1. originalText가 있으면 추가
    if original_text:
        original = str(original_text).strip()
        if original:
            contexts.append(f"[원본 텍스트]\n{original}")
    
    # 2. 첨부 문서가 있으면 추가
    if parsed_documents:
        for i, parsed_document in enumerate(parsed_documents, 1):
            title = parsed_document.get("title", "Unknown")
            page = parsed_document.get("page")
            text = parsed_document.get("text", "")
            
            source_info = f"[문서 {i}: {title}"
            if page:
                source_info += f", 페이지 {page}"
            source_info += "]"
            
            contexts.append(f"{source_info}\n{text}")
    
    # 3. 둘 다 없으면 에러
    if not contexts:
        raise BadRequestError("originalText 또는 Documents(첨부파일) 중 하나는 필수입니다.")
    
    # 4. 모든 CONTEXTS 결합
    combined_contexts = "\n\n---\n\n".join(contexts)
    
    # 5. userPrompt가 있으면 추가 지시사항으로 붙임
    if user_prompt:
        detail = str(user_prompt).strip()
        if detail:
            return f"{combined_contexts}\n\n[추가 요청사항]\n{detail}"
    
    return combined_contexts


def stream_chat_for_summary(
    user_id: int,
    slug: str,
    category: str,
    body: Dict[str, Any],
) -> Generator[str, None, None]:
    """
    Summary 카테고리 스트리밍
    
    특징:
    - 전체 문서 로드 (벡터화 X)
    - originalText 또는 Document 요약
    - 추가 요청사항 처리
    """
    # 1. Preflight 검증
    pre = preflight_stream_chat_for_workspace(user_id, slug, category, body)
    ws = pre["ws"]

    # 2. Body 준비
    body = dict(body)

    # 3. LLM runner 준비
    runner = resolve_runner(body["provider"], body["model"])

    # 4. 문서 전체 로드
    parsed_documents = get_full_documents_for_summary(body)
    temp_doc_ids = [doc["doc_id"] for doc in parsed_documents]
    
    # 5. 메시지 구성 (originalText 또는 document_snippets)
    body["message"] = _compose_summary_message(
        user_prompt=body.get("userPrompt"),
        original_text=body.get("originalText") if body.get("originalText") else None,
        parsed_documents=parsed_documents if parsed_documents else None,
    )
    
    # 6. 메시지 목록 구성
    messages: List[Dict[str, Any]] = []
    
    # 시스템 프롬프트 추가
    system_prompt = str(body.get("systemPrompt") or "").strip()
    messages.append(build_system_message(system_prompt, category, body))

    # User message 추가 (context는 이미 message에 포함되어 있으므로 빈 리스트)
    user_message = build_user_message_with_context(body["message"], [])
    messages.append({"role": "user", "content": user_message})

    # 7. 스트리밍 및 저장
    yield from stream_and_persist(
        user_id, 
        category, 
        ws, 
        body, 
        runner, 
        messages, 
        parsed_documents,  # sources에 문서 정보 포함
        temp_doc_ids
    )

