"""Doc Gen 카테고리 스트리밍 로직"""
from typing import Dict, Any, Generator, List
from ..documents.full_document_loader import get_full_documents_texts
from utils import logger
from .common import (
    build_system_message,
    resolve_runner,
    stream_and_persist,
    build_user_message_with_context
)

logger = logger(__name__)


def _compose_doc_gen_message(user_prompt: Any, template_vars: dict[str, Any], parsed_documents: List[Dict[str, Any]]) -> str:
    """
    Doc Gen용 메시지 구성
    
    특징:
    - 문서 contexts (있으면)
    - 템플릿 변수를 포맷팅
    - 사용자 프롬프트와 결합
    """
    parts = []
    
    # 1. 문서 contexts (있으면 추가)
    if parsed_documents:
        contexts = []
        for i, parsed_document in enumerate(parsed_documents, 1):
            title = parsed_document.get("title", "Unknown")
            page = parsed_document.get("page")
            text = parsed_document.get("text", "")
            
            source_info = f"<document>\n[문서 {i}: {title}"
            if page:
                source_info += f", 페이지 {page}"
            source_info += "]"
            
            contexts.append(f"{source_info}\n{text}</document>")
        
        combined_contexts = "\n\n---\n\n".join(contexts)
        parts.append(combined_contexts)
    
    parts.append("<user_prompt>")
    
    # 2. 템플릿 변수 (있으면 추가)
    if template_vars:
        var_lines = "\n".join(f"- {key}: {value}" for key, value in template_vars.items())
        parts.append(f"\n{var_lines}\n")

    # 3. 사용자 프롬프트 (있으면 추가)
    user_prompt_text = str(user_prompt or "").strip()
    if user_prompt_text:
        parts.append(f"Prompt: {user_prompt_text}\n**The Answer should be in Korean.**")
    parts.append("</user_prompt>")

    # 4. 모든 parts 결합
    if parts:
        return "\n\n".join(parts)
    
    # 5. 아무것도 없으면 기본 메시지
    return "요청된 템플릿에 따라 문서를 작성해 주세요."



def stream_chat_for_doc_gen(
    user_id: int,
    ws: str,
    category: str,
    body: Dict[str, Any],
) -> Generator[str, None, None]:
    """
    Doc Gen 카테고리 스트리밍
    
    특징:
    - 템플릿 기반 문서 생성
    - 변수 치환
    - 양식 생성
    """
    # 2. Body 준비
    body = dict(body)

    # 3. LLM runner 준비
    runner = resolve_runner(body["provider"], body["model"])

    # 4. 워크스페이스 ID 조회
    parsed_documents = get_full_documents_texts(ws["id"])
    doc_ids = [doc["doc_id"] for doc in parsed_documents]
    
    logger.debug(f"userPrompt: {body.get('userPrompt')}")
    logger.debug(f"templateVariables: {body.get('templateVariables')}")
    logger.debug(f"parsed_documents: {parsed_documents}")

    
    # 메시지 구성 (variables + documents + userPrompt 결합)
    body["message"] = _compose_doc_gen_message(
        user_prompt=body.get("userPrompt"),
        template_vars=body.get("templateVariables") or {},
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


    logger.debug(f"## 메시지: \n{messages}")

    # 5. 스트리밍 및 저장
    yield from stream_and_persist(
        user_id, 
        category, 
        ws, 
        body, 
        runner, 
        messages, 
        parsed_documents,
        doc_ids
    )

