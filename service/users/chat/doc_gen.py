"""Doc Gen 카테고리 스트리밍 로직"""
from typing import Dict, Any, Generator, List
from utils import logger
from .common import (
    preflight_stream_chat_for_workspace,
    build_system_message,
    resolve_runner,
    stream_and_persist,
)
from .retrieval import extract_doc_ids_from_attachments
from repository.documents import list_doc_ids_by_workspace
from .retrieval import retrieve_contexts_local

logger = logger(__name__)


def _compose_doc_gen_message(user_prompt: Any, template_vars: dict[str, Any]) -> str:
    """
    Doc Gen용 메시지 구성
    
    특징:
    - 템플릿 변수를 포맷팅
    - 사용자 프롬프트와 결합
    """
    base = str(user_prompt or "").strip()
    if template_vars:
        var_lines = "\n".join(f"- {key} : {value}" for key, value in template_vars.items())
        block = "[User Prompt]\n" + var_lines
        return f"{base}\n\n{block}" if base else block
    return base or "요청된 템플릿에 따라 문서를 작성해 주세요."


def _build_messages_with_attachments(ws: Dict[str, Any], body: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    첨부파일(이미지)을 포함한 메시지 구성
    
    특징:
    - OpenAI의 vision 모델용 이미지 첨부 지원
    """
    system_prompt = ws.get("system_prompt")
    provider = (ws.get("provider") or "").lower()
    attachments = body.get("attachments") or []
    content = body["message"]

    msgs: List[Dict[str, Any]] = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt + ". 반드시 한국어로 대답하세요."})

    if provider == "openai" and attachments:
        parts = [{"type": "text", "text": content}]
        for att in attachments:
            cs = att.get("contentString")
            if cs:
                parts.append({"type": "image_url", "image_url": {"url": cs}})
        msgs.append({"role": "user", "content": parts})
    else:
        msgs.append({"role": "user", "content": content})

    logger.info(f"msgs: {msgs}")
    return msgs


def _insert_rag_context(ws: Dict[str, Any], body: Dict[str, Any]) -> tuple[List[Dict[str, Any]], List[str]]:
    """
    Doc Gen용 RAG 컨텍스트 검색 (선택적)
    
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
        
        candidate_doc_ids.extend(temp_doc_ids)
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


def stream_chat_for_doc_gen(
    user_id: int,
    slug: str,
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
    # 1. Preflight 검증
    pre = preflight_stream_chat_for_workspace(user_id, slug, category, body)
    ws = pre["ws"]

    # 2. Body 준비
    body = dict(body)
    body["message"] = _compose_doc_gen_message(
        user_prompt=body.get("userPrompt"),
        template_vars=body.get("templateVariables") or {},
    )
    
    # 3. LLM runner 준비
    runner = resolve_runner(body["provider"], body["model"])

    # 4. 메시지 구성
    messages: List[Dict[str, Any]] = []
    
    # RAG context (선택적)
    snippets, temp_doc_ids = _insert_rag_context(ws, body)
    
    # 시스템 프롬프트 추가
    system_prompt = str(body.get("systemPrompt") or "").strip()
    messages.append(build_system_message(system_prompt, category, body))
    
    # 첨부파일 포함한 메시지 추가
    ws_no_sys = dict(ws)
    ws_no_sys["system_prompt"] = None
    messages.extend(_build_messages_with_attachments(ws_no_sys, body))

    # 5. 스트리밍 및 저장
    yield from stream_and_persist(
        user_id, 
        category, 
        ws, 
        body, 
        runner, 
        messages, 
        snippets,
        temp_doc_ids
    )

