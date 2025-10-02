from typing import Dict, Any, Generator, List
from errors import NotFoundError, BadRequestError
from utils.llms.registry import LLM, Streamer
from repository.workspace import get_workspace_by_workspace_id, get_workspace_id_by_slug_for_user
from repository.workspace_thread import get_thread_id_by_slug_for_user
from repository.documents import list_doc_ids_by_workspace, delete_document_vectors_by_doc_ids
from repository.workspace_chat import (
    get_chat_history_by_thread_id,
    insert_chat_history,
)
from service.commons.doc_gen_templates import get_doc_gen_template
from service.commons.summary_templates import get_summary_template
from utils import logger
import json, time
from .retrieval import (
    retrieve_contexts_local,
    extract_doc_ids_from_attachments,
)

logger = logger(__name__)

def _build_messages(ws: Dict[str, Any], body: Dict[str, Any]) -> List[Dict[str, Any]]:
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

def _render_template(category: str, body: Dict[str, Any]) -> str:
    """
    화면 입력값(templateVariables)과 템플릿(content/sub_content 또는 templateText)을 병합해
    시스템 프롬프트로 주입할 텍스트 생성.
    """
    tpl_text = body.get("templateText") or ""
    tpl_id = body.get("templateId")
    vars_map: Dict[str, str] = dict(body.get("templateVariables") or {})

    row = None
    try:
        if not tpl_text and tpl_id:
            if category == "doc_gen":
                row = get_doc_gen_template(int(tpl_id))
                if row:
                    base = str(row.get("content") or "")
                    sub = str(row.get("sub_content") or "") or ""
                    defaults = {}
                    for it in row.get("variables") or []:
                        k = str(it.get("key") or "")
                        v = str(it.get("value") or "")
                        if k and v:
                            defaults[k] = v
                    merged = {**defaults, **vars_map}
                    txt = base + (("\n\n" + sub) if sub else "")
                    for k, v in merged.items():
                        txt = txt.replace("{{" + k + "}}", str(v))
                    tpl_text = txt
            elif category == "summary":
                row = get_summary_template(int(tpl_id))
                if row:
                    tpl_text = str(row.get("content") or "")
    except Exception:
        pass

    if tpl_text and vars_map:
        kv_lines = "\n".join(f"- {k}: {v}" for k, v in vars_map.items())
        tpl_text = tpl_text + f"\n\n### 입력값\n{kv_lines}"
    return tpl_text.strip()


def preflight_stream_chat_for_workspace(
    user_id: int,
    slug: str,
    category: str,
    body: Dict[str, Any],
    thread_slug: str | None = None,
) -> Dict[str, Any]:
    """
    스트리밍 시작 전 모든 유효성 검사를 수행하고 필요 리소스를 준비한다.
    예외는 여기서 발생시켜 StreamingResponse 시작 전 FastAPI 핸들러로 전달되게 한다.
    """
    # 카테고리 검증(옵션)
    if category not in ("qa", "doc_gen", "summary"):
        raise BadRequestError("category must be one of: qa, doc_gen, summary")
    workspace_id = get_workspace_id_by_slug_for_user(user_id, slug)

    if not workspace_id:
        raise NotFoundError("워크스페이스를 찾을 수 없습니다")

    ws = get_workspace_by_workspace_id(user_id, workspace_id)
    if not ws:
        raise NotFoundError("워크스페이스를 찾을 수 없습니다")

    if category == "qa":
        if not thread_slug:
            raise BadRequestError("qa 카테고리는 thread_slug가 필요합니다")
        thread_id = get_thread_id_by_slug_for_user(user_id, thread_slug)
        if not thread_id:
            raise NotFoundError("채팅 스레드를 찾을 수 없습니다")
    else:
        thread_id = None

    # 모드 검증
    mode = (body.get("mode") or ws.get("chat_mode") or "chat").lower()
    if mode not in ("chat", "query"):
        raise BadRequestError("mode must be 'chat' or 'query'")

    return {"ws": ws, "workspace_id": workspace_id, "thread_id": thread_id, "mode": mode}


def _build_context_string(snippets: List[Dict[str, Any]]) -> str:
    """snippets를 user message용 context 문자열로 변환"""
    if not snippets:
        return ""
    
    parts = []
    for i, snippet in enumerate(snippets, 1):
        title = snippet.get("title", "Unknown")
        page = snippet.get("page")
        text = snippet.get("text", "")
        
        source_info = f"[{i}] (출처: {title}"
        if page:
            source_info += f", 페이지 {page}"
        source_info += ")"
        
        parts.append(f"{source_info}\n{text}")
    
    return "\n\n---\n\n".join(parts)


def insert_rag_context(ws: Dict[str, Any], body: Dict[str, Any]) -> Generator[str, None, None]:
    """
    RAG 컨텍스트 주입
    """
    try:
        candidate_doc_ids = []
        #후보 문서 : 워크 스페이스 전역 + 첨부 임시 문서
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

        snippets =[]
        if candidate_doc_ids:
            top_k = int(ws.get("top_n") or 4)
            thr = float(ws.get("similarity_threshold") or 0.0)
            snippets = retrieve_contexts_local(body["message"], candidate_doc_ids, top_k=top_k, threshold=thr)
        else:
            logger.info(f"## 참조문서 없음.") # 비정상 종료 방지
        return snippets, temp_doc_ids
    except Exception as e:
        logger.error(f"RAG context build failed: {e}")
        return [], []

# TODO: 추후 수정 UserPrompt -> Document 
def _compose_summary_message(user_prompt: str, original_text: str) -> str:
    if original_text:
        original = str(original_text or "").strip()
    else:
        original = ""
    detail = str(user_prompt or "").strip()
    if not original and not detail:
        raise BadRequestError("originalText 또는 Documents 중 하나는 필수입니다.")
    if detail:
        suffix = "[User Prompt]\n" + detail
        return f"{original}\n\n{suffix}" if original else suffix
    return original



def _compose_doc_gen_message(user_prompt: Any, template_vars: dict[str, Any]) -> str:
    base = str(user_prompt or "").strip()
    if template_vars:
        var_lines = "\n".join(f"- {key} : {value}" for key, value in template_vars.items())
        block = "[User Prompt]\n" + var_lines
        return f"{base}\n\n{block}" if base else block
    return base or "요청된 템플릿에 따라 문서를 작성해 주세요."

    
def _resolve_runner(provider, model) -> Streamer:
    if not provider or not model:
        raise BadRequestError("provider와 model 정보를 확인할 수 없습니다.")
    return LLM.from_workspace(provider, model)


def _build_system_message(base_prompt: str, category: str, body: Dict[str, Any]) -> Dict[str, str]:
    system_text = (base_prompt + "\n\n반드시 한국어로 대답하세요.").strip()
    segments = [system_text or "반드시 한국어로 대답하세요."]

    if category == "doc_gen":
        tpl = _render_template("doc_gen", body)
        if tpl:
            segments.append("### TEMPLATE\n" + tpl)
    return {"role": "system", "content": "\n\n".join(segments)}

def _build_user_message_with_context(message: str, snippets: str, query_refusal_response: str="") -> str :
    """User message에 RAG context 포함 """
    if not snippets: return message
    parts = [f"[{i}] {h['text']}" for i, h in enumerate(snippets, 1)]
    contexts = f"아래 CONTEXTS 를 근거로 USER QUESTION에 대해 한국어로 답변하세요.\n\n### CONTEXTS\n" + "\n---\n".join(parts)
    return (
        f"{contexts}\n\n"
        f"### USER QUESTION\n- {message}\n\n"
        f"### QUERY REFUSAL RESPONSE\n- {query_refusal_response}\n\n" if query_refusal_response else ""
    )


def stream_chat_for_qa(
    user_id: int,
    slug: str,
    category: str,
    body: Dict[str, Any],
    thread_slug: str | None = None,
) -> Generator[str, None, None]:
    pre = preflight_stream_chat_for_workspace(user_id, slug, category, body, thread_slug)
    ws = pre["ws"]
    thread_id = pre["thread_id"]

    body = dict(body)
    body["message"] = str(body.get("message") or "").strip()
    if not body["message"]:
        raise BadRequestError("message is required")
    logger.info(f"BODY : {body}")

    runner = _resolve_runner(body["provider"], body["model"])

    messages: List[Dict[str, Any]] = []
    if category == "qa" and ws["chat_history"] > 0 and thread_id is not None:
        history = get_chat_history_by_thread_id(user_id, thread_id, ws["chat_history"])
        for chat in history[::-1]:
            messages.append({"role": "user", "content": chat["prompt"]})
            assistant_text = chat["response"]
            try:
                assistant_text = json.loads(assistant_text).get("text", assistant_text)
            except Exception:
                pass
            messages.append({"role": "assistant", "content": assistant_text})

    temp_doc_ids: List[str] = []
    # RAG context 검색
    snippets, temp_doc_ids = insert_rag_context(ws, body)
    logger.info(f"\n## 검색된 SNIPPETS 목록: \n{snippets}\n")

    # User message에 context 포함
    user_message = _build_user_message_with_context(body["message"], snippets, ws["query_refusal_response"])
    messages.append({"role": "user", "content": user_message})
    logger.info(f"\nMESSAGES:\n{messages}")

    yield from _stream_and_persist(user_id, category, ws, body, runner, messages, snippets, temp_doc_ids, thread_id)


def stream_chat_for_summary(
    user_id: int,
    slug: str,
    category: str,
    body: Dict[str, Any],
) -> Generator[str, None, None]:
    pre = preflight_stream_chat_for_workspace(user_id, slug, category, body)
    ws = pre["ws"]

    body = dict(body)
    body["message"] = _compose_summary_message(
        original_text=body.get("originalText"),
        user_prompt=body.get("userPrompt"),
    )

    runner = _resolve_runner(body["provider"], body["model"])

    messages: List[Dict[str, Any]] = []
    ctx = ""
    temp_doc_ids: List[str] = []
    ctx_result = insert_rag_context(ws, body, messages)
    if ctx_result:
        ctx, temp_doc_ids = ctx_result

    system_prompt = str(body.get("systemPrompt") or "").strip()
    messages.insert(0, _build_system_message(ctx, system_prompt, category, body))

    ws_no_sys = dict(ws)
    ws_no_sys["system_prompt"] = None
    messages.extend(_build_messages(ws_no_sys, body))

    yield from _stream_and_persist(user_id, category, ws, body, runner, messages, temp_doc_ids)


def stream_chat_for_doc_gen(
    user_id: int,
    slug: str,
    category: str,
    body: Dict[str, Any],
) -> Generator[str, None, None]:
    pre = preflight_stream_chat_for_workspace(user_id, slug, category, body)
    ws = pre["ws"]

    body = dict(body)
    body["message"] = _compose_doc_gen_message(
        user_prompt=body.get("userPrompt"),
        template_vars=body.get("templateVariables") or {},
    )
    runner = _resolve_runner(body["provider"], body["model"])

    messages: List[Dict[str, Any]] = []
    ctx = ""
    temp_doc_ids: List[str] = []
    ctx_result = insert_rag_context(ws, body, messages)
    if ctx_result:
        ctx, temp_doc_ids = ctx_result

    system_prompt = str(body.get("systemPrompt") or "").strip()
    messages.insert(0, _build_system_message(ctx, system_prompt, category, body))

    ws_no_sys = dict(ws)
    ws_no_sys["system_prompt"] = None
    messages.extend(_build_messages(ws_no_sys, body))

    yield from _stream_and_persist(user_id, category, ws, body, runner, messages,  temp_doc_ids)

def _stream_and_persist(
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
    temperature = ws.get("temperature")
    acc_text: List[str] = []
    t0 = time.perf_counter()
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
    response_json = {
        "text": "".join(acc_text),
        "sources": sources,
        "type": "chat", # 일단 chat으로 고정 query 모드는 사용하지 않음
        "attachments": body.get("attachments") or [],
        "metrics": {
            "completion_tokens": 0,
            "prompt_tokens": 0,
            "total_tokens": 0,
            "outputTps": 0.0 if duration == 0 else len("".join(acc_text)) / max(duration, 1e-6),
            "duration": round(duration, 3),
        },
    }
    insert_chat_history(
        user_id=user_id,
        category=category,
        workspace_id=ws["id"],
        prompt=body["message"],
        response=json.dumps(response_json, ensure_ascii=False),
        thread_id=thread_id,
    )
    try:
        if temp_doc_ids:
            delete_document_vectors_by_doc_ids(temp_doc_ids)
    except Exception as exc:
        logger.error(f"vector cleanup failed: {exc}")