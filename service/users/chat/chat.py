from typing import Dict, Any, Generator, List
from errors import NotFoundError, BadRequestError
from utils.llms.registry import LLM
from repository.users.workspace import get_workspace_by_workspace_id, get_workspace_id_by_slug_for_user
from repository.users.workspace_thread import get_thread_id_by_slug_for_user
from repository.documents import list_doc_ids_by_workspace, delete_document_vectors_by_doc_ids
from repository.users.workspace_chat import (
    get_chat_history_by_thread_id,
    insert_chat_history,
)
from service.commons.doc_gen_templates import get_doc_gen_template
from service.commons.summary_templates import get_summary_template
from utils import logger
import json, time
from .retrieval import (
    retrieve_contexts_local,
    build_context_message,
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



def stream_chat_for_workspace(
    user_id: int, 
    slug: str, 
    category: str,
    body: Dict[str, Any], 
    thread_slug: str=None
) -> Generator[str, None, None]:
    """
    주: 이 함수는 스트리밍만 담당한다고 가정. 검증/조회 예외는 preflight에서 끝낸다.
    """
    # 프리플라이트 결과를 재활용하려면 엔드포인트에서 전달받아도 되고,
    # 간단히 여기서 한 번 더 호출해도 됨(중복 조회 허용 시).
    pre = preflight_stream_chat_for_workspace(user_id, slug, category, body, thread_slug)
    ws = pre["ws"]
    thread_id = pre["thread_id"]

    # QA일 때만 history 포함
    messages: List[Dict[str, Any]] = []
    if category == "qa":
        limit = ws["chat_history"]
        if limit > 0 and thread_id is not None:
            chat_history = get_chat_history_by_thread_id(user_id, thread_id, limit)
            for chat in chat_history[::-1]:  # 오래된 것부터 추가
                messages.append({"role": "user", "content": chat["prompt"]})
                # response는 문자열 -> text만 추출
                assistant_text = chat["response"]
                try:
                    assistant_text = json.loads(assistant_text).get("text", assistant_text)
                except Exception:
                    pass
                messages.append({"role": "assistant", "content": assistant_text})

    runner = LLM.from_workspace(ws) 

    ###### RAG 컨텍스트 주입 ######
    temp_docs_ids : List[str] = []
    ctx = ""
    try:
        #후보 문서 : 워크 스페이스 전역 + 첨부 임시 문서
        candidate_doc_ids: List[str] = []
        try:
            if ws.get("id"):
                ws_docs = list_doc_ids_by_workspace(ws["id"]) or []
                logger.info(f"\n## 워크스페이스 문서 목록: \n{ws_docs}\n")
                candidate_doc_ids.extend([str(d["doc_id"]) if isinstance(d, dict) else str(d) for d in ws_docs])
        except Exception:
            pass
        temp_doc_ids = extract_doc_ids_from_attachments(body.get("attachments"))
        logger.info(f"\n## 첨부 문서 목록: \n{temp_doc_ids}\n")
        # 첨부에서 온 임시 문서 Retrieval 추가
        candidate_doc_ids.extend(temp_doc_ids)
        # 중복 제거
        candidate_doc_ids = list(dict.fromkeys(candidate_doc_ids))
        logger.info(f"\n## 후보 문서 목록: \n{candidate_doc_ids}\n")

        if candidate_doc_ids:
            top_k = int(ws.get("top_n") or 4)
            thr = float(ws.get("similarity_threshold") or 0.0)
            snippets = retrieve_contexts_local(body["message"], candidate_doc_ids, top_k=top_k, threshold=thr)
            ctx = build_context_message(snippets) or ""
            logger.info(f"\n## CONTEXT 주입 결과: \n{ctx}\n")
            if ctx:
                messages.insert(0, {"role": "system", "content": ctx})
    except Exception as e:
        logger.error(f"RAG context build failed: {e}")

    # 메시지 구성 : [system = system_prompt + ctx] + [history] + [user]
    messages : List[Dict[str, Any]] = []

    # 1) system : (요청 오버라이드 또는 워크스페이스) 시스템 프롬프트 + RAG 컨텍스트 + 템플릿
    system_parts: List[str] = []
    effective_system_prompt = (body.get("systemPrompt") or ws.get("system_prompt"))
    if effective_system_prompt:
        system_parts.append(str(effective_system_prompt) + ". 반드시 한국어로 대답하세요.")
    if ctx:
        system_parts.append(ctx)
    if category in ("doc_gen", "summary"):
        try:
            tpl = _render_template(category, body)
            if tpl:
                system_parts.append("### TEMPLATE\n" + tpl)
        except Exception as e:
            logger.error(f"template render failed: {e}")
    if system_parts:
        messages.append({"role": "system", "content": "\n\n".join(system_parts)})

    # 2) history (QA일 때만)
    if category == "qa":
        limit = ws["chat_history"]
        if limit > 0  and thread_id is not None:
            chat_history = get_chat_history_by_thread_id(user_id, thread_id, limit)
            for chat in chat_history[::-1]:
                messages.append({"role": "user", "content": chat["prompt"]})
                assistant_text = chat["response"]
                try:
                    assistant_text = json.loads(assistant_text).get("text", assistant_text)
                except Exception:
                    pass
                messages.append({"role": "assistant", "content": assistant_text})
    
    # 3) user: system 중복 방지를 위해 system_prompt 제거한 ws로 사용자 메시지만 추가
    ws_no_sys = dict(ws)
    ws_no_sys["system_prompt"] = None
    messages.extend(_build_messages(ws_no_sys, body))

    logger.info(f"\n\nretrieval: \n\n{messages}\n\n")
    temperature = ws.get("temperature")

    # logger.info(f"\n\nmessages: \n\n{messages}\n\n")
    # logger.info(f"temperature: {temperature}\n\n")

    acc_text = []
    t0 = time.perf_counter()
    for chunk in runner.stream(messages, temperature=temperature):
        if chunk:
            acc_text.append(chunk)
            yield chunk
    duration = max(time.perf_counter() - t0, 0.0)
    # 스트리밍 완료 후 저장
    response_json = {
        "text": "".join(acc_text),
        "sources": [],                          # TODO: 소스 추가
        "type": "chat",
        "attachments": body.get("attachments") or [], # TODO: 첨부파일 추가
        "metrics": {
            "completion_tokens": 0,             # TODO: 토큰 카운트 추가
            "prompt_tokens": 0,                 # TODO: 토큰 카운트 추가
            "total_tokens": 0,                  # TODO: 토큰 카운트 추가
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
        if temp_doc_ids:  # 임시 문서 삭제
            delete_document_vectors_by_doc_ids(temp_doc_ids)
    except Exception as e:
        logger.error(f"vector clenup failed: {e}")
