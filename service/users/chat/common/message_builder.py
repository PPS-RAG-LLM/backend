"""메시지 구성 로직"""
from typing import Dict, Any, List
from errors import BadRequestError
from utils.llms.registry import LLM, Streamer
from service.commons.doc_gen_templates import get_doc_gen_template
from service.commons.summary_templates import get_summary_template


def render_template(category: str, body: Dict[str, Any]) -> str:
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


def build_system_message(base_prompt: str, category: str, body: Dict[str, Any]) -> Dict[str, str]:
    """시스템 메시지 구성"""
    system_text = (base_prompt + f"\n\n **Please think in <think> tag and answer in <answer> tag.**").strip()
    segments = [system_text or "**Please think in <think> tag and answer in <answer> tag.**"]

    if category == "doc_gen":
        tpl = render_template("doc_gen", body)
        if tpl:
            segments.append("### TEMPLATE\n" + tpl)
    
    return {"role": "system", "content": "\n\n".join(segments)}


def build_user_message_with_context(message: str, snippets: List[Dict[str, Any]], query_refusal_response: str = "") -> str:
    """User message에 RAG context 포함"""
    if not snippets:
        return message
    
    if parts:
        parts = [f"[{i}] {h['text']}" for i, h in enumerate(snippets, 1)]
        contexts = (
            "Answer the `<user_prompt>` based on the following `<documents>`.\n"
            "<documents>\n" + "<divider/>\n".join(parts) + "\n</documents>\n"
        )
        
    result = f"{contexts}\n\n<user_prompt>\n- {message}\n</user_prompt>\n"
    if parts and query_refusal_response:
        result += f"\nIf the question in `<user_prompt>` is not related to the `<documents>`, answer the following query refusal response.\n- {query_refusal_response}\n\n"
    return result


def resolve_runner(provider, model) -> Streamer:
    """LLM runner 생성"""
    if not provider or not model:
        raise BadRequestError("provider와 model 정보를 확인할 수 없습니다.")
    return LLM.from_workspace(provider, model)

