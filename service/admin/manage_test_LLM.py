# /home/work/CoreIQ/backend/service/admin/manage_test_LLM.py
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

# ===== 경로/세션 고정값 =====
_BACKEND_ROOT = Path(__file__).resolve().parents[2]     # .../backend
VAL_DIR = _BACKEND_ROOT / "storage" / "val_data"        # 공유 원본 저장소
VAL_DIR.mkdir(parents=True, exist_ok=True)

_SHARED_META = VAL_DIR / ".shared_session.json"         # 세션 SID 유지용

# ===== 외부 의존 (기존 모듈 재사용) =====
from service.admin.manage_admin_LLM import (
    _connect,
    _lookup_model_by_name,
    _fill_template,
    _fetch_prompt_full,
    _norm_category,
    _simple_generate,
)

from service.admin.manage_vator_DB import (
    create_test_session,         # (sid, dir) 생성
    get_test_session,            # sid -> meta
    ingest_test_pdfs,            # (sid, paths, task_types)
    delete_test_files_by_names,  # (sid, file_names, task_type)
    search_documents_test,       # (req, sid, search_type_override)
    RAGSearchRequest,
)

# ===== (옵션) 모델 스트리머 (가능 시 스트림, 실패 시 _simple_generate 폴백) =====
try:
    from utils.llms.huggingface.qwen_7b import stream_chat as _qwen_stream
except Exception:
    _qwen_stream = None

try:
    from utils.llms.huggingface.gpt_oss_20b import stream_chat as _gptoss_stream
except Exception:
    _gptoss_stream = None

from service.admin.manage_admin_LLM import (
    _db_get_model_path as _admin_db_get_model_path,
    _resolve_model_fs_path as _admin_resolve_model_fs_path,
)

def _resolve_local_model_dir_for_infer(model_name: str) -> Optional[str]:
    p = _admin_db_get_model_path(model_name) or _admin_resolve_model_fs_path(model_name)
    if not p:
        return None
    p = p.replace("\\", "/")
    if p.startswith("./"):
        return str((_BACKEND_ROOT / p.lstrip("./")).resolve())
    return p

def _select_stream_backend(model_name: str):
    row = _lookup_model_by_name(model_name)
    prov = (row["provider"] if row and "provider" in row.keys() else "") or ""
    name = (model_name or "")
    key = (prov + " " + name).lower()
    if "qwen" in key and _qwen_stream:
        return _qwen_stream
    if ("gpt-oss" in key or "gpt_oss" in key) and _gptoss_stream:
        return _gptoss_stream
    return None

def _infer_answer(prompt_text: str, model_name: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
    try:
        model_dir = _resolve_local_model_dir_for_infer(model_name)
        backend = _select_stream_backend(model_name)
        if backend and model_dir:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt_text},
            ]
            chunks: List[str] = []
            for token in backend(messages, model_path=model_dir, temperature=temperature, max_new_tokens=max_tokens):
                chunks.append(token)
            out = "".join(chunks).strip()
            if out:
                return out
    except Exception:
        logger.exception("stream backend inference failed; fallback to _simple_generate")
    try:
        return _simple_generate(prompt_text, model_name, max_tokens=max_tokens, temperature=temperature)
    except Exception:
        logger.exception("_simple_generate failed; returning stub text")
        return "⚠️ 로컬 모델이 로드되지 않아 샘플 응답을 반환합니다. (테스트 전용)"

# ===== 유틸 =====
def _canon_pdf_list(pdf_list: List[str]) -> List[str]:
    names = []
    for f in pdf_list or []:
        name = Path(str(f)).name.strip()
        if name:
            names.append(name)
    # 순서보존 중복제거
    seen = set()
    out = []
    for n in names:
        if n in seen:
            continue
        seen.add(n)
        out.append(n)
    # 소문자 정렬 (표준화)
    out = sorted(out, key=lambda s: s.lower())
    return out

def _token_set(s: str) -> set[str]:
    import re
    if not s:
        return set()
    toks = re.findall(r"[0-9A-Za-z가-힣]+", s.lower())
    return set(toks)

def _acc_overlap(a: str, b: str) -> float:
    A, B = _token_set(a), _token_set(b)
    if not A or not B:
        return 0.0
    inter = len(A & B)
    union = len(A | B)
    return round(100.0 * inter / max(1, union), 2)

# ===== 공유 세션 보장 =====
def _load_shared_sid_from_disk() -> Optional[str]:
    try:
        if _SHARED_META.is_file():
            data = json.loads(_SHARED_META.read_text(encoding="utf-8"))
            sid = data.get("sid")
            if sid and isinstance(sid, str):
                return sid
    except Exception:
        logger.exception("failed to load shared sid")
    return None

def _save_shared_sid_to_disk(sid: str) -> None:
    try:
        _SHARED_META.write_text(json.dumps({"sid": sid}, ensure_ascii=False), encoding="utf-8")
    except Exception:
        logger.exception("failed to save shared sid")

def _ensure_shared_session() -> str:
    """
    공유 세션 SID를 파일로 유지한다.
    - 파일에 없으면 create_test_session()으로 신규 생성 후 저장
    - 재시작해도 동일 SID 사용
    """
    sid = _load_shared_sid_from_disk()
    if sid:
        meta = get_test_session(sid)
        if meta:
            return sid
        # 세션 메타가 사라졌으면 재생성
    meta = create_test_session()
    if isinstance(meta, dict):
        sid = meta.get("sid") or meta.get("id") or meta.get("session_id")
    elif isinstance(meta, (list, tuple)) and len(meta) >= 1:
        sid = meta[0]
    else:
        sid = None
    if not sid:
        import uuid
        sid = f"shared-{uuid.uuid4().hex[:8]}"
    _save_shared_sid_to_disk(sid)
    return sid

# ===== 파일 관리 =====
def list_shared_files() -> Dict[str, Any]:
    files = []
    for p in sorted(VAL_DIR.glob("*")):
        if p.is_file() and not p.name.startswith("."):
            files.append(p.name)
    return {"success": True, "files": files, "dir": str(VAL_DIR)}

async def upload_shared_files(mem_files: List[tuple[str, bytes]]) -> Dict[str, Any]:
    """
    mem_files: [(filename, bytes), ...]
    - VAL_DIR에 파일 저장(동일 파일명 존재 시 덮어씌움)
    - 공유 세션(sid)에 인제스트
    """
    saved: List[str] = []
    for name, data in mem_files:
        name = Path(name).name.strip()
        if not name:
            continue
        dst = VAL_DIR / name
        dst.write_bytes(data)
        saved.append(str(dst))

    if not saved:
        return {"success": False, "error": "no valid files"}

    sid = _ensure_shared_session()
    # 인제스트 (doc_gen/summary/qna 모두 검색 가능하도록 task_types=None)
    res = await ingest_test_pdfs(sid, saved, task_types=None)
    return {
        "success": True,
        "sid": sid,
        "saved": [Path(p).name for p in saved],
        "ingest": res,
    }

async def delete_shared_files(file_names: List[str]) -> Dict[str, Any]:
    names = [Path(n).name.strip() for n in (file_names or []) if str(n).strip()]
    if not names:
        return {"success": False, "error": "fileNames empty"}

    # 디스크 삭제
    removed: List[str] = []
    for n in names:
        f = VAL_DIR / n
        try:
            if f.is_file():
                f.unlink()
                removed.append(n)
        except Exception:
            logger.exception("failed to remove file: %s", n)

    sid = _ensure_shared_session()
    # 인덱스에서도 제거
    res = await delete_test_files_by_names(sid, file_names=removed, task_type=None)
    return {"success": True, "sid": sid, "removed": removed, "index": res}

# ===== 모델 선택 (기본 템플릿/매핑 기반) =====
def _select_model_for_task(category: str, subcategory: Optional[str]) -> Optional[str]:
    cat = _norm_category(category)
    sub = (subcategory or "").strip().lower() or None
    conn = _connect()
    cur = conn.cursor()
    try:
        # 1) default 템플릿 찾기
        if sub:
            cur.execute(
                """
                SELECT id, name FROM system_prompt_template
                 WHERE category=? AND lower(name)=? AND ifnull(is_default,0)=1 AND ifnull(is_active,1)=1
                 ORDER BY id DESC LIMIT 1
                """,
                (cat, sub),
            )
        else:
            cur.execute(
                """
                SELECT id, name FROM system_prompt_template
                 WHERE category=? AND ifnull(is_default,0)=1 AND ifnull(is_active,1)=1
                 ORDER BY id DESC LIMIT 1
                """,
                (cat,),
            )
        tmpl = cur.fetchone()
        if not tmpl:
            # 2) 카테고리 활성 모델
            cur.execute(
                """
                SELECT name FROM llm_models
                 WHERE is_active=1 AND (category=? OR category='all')
                 ORDER BY trained_at DESC, id DESC LIMIT 1
                """,
                (cat,),
            )
            r = cur.fetchone()
            return r[0] if r else None

        prompt_id = int(tmpl["id"])
        # 3) 해당 템플릿 매핑 중 최고 rouge
        cur.execute(
            """
            SELECT llm_id FROM llm_prompt_mapping
             WHERE prompt_id=?
             ORDER BY IFNULL(rouge_score,-1) DESC, llm_id DESC
             LIMIT 1
            """,
            (prompt_id,),
        )
        mp = cur.fetchone()
        if not mp:
            return None
        cur.execute("SELECT name FROM llm_models WHERE id=?", (mp["llm_id"],))
        r = cur.fetchone()
        return r[0] if r else None
    finally:
        conn.close()

# ===== 핵심: 공유 세션 전체에서 평가 실행 =====
async def ensure_eval_on_shared_session(
    *,
    category: str,
    subcategory: Optional[str],
    prompt_id: int,
    model_name: Optional[str],
    user_prompt: Optional[str],
    top_k: int = 5,
    user_level: int = 1,
    search_type: Optional[str] = None,
) -> Dict[str, Any]:
    """
    - 공유 세션(유일) 전체에서 RAG 검색 → LLM 생성 → llm_eval_runs 저장
    - 동일키(카테고리/서브카테고리/프롬프트ID/모델명/**user_prompt/pdf_list**) 완전 일치 시 DB 결과를 재사용(모델 미실행)
    - rag_refs에는 RAG 히트의 '실제 출처'를 저장 (예: milvus://<sid>/<doc_id>)
    """
    cat = _norm_category(category)
    sub = (subcategory or "").strip().lower() or None
    user_prompt = (user_prompt or "").strip()

    # 현재 공유 세션 및 파일 목록(정규화)
    sid = _ensure_shared_session()
    current_files = [p.name for p in VAL_DIR.glob("*") if p.is_file() and not p.name.startswith(".")]
    pdf_list = _canon_pdf_list(current_files)
    pdf_json = json.dumps(pdf_list, ensure_ascii=False)

    conn = _connect()
    cur = conn.cursor()
    try:
        # ===== 재사용 조건에 user_prompt, pdf_list까지 포함 =====
        cur.execute(
            """
            SELECT id, answer_text, acc_score, rag_refs, pdf_list, created_at, user_prompt, prompt_text
              FROM llm_eval_runs
             WHERE category=?
               AND IFNULL(LOWER(subcategory),'') = IFNULL(LOWER(?),'')
               AND prompt_id=?
               AND model_name=?
               AND IFNULL(user_prompt,'') = IFNULL(?, '')
               AND IFNULL(pdf_list,'[]') = ?
             ORDER BY id DESC
             LIMIT 1
            """,
            (cat, sub, int(prompt_id), (model_name or ""), user_prompt, pdf_json),
        )
        old = cur.fetchone()
        if old:
            try:
                prev_rag_refs = json.loads(old["rag_refs"] or "[]")
            except Exception:
                prev_rag_refs = []
            try:
                prev_pdf_list = json.loads(old["pdf_list"] or "[]")
            except Exception:
                prev_pdf_list = []
            return {
                "success": True,
                "skipped": True,
                "reason": "reuse previous answer (all keys incl. userPrompt & pdf_list matched)",
                "runId": int(old["id"]),
                "category": cat,
                "subcategory": sub,
                "modelName": (model_name or ""),
                "promptId": int(prompt_id),
                "answer": old["answer_text"],
                "acc": old["acc_score"],
                "ragRefs": prev_rag_refs,
                "pdfList": prev_pdf_list,
                "sid": sid,
                "createdAt": old["created_at"],
            }

        # ===== 템플릿 구성 =====
        tmpl, _ = _fetch_prompt_full(prompt_id)
        td = {k: tmpl[k] for k in tmpl.keys()} if not isinstance(tmpl, dict) else tmpl
        system_raw = (td.get("content") or td.get("system_prompt") or "").strip()
        user_raw   = (td.get("sub_content") or td.get("user_prompt") or "").strip()
        tmpl_name  = (td.get("name") or td.get("template_name") or "untitled").strip()

        system_prompt_text = _fill_template(system_raw, {})
        user_prompt_text_from_tmpl = _fill_template(user_raw, {})

        if not model_name:
            model_name = _select_model_for_task(cat, (sub or tmpl_name))
            if not model_name:
                return {"success": False, "error": "모델을 찾을 수 없습니다. 기본/활성 모델을 지정하세요."}

        merged_user = (user_prompt or user_prompt_text_from_tmpl)
        base_prompt_text = (system_prompt_text + ("\n" + merged_user if merged_user else "")).strip()

        # ===== RAG 검색 (공유 세션 전체) =====
        task_for_rag = cat if cat in ("doc_gen", "summary", "qna") else "qna"
        req = RAGSearchRequest(
            query=(user_prompt or tmpl_name or "검색"),
            top_k=top_k,
            user_level=user_level,
            task_type=task_for_rag,
            model=None,
        )
        rag_res = await search_documents_test(req, sid=sid, search_type_override=search_type)
        hits = rag_res.get("hits", []) if isinstance(rag_res, dict) else []

        # rag_refs: 실제 출처를 저장
        rag_refs: List[str] = []
        seen = set()
        for h in hits:
            doc_id = h.get("doc_id") or h.get("document_id") or ""
            if not doc_id:
                path = h.get("path") or ""
                doc_id = Path(path).stem if path else ""
            if not doc_id:
                continue
            key = f"milvus://{sid}/{doc_id}"
            if key in seen:
                continue
            seen.add(key)
            rag_refs.append(key)
        rag_json = json.dumps(rag_refs, ensure_ascii=False)

        # 컨텍스트 결합
        context = "\n---\n".join([h.get("snippet") or "" for h in hits if (h.get("snippet"))]).strip()
        full_prompt = base_prompt_text
        if context:
            full_prompt = f"{base_prompt_text}\n\n[CONTEXT]\n{context}"

        # ===== LLM 생성 =====
        answer = _infer_answer(full_prompt, model_name, max_tokens=512, temperature=0.7)

        # ===== 점수: RAG 컨텍스트 기반 + 프롬프트 겹침 혼합 =====
        try:
            from rouge import Rouge
            _r = Rouge()
            sc = _r.get_scores(answer or "", context or "", avg=True)
            rouge_ctx = (sc.get("rouge-l", {}).get("f", 0.0) or 0.0) * 100.0
        except Exception:
            rouge_ctx = _acc_overlap(context, answer)
        overlap_prompt = _acc_overlap(base_prompt_text, answer)
        acc = round(0.7 * rouge_ctx + 0.3 * overlap_prompt, 2)

        # ===== 저장 =====
        mrow = _lookup_model_by_name(model_name)
        llm_id = int(mrow["id"]) if mrow else None

        cur.execute(
            """
            INSERT INTO llm_eval_runs(
              mapping_id, llm_id, prompt_id, category, subcategory, model_name,
              prompt_text, user_prompt, rag_refs, answer_text, acc_score, meta, pdf_list
            )
            VALUES(NULL,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                llm_id,
                int(prompt_id),
                cat,
                (sub or tmpl_name),
                model_name,
                full_prompt,
                user_prompt,
                rag_json,
                answer,
                acc,
                json.dumps({"source": "ensure-on-shared", "sid": sid, "top_k": top_k, "user_level": user_level}, ensure_ascii=False),
                pdf_json,
            ),
        )
        conn.commit()
        run_id = int(cur.lastrowid)

        return {
            "success": True,
            "created": True,
            "runId": run_id,
            "category": cat,
            "subcategory": (sub or tmpl_name),
            "modelName": model_name,
            "promptId": int(prompt_id),
            "answer": answer,
            "acc": acc,
            "ragRefs": rag_refs,
            "pdfList": pdf_list,
            "sid": sid,
        }
    except Exception:
        logger.exception("ensure_eval_on_shared_session failed")
        return {"success": False, "error": "ensure_eval_on_shared_session failed"}
    finally:
        conn.close()

# ===== 과거 답 삭제 =====
def delete_past_runs(
    *,
    run_id: Optional[int] = None,
    category: Optional[str] = None,
    subcategory: Optional[str] = None,
    prompt_id: Optional[int] = None,
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    - run_id가 오면 해당 1건 삭제
    - 아니면 (category, subcategory, prompt_id, model_name) 조합으로 삭제
    - 최소 1가지 조건은 필요
    """
    if not run_id and not any([category, subcategory, prompt_id, model_name]):
        return {"success": False, "error": "no criteria"}

    conn = _connect()
    cur = conn.cursor()
    try:
        if run_id:
            cur.execute("DELETE FROM llm_eval_runs WHERE id=?", (int(run_id),))
            conn.commit()
            return {"success": True, "deleted": cur.rowcount, "by": {"runId": run_id}}

        cat = _norm_category(category) if category else None
        sub = (subcategory or "").strip().lower() if subcategory is not None else None

        sql = "DELETE FROM llm_eval_runs WHERE 1=1"
        params: List[Any] = []
        if cat is not None:
            sql += " AND category=?"
            params.append(cat)
        if sub is not None:
            sql += " AND IFNULL(LOWER(subcategory),'') = IFNULL(LOWER(?),'')"
            params.append(sub)
        if prompt_id is not None:
            sql += " AND prompt_id=?"
            params.append(int(prompt_id))
        if model_name is not None:
            sql += " AND model_name=?"
            params.append(model_name)

        cur.execute(sql, tuple(params))
        conn.commit()
        return {"success": True, "deleted": cur.rowcount, "by": {"category": cat, "subcategory": sub, "promptId": prompt_id, "modelName": model_name}}
    except Exception:
        logger.exception("delete_past_runs failed")
        return {"success": False, "error": "delete_past_runs failed"}
    finally:
        conn.close()
