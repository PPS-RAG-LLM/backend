# /home/work/CoreIQ/backend/service/admin/manage_test_LLM.py
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ===== 공용 유틸/DB/LLM 헬퍼 재사용 =====
from service.admin.manage_admin_LLM import (
    _connect,
    _lookup_model_by_name,
    _fetch_prompt_full,
    _fill_template,
    _simple_generate,
    _norm_category,
)

# ===== RAG 파이프라인(서비스 레벨 함수) 재사용 =====
from service.admin.manage_vator_DB import (
    RAGSearchRequest,
    create_test_session,
    ingest_test_pdfs,
    search_documents_test,
    drop_test_session,
)

# =========================
# 로컬 유틸
# =========================
def _row_to_dict(row) -> dict:
    if row is None:
        return {}
    if isinstance(row, dict):
        return row
    try:
        return {k: row[k] for k in row.keys()}
    except Exception:
        try:
            return dict(row)
        except Exception:
            return {}

def _dedup_preserve(seq: List[str]) -> List[str]:
    seen, out = set(), []
    for s in seq:
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out

def _token_set(s: str) -> set[str]:
    if not s:
        return set()
    return set(re.findall(r"[0-9A-Za-z가-힣]+", s.lower()))

def _acc_fallback(reference: str, answer: str) -> float:
    A, B = _token_set(reference), _token_set(answer)
    if not A or not B:
        return 0.0
    inter = len(A & B)
    union = len(A | B)
    return round(100.0 * inter / max(1, union), 2)

def _canon_pdf_list(pdf_list: List[str]) -> List[str]:
    # 파일명만, 공백제거, 중복제거(순서보존), 소문자 정렬
    names = [Path(str(x)).name.strip() for x in (pdf_list or []) if str(x).strip()]
    names = _dedup_preserve(names)
    names = sorted(names, key=lambda s: s.lower())
    return names

def _extract_template_texts(tmpl_row, variables: dict[str, str]) -> tuple[str, str, str]:
    td = _row_to_dict(tmpl_row)
    system_raw = (td.get("content") or td.get("system_prompt") or "").strip()
    user_raw   = (td.get("sub_content") or td.get("user_prompt") or "").strip()
    name       = (td.get("name") or td.get("template_name") or "untitled").strip()
    return _fill_template(system_raw, variables or {}), _fill_template(user_raw, variables or {}), name

def _find_mapping_id(conn, prompt_id: int, model_name: str) -> Optional[int]:
    cur = conn.cursor()
    cur.execute("SELECT id FROM llm_models WHERE name=?", (model_name,))
    r = cur.fetchone()
    llm_id = int(r["id"]) if r else None
    if not llm_id:
        return None
    cur.execute(
        "SELECT id FROM llm_prompt_mapping WHERE prompt_id=? AND llm_id=? ORDER BY id DESC LIMIT 1",
        (prompt_id, llm_id),
    )
    m = cur.fetchone()
    return int(m["id"]) if m else None

def _select_model_for_task(category: str, subcategory: Optional[str]) -> Optional[str]:
    """
    기본/활성/베이스 순으로 모델 선택:
      1) llm_task_defaults(category, subcategory)
      2) llm_models.is_active=1 AND category=?
      3) llm_models.is_active=1 AND category='all'
    """
    cat = _norm_category(category)
    sub = (subcategory or "").strip().lower() or None
    conn = _connect(); cur = conn.cursor()
    try:
        cur.execute("""
            SELECT m.name
              FROM llm_task_defaults d JOIN llm_models m ON m.id=d.model_id
             WHERE d.category=? AND IFNULL(d.subcategory,'')=IFNULL(?, '')
             LIMIT 1
        """, (cat, sub))
        r = cur.fetchone()
        if r:
            return r[0]

        cur.execute("""
            SELECT name FROM llm_models
             WHERE is_active=1 AND category=?
             ORDER BY trained_at DESC, id DESC
             LIMIT 1
        """, (cat,))
        r = cur.fetchone()
        if r:
            return r[0]

        cur.execute("""
            SELECT name FROM llm_models
             WHERE is_active=1 AND category='all'
             ORDER BY trained_at DESC, id DESC
             LIMIT 1
        """)
        r = cur.fetchone()
        return r[0] if r else None
    finally:
        conn.close()

# =========================
# 로컬 추론 (stream 우선, 실패시 간단생성기)
# =========================
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
    _lookup_model_by_name as _admin_lookup_model,
)

_BACKEND_ROOT = Path(__file__).resolve().parents[2]

def _resolve_local_model_dir(model_name: str) -> Optional[str]:
    p = _admin_db_get_model_path(model_name) or _admin_resolve_model_fs_path(model_name)
    if not p:
        return None
    p = p.replace("\\", "/")
    if p.startswith("./"):
        return str((_BACKEND_ROOT / p.lstrip("./")).resolve())
    return p

def _pick_stream_backend(model_name: str):
    row = _admin_lookup_model(model_name)
    prov = (row["provider"] if row and "provider" in row.keys() else "") or ""
    key = (prov + " " + model_name).lower()
    if "qwen" in key and _qwen_stream:
        return _qwen_stream
    if ("gpt-oss" in key or "gpt_oss" in key) and _gptoss_stream:
        return _gptoss_stream
    return None

def _infer_answer(prompt_text: str, model_name: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
    try:
        model_dir = _resolve_local_model_dir(model_name)
        backend = _pick_stream_backend(model_name)
        if backend and model_dir:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt_text},
            ]
            buf: List[str] = []
            for tok in backend(messages, model_path=model_dir, temperature=temperature, max_new_tokens=max_tokens):
                buf.append(tok)
            out = "".join(buf).strip()
            if out:
                return out
    except Exception:
        logger.exception("stream backend inference failed; fallback to _simple_generate()")

    # 폴백
    try:
        return _simple_generate(prompt_text, model_name, max_tokens=max_tokens, temperature=temperature)
    except Exception:
        logger.exception("_simple_generate failed; returning stub text")
        return "⚠️ 로컬 모델이 로드되지 않아 샘플 응답을 반환합니다. (테스트 전용)"

# =========================
# 핵심 서비스: Ensure & Run (세션 내부적으로 생성/정리)
# =========================
async def ensure_run_if_empty_uploaded(
    *,
    category: str,
    subcategory: Optional[str] = None,
    prompt_id: Optional[int] = None,          # doc_gen일 때 권장
    model_name: Optional[str] = None,
    user_prompt: Optional[str] = None,
    uploaded_files: List[Tuple[str, bytes]],   # [(filename, bytes), ...]
    top_k: int = 5,
    user_level: int = 1,
) -> Dict[str, Any]:

    # ---- 입력 정리 ----
    cat = _norm_category(category)
    sub = (subcategory or "").strip().lower() or None
    user_prompt = (user_prompt or "").strip()
    if cat == "doc_gen" and not prompt_id:
        return {"success": False, "error": "doc_gen에서는 promptId가 필요합니다."}

    # 업로드 파일 이름 정규화 (동일키 판단에 사용)
    canon_pdfs = _canon_pdf_list([fn for (fn, _) in (uploaded_files or [])])
    pdf_json = json.dumps(canon_pdfs, ensure_ascii=False)

    # 템플릿 로드(있으면)
    tmpl_name = None
    if prompt_id:
        tmpl, _ = _fetch_prompt_full(int(prompt_id))
        system_prompt_text, user_prompt_text_from_tmpl, tmpl_name = _extract_template_texts(tmpl, {})
    else:
        # 템플릿이 없으면 system/user 프롬프트는 사용자 입력만 사용
        system_prompt_text, user_prompt_text_from_tmpl, tmpl_name = "", "", (sub or "untitled")

    # 모델 선택 (명시 없으면 디폴트 로직)
    if not model_name:
        model_name = _select_model_for_task(cat, (sub or tmpl_name))
        if not model_name:
            return {"success": False, "error": "기본/활성/베이스 모델을 찾을 수 없습니다. 먼저 기본 모델을 지정하거나 로드하세요."}

    # ---- 동일키 존재 여부 체크 ----
    conn = _connect(); cur = conn.cursor()
    try:
        sql = """
          SELECT id, mapping_id, llm_id, prompt_id, category, subcategory, model_name,
                 acc_score, created_at, user_prompt, prompt_text, answer_text, rag_refs, pdf_list
            FROM llm_eval_runs
           WHERE category=?
             AND IFNULL(LOWER(subcategory),'') = IFNULL(LOWER(?),'')
             AND model_name=?
             AND IFNULL(user_prompt,'') = IFNULL(?, '')
             AND IFNULL(pdf_list,'[]') = ?
        """
        params: List[Any] = [cat, sub, model_name, user_prompt, pdf_json]
        if cat == "doc_gen":
            sql += " AND prompt_id=?"
            params.append(int(prompt_id or 0))
        sql += " ORDER BY id DESC LIMIT 1"

        cur.execute(sql, tuple(params))
        row = cur.fetchone()
        if row:
            # 이미 저장된 결과 반환
            r = _row_to_dict(row)
            # rag_refs / pdf_list 정리
            try:
                rag_refs = _dedup_preserve([str(x).strip() for x in json.loads(r.get("rag_refs") or "[]") if str(x).strip()])
            except Exception:
                rag_refs = []
            try:
                pdf_list = _canon_pdf_list(json.loads(r.get("pdf_list") or "[]"))
            except Exception:
                pdf_list = canon_pdfs

            return {
                "success": True,
                "skipped": True,
                "reason": "already exists",
                "runId": r["id"],
                "category": r["category"],
                "subcategory": r["subcategory"],
                "modelName": r["model_name"],
                "promptId": r.get("prompt_id"),
                "answer": r.get("answer_text"),
                "acc": r.get("acc_score"),
                "ragRefs": rag_refs,
                "pdfList": pdf_list,
                "createdAt": r.get("created_at"),
            }
    finally:
        conn.close()

    # ---- 임시 세션 생성 → 파일 저장 → 인제스트 ----
    sid_meta = create_test_session()  # { sid, dir }
    sid = sid_meta["sid"]
    sess_dir = Path(sid_meta["dir"]); sess_dir.mkdir(parents=True, exist_ok=True)

    pdf_paths: List[str] = []
    for (fn, data) in (uploaded_files or []):
        stem = Path(fn).stem
        ext = Path(fn).suffix or ".pdf"
        dst = sess_dir / f"{stem}_{int(time.time())}_{uuid.uuid4().hex[:6]}{ext}"
        with dst.open("wb") as out:
            out.write(data)
        pdf_paths.append(str(dst))

    # 인제스트 (세션 컬렉션에만 들어간다)
    try:
        await ingest_test_pdfs(sid, pdf_paths, task_types=None)
    except Exception:
        logger.exception("ingest_test_pdfs failed")

    # ---- 세션 한정 RAG 검색 ----
    task_for_rag = cat if cat in ("doc_gen", "summary", "qna") else "qna"
    req = RAGSearchRequest(
        query=(user_prompt or tmpl_name or "검색"),
        top_k=int(top_k),
        user_level=int(user_level),
        task_type=task_for_rag,
        model=None,
    )
    try:
        rag_res = await search_documents_test(req, sid=sid, search_type_override=None)
    except Exception:
        logger.exception("search_documents_test failed")
        rag_res = {}

    hits = rag_res.get("hits", []) if isinstance(rag_res, dict) else []

    # 문서단위 ref dedup
    milvus_refs: List[str] = []
    seen_docs = set()
    for h in hits:
        doc_id = h.get("doc_id") or Path(h.get("path") or "").stem or None
        if not doc_id:
            continue
        key = f"milvus://{sid}/{doc_id}"
        if key in seen_docs:
            continue
        seen_docs.add(key)
        milvus_refs.append(key)
    file_refs = [f"file://{nm}" for nm in canon_pdfs]
    rag_refs = _dedup_preserve(milvus_refs + file_refs)
    rag_json = json.dumps(rag_refs, ensure_ascii=False)

    # ---- 프롬프트 구성 ----
    merged_user = user_prompt or user_prompt_text_from_tmpl
    base_prompt_text = (system_prompt_text + ("\n" + merged_user if merged_user else "")).strip()
    context = "\n---\n".join([h.get("snippet") or "" for h in hits if (h.get("snippet"))]).strip()
    final_prompt_text = base_prompt_text + (f"\n\n[CONTEXT]\n{context}" if context else "")

    # ---- LLM 추론 ----
    answer = _infer_answer(final_prompt_text, model_name, max_tokens=512, temperature=0.7)

    # ---- ACC(rouge-l f1 * 100) 계산 ----
    try:
        from rouge import Rouge
        _r = Rouge()
        sc = _r.get_scores(answer or "", context or "", avg=True)
        acc = round((sc.get("rouge-l", {}).get("f", 0.0) or 0.0) * 100.0, 2)
    except Exception:
        acc = _acc_fallback(context or base_prompt_text, answer or "")

    # ---- 저장 ----
    conn = _connect(); cur = conn.cursor()
    try:
        row = _lookup_model_by_name(model_name)
        llm_id = int(row["id"]) if row else None
        mapping_id = _find_mapping_id(conn, int(prompt_id or 0), model_name) if prompt_id else None

        meta_json = json.dumps({
            "source": "ensure-upload",
            "sid": sid,
            "top_k": top_k,
            "user_level": user_level,
            "rag_settings_used": (rag_res.get("settings_used") if isinstance(rag_res, dict) else None),
        }, ensure_ascii=False)

        cur.execute("""
            INSERT INTO llm_eval_runs(
                mapping_id, llm_id, prompt_id, category, subcategory, model_name,
                prompt_text, user_prompt, rag_refs, answer_text, acc_score, meta, pdf_list
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            mapping_id, llm_id, (int(prompt_id) if prompt_id else None), cat, (sub or tmpl_name), model_name,
            final_prompt_text, user_prompt, rag_json, answer, acc, meta_json, pdf_json
        ))
        conn.commit()
        run_id = int(cur.lastrowid)
    finally:
        try:
            conn.close()
        except Exception:
            pass

    # ---- 세션/파일 정리 ----
    try:
        await drop_test_session(sid)
    except Exception:
        logger.exception("drop_test_session failed")
        # 컬렉션만 드롭 실패시 폴더는 별도로 정리 시도
        try:
            shutil.rmtree(sess_dir, ignore_errors=True)
        except Exception:
            logger.exception("session dir cleanup failed")

    return {
        "success": True,
        "created": True,
        "runId": run_id,
        "category": cat,
        "subcategory": (sub or tmpl_name),
        "modelName": model_name,
        "promptId": (int(prompt_id) if prompt_id else None),
        "answer": answer,
        "acc": acc,
        "ragRefs": rag_refs,
        "pdfList": canon_pdfs,
    }
