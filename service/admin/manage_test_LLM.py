# /home/work/CoreIQ/backend/service/admin/manage_test_LLM.py
from __future__ import annotations

import json
import logging
import os
import re
import uuid
import shutil
import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ===== Paths =====
_BACKEND_ROOT = Path(__file__).resolve().parents[2]   # .../backend
VAL_DIR = _BACKEND_ROOT / "storage" / "val_data"

def _ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)
def _now_iso(): 
    from datetime import datetime
    return datetime.utcnow().isoformat()

# ====== Small utils ======
def _dedup_preserve(seq: List[str]) -> List[str]:
    seen, out = set(), []
    for x in seq:
        if x in seen: 
            continue
        seen.add(x); out.append(x)
    return out

def _canon_pdf_list(pdf_list: List[str]) -> List[str]:
    names = [Path(str(f)).name.strip() for f in (pdf_list or []) if str(f).strip()]
    names = _dedup_preserve(names)
    names.sort(key=lambda s: s.lower())
    return names

def _row_to_dict(row) -> dict:
    if row is None: return {}
    if isinstance(row, dict): return row
    try: return {k: row[k] for k in row.keys()}
    except Exception:
        try: return dict(row)
        except Exception: return {}

def _token_set(s: str) -> set[str]:
    if not s: return set()
    return set(re.findall(r"[0-9A-Za-z가-힣]+", s.lower()))

def _acc_fallback(a: str, b: str) -> float:
    A, B = _token_set(a), _token_set(b)
    if not A or not B: return 0.0
    inter = len(A & B); uni = len(A | B)
    return round(100.0 * inter / max(1, uni), 2)

# ===== Admin helpers we reuse =====
from service.admin.manage_admin_LLM import (
    _connect,
    _json,
    _fetch_prompt_full,
    _fill_template,
    _simple_generate,
    _lookup_model_by_name,
    _norm_category,
)

# ===== Vector test helpers =====
from service.admin.manage_vator_DB import (
    RAGSearchRequest,
    create_test_session,
    ingest_test_pdfs,
    search_documents_test,
    drop_test_session,
)

# ===== Prompt extraction =====
def _extract_template_texts(tmpl, variables: dict[str, str]):
    d = _row_to_dict(tmpl)
    system_raw = d.get("content") or d.get("system_prompt") or d.get("systemPrompt") or ""
    user_raw   = d.get("sub_content") or d.get("user_prompt") or d.get("userPrompt") or ""
    name       = d.get("name") or d.get("template_name") or d.get("templateName") or "untitled"
    return _fill_template(system_raw, variables or {}), _fill_template(user_raw, variables or {}), name

# ===== Select model for task =====
def select_model_for_task(category: str, subcategory: Optional[str] = None) -> Optional[str]:
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
        if r: return r[0]

        cur.execute("""
            SELECT name FROM llm_models
             WHERE is_active=1 AND category=?
             ORDER BY trained_at DESC, id DESC LIMIT 1
        """, (cat,))
        r = cur.fetchone()
        if r: return r[0]

        cur.execute("""
            SELECT name FROM llm_models
             WHERE is_active=1 AND category='all'
             ORDER BY trained_at DESC, id DESC LIMIT 1
        """)
        r = cur.fetchone()
        return r[0] if r else None
    finally:
        conn.close()

# ===== Inference backends =====
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

def _resolve_local_model_dir_for_infer(model_name: str) -> Optional[str]:
    p = _admin_db_get_model_path(model_name) or _admin_resolve_model_fs_path(model_name)
    if not p: return None
    p = p.replace("\\", "/")
    if p.startswith("./"):
        return str((_BACKEND_ROOT / p.lstrip("./")).resolve())
    return p

def _select_stream_backend(model_name: str):
    row = _admin_lookup_model(model_name)
    prov = (row["provider"] if row and "provider" in row.keys() else "") or ""
    key = (prov + " " + (model_name or "")).lower()
    if "qwen" in key and _qwen_stream: return _qwen_stream
    if ("gpt-oss" in key or "gpt_oss" in key) and _gptoss_stream: return _gptoss_stream
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
            out_chunks = []
            for token in backend(messages, model_path=model_dir, temperature=temperature, max_new_tokens=max_tokens):
                out_chunks.append(token)
            out = "".join(out_chunks).strip()
            if out: return out
    except Exception:
        logger.exception("stream backend inference failed; fallback to _simple_generate")
    try:
        return _simple_generate(prompt_text, model_name, max_tokens=max_tokens, temperature=temperature)
    except Exception:
        logger.exception("_simple_generate failed; returning stub")
        return "⚠️ 로컬 모델이 로드되지 않아 샘플 응답을 반환합니다. (테스트 전용)"

# ------------------------------------------------------------------------------
# Ensure-run with uploads (원본은 val_data에 저장, 세션은 일시적으로만 사용)
# ------------------------------------------------------------------------------
async def ensure_run_if_empty_uploaded(
    *,
    category: str,
    subcategory: Optional[str] = None,
    prompt_id: Optional[int] = None,        # doc_gen 권장
    model_name: Optional[str] = None,
    user_prompt: Optional[str] = None,
    uploaded_files: List[Tuple[str, bytes]], # [(filename, bytes), ...]
    top_k: int = 5,
    user_level: int = 1,
) -> Dict[str, Any]:

    cat = _norm_category(category)
    sub = (subcategory or "").strip().lower() or None
    user_prompt = (user_prompt or "").strip()
    if cat == "doc_gen" and not prompt_id:
        return {"success": False, "error": "doc_gen에서는 promptId가 필요합니다."}

    # 템플릿/프롬프트
    if prompt_id:
        tmpl, _ = _fetch_prompt_full(int(prompt_id))
        sys_t, user_t_from_tmpl, tmpl_name = _extract_template_texts(tmpl, {})
    else:
        sys_t, user_t_from_tmpl, tmpl_name = "", "", (sub or "untitled")

    # 모델
    if not model_name:
        model_name = select_model_for_task(cat, (sub or tmpl_name))
        if not model_name:
            return {"success": False, "error": "기본/활성/베이스 모델을 찾을 수 없습니다. 먼저 기본 모델을 지정하거나 로드하세요."}

    canon_pdfs = _canon_pdf_list([fn for (fn, _) in (uploaded_files or [])])
    pdf_json = json.dumps(canon_pdfs, ensure_ascii=False)

    # 동일키 존재 시 DB 재사용
    conn = _connect(); cur = conn.cursor()
    try:
        sql = """
          SELECT id, mapping_id, llm_id, prompt_id, category, subcategory, model_name,
                 acc_score, created_at, user_prompt, prompt_text, answer_text, rag_refs, pdf_list
            FROM llm_eval_runs
           WHERE category=?
             AND IFNULL(LOWER(subcategory),'')=IFNULL(LOWER(?),'')
             AND model_name=? AND IFNULL(user_prompt,'')=IFNULL(?, '')
             AND IFNULL(pdf_list,'[]')=?
        """
        params: List[Any] = [cat, sub, model_name, user_prompt, pdf_json]
        if cat == "doc_gen":
            sql += " AND prompt_id=?"
            params.append(int(prompt_id or 0))
        sql += " ORDER BY id DESC LIMIT 1"
        cur.execute(sql, tuple(params))
        row = cur.fetchone()
        if row:
            r = _row_to_dict(row)
            try: rag_refs = _dedup_preserve([str(x).strip() for x in json.loads(r.get("rag_refs") or "[]") if str(x).strip()])
            except Exception: rag_refs = []
            try: pdf_list = _canon_pdf_list(json.loads(r.get("pdf_list") or "[]"))
            except Exception: pdf_list = canon_pdfs
            return {
                "success": True, "skipped": True, "reason": "already exists",
                "runId": r["id"], "category": r["category"], "subcategory": r["subcategory"],
                "modelName": r["model_name"], "promptId": r.get("prompt_id"),
                "answer": r.get("answer_text"), "acc": r.get("acc_score"),
                "ragRefs": rag_refs, "pdfList": pdf_list, "createdAt": r.get("created_at"),
            }
    finally:
        conn.close()

    # 원본 보관소 저장: storage/val_data/YYYYMMDD/<job_id>/
    today = datetime.datetime.utcnow().strftime("%Y%m%d")
    job_id = uuid.uuid4().hex[:8]
    job_dir = VAL_DIR / today / job_id
    _ensure_dir(job_dir)

    pdf_paths: List[str] = []
    for (fn, data) in (uploaded_files or []):
        name = Path(fn).name
        dst = job_dir / name
        with dst.open("wb") as out:
            out.write(data)
        pdf_paths.append(str(dst))

    # 세션 생성 → 인제스트 → 검색
    sid_meta = create_test_session()
    sid = sid_meta["sid"]
    try:
        await ingest_test_pdfs(sid, pdf_paths, task_types=None)
    except Exception:
        logger.exception("ingest_test_pdfs failed")
    try:
        req = RAGSearchRequest(
            query=(user_prompt or tmpl_name or "검색"),
            top_k=int(top_k), user_level=int(user_level),
            task_type=(cat if cat in ("doc_gen", "summary", "qna") else "qna"),
            model=None,
        )
        rag_res = await search_documents_test(req, sid=sid, search_type_override=None)
    except Exception:
        logger.exception("search_documents_test failed")
        rag_res = {}

    hits = rag_res.get("hits", []) if isinstance(rag_res, dict) else []
    seen_docs, milvus_refs = set(), []
    for h in hits:
        doc_id = h.get("doc_id") or Path(h.get("path") or "").stem or None
        if not doc_id: continue
        key = f"milvus://{sid}/{doc_id}"
        if key in seen_docs: continue
        seen_docs.add(key); milvus_refs.append(key)
    file_refs = [f"file://{nm}" for nm in canon_pdfs]
    rag_refs = _dedup_preserve(milvus_refs + file_refs)
    rag_json = json.dumps(rag_refs, ensure_ascii=False)

    merged_user = user_prompt or user_t_from_tmpl
    base_prompt_text = (sys_t + ("\n" + merged_user if merged_user else "")).strip()
    context = "\n---\n".join([h.get("snippet") or "" for h in hits if (h.get("snippet"))]).strip()
    final_prompt_text = base_prompt_text + (f"\n\n[CONTEXT]\n{context}" if context else "")

    answer = _infer_answer(final_prompt_text, model_name, max_tokens=512, temperature=0.7)

    try:
        from rouge import Rouge
        _r = Rouge()
        sc = _r.get_scores(answer or "", context or "", avg=True)
        acc = round((sc.get("rouge-l", {}).get("f", 0.0) or 0.0) * 100.0, 2)
    except Exception:
        acc = _acc_fallback(context or base_prompt_text, answer or "")

    conn = _connect(); cur = conn.cursor()
    try:
        row = _lookup_model_by_name(model_name)
        llm_id = int(row["id"]) if row else None
        mapping_id = None
        if prompt_id:
            cur.execute(
                "SELECT id FROM llm_models WHERE name=?", (model_name,)
            )
            r = cur.fetchone()
            if r:
                cur.execute(
                    "SELECT id FROM llm_prompt_mapping WHERE prompt_id=? AND llm_id=? ORDER BY id DESC LIMIT 1",
                    (int(prompt_id), int(r[0])),
                )
                m = cur.fetchone()
                mapping_id = int(m[0]) if m else None

        meta_json = json.dumps({
            "source": "ensure-upload",
            "sid": sid,
            "top_k": top_k,
            "user_level": user_level,
            "rag_settings_used": (rag_res.get("settings_used") if isinstance(rag_res, dict) else None),
            "val_dir": str(job_dir),
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
        try: conn.close()
        except Exception: pass

    try:
        await drop_test_session(sid)  # 벡터 컬렉션만 정리. 원본은 val_data 보존.
    except Exception:
        logger.exception("drop_test_session failed")

    return {
        "success": True, "created": True, "runId": run_id,
        "category": cat, "subcategory": (sub or tmpl_name),
        "modelName": model_name, "promptId": (int(prompt_id) if prompt_id else None),
        "answer": answer, "acc": acc,
        "ragRefs": rag_refs, "pdfList": canon_pdfs,
        "valDir": str(job_dir),
    }

# ==============================================================================
#                      NEW: Val-data file list / delete APIs
# ==============================================================================
class DeleteValFilesBody(BaseModel):
    paths: List[str] = Field(..., description="VAL_DIR 기준 상대경로 목록 (예: '20250930/abcd1234/a.pdf')")

ALLOWED_EXTS = {".pdf", ".pptx", ".ppt", ".docx", ".xlsx", ".csv", ".txt", ".md"}

def _is_safe_rel(rel: str) -> bool:
    rel = rel.replace("\\", "/").lstrip("/")
    return not (".." in rel or rel.startswith("/"))

def _under_val(path: Path) -> bool:
    try:
        return str(path.resolve()).startswith(str(VAL_DIR.resolve()))
    except Exception:
        return False

def list_val_files(date: Optional[str] = None, job_id: Optional[str] = None, exts: Optional[List[str]] = None) -> Dict[str, Any]:
    base = VAL_DIR
    if date:
        if not re.fullmatch(r"\d{8}", str(date)):
            return {"success": False, "error": "date 형식은 YYYYMMDD"}
        base = base / str(date)
    if job_id:
        base = base / str(job_id)

    if not base.exists():
        return {"success": True, "items": [], "total": 0}

    exts_set = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in (exts or [])} or ALLOWED_EXTS
    items: List[Dict[str, Any]] = []

    for p in base.rglob("*"):
        if not p.is_file(): 
            continue
        if exts_set and p.suffix.lower() not in exts_set:
            continue
        rel = p.relative_to(VAL_DIR)
        parts = rel.parts
        _date = parts[0] if len(parts) >= 1 else None
        _job  = parts[1] if len(parts) >= 2 else None
        stat = p.stat()
        items.append({
            "name": p.name,
            "relPath": str(rel).replace("\\", "/"),
            "size": stat.st_size,
            "mtime": datetime.datetime.utcfromtimestamp(stat.st_mtime).isoformat(),
            "date": _date,
            "jobId": _job,
            "ext": p.suffix.lower(),
        })

    # 최신 수정순 정렬
    items.sort(key=lambda x: x["mtime"], reverse=True)
    return {"success": True, "items": items, "total": len(items)}

def delete_val_files(paths: List[str]) -> Dict[str, Any]:
    if not paths:
        return {"success": False, "error": "paths 비어있음"}
    ok, failed = [], []
    for rel in paths:
        if not _is_safe_rel(rel):
            failed.append({"path": rel, "reason": "unsafe path"}); continue
        tgt = (VAL_DIR / rel.replace("\\", "/").lstrip("/"))
        if not _under_val(tgt):
            failed.append({"path": rel, "reason": "outside val_data"}); continue
        if not tgt.exists():
            failed.append({"path": rel, "reason": "not found"}); continue
        if tgt.is_dir():
            failed.append({"path": rel, "reason": "is a directory"}); continue
        try:
            tgt.unlink()
            # 상위 빈 폴더 정리(job → date)
            parent = tgt.parent
            for _ in range(2):
                try:
                    if parent.exists() and not any(parent.iterdir()):
                        parent.rmdir()
                        parent = parent.parent
                    else:
                        break
                except Exception:
                    break
            ok.append(rel)
        except Exception as e:
            failed.append({"path": rel, "reason": str(e)})

    return {"success": True, "deleted": ok, "failed": failed}

def delete_val_job(date: Optional[str] = None, job_id: Optional[str] = None, rel_dir: Optional[str] = None) -> Dict[str, Any]:
    # date+job_id 또는 rel_dir(VAL_DIR 상대)을 허용
    if rel_dir:
        if not _is_safe_rel(rel_dir):
            return {"success": False, "error": "unsafe rel_dir"}
        base = (VAL_DIR / rel_dir.replace("\\", "/").lstrip("/"))
    else:
        if not (date and job_id):
            return {"success": False, "error": "date/jobId 또는 rel_dir 필요"}
        if not re.fullmatch(r"\d{8}", str(date)):
            return {"success": False, "error": "date 형식은 YYYYMMDD"}
        base = VAL_DIR / str(date) / str(job_id)

    if not _under_val(base):
        return {"success": False, "error": "outside val_data"}
    if not base.exists():
        return {"success": True, "deleted": 0, "note": "already gone"}

    try:
        cnt = 0
        for p in base.rglob("*"):
            if p.is_file():
                cnt += 1
        shutil.rmtree(base)
        # date 폴더가 비면 정리
        parent = base.parent
        try:
            if parent.exists() and not any(parent.iterdir()):
                parent.rmdir()
        except Exception:
            pass
        return {"success": True, "deleted": cnt, "dir": str(base.relative_to(VAL_DIR))}
    except Exception as e:
        return {"success": False, "error": f"delete failed: {e}"}
