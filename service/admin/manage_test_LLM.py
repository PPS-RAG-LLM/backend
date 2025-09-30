# /home/work/CoreIQ/backend/service/admin/manage_test_LLM.py

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

from pydantic import BaseModel, Field

from pathlib import Path

def _dedup_preserve(seq: list[str]) -> list[str]:
    """중복 제거(순서 보존)."""
    seen = set()
    out: list[str] = []
    for x in seq:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out

def _canon_pdf_list(pdf_list: list[str]) -> list[str]:
    """
    pdf_list를 비교/저장용으로 정규화:
      - 파일명만 추출(Path(...).name)
      - 앞뒤 공백 제거
      - 중복 제거(순서 보존)
      - 소문자 기준 정렬
    """
    names = []
    for f in pdf_list or []:
        name = Path(str(f)).name.strip()
        if name:
            names.append(name)
    names = _dedup_preserve(names)
    names = sorted(names, key=lambda s: s.lower())
    return names

def _to_dict(obj) -> dict:
    """sqlite3.Row, dict 모두 안전하게 dict로 변환."""
    if isinstance(obj, dict):
        return obj
    try:
        # sqlite3.Row는 keys() 제공
        return {k: obj[k] for k in obj.keys()}
    except Exception:
        try:
            return dict(obj)
        except Exception:
            return {}

def _extract_template_texts(tmpl_row, variables: dict[str, str]) -> tuple[str, str, str]:
    """
    템플릿 레코드( Row / dict 모두 지원 )에서 system_prompt(content), sub_content, name 추출 + 치환.
    반환: (system_prompt_text, user_prompt_text, template_name)
    """
    from service.admin.manage_admin_LLM import _fill_template  # 순환 import 방지용
    td = _to_dict(tmpl_row)
    system_raw = (td.get("content") or td.get("system_prompt") or "").strip()
    user_raw   = (td.get("sub_content") or td.get("user_prompt") or "").strip()
    name       = (td.get("name") or td.get("template_name") or "untitled").strip()

    system_filled = _fill_template(system_raw, variables or {})
    user_filled   = _fill_template(user_raw, variables or {})

    return system_filled, user_filled, name

def _acc_from_prompt_and_answer(prompt_text: str, answer_text: str) -> float:
    p = _token_set(prompt_text)
    a = _token_set(answer_text)
    if not p:
        return 0.0
    return round(100.0 * len(p & a) / len(p), 2)

def _row_to_dict(row) -> dict:
    """sqlite3.Row, SQLAlchemy Row 등 다양한 row를 dict로 안정 변환."""
    if row is None:
        return {}
    if isinstance(row, dict):
        return row
    # sqlite3.Row: keys() 지원
    try:
        return {k: row[k] for k in row.keys()}
    except Exception:
        # 최후의 보루
        try:
            return dict(row)
        except Exception:
            return {}


# Reuse DB and prompt helpers from admin service
from service.admin.manage_admin_LLM import (
    _connect,
    _json,
    _fetch_prompt_full,
    _fill_template,
    _simple_generate,
    _lookup_model_by_name,
    _norm_category,
    _migrate_llm_models_if_needed,
)

# RAG 테스트 세션 유틸(이미 구현돼 있다고 가정)
from service.admin.manage_vator_DB import (
    RAGSearchRequest,
    create_test_session,
    get_test_session,
    ingest_test_pdfs,
    search_documents_test,
    drop_test_session,
)

logger = logging.getLogger(__name__)

VAL_DIR = "./storage/val_data"


def _ensure_val_dir():
    try:
        os.makedirs(VAL_DIR, exist_ok=True)
    except Exception:
        logger.exception("failed to create val_data directory")


# ===========================
# Pydantic models
# ===========================
class RunEvalBody(BaseModel):
    category: str = Field(..., description="qa | qna | doc_gen | summary")
    subcategory: Optional[str] = Field(None, description="세부테스크(= template.name)")
    promptId: int = Field(..., description="system_prompt_template.id")
    modelName: Optional[str] = Field(None, description="명시 모델명(없으면 테스크 기본 선정)")
    userPrompt: Optional[str] = Field(None, description="사용자 추가 프롬프트")
    ragRefs: Optional[List[str]] = Field(default=None, description="['milvus://collection/id', 'file://...']")
    max_tokens: int = 512
    temperature: float = 0.7


class EvalQuery(BaseModel):
    category: str
    subcategory: Optional[str] = None
    modelName: Optional[str] = None
    userPrompt: Optional[str] = None
    promptId: Optional[int] = None         # doc_gen일 때만 실제 필터링
    pdfList: Optional[List[str]] = None    # 파일명 리스트(순서 무시, 정확 일치)


class DefaultModelBody(BaseModel):
    category: str
    subcategory: Optional[str] = None
    modelName: str


class SelectModelQuery(BaseModel):
    category: str
    subcategory: Optional[str] = None  # template.name (예: doc_gen의 세부 테스크명)


# ===========================
# Helpers
# ===========================
def _token_set(s: str) -> set[str]:
    if not s:
        return set()
    toks = re.findall(r"[0-9A-Za-z가-힣]+", s.lower())
    return set(toks)


def _acc_from_tokens(a: str, b: str) -> float:
    """간단 토큰 Jaccard 기반 % (ROUGE 미설치 시 폴백)"""
    A, B = _token_set(a), _token_set(b)
    if not A or not B:
        return 0.0
    inter = len(A & B)
    union = len(A | B)
    return round(100.0 * inter / max(1, union), 2)


def _safe_json_loads(s: Optional[str], default) -> Any:
    try:
        return json.loads(s or "")
    except Exception:
        return default


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


# 기존 _extract_template_texts(tmpl, variables) 자리에 넣어주세요.
# 기존 _extract_template_texts(tmpl, variables) 자리에 넣어주세요.
def _extract_template_texts(tmpl, variables: dict[str, str]):
    """
    tmpl 은 sqlite3.Row / dict 등일 수 있음 -> dict로 변환 후
    - system(=content 또는 system_prompt)
    - user(=sub_content 또는 user_prompt)
    - name(=name 또는 template_name)
    를 유연하게 추출.
    """
    d = _row_to_dict(tmpl)

    # system prompt 본문
    system_raw = (
        d.get("content") or
        d.get("system_prompt") or
        d.get("systemPrompt") or
        ""
    )
    # user prompt(서브)
    user_raw = (
        d.get("sub_content") or
        d.get("user_prompt") or
        d.get("userPrompt") or
        ""
    )
    # 템플릿 명
    name = (
        d.get("name") or
        d.get("template_name") or
        d.get("templateName") or
        "untitled"
    )

    system_prompt_text = _fill_template(system_raw, variables or {})
    user_prompt_text = _fill_template(user_raw, variables or {})
    return system_prompt_text, user_prompt_text, name

def _normalize_pdf_names(pdf_list: List[str]) -> List[str]:
    return sorted([Path(x).name.strip() for x in (pdf_list or []) if str(x).strip()])


def _same_pdf_list(a: Optional[str], b_list: Optional[List[str]]) -> bool:
    try:
        a_list = _safe_json_loads(a, [])
        return _normalize_pdf_names(a_list) == _normalize_pdf_names(b_list or [])
    except Exception:
        return False

def select_model_for_task(category: str, subcategory: Optional[str] = None) -> Optional[str]:
    from service.admin.manage_admin_LLM import _connect, _norm_category

    cat = _norm_category(category)
    sub = (subcategory or "").strip().lower() or None
    conn = _connect()
    cur = conn.cursor()
    try:
        # 1) explicit default mapping
        cur.execute(
            """
            SELECT m.name
              FROM llm_task_defaults d JOIN llm_models m ON m.id=d.model_id
             WHERE d.category=? AND IFNULL(d.subcategory,'')=IFNULL(?, '')
             LIMIT 1
            """,
            (cat, sub),
        )
        r = cur.fetchone()
        if r:
            return r[0]

        # 2) task-specific active model
        cur.execute(
            """
            SELECT name FROM llm_models
             WHERE is_active=1 AND category=?
             ORDER BY trained_at DESC, id DESC LIMIT 1
            """,
            (cat,),
        )
        r = cur.fetchone()
        if r:
            return r[0]

        # 3) fallback base(all)
        cur.execute(
            """
            SELECT name FROM llm_models
             WHERE is_active=1 AND category='all'
             ORDER BY trained_at DESC, id DESC LIMIT 1
            """
        )
        r = cur.fetchone()
        return r[0] if r else None
    finally:
        conn.close()

# ===========================
# Core APIs (service)
# ===========================
def run_eval_once(body: RunEvalBody) -> Dict[str, Any]:
    from service.admin.manage_admin_LLM import (
        _connect, _fetch_prompt_full, _simple_generate, _lookup_model_by_name,
        _norm_category
    )

    category = _norm_category(body.category)
    subcat = (body.subcategory or "").strip().lower() or None

    # 1) 템플릿 로드 + 변수 치환
    tmpl, _ = _fetch_prompt_full(body.promptId)
    system_prompt_text, user_prompt_text_from_tmpl, tmpl_name = _extract_template_texts(tmpl, {})
    user_prompt_text = (body.userPrompt or "").strip()
    prompt_text = (system_prompt_text + ("\n" + (user_prompt_text or user_prompt_text_from_tmpl) if (user_prompt_text or user_prompt_text_from_tmpl) else "")).strip()

    # 2) 모델 선택
    if body.modelName:
        model_name = body.modelName
    else:
        model_name = select_model_for_task(category, (subcat or tmpl_name))
        if not model_name:
            return {"success": False, "error": "기본/활성/베이스 모델을 찾을 수 없습니다. 먼저 기본 모델을 지정하거나 모델을 로드하세요."}

    # 3) 생성
    answer = _infer_answer(prompt_text, model_name, body.max_tokens, body.temperature)

    # 4) acc 계산(간이: prompt 대비 토큰 겹침)
    acc = _acc_from_prompt_and_answer(prompt_text, answer)

    # 5) 메타/매핑 찾기 + 저장
    conn = _connect()
    cur = conn.cursor()
    try:
        row = _lookup_model_by_name(model_name)
        llm_id = int(row["id"]) if row else None

        # 매핑
        mapping_id = _find_mapping_id(conn, body.promptId, model_name)

        # ragRefs 중복 제거
        _deduped_rag = _dedup_preserve([str(x).strip() for x in (body.ragRefs or []) if str(x).strip()])
        rag_json = json.dumps(_deduped_rag, ensure_ascii=False)

        cur.execute("""
            INSERT INTO llm_eval_runs(mapping_id, llm_id, prompt_id, category, subcategory, model_name,
                                      prompt_text, user_prompt, rag_refs, answer_text, acc_score, meta, pdf_list)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            mapping_id, llm_id, body.promptId, category, (subcat or tmpl_name), model_name,
            prompt_text, user_prompt_text, rag_json, answer, acc, None, json.dumps([], ensure_ascii=False)
        ))
        conn.commit()
        run_id = int(cur.lastrowid)
    except Exception:
        logging.getLogger(__name__).exception("failed to save eval run")
        return {"success": False, "error": "평가 결과 저장 실패"}
    finally:
        conn.close()

    return {
        "success": True,
        "runId": run_id,
        "category": category,
        "subcategory": (subcat or tmpl_name),
        "modelName": model_name,
        "promptId": body.promptId,
        "answer": answer,
        "acc": acc,
    }


def list_eval_runs(q: EvalQuery) -> Dict[str, Any]:
    from service.admin.manage_admin_LLM import _connect, _norm_category

    cat = _norm_category(q.category)
    sub = (q.subcategory or "").strip().lower() or None

    conn = _connect()
    cur = conn.cursor()
    try:
        sql = """
          SELECT id, mapping_id, llm_id, prompt_id, category, subcategory, model_name,
                 acc_score, created_at, user_prompt, prompt_text, answer_text, rag_refs, pdf_list
            FROM llm_eval_runs
           WHERE category=?
        """
        params: List[Any] = [cat]
        if sub is not None:
            sql += " AND lower(ifnull(subcategory,'')) = ?"
            params.append(sub)
        if q.modelName:
            sql += " AND model_name = ?"
            params.append(q.modelName)
        if q.userPrompt:
            sql += " AND user_prompt = ?"
            params.append(q.userPrompt)
        sql += " ORDER BY id DESC"  # 전체 조회(제한 제거)

        cur.execute(sql, tuple(params))
        rows = cur.fetchall()
        items = []
        for r in rows:
            # ragRefs 정리
            try:
                rag_refs_raw = json.loads(r["rag_refs"] or "[]")
            except Exception:
                rag_refs_raw = []
            rag_refs = _dedup_preserve([str(x).strip() for x in rag_refs_raw if str(x).strip()])

            # pdf_list 정규화
            try:
                pdf_raw = json.loads((r["pdf_list"] or "[]"))
            except Exception:
                pdf_raw = []
            pdf_list = _canon_pdf_list(pdf_raw)

            items.append({
                "id": r["id"],
                "mappingId": r["mapping_id"],
                "llmId": r["llm_id"],
                "promptId": r["prompt_id"],
                "category": r["category"],
                "subcategory": r["subcategory"],
                "modelName": r["model_name"],
                "acc": r["acc_score"],
                "createdAt": r["created_at"],
                "userPrompt": r["user_prompt"],
                "promptText": r["prompt_text"],
                "answerText": r["answer_text"],
                "ragRefs": rag_refs,
                "pdfList": pdf_list,
            })
        return {"success": True, "total": len(items), "items": items}
    except Exception:
        logging.getLogger(__name__).exception("failed to list eval runs")
        return {"success": False, "error": "평가 결과 조회 실패"}
    finally:
        conn.close()

def set_default_model(body: DefaultModelBody) -> Dict[str, Any]:
    _migrate_llm_models_if_needed()
    cat = _norm_category(body.category)
    sub = (body.subcategory or "").strip().lower() or None
    conn = _connect()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT id FROM llm_eval_runs
            WHERE category=?
            AND IFNULL(LOWER(subcategory),'') = IFNULL(LOWER(?),'')
            AND model_name=?
            AND prompt_id=?
            AND IFNULL(user_prompt,'') = IFNULL(?, '')
            AND IFNULL(pdf_list,'[]') = ?
            ORDER BY id DESC LIMIT 1
        """, (category, subcategory, model_name, prompt_id, user_prompt, pdf_json))
        row = cur.fetchone()
        if not row:
            return {"success": True, "skipped": True, "reason": "already exists", "runId": int(row["id"])}
        model_id = int(row[0])
        cur.execute(
            """
            INSERT INTO llm_task_defaults(category, subcategory, model_id)
            VALUES(?,?,?)
            ON CONFLICT(category, IFNULL(subcategory,'')) DO UPDATE SET model_id=excluded.model_id, updated_at=CURRENT_TIMESTAMP
            """,
            (cat, sub, model_id),
        )
        conn.commit()
        return {"success": True, "category": cat, "subcategory": sub, "modelId": model_id, "modelName": body.modelName}
    finally:
        conn.close()


def get_default_model(category: str, subcategory: Optional[str] = None) -> Dict[str, Any]:
    cat = _norm_category(category)
    sub = (subcategory or "").strip().lower() or None
    conn = _connect()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT m.id, m.name FROM llm_task_defaults d
            JOIN llm_models m ON m.id=d.model_id
            WHERE d.category=? AND IFNULL(d.subcategory,'')=IFNULL(?, '')
            LIMIT 1
            """,
            (cat, sub),
        )
        row = cur.fetchone()
        model = {"id": row[0], "name": row[1]} if row else None
        return {"category": cat, "subcategory": sub, "model": model}
    finally:
        conn.close()


def get_selected_model(q: SelectModelQuery) -> Dict[str, Any]:
    category = _norm_category(q.category)
    subcat = (q.subcategory or "").strip().lower() or None

    conn = _connect()
    cur = conn.cursor()
    try:
        # 1) default 템플릿 선택
        if subcat:
            cur.execute(
                """
                SELECT id, name FROM system_prompt_template
                 WHERE category=? AND lower(name)=? AND ifnull(is_default,0)=1 AND ifnull(is_active,1)=1
                 ORDER BY id DESC LIMIT 1
            """,
                (category, subcat),
            )
        else:
            cur.execute(
                """
                SELECT id, name FROM system_prompt_template
                 WHERE category=? AND ifnull(is_default,0)=1 AND ifnull(is_active,1)=1
                 ORDER BY id DESC LIMIT 1
            """,
                (category,),
            )
        tmpl = cur.fetchone()
        if not tmpl:
            return {"category": category, "subcategory": subcat, "default": None, "note": "기본 템플릿 없음"}

        prompt_id = int(tmpl["id"])

        # 2) prompt 매핑 중 최고 rouge_score 1건
        cur.execute(
            """
            SELECT llm_id, prompt_id, rouge_score
              FROM llm_prompt_mapping
             WHERE prompt_id=?
             ORDER BY IFNULL(rouge_score, -1) DESC, llm_id DESC
             LIMIT 1
        """,
            (prompt_id,),
        )
        mp = cur.fetchone()
        if not mp:
            return {"category": category, "subcategory": tmpl["name"], "default": None, "note": "프롬프트-모델 매핑 없음"}

        # 3) llm_models 메타 (is_active는 강제 안함 — 조회용)
        cur.execute(
            """
            SELECT id, name, provider, type, model_path, mather_path, category, is_active, trained_at, created_at
              FROM llm_models WHERE id=?
        """,
            (mp["llm_id"],),
        )
        mdl = cur.fetchone()
        if not mdl:
            return {"category": category, "subcategory": tmpl["name"], "default": None, "note": "llm_models에 모델 없음"}

        # 스키마 제약 확인
        cur.execute(
            """
            SELECT COUNT(*) AS cnt
              FROM system_prompt_template
             WHERE category=? AND ifnull(is_default,0)=1 AND ifnull(is_active,1)=1
               AND (? IS NULL OR lower(name)=?)
        """,
            (category, subcat, subcat),
        )
        cnt = (cur.fetchone() or {"cnt": 0})["cnt"]

        return {
            "category": category,
            "subcategory": (tmpl["name"] or None),
            "default": {
                "promptId": prompt_id,
                "llmId": mdl["id"],
                "modelName": mdl["name"],
                "type": mdl["type"],
                "modelPath": mdl["model_path"],
                "matherPath": mdl["mather_path"],
                "rougeScore": mp["rouge_score"],
            },
            "note": ("default 템플릿 다수" if (cnt and cnt > 1) else None),
        }
    finally:
        conn.close()


# ===========================
# Ensure-run with uploads
# ===========================
# service/admin/manage_test_LLM.py 안의 기존 ensure_run_if_empty_uploaded를 이 버전으로 교체

async def ensure_run_if_empty_uploaded(
    *,
    sid: str,
    category: str,
    subcategory: Optional[str] = None,

    # ✅ 파라미터 별칭 모두 허용
    model_name: Optional[str] = None,
    modelName: Optional[str] = None,

    user_prompt: Optional[str] = None,
    userPrompt: Optional[str] = None,

    prompt_id: Optional[int] = None,
    promptId: Optional[int] = None,

    uploaded_pdf_names: Optional[List[str]] = None,
    pdf_list: Optional[List[str]] = None,
    uploaded_files: Optional[List[str]] = None,

    top_k: int = 5,
    user_level: int = 1,
    search_type: Optional[str] = None,
) -> Dict[str, Any]:
    """
    1) 동일 조건(카테고리/서브카테고리/모델/유저프롬프트/프롬프트ID/pdf_list 정규화)이 이미 저장되어 있으면 스킵
    2) 없으면: 세션 컬렉션(sid)만 대상으로 RAG 검색 → LLM 생성 → Rouge(L)-F1 * 100 = acc_score → 저장
    - 라우터/클라이언트가 snake_case 또는 camelCase 어느 쪽을 보내도 동작하도록 별칭 모두 지원
    """
    from service.admin.manage_admin_LLM import (
        _connect, _lookup_model_by_name, _simple_generate, _norm_category, _fetch_prompt_full
    )
    from service.admin.manage_vator_DB import RAGSearchRequest, search_documents_test

    # ---- 별칭 정리 ----
    model_name = (model_name or modelName or "").strip() or None
    user_prompt = (user_prompt or userPrompt or "").strip()
    prompt_id = int(prompt_id if prompt_id is not None else (promptId if promptId is not None else 0))
    canon_pdfs = _canon_pdf_list(
        (uploaded_pdf_names or []) + (pdf_list or []) + (uploaded_files or [])
    )

    if not prompt_id:
        return {"success": False, "error": "prompt_id (또는 promptId)가 필요합니다."}

    # --- 입력 정리 ---
    cat = _norm_category(category)
    sub = (subcategory or "").strip().lower() or None
    pdf_json = json.dumps(canon_pdfs, ensure_ascii=False)

    # --- 이미 동일 조건이 저장되어 있는지 검사 ---
    conn = _connect()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT id FROM llm_eval_runs
             WHERE category=?
               AND IFNULL(LOWER(subcategory),'') = IFNULL(LOWER(?),'')
               AND model_name=?
               AND prompt_id=?
               AND IFNULL(user_prompt,'') = IFNULL(?, '')
               AND IFNULL(pdf_list,'[]') = ?
             ORDER BY id DESC LIMIT 1
        """, (cat, sub, (model_name or ""), int(prompt_id), user_prompt, pdf_json))
        already = cur.fetchone()
        if already:
            return {"success": True, "skipped": True, "reason": "already exists", "runId": int(already["id"])}

        # --- 템플릿 로드 및 프롬프트 구성 ---
        tmpl, _ = _fetch_prompt_full(prompt_id)
        system_prompt_text, user_prompt_text_from_tmpl, tmpl_name = _extract_template_texts(tmpl, {})
        if not model_name:
            # 기본 선택
            model_name = select_model_for_task(cat, (sub or tmpl_name))
            if not model_name:
                return {"success": False, "error": "기본/활성/베이스 모델을 찾을 수 없습니다. 먼저 기본 모델을 지정하거나 모델을 로드하세요."}

        merged_user = (user_prompt or user_prompt_text_from_tmpl)
        base_prompt_text = (system_prompt_text + ("\n" + merged_user if merged_user else "")).strip()

        # --- RAG: 세션 컬렉션에서만 검색 ---
        task_for_rag = cat if cat in ("doc_gen", "summary", "qna") else "qna"
        req = RAGSearchRequest(query=(user_prompt or tmpl_name or "검색"), top_k=top_k, user_level=user_level, task_type=task_for_rag, model=None)
        rag_res = await search_documents_test(req, sid=sid, search_type_override=search_type)
        hits = rag_res.get("hits", []) if isinstance(rag_res, dict) else []

        # 문서 단위 milvus:// dedup
        milvus_refs: list[str] = []
        seen_docs = set()
        for h in hits:
            doc_id = h.get("doc_id")
            if not doc_id:
                p = h.get("path") or ""
                doc_id = Path(p).stem if p else None
            if not doc_id:
                continue
            key = f"milvus://{sid}/{doc_id}"
            if key not in seen_docs:
                seen_docs.add(key)
                milvus_refs.append(key)
        file_refs = [f"file://{nm}" for nm in canon_pdfs]
        rag_refs = _dedup_preserve(milvus_refs + file_refs)
        rag_json = json.dumps(rag_refs, ensure_ascii=False)

        # 컨텍스트 조립
        context = "\n---\n".join([h.get("snippet") or "" for h in hits if (h.get("snippet"))]).strip()
        full_prompt = base_prompt_text
        if context:
            full_prompt = f"{base_prompt_text}\n\n[CONTEXT]\n{context}"

        # --- LLM 생성 ---
        answer = _infer_answer(base_prompt_text, model_name, max_tokens, temperature)

        # --- Rouge(L)-F1 * 100 = acc_score ---
        try:
            from rouge import Rouge
            _r = Rouge()
            sc = _r.get_scores(answer or "", context or "", avg=True)
            acc = round((sc.get("rouge-l", {}).get("f", 0.0) or 0.0) * 100.0, 2)
        except Exception:
            acc = _acc_from_prompt_and_answer(context or base_prompt_text, answer or "")

        # --- 저장 ---
        row = _lookup_model_by_name(model_name)
        llm_id = int(row["id"]) if row else None
        mapping_id = _find_mapping_id(conn, prompt_id, model_name)

        meta_json = json.dumps({
            "source": "ensure-upload",
            "sid": sid,
            "top_k": top_k,
            "user_level": user_level,
            "rag_settings_used": (rag_res.get("settings_used") if isinstance(rag_res, dict) else None),
        }, ensure_ascii=False)

        cur.execute("""
            INSERT INTO llm_eval_runs(mapping_id, llm_id, prompt_id, category, subcategory, model_name,
                                      prompt_text, user_prompt, rag_refs, answer_text, acc_score, meta, pdf_list)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            mapping_id, llm_id, int(prompt_id), cat, (sub or tmpl_name), model_name,
            full_prompt, user_prompt, rag_json, answer, acc, meta_json, pdf_json
        ))
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
            "acc": acc,
            "ragRefs": rag_refs,
            "pdfList": canon_pdfs,
            "answer": answer,
        }
    except Exception:
        logging.getLogger(__name__).exception("ensure_run_if_empty_uploaded failed")
        return {"success": False, "error": "ensure_run_if_empty_uploaded failed"}
    finally:
        conn.close()

# ==== Inference (backend-local) =================================================
from pathlib import Path
from typing import Optional

# 스트리머 백엔드(둘 중 필요한 것만 import 해도 됨)
try:
    from utils.llms.huggingface.qwen_7b import stream_chat as _qwen_stream
except Exception:
    _qwen_stream = None

try:
    from utils.llms.huggingface.gpt_oss_20b import stream_chat as _gptoss_stream
except Exception:
    _gptoss_stream = None

# admin의 경로 해석 유틸 재사용
from service.admin.manage_admin_LLM import _db_get_model_path as _admin_db_get_model_path
from service.admin.manage_admin_LLM import _resolve_model_fs_path as _admin_resolve_model_fs_path
from service.admin.manage_admin_LLM import _lookup_model_by_name as _admin_lookup_model

_BACKEND_ROOT = Path(__file__).resolve().parents[2]  # .../backend

def _resolve_local_model_dir_for_infer(model_name: str) -> Optional[str]:
    """
    모델 이름 -> 로컬 디렉터리(절대경로) 추출.
    - DB(model_path) 우선, 없으면 FS 규칙 탐색
    - './' 시작이면 backend 루트 기준으로 환원
    """
    p = _admin_db_get_model_path(model_name) or _admin_resolve_model_fs_path(model_name)
    if not p:
        return None
    p = p.replace("\\", "/")
    if p.startswith("./"):
        return str((_BACKEND_ROOT / p.lstrip("./")).resolve())
    return p

def _select_stream_backend(model_name: str):
    """
    모델명/DB provider로 스트리머 선택.
    - qwen -> qwen_7b.stream_chat
    - gpt-oss -> gpt_oss_20b.stream_chat
    - 없으면 None (fallback: _simple_generate)
    """
    row = _admin_lookup_model(model_name)
    prov = (row["provider"] if row and "provider" in row.keys() else "") or ""
    name = (model_name or "")
    key = (prov + " " + name).lower()
    if "qwen" in key and _qwen_stream:
        return _qwen_stream
    if ("gpt-oss" in key or "gpt_oss" in key) and _gptoss_stream:
        return _gptoss_stream
    # 필요시 더 추가
    return None

def _infer_answer(prompt_text: str, model_name: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
    """
    우선 utils 스트리머(로컬 캐시 lru_cache 활용)로 추론 시도 → 실패 시 기존 _simple_generate 폴백.
    """
    try:
        model_dir = _resolve_local_model_dir_for_infer(model_name)
        backend = _select_stream_backend(model_name)
        if backend and model_dir:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt_text},
            ]
            chunks = []
            # stream_chat는 generator를 반환. 모두 모아서 문자열로 합칩니다.
            for token in backend(messages, model_path=model_dir, temperature=temperature, max_new_tokens=max_tokens):
                chunks.append(token)
            out = "".join(chunks).strip()
            if out:
                return out
    except Exception:
        logging.getLogger(__name__).exception("stream backend inference failed; will fallback to _simple_generate")

    # 폴백: (기존) 간단 로컬 생성기
    try:
        return _simple_generate(prompt_text, model_name, max_tokens=max_tokens, temperature=temperature)
    except Exception:
        logging.getLogger(__name__).exception("_simple_generate failed; returning stub text")
        return "⚠️ 로컬 모델이 로드되지 않아 샘플 응답을 반환합니다. (테스트 전용)"
