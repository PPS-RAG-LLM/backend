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
    """
    우선순위:
      1) llm_task_defaults 매핑
      2) 해당 과업 활성 모델(is_active=1)
      3) category='all' 활성 베이스
    """
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
    _ensure_val_dir()

    category = _norm_category(body.category)
    subcat = (body.subcategory or "").strip().lower() or None

    # 1) 템플릿 로드 + 변수 치환
    tmpl, _ = _fetch_prompt_full(body.promptId)
    system_prompt_text, user_prompt_from_tmpl, _tmpl_name = _extract_template_texts(tmpl, {})
    user_prompt_text = (body.userPrompt or "").strip() or user_prompt_from_tmpl
    prompt_text = (system_prompt_text + ("\n" + user_prompt_text if user_prompt_text else "")).strip()

    # 2) 모델 선택
    if body.modelName:
        model_name = body.modelName
    else:
        model_name = select_model_for_task(category, subcat or tmpl_name)
        if not model_name:
            return {"success": False, "error": "기본/활성 모델을 찾을 수 없습니다. 먼저 모델을 지정/활성화하세요."}

    # 3) 생성
    answer = _simple_generate(prompt_text, model_name, body.max_tokens, body.temperature)

    # 4) acc(간단 토큰기반) — 여기서는 RAG 컨텍스트 없이 프롬프트 대비로 계산
    acc = _acc_from_tokens(answer, prompt_text)

    # 5) 저장
    conn = _connect()
    cur = conn.cursor()
    try:
        row = _lookup_model_by_name(model_name)
        llm_id = int(row["id"]) if row else None
        mapping_id = _find_mapping_id(conn, body.promptId, model_name)
        rag_json = json.dumps(body.ragRefs or [], ensure_ascii=False)

        cur.execute(
            """
            INSERT INTO llm_eval_runs(
                mapping_id, llm_id, prompt_id, category, subcategory, model_name,
                prompt_text, user_prompt, rag_refs, answer_text, acc_score, meta, pdf_list
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                mapping_id, llm_id, body.promptId, category, (subcat or tmpl_name), model_name,
                prompt_text, user_prompt_text, rag_json, answer, acc, None, json.dumps([], ensure_ascii=False),
            ),
        )
        conn.commit()
        run_id = int(cur.lastrowid)
    except Exception:
        logger.exception("failed to save eval run")
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
            sql += " AND lower(IFNULL(subcategory,'')) = ?"
            params.append(sub)
        if q.modelName:
            sql += " AND model_name = ?"
            params.append(q.modelName)
        if q.userPrompt:
            sql += " AND user_prompt = ?"
            params.append(q.userPrompt)
        if cat == "doc_gen" and q.promptId is not None:
            sql += " AND prompt_id = ?"
            params.append(int(q.promptId))

        sql += " ORDER BY id DESC"

        cur.execute(sql, tuple(params))
        rows = cur.fetchall()

        # pdf_list 완전 일치(이름 기준, 순서 무시) 필터는 파이썬에서
        want_pdf = _normalize_pdf_names(q.pdfList or [])
        items = []
        for r in rows:
            row_pdf = _safe_json_loads(r["pdf_list"], [])
            if want_pdf and _normalize_pdf_names(row_pdf) != want_pdf:
                continue
            try:
                rag_refs = json.loads(r["rag_refs"] or "[]")
            except Exception:
                rag_refs = []
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
                "pdfList": row_pdf,
            })
        return {"success": True, "total": len(items), "items": items}
    except Exception:
        logger.exception("failed to list eval runs")
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
        cur.execute("SELECT id FROM llm_models WHERE name=?", (body.modelName,))
        row = cur.fetchone()
        if not row:
            return {"success": False, "message": f"모델이 없습니다: {body.modelName}"}
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
async def ensure_run_if_empty_uploaded(
    category: str,
    subcategory: Optional[str],
    model_name: Optional[str],
    user_prompt: Optional[str],
    prompt_id: Optional[int],
    uploaded_files: List[Tuple[str, bytes]],
    sid: Optional[str] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    1) 동일 (category, subcategory, model_name, user_prompt, (doc_gen이면 prompt_id), pdf_list) 가 이미 있으면 그대로 반환
    2) 없으면:
       - (sid가 없으면) 임시 세션 생성 → 업로드 파일 저장 → 인제스트
       - 세션 컬렉션에서 RAG 검색 → 상위 스니펫 합쳐 컨텍스트 구성
       - 템플릿 로드(system/user prompt) + 컨텍스트로 모델 호출
       - answer vs RAG context로 ROUGE(L) 시도, 실패 시 토큰 Jaccard 폴백
       - llm_eval_runs 저장(pdf_list 포함) 후 결과 반환
    """
    cat = _norm_category(category)
    sub = (subcategory or "").strip().lower() or None
    pdf_names = _normalize_pdf_names([fn for (fn, _data) in uploaded_files])

    # 0) 존재 여부 확인
    exists = list_eval_runs(
        EvalQuery(category=cat, subcategory=sub, modelName=model_name, userPrompt=user_prompt, promptId=prompt_id, pdfList=pdf_names)
    )
    if exists.get("success") and exists.get("total", 0) > 0:
        return {"success": True, "skipped": True, "reason": "already-exists", **exists}

    # 1) 템플릿 로드
    if cat == "doc_gen" and not prompt_id:
        return {"success": False, "error": "doc_gen의 경우 promptId가 필요합니다."}
    use_prompt_id = int(prompt_id or 0)

    tmpl, _ = _fetch_prompt_full(use_prompt_id) if use_prompt_id else ({"system_prompt": "", "user_prompt": "", "name": "ad-hoc"}, {})
    system_prompt_text, user_prompt_text_from_tmpl, tmpl_name = _extract_template_texts(tmpl, {})
    user_prompt_text = (user_prompt or "").strip() or user_prompt_text_from_tmpl

    # 2) 모델 선택
    if model_name:
        model = model_name
    else:
        model = select_model_for_task(cat, sub or tmpl_name)
        if not model:
            return {"success": False, "error": "기본/활성 모델을 찾을 수 없습니다. 먼저 모델을 지정/활성화하세요."}

    # 3) 세션 준비 + 파일 저장/인제스트
    created_tmp_session = False
    used_sid = sid
    session_meta = None
    try:
        if not used_sid:
            session_meta = create_test_session()
            used_sid = session_meta.get("sid")
            created_tmp_session = True
        else:
            session_meta = get_test_session(used_sid)
            if not session_meta:
                return {"success": False, "error": "invalid sid"}

        sess_dir = Path(session_meta["dir"])
        sess_dir.mkdir(parents=True, exist_ok=True)

        # 파일 저장
        saved_paths: List[str] = []
        for fn, data in uploaded_files:
            dst = sess_dir / Path(fn).name
            with dst.open("wb") as out:
                out.write(data)
            saved_paths.append(str(dst))

        # 인제스트(해당 카테고리 매핑)
        task_map = {"qa": "qna", "qna": "qna", "doc_gen": "doc_gen", "summary": "summary"}
        task_type = task_map.get(cat, "qna")

        await ingest_test_pdfs(used_sid, saved_paths, task_types=[task_type])

        # 4) RAG 검색
        query_text = user_prompt_text or tmpl_name or "test"
        req = RAGSearchRequest(query=query_text, top_k=int(top_k), user_level=1, task_type=task_type, model=None)
        rag_res = await search_documents_test(req, sid=used_sid, search_type_override=None)
        hits = rag_res.get("hits", []) or []
        # context
        context = "\n---\n".join([h.get("snippet", "") for h in hits if h.get("snippet")])

        # 5) LLM 호출(컨텍스트 주입)
        prompt_text = system_prompt_text
        if user_prompt_text:
            prompt_text = (prompt_text + "\n" + user_prompt_text).strip()
        if context:
            prompt_text = (prompt_text + "\n\n[Context]\n" + context).strip()

        answer = _simple_generate(prompt_text, model, max_tokens=512, temperature=0.7)

        # 6) ROUGE-L 시도 → 실패 시 토큰 폴백
        acc = 0.0
        try:
            from rouge import Rouge  # type: ignore
            rouge = Rouge()
            # 단일 문자열 비교
            scores = rouge.get_scores(answer, context, avg=True)
            acc = round(100.0 * float(scores["rouge-l"]["f"]), 2)
        except Exception:
            acc = _acc_from_tokens(answer, context)

        # 7) 저장
        conn = _connect()
        cur = conn.cursor()
        try:
            row = _lookup_model_by_name(model)
            llm_id = int(row["id"]) if row else None
            mapping_id = _find_mapping_id(conn, use_prompt_id, model) if use_prompt_id else None

            rag_refs = []
            for h in hits:
                did = h.get("doc_id")
                if did:
                    rag_refs.append(f"milvus://{used_sid}/{did}")
            # 파일도 ref에 포함
            rag_refs += [f"file://{Path(x).name}" for x in saved_paths]

            cur.execute(
                """
                INSERT INTO llm_eval_runs(
                    mapping_id, llm_id, prompt_id, category, subcategory, model_name,
                    prompt_text, user_prompt, rag_refs, answer_text, acc_score, meta, pdf_list
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    mapping_id, llm_id, (use_prompt_id or None), cat, (sub or tmpl_name), model,
                    prompt_text, user_prompt_text, json.dumps(rag_refs, ensure_ascii=False),
                    answer, acc, json.dumps({"sid": used_sid, "top_k": top_k}, ensure_ascii=False),
                    json.dumps(pdf_names, ensure_ascii=False),
                ),
            )
            conn.commit()
            run_id = int(cur.lastrowid)
        finally:
            conn.close()

        return {
            "success": True,
            "created": True,
            "runId": run_id,
            "category": cat,
            "subcategory": (sub or tmpl_name),
            "modelName": model,
            "promptId": use_prompt_id or None,
            "pdfList": pdf_names,
            "acc": acc,
            "answer": answer,
            "ragRefsCount": len(rag_refs),
        }
    finally:
        # 임시 세션이면 정리
        if created_tmp_session and used_sid:
            try:
                await drop_test_session(used_sid)
            except Exception:
                logger.exception("failed to drop temp test session")
