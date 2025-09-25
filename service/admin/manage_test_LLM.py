# service/admin/manage_test_LLM.py
from __future__ import annotations

import json
import logging
import os
import re
from typing import Optional, Dict, Any, List

from pydantic import BaseModel, Field

# Reuse DB and prompt helpers from admin service (do not duplicate logic)
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

VAL_DIR = "./storage/val_data"

def _ensure_val_dir():
    try:
        os.makedirs(VAL_DIR, exist_ok=True)
    except Exception:
        logging.getLogger(__name__).exception("failed to create val_data directory")

def _ensure_llm_eval_runs_table():
    conn = _connect()
    cur = conn.cursor()
    try:
        cur.execute("""
        CREATE TABLE IF NOT EXISTS llm_eval_runs (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          mapping_id INTEGER,
          llm_id INTEGER,
          prompt_id INTEGER,
          category TEXT NOT NULL,
          subcategory TEXT,
          model_name TEXT,
          prompt_text TEXT NOT NULL,
          user_prompt TEXT,
          rag_refs TEXT,
          answer_text TEXT NOT NULL,
          acc_score REAL NOT NULL DEFAULT 0,
          meta TEXT,
          created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_eval_runs_cat_sub_model ON llm_eval_runs (category, subcategory, model_name)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_eval_runs_prompt ON llm_eval_runs (prompt_id, llm_id, mapping_id)")
        conn.commit()
    except Exception:
        logging.getLogger(__name__).exception("failed to create llm_eval_runs table")
    finally:
        conn.close()

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
    limit: int = 50


class DefaultModelBody(BaseModel):
    category: str
    subcategory: Optional[str] = None
    modelName: str


class SelectModelQuery(BaseModel):
    category: str
    subcategory: Optional[str] = None  # template.name (예: doc_gen의 세부 테스크명)


def _token_set(s: str) -> set[str]:
    """한글/영문/숫자 토큰 단순 분해 → 소문자"""
    if not s:
        return set()
    toks = re.findall(r"[0-9A-Za-z가-힣]+", s.lower())
    return set(toks)

def _acc_from_prompt_and_answer(prompt_text: str, answer_text: str) -> float:
    """
    정답 텍스트가 없으므로 '입력 프롬프트(= system+user+RAG 주입)' 대비
    모델 답변의 토큰 겹침 비율을 acc로 정의.
      acc = |tokens(answer) ∩ tokens(prompt)| / |tokens(prompt)| * 100
    """
    p = _token_set(prompt_text)
    a = _token_set(answer_text)
    if not p:
        return 0.0
    return round(100.0 * len(p & a) / len(p), 2)

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

def run_eval_once(body: RunEvalBody) -> Dict[str, Any]:
    _ensure_val_dir()
    _ensure_llm_eval_runs_table()

    category = _norm_category(body.category)
    subcat = (body.subcategory or "").strip().lower() or None

    # 1) 템플릿 로드 + 변수 치환
    tmpl, _ = _fetch_prompt_full(body.promptId)
    system_prompt_text = _fill_template(tmpl["content"], {})
    user_prompt_text = (body.userPrompt or "").strip()
    prompt_text = (system_prompt_text + ("\n" + user_prompt_text if user_prompt_text else "")).strip()

    # 2) 모델 선택
    if body.modelName:
        model_name = body.modelName
    else:
        model_name = select_model_for_task(category, subcat or tmpl["name"])
        if not model_name:
            return {"success": False, "error": "기본/활성/베이스 모델을 찾을 수 없습니다. 먼저 기본 모델을 지정하거나 모델을 로드하세요."}

    # 3) 생성
    answer = _simple_generate(prompt_text, model_name, body.max_tokens, body.temperature)

    # 4) acc 계산
    acc = _acc_from_prompt_and_answer(prompt_text, answer)

    # 5) 메타/매핑 찾기 + 저장
    conn = _connect()
    cur = conn.cursor()
    try:
        row = _lookup_model_by_name(model_name)
        llm_id = int(row["id"]) if row else None
        mapping_id = _find_mapping_id(conn, body.promptId, model_name)
        rag_json = json.dumps(body.ragRefs or [], ensure_ascii=False)

        cur.execute("""
            INSERT INTO llm_eval_runs(mapping_id, llm_id, prompt_id, category, subcategory, model_name,
                                      prompt_text, user_prompt, rag_refs, answer_text, acc_score, meta)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            mapping_id, llm_id, body.promptId, category, (subcat or tmpl["name"]), model_name,
            prompt_text, user_prompt_text, rag_json, answer, acc, None
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
        "subcategory": (subcat or tmpl["name"]),
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
                 acc_score, created_at, user_prompt, prompt_text, answer_text, rag_refs
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
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(int(q.limit))

        cur.execute(sql, tuple(params))
        rows = cur.fetchall()
        items = []
        for r in rows:
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

def select_model_for_task(category: str, subcategory: Optional[str] = None) -> Optional[str]:
    """
    우선순위:
      1) llm_task_defaults 매핑
      2) 해당 과업(및 서브테스크) 활성 모델
      3) category='all' 활성 베이스
    반환: 모델 name (없으면 None)
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

        # 2) task-specific active model (subcategory 로직 제거됨)

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

def get_selected_model(q: SelectModelQuery) -> Dict[str, Any]:
    """
    요구사항: 테스크별 default model은 '프롬프트 테이블'에서 확인.
    절차:
      1) system_prompt_template에서 category(및 선택적 name=subcategory) + is_default=1 템플릿 1건을 고른다.
      2) 해당 템플릿(prompt_id)에 연결된 llm_prompt_mapping 중 rouge_score가 가장 높은 1건을 선택
      3) 그 llm_id로 llm_models 조회 후 모델 메타 반환
    """
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

        # 3) llm_models 메타
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

        # 스키마 제약 확인(참고 메시지)
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

def test_prompt(prompt_id: int, body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    프롬프트 테스트 실행:
      1) 템플릿/변수 로드 후 치환
      2) 카테고리별 활성모델 선택
      3) 로컬 모델 추론(가능 시) -> 답변 생성
      4) event_logs에 평가 JSON 저장(rougeScore는 제공되면 사용, 아니면 0)
    요청 body 예시:
      {
        "variables": {"date":"2025-01-01", "location":"서울"},
        "category": "summary",
        "modelName": "Qwen2.5-7B-RAG-FT",
        "reference": "정답 텍스트(있을 때만)",
        "max_tokens": 512,
        "temperature": 0.7
      }
    """
    body = body or {}
    variables = body.get("variables", {}) or {}
    category = body.get("category") or "summary"
    max_tokens = int(body.get("max_tokens", 512))
    temperature = float(body.get("temperature", 0.7))

    tmpl, tmpl_vars = _fetch_prompt_full(prompt_id)
    subcategory = (tmpl["name"] if isinstance(tmpl, dict) else getattr(tmpl, "name", None)) or None

    if body.get("modelName"):
        model_name = body["modelName"]
    else:
        model_name = select_model_for_task(category, subcategory)
        if not model_name:
            return {"success": False, "error": "기본/활성/베이스 모델을 찾을 수 없습니다. 먼저 기본 모델을 지정하거나 모델을 로드하세요."}
    required = json.loads(tmpl["required_vars"] or "[]")
    missing = [k for k in required if k not in variables]
    if missing:
        return {"success": False, "error": f"필수 변수 누락: {', '.join(missing)}"}

    system_prompt_text = _fill_template(tmpl["content"], variables)
    user_prompt_raw = (tmpl.get("sub_content") if isinstance(tmpl, dict) else getattr(tmpl, "sub_content", None)) or ""
    user_prompt_text = _fill_template(user_prompt_raw, variables)
    prompt_text = (system_prompt_text + ("\n" + user_prompt_text if user_prompt_text else "")).strip()

    answer = _simple_generate(prompt_text, model_name, max_tokens=max_tokens, temperature=temperature)

    rouge = 0
    ref = body.get("reference")
    if isinstance(ref, str) and ref.strip():
        try:
            ref_tokens = ref.strip().split()
            ans_tokens = (answer or "").strip().split()
            if ref_tokens:
                overlap = len(set(ref_tokens) & set(ans_tokens))
                rouge = int(100 * overlap / len(set(ref_tokens)))
        except Exception:
            rouge = 0

    row = _lookup_model_by_name(model_name)
    model_id = int(row["id"]) if row else None

    meta = {
        "category": category,
        "subcategory": subcategory,
        "promptId": prompt_id,
        "promptText": prompt_text,
        "variables": variables,
        "modelId": model_id,
        "modelName": model_name,
        "answer": answer,
        "rougeScore": rouge,
    }
    conn = _connect()
    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO event_logs(event, metadata, user_id, occurred_at) VALUES(?,?,NULL,CURRENT_TIMESTAMP)",
            ("model_eval", _json(meta)),
        )
        conn.commit()
    except Exception:
        logging.getLogger(__name__).exception("failed to insert model_eval event")
    finally:
        conn.close()

    return {"success": True, "result": "테스트 실행 완료", "promptId": prompt_id, "answer": answer, "rougeScore": rouge}
