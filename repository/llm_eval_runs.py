from __future__ import annotations
from typing import Optional, Dict, Any, List
from sqlalchemy import select, delete, func
from sqlalchemy.exc import IntegrityError
from storage.db_models import LlmEvalRun
from utils.database import get_session
from utils import logger

logger = logger(__name__)

def find_reusable_run(
    category: str,
    subcategory: Optional[str],
    prompt_id: int,
    model_name: str,
    user_prompt: Optional[str],
    pdf_json: str
) -> Optional[Dict[str, Any]]:
    """
    재사용 가능한 이전 평가 실행 결과를 찾습니다.
    동일 키(카테고리/서브/프롬프트/모델/유저프롬프트/PDF목록)가 완전 일치하는 최신 항목 1개를 반환합니다.
    """
    stmt = (
        select(LlmEvalRun)
        .where(
            LlmEvalRun.category == category,
            LlmEvalRun.prompt_id == prompt_id,
            LlmEvalRun.model_name == model_name,
        )
        .order_by(LlmEvalRun.id.desc())
    )

    # 동적 필터링 (SQLAlchemy 식)
    # subcategory
    sub = (subcategory or "").strip().lower()
    if sub:
        stmt = stmt.where(func.lower(func.coalesce(LlmEvalRun.subcategory, "")) == sub)
    else:
        stmt = stmt.where(func.coalesce(LlmEvalRun.subcategory, "") == "")
    
    # user_prompt
    usr = (user_prompt or "").strip()
    if usr:
        stmt = stmt.where(func.coalesce(LlmEvalRun.user_prompt, "") == usr)
    else:
        stmt = stmt.where(func.coalesce(LlmEvalRun.user_prompt, "") == "")

    # pdf_list (JSON string exact match)
    stmt = stmt.where(func.coalesce(LlmEvalRun.pdf_list, "[]") == pdf_json)

    with get_session() as session:
        row = session.execute(stmt.limit(1)).scalars().first()
        if not row:
            return None
        
        return {
            "id": row.id,
            "answer_text": row.answer_text,
            "acc_score": row.acc_score,
            "rag_refs": row.rag_refs,
            "pdf_list": row.pdf_list,
            "created_at": row.created_at,
            "user_prompt": row.user_prompt,
            "prompt_text": row.prompt_text,
        }


def insert_llm_eval_run(
    llm_id: int,
    prompt_id: int,
    category: str,
    subcategory: Optional[str],
    model_name: str,
    prompt_text: str,
    user_prompt: Optional[str],
    rag_json: str,
    answer: str,
    acc: float,
    meta_json: str,
    pdf_json: str
) -> int:
    """평가 실행 결과를 DB에 저장하고 생성된 ID를 반환합니다."""
    with get_session() as session:
        try:
            obj = LlmEvalRun(
                mapping_id=None,
                llm_id=llm_id,
                prompt_id=prompt_id,
                category=category,
                subcategory=subcategory,
                model_name=model_name,
                prompt_text=prompt_text,
                user_prompt=user_prompt,
                rag_refs=rag_json,
                answer_text=answer,
                acc_score=acc,
                meta=meta_json,
                pdf_list=pdf_json
            )
            session.add(obj)
            session.commit()
            session.refresh(obj)
            return obj.id
        except IntegrityError as e:
            session.rollback()
            logger.error(f"insert_llm_eval_run failed: {e}")
            raise

def delete_past_eval_runs(
    run_id: Optional[int] = None,
    category: Optional[str] = None,
    subcategory: Optional[str] = None,
    prompt_id: Optional[int] = None,
    model_name: Optional[str] = None
) -> int:
    """조건에 맞는 과거 평가 기록을 삭제하고 삭제된 건수를 반환합니다."""
    with get_session() as session:
        stmt = delete(LlmEvalRun)
        
        has_cond = False
        if run_id:
            stmt = stmt.where(LlmEvalRun.id == run_id)
            has_cond = True
        else:
            if category:
                stmt = stmt.where(LlmEvalRun.category == category)
                has_cond = True
            if subcategory is not None:
                sub = subcategory.strip().lower()
                if sub:
                    stmt = stmt.where(func.lower(func.coalesce(LlmEvalRun.subcategory, "")) == sub)
                else:
                    stmt = stmt.where(func.coalesce(LlmEvalRun.subcategory, "") == "")
                has_cond = True
            if prompt_id is not None:
                stmt = stmt.where(LlmEvalRun.prompt_id == prompt_id)
                has_cond = True
            if model_name is not None:
                stmt = stmt.where(LlmEvalRun.model_name == model_name)
                has_cond = True
            
        if not has_cond:
            return 0

        result = session.execute(stmt)
        session.commit()
        return result.rowcount