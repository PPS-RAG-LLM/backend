# /home/work/CoreIQ/yb/backend/repository/users/summary_templates.py
from typing import List, Dict, Any, Optional
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from utils.database import get_session
from storage.db_models import SystemPromptTemplate
from errors import DatabaseError

def repo_list_qa_templates() -> List[Dict[str, str]]:
    with get_session() as session:
        stmt = (
            select(
                SystemPromptTemplate.id,
                SystemPromptTemplate.name,
                SystemPromptTemplate.system_prompt,
                SystemPromptTemplate.user_prompt,
            )
            .where(
                SystemPromptTemplate.category == "qa",
                SystemPromptTemplate.is_active == True,
            )
            .order_by(SystemPromptTemplate.id.desc())
        )
        rows = session.execute(stmt).all()
        return [{"id": r.id, "name": r.name, "system_prompt": r.system_prompt, "user_prompt":r.user_prompt} for r in rows]
        

def repo_create_qa_template(
    system_prompt: str, user_prompt: Optional[str]=""
) -> Dict[str, Any]:
    with get_session() as session:
        """QnA 시스템프롬프트 생성"""
        try:
            template = SystemPromptTemplate(
                name="qa_prompt",
                category="qa",
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                is_default=False,
                is_active=True,
            )
            session.add(template)
            session.commit()

            return {
                "id": int(template.id),
                "name": template.name,
                "system_prompt": template.system_prompt,
                "is_default": bool(template.is_default),
                "is_active": bool(template.is_active),
            }
        except IntegrityError as exc:
            session.rollback()
            raise DatabaseError(f"QA Prompt template create failed: {exc}") from exc
